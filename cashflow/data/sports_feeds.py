"""Sports data feeds for Kalshi sports contract trading.

Provides four clients:
  - KalshiSportsClient: Fetches and parses sports contracts from Kalshi,
    reusing the RSA auth infrastructure from KalshiWeatherClient.
  - NBAStatsClient: Fetches free NBA stats from balldontlie.io API.
  - NCAStatsClient: Fetches college basketball ratings from public sources.
  - MLBStatsClient: Fetches MLB stats from the free MLB Stats API.

The KalshiSportsClient shares auth logic with KalshiWeatherClient to avoid
code duplication — both inherit signing/auth from the same base methods.
"""

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import requests

from cashflow.data.weather_feeds import KalshiWeatherClient
from cashflow.utils.config import get_kalshi_credentials

log = logging.getLogger("cashflow.sports")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SportsContract:
    """A parsed Kalshi sports contract."""
    ticker: str
    title: str
    contract_type: str           # game_winner, player_prop, spread, over_under
    sport: str                   # nba, ncaa, nfl, etc.
    yes_price: float             # 0.0-1.0
    no_price: float              # 0.0-1.0
    volume: int
    status: str
    expiration: str              # ISO date string
    # Parsed details (depend on contract_type)
    team_a: str                  # e.g. "Cleveland"
    team_b: str                  # e.g. "Detroit"
    player_name: str             # e.g. "Donovan Mitchell"
    stat_type: str               # e.g. "points", "rebounds", "assists"
    threshold: float             # e.g. 25.0 for "25+ points"
    spread: float                # e.g. 12.5 for "wins by over 12.5"
    total: float                 # e.g. 220.5 for over/under
    home_team: str
    away_team: str
    event_ticker: str
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class PlayerStats:
    """Season and recent stats for an NBA player."""
    player_id: int
    name: str
    team: str
    games_played: int
    # Season averages
    pts_avg: float
    reb_avg: float
    ast_avg: float
    min_avg: float
    # Computed
    pts_std: float               # Standard deviation of points
    reb_std: float
    ast_std: float
    # Recent form (last 10 games)
    recent_pts_avg: float
    recent_reb_avg: float
    recent_ast_avg: float
    recent_games: int


@dataclass
class TeamStats:
    """Team-level stats for game winner / spread / total models."""
    team_id: int
    name: str
    abbreviation: str
    wins: int
    losses: int
    win_pct: float
    # Derived or fetched
    avg_points_for: float        # Offensive rating proxy
    avg_points_against: float    # Defensive rating proxy
    point_differential: float
    home_record: str             # e.g. "20-5"
    away_record: str             # e.g. "15-10"
    pace: float                  # Possessions per game estimate


# ---------------------------------------------------------------------------
# NBA team name normalization
# ---------------------------------------------------------------------------

NBA_TEAM_ALIASES = {
    "hawks": "Atlanta Hawks", "celtics": "Boston Celtics",
    "nets": "Brooklyn Nets", "hornets": "Charlotte Hornets",
    "bulls": "Chicago Bulls", "cavaliers": "Cleveland Cavaliers",
    "cavs": "Cleveland Cavaliers", "cleveland": "Cleveland Cavaliers",
    "mavericks": "Dallas Mavericks", "mavs": "Dallas Mavericks",
    "nuggets": "Denver Nuggets", "pistons": "Detroit Pistons",
    "detroit": "Detroit Pistons", "warriors": "Golden State Warriors",
    "rockets": "Houston Rockets", "pacers": "Indiana Pacers",
    "clippers": "LA Clippers", "lakers": "Los Angeles Lakers",
    "grizzlies": "Memphis Grizzlies", "heat": "Miami Heat",
    "bucks": "Milwaukee Bucks", "timberwolves": "Minnesota Timberwolves",
    "wolves": "Minnesota Timberwolves", "pelicans": "New Orleans Pelicans",
    "knicks": "New York Knicks", "thunder": "Oklahoma City Thunder",
    "okc": "Oklahoma City Thunder", "magic": "Orlando Magic",
    "76ers": "Philadelphia 76ers", "sixers": "Philadelphia 76ers",
    "suns": "Phoenix Suns", "blazers": "Portland Trail Blazers",
    "trail blazers": "Portland Trail Blazers",
    "kings": "Sacramento Kings", "spurs": "San Antonio Spurs",
    "raptors": "Toronto Raptors", "jazz": "Utah Jazz",
    "wizards": "Washington Wizards",
}


MLB_TEAM_ALIASES = {
    "diamondbacks": "Arizona Diamondbacks", "dbacks": "Arizona Diamondbacks",
    "braves": "Atlanta Braves", "orioles": "Baltimore Orioles",
    "red sox": "Boston Red Sox", "cubs": "Chicago Cubs",
    "white sox": "Chicago White Sox", "reds": "Cincinnati Reds",
    "guardians": "Cleveland Guardians", "rockies": "Colorado Rockies",
    "tigers": "Detroit Tigers", "astros": "Houston Astros",
    "royals": "Kansas City Royals", "angels": "Los Angeles Angels",
    "dodgers": "Los Angeles Dodgers", "marlins": "Miami Marlins",
    "brewers": "Milwaukee Brewers", "twins": "Minnesota Twins",
    "mets": "New York Mets", "yankees": "New York Yankees",
    "athletics": "Oakland Athletics", "a's": "Oakland Athletics",
    "phillies": "Philadelphia Phillies", "pirates": "Pittsburgh Pirates",
    "padres": "San Diego Padres", "giants": "San Francisco Giants",
    "mariners": "Seattle Mariners", "cardinals": "St. Louis Cardinals",
    "rays": "Tampa Bay Rays", "rangers": "Texas Rangers",
    "blue jays": "Toronto Blue Jays", "nationals": "Washington Nationals",
}


def _normalize_team(name: str) -> str:
    """Normalize a team name fragment to a full team name (NBA or MLB)."""
    name_lower = name.strip().lower()
    # Direct alias match (check both dicts)
    if name_lower in NBA_TEAM_ALIASES:
        return NBA_TEAM_ALIASES[name_lower]
    if name_lower in MLB_TEAM_ALIASES:
        return MLB_TEAM_ALIASES[name_lower]
    # Partial match
    for alias, full in NBA_TEAM_ALIASES.items():
        if alias in name_lower or name_lower in full.lower():
            return full
    for alias, full in MLB_TEAM_ALIASES.items():
        if alias in name_lower or name_lower in full.lower():
            return full
    return name.strip()


# ---------------------------------------------------------------------------
# Kalshi Sports Client
# ---------------------------------------------------------------------------

# Patterns for parsing Kalshi sports contract titles
_PLAYER_PROP_PATTERN = re.compile(
    r"(?:yes\s+)?(.+?):\s*(\d+\.?\d*)\+?\s*(points?|rebounds?|assists?|pts|reb|ast|3-?pointers?)?",
    re.IGNORECASE,
)
_GAME_WINNER_PATTERN = re.compile(
    r"(?:yes\s+)?(.+?)\s+(?:wins?|beats?|defeats?|over)\b",
    re.IGNORECASE,
)
_SPREAD_PATTERN = re.compile(
    r"(.+?)\s+wins?\s+by\s+(?:over|more than)\s+(\d+\.?\d*)\s*(?:points?)?",
    re.IGNORECASE,
)
_OVER_UNDER_PATTERN = re.compile(
    r"(?:over|under)\s+(\d+\.?\d*)\s*(?:points?\s+scored|total)?",
    re.IGNORECASE,
)

# Sport detection keywords
_SPORT_KEYWORDS = {
    "nba": ["nba", "pro basketball",
            # All 30 NBA teams (full names)
            "cavaliers", "celtics", "lakers", "warriors", "nuggets",
            "bucks", "heat", "knicks", "nets", "bulls", "76ers", "suns",
            "mavericks", "thunder", "clippers", "rockets", "grizzlies",
            "pacers", "hawks", "pistons", "magic", "hornets", "wizards",
            "pelicans", "kings", "spurs", "blazers", "raptors", "jazz",
            "timberwolves",
            # City / market names
            "cleveland", "detroit", "boston", "golden state", "milwaukee",
            "miami heat", "new york knicks", "brooklyn", "chicago bulls",
            "phoenix suns", "dallas mavericks", "oklahoma city thunder",
            "la clippers", "los angeles lakers", "houston rockets",
            "memphis grizzlies", "indiana pacers", "atlanta hawks",
            "orlando magic", "charlotte hornets", "washington wizards",
            "new orleans pelicans", "sacramento kings", "san antonio spurs",
            "portland trail blazers", "toronto raptors", "utah jazz",
            "minnesota timberwolves", "denver nuggets",
            # Star players for contract matching
            "lebron james", "stephen curry", "luka doncic", "giannis",
            "nikola jokic", "joel embiid", "jayson tatum",
            "shai gilgeous-alexander", "anthony edwards", "donovan mitchell",
            "kevin durant", "jimmy butler", "damian lillard",
            "devin booker", "james harden", "anthony davis"],
    "ncaa": ["ncaa", "college basketball", "march madness", "tournament",
             "sweet 16", "elite 8", "final four", "first four",
             "college", "acc", "big ten", "big 12", "big east", "sec",
             # SEC teams
             "alabama", "auburn", "arkansas", "florida", "georgia",
             "kentucky", "lsu", "mississippi state", "ole miss",
             "missouri", "oklahoma", "south carolina", "tennessee",
             "texas", "texas a&m", "vanderbilt",
             # Big Ten / Big 12 / ACC / Big East
             "michigan", "michigan state", "ohio state", "purdue",
             "indiana", "illinois", "iowa", "wisconsin", "minnesota",
             "kansas", "baylor", "texas tech", "houston", "cincinnati",
             "duke", "unc", "north carolina", "virginia", "clemson",
             "louisville", "syracuse", "villanova", "creighton",
             "marquette", "st. john's", "uconn", "xavier",
             # Other top programs
             "gonzaga", "arizona", "ucla", "usc", "oregon",
             "memphis", "dayton", "san diego state"],
    "pga": ["pga", "golf", "golfer", "players championship", "masters",
            "us open golf", "open championship", "pga championship",
            "ryder cup", "tour championship", "fedex cup",
            "scottie scheffler", "rory mcilroy", "jon rahm",
            "xander schauffele", "collin morikawa", "bryson dechambeau",
            "wyndham clark", "viktor hovland", "patrick cantlay",
            "ludvig aberg", "sahith theegala", "max homa",
            "jordan spieth", "justin thomas", "brooks koepka",
            "tpc sawgrass", "augusta national"],
    "nfl": ["nfl", "football", "quarterback", "touchdown", "passing yards"],
    "tennis": ["tennis", "atp", "wta", "grand slam", "sinner", "alcaraz",
               "djokovic", "medvedev", "swiatek", "sabalenka", "wimbledon",
               "us open", "french open", "australian open"],
    "soccer": ["soccer", "football match", "premier league", "la liga",
               "champions league", "mls", "serie a", "bundesliga",
               "manchester", "liverpool", "arsenal", "chelsea", "real madrid",
               "barcelona", "bayern", "psg", "inter milan"],
    "mlb": ["mlb", "baseball", "strikeouts", "home runs", "innings",
            "pitcher", "batter", "hits", "rbi", "era",
            "diamondbacks", "braves", "orioles", "red sox", "cubs",
            "white sox", "reds", "guardians", "rockies", "astros",
            "royals", "angels", "dodgers", "marlins", "brewers",
            "twins", "mets", "yankees", "athletics", "phillies",
            "pirates", "padres", "giants", "mariners", "cardinals",
            "rays", "rangers", "blue jays", "nationals"],
}

# Ticker prefix patterns for sports
_SPORTS_TICKER_PREFIXES = [
    "KXMVESPORTS", "KXMVECROSS", "KXNBA", "KXNCAA", "KXNFL", "KXMLB",
    "KXPGA", "KXGOLF",
    "NBA", "NCAA", "NFL", "SPORTS", "PGA", "GOLF",
]


class KalshiSportsClient(KalshiWeatherClient):
    """Kalshi client specialized for sports contracts.

    Inherits RSA auth, _get(), _sign_request(), and authenticate() from
    KalshiWeatherClient. Only adds sports-specific fetching and parsing.
    """

    def get_sports_events(self) -> list:
        """Fetch sports-related events from Kalshi.

        Searches events by category and keyword filtering for sports.
        Returns raw event dicts from the Kalshi API.
        """
        try:
            all_events = []
            # Fetch with category filter if API supports it
            for category in ["sports", "entertainment"]:
                params = {"limit": 200, "status": "open"}
                try:
                    resp = self._get("/events", params=params)
                    if resp.status_code == 200:
                        events = resp.json().get("events", [])
                        all_events.extend(events)
                        break  # Got all events, filter below
                except Exception:
                    continue

            # Deduplicate by event_ticker
            seen = set()
            unique = []
            for event in all_events:
                ticker = event.get("event_ticker", "")
                if ticker not in seen:
                    seen.add(ticker)
                    unique.append(event)

            # Log categories for debugging
            categories = {}
            for event in unique:
                cat = event.get("category", "unknown")
                categories[cat] = categories.get(cat, 0) + 1
            log.info(f"Kalshi event categories: {dict(sorted(categories.items(), key=lambda x: -x[1]))}")

            # Filter for sports
            sports_events = []
            for event in unique:
                if self._is_sports_event(event):
                    sports_events.append(event)

            log.info(f"Found {len(sports_events)} sports events out of "
                     f"{len(unique)} total events")
            return sports_events

        except requests.RequestException as e:
            log.error(f"Kalshi sports events fetch failed: {e}")
            return []

    def _is_sports_event(self, event: dict) -> bool:
        """Determine if an event is sports-related."""
        ticker = event.get("event_ticker", "").upper()
        title = event.get("title", "").lower()
        category = event.get("category", "").lower()
        sub_title = event.get("sub_title", "").lower()
        text = f"{title} {sub_title}"

        # Category check (Kalshi uses these category strings)
        sports_categories = {
            "sports", "nba", "ncaa", "nfl", "mlb", "tennis", "soccer",
            "pga", "golf", "sports & gaming",
        }
        if category in sports_categories:
            return True

        # Ticker prefix check (most reliable)
        if any(ticker.startswith(p) for p in _SPORTS_TICKER_PREFIXES):
            return True

        # Keyword check — require at least 2 keyword hits to reduce false positives
        # on events like "Will Britain win..." or "Johnny Depp casted..."
        for sport, keywords in _SPORT_KEYWORDS.items():
            hits = sum(1 for kw in keywords if kw in text)
            if hits >= 2:
                return True
            # Single hit OK for very specific keywords (league names)
            specific = {"nba", "ncaa", "nfl", "mlb", "pga", "golf",
                        "atp", "wta", "premier league", "la liga",
                        "champions league", "march madness", "mls",
                        "players championship", "masters", "pga championship"}
            if hits == 1 and any(kw in text for kw in specific):
                return True

        return False

    def get_sports_contracts(self, event_ticker: str = None,
                             sport_filter: str = None) -> list[SportsContract]:
        """Fetch and parse sports contracts from Kalshi.

        Args:
            event_ticker: If provided, fetch only markets for this event.
            sport_filter: If provided, only return contracts for this sport
                         ("nba", "ncaa", etc.).

        Returns list of parsed SportsContract objects, filtering out
        multi-leg parlays and focusing on single-outcome contracts.
        """
        raw_markets = []

        if event_ticker:
            raw_markets = self._fetch_sports_markets(event_ticker=event_ticker)
        else:
            events = self.get_sports_events()
            for event in events:
                ticker = event.get("event_ticker", "")
                if ticker:
                    markets = self._fetch_sports_markets(event_ticker=ticker)
                    raw_markets.extend(markets)
                    time.sleep(0.2)  # Rate limit

        # Parse into structured objects
        parsed = []
        skipped_no_price = 0
        skipped_parlay = 0
        for m in raw_markets:
            contract = self._parse_sports_market(m)
            if contract is None:
                continue
            # Filter out contracts with no price data
            if contract.yes_price <= 0:
                skipped_no_price += 1
                log.info(f"  SKIPPED (no price): {contract.title[:60]} | "
                         f"raw keys: {list(m.keys())[:15]} | "
                         f"yes_price={m.get('yes_price')} "
                         f"last_price={m.get('last_price')} "
                         f"yes_bid={m.get('yes_bid')} "
                         f"yes_ask={m.get('yes_ask')} "
                         f"previous_yes_price={m.get('previous_yes_price')} "
                         f"floor_strike={m.get('floor_strike')} "
                         f"cap_strike={m.get('cap_strike')}")
                continue
            # Filter out multi-leg parlays (title contains "AND" or multiple bets)
            if self._is_parlay(contract):
                skipped_parlay += 1
                continue
            if sport_filter and contract.sport != sport_filter.lower():
                continue
            parsed.append(contract)

        log.info(f"Found {len(parsed)} tradeable sports contracts "
                 f"(skipped {skipped_no_price} no-price, {skipped_parlay} parlays)")
        return parsed

    def _fetch_sports_markets(self, event_ticker: str = None) -> list:
        """Fetch raw market data for sports events."""
        try:
            params = {"limit": 200, "status": "open"}
            if event_ticker:
                params["event_ticker"] = event_ticker

            resp = self._get("/markets", params=params)
            if resp.status_code != 200:
                log.debug(f"Kalshi sports markets fetch failed ({resp.status_code})")
                return []

            return resp.json().get("markets", [])
        except requests.RequestException as e:
            log.error(f"Kalshi sports markets fetch failed: {e}")
            return []

    def _parse_sports_market(self, data: dict) -> Optional[SportsContract]:
        """Parse a raw Kalshi market dict into a SportsContract."""
        title = data.get("title", "") or ""
        subtitle = data.get("subtitle", "") or ""
        ticker = data.get("ticker", "")
        event_ticker = data.get("event_ticker", "")
        text = f"{title} {subtitle}".strip()

        if not text:
            return None

        # Detect sport
        sport = self._detect_sport(text, ticker, data)
        if not sport:
            return None

        # Detect contract type and parse details
        contract_type, details = self._classify_contract(text)

        # Parse prices (Kalshi returns cents 0-100)
        yes_price = float(data.get("yes_price", 0) or 0)
        no_price = float(data.get("no_price", 0) or 0)
        if yes_price > 1:
            yes_price /= 100.0
        if no_price > 1:
            no_price /= 100.0

        # Fallback pricing
        if yes_price == 0:
            last = float(data.get("last_price", 0) or 0)
            if last > 1:
                yes_price = last / 100.0
            elif last > 0:
                yes_price = last

        if yes_price == 0:
            yes_bid = float(data.get("yes_bid", 0) or 0)
            yes_ask = float(data.get("yes_ask", 0) or 0)
            if yes_bid > 1:
                yes_bid /= 100.0
            if yes_ask > 1:
                yes_ask /= 100.0
            if yes_bid > 0 and yes_ask > 0:
                yes_price = (yes_bid + yes_ask) / 2.0

        if no_price == 0 and yes_price > 0:
            no_price = round(1.0 - yes_price, 4)

        # Parse expiration
        expiration = ""
        for tf in ["close_time", "expiration_time", "expected_expiration_time"]:
            ts = data.get(tf, "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    expiration = dt.strftime("%Y-%m-%d %H:%M")
                    break
                except (ValueError, TypeError):
                    continue

        return SportsContract(
            ticker=ticker,
            title=text,
            contract_type=contract_type,
            sport=sport,
            yes_price=round(yes_price, 4),
            no_price=round(no_price, 4),
            volume=int(data.get("volume", 0) or 0),
            status=data.get("status", ""),
            expiration=expiration,
            team_a=details.get("team_a", ""),
            team_b=details.get("team_b", ""),
            player_name=details.get("player_name", ""),
            stat_type=details.get("stat_type", "points"),
            threshold=details.get("threshold", 0.0),
            spread=details.get("spread", 0.0),
            total=details.get("total", 0.0),
            home_team=details.get("home_team", ""),
            away_team=details.get("away_team", ""),
            event_ticker=event_ticker,
            raw=data,
        )

    def _detect_sport(self, text: str, ticker: str, data: dict) -> str:
        """Detect which sport a contract belongs to."""
        combined = f"{text} {ticker}".lower()
        category = data.get("category", "").lower()

        if category in ("nba", "ncaa", "nfl", "mlb", "pga", "golf"):
            return "pga" if category == "golf" else category

        for sport, keywords in _SPORT_KEYWORDS.items():
            if any(kw in combined for kw in keywords):
                return sport

        # Default to empty (unknown sport)
        return ""

    def _classify_contract(self, text: str) -> tuple[str, dict]:
        """Classify a sports contract by type and extract details.

        Returns (contract_type, details_dict).
        """
        details: dict = {
            "team_a": "", "team_b": "", "player_name": "",
            "stat_type": "points", "threshold": 0.0,
            "spread": 0.0, "total": 0.0,
            "home_team": "", "away_team": "",
        }

        # 1. Over/Under total
        m = _OVER_UNDER_PATTERN.search(text)
        if m:
            details["total"] = float(m.group(1))
            return "over_under", details

        # 2. Spread
        m = _SPREAD_PATTERN.search(text)
        if m:
            details["team_a"] = _normalize_team(m.group(1))
            details["spread"] = float(m.group(2))
            return "spread", details

        # 3. Player prop (must check before game_winner since both can match names)
        m = _PLAYER_PROP_PATTERN.search(text)
        if m:
            name = m.group(1).strip()
            threshold = float(m.group(2))
            stat = (m.group(3) or "points").lower().rstrip("s")
            # Normalize stat names
            stat_map = {"pt": "points", "point": "points", "reb": "rebounds",
                        "rebound": "rebounds", "ast": "assists",
                        "assist": "assists", "3-pointer": "threes",
                        "3pointer": "threes",
                        "strikeout": "strikeouts", "k": "strikeouts",
                        "hit": "hits", "home run": "home_runs",
                        "hr": "home_runs", "rbi": "rbis",
                        "run": "runs", "base": "bases"}
            stat = stat_map.get(stat, stat)

            # Only treat as player prop if the name looks like a person
            # (at least two words, not a team name)
            if " " in name and name.lower() not in NBA_TEAM_ALIASES:
                details["player_name"] = name
                details["threshold"] = threshold
                details["stat_type"] = stat
                return "player_prop", details

        # 4. Game winner (fallback)
        m = _GAME_WINNER_PATTERN.search(text)
        if m:
            details["team_a"] = _normalize_team(m.group(1))
            return "game_winner", details

        # 5. If text starts with "yes <TeamName>", treat as game_winner
        yes_match = re.match(r"yes\s+(.+)", text, re.IGNORECASE)
        if yes_match:
            team_text = yes_match.group(1).strip()
            # Check if it has a threshold (player prop without explicit stat)
            prop_check = re.match(r"(.+?):\s*(\d+\.?\d*)\+?", team_text)
            if prop_check and " " in prop_check.group(1):
                details["player_name"] = prop_check.group(1).strip()
                details["threshold"] = float(prop_check.group(2))
                details["stat_type"] = "points"
                return "player_prop", details
            details["team_a"] = _normalize_team(team_text)
            return "game_winner", details

        # Unknown
        return "unknown", details

    def _is_parlay(self, contract: SportsContract) -> bool:
        """Check if a contract is a multi-leg parlay we should skip."""
        title_lower = contract.title.lower()
        # Multi-leg indicators
        parlay_indicators = [
            " and ", " & ", "parlay", "multi",
            "combo", "both teams", "all of",
        ]
        # Count how many independent conditions are in the title
        condition_count = title_lower.count(" and ") + title_lower.count(" & ")
        if condition_count >= 1:
            return True
        return any(ind in title_lower for ind in parlay_indicators[2:])

    def place_order(self, ticker: str, side: str, count: int,
                    price_cents: int) -> Optional[dict]:
        """Place an order on Kalshi (live trading only).

        Args:
            ticker: Contract ticker.
            side: "yes" or "no".
            count: Number of contracts.
            price_cents: Limit price in cents (1-99).

        Returns order response dict or None on failure.
        """
        path = "/trade-api/v2/portfolio/orders"
        payload = {
            "ticker": ticker,
            "action": "buy",
            "side": side,
            "count": count,
            "type": "limit",
            "yes_price" if side == "yes" else "no_price": price_cents,
        }

        try:
            import base64
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding

            url = f"{self.BASE_URL}/portfolio/orders"
            api_path = "/trade-api/v2/portfolio/orders"
            timestamp_ms = str(int(datetime.utcnow().timestamp() * 1000))
            message = timestamp_ms + "POST" + api_path
            signature = self._private_key.sign(
                message.encode(), padding.PKCS1v15(), hashes.SHA256(),
            )
            headers = {
                "KALSHI-ACCESS-KEY": self._api_key,
                "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
                "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
                "Content-Type": "application/json",
            }
            resp = self.session.post(url, json=payload, headers=headers,
                                     timeout=self.timeout)
            if resp.status_code in (200, 201):
                log.info(f"Order placed: {side} {count}x {ticker} @ {price_cents}c")
                return resp.json()
            else:
                log.error(f"Order failed ({resp.status_code}): {resp.text[:300]}")
                return None
        except Exception as e:
            log.error(f"Order placement failed: {e}")
            return None


# ---------------------------------------------------------------------------
# NBA Stats Client (balldontlie.io)
# ---------------------------------------------------------------------------

class NBAStatsClient:
    """Client for the balldontlie.io API (free NBA stats).

    API docs: https://api.balldontlie.io/v1/
    Requires a free API key set as BALLDONTLIE_API_KEY env var.

    Provides player season averages, game logs, and team info for use
    in player prop and game winner models.
    """

    BASE_URL = "https://api.balldontlie.io/v1"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.api_key = os.getenv("BALLDONTLIE_API_KEY", "")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CashFlow/1.0"})
        if self.api_key:
            self.session.headers["Authorization"] = self.api_key
        self._player_cache: dict = {}
        self._team_cache: dict = {}

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        """Make a GET request to balldontlie API."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}{path}",
                params=params,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                log.warning("balldontlie rate limited, backing off")
                time.sleep(5)
                return None
            else:
                log.debug(f"balldontlie {path} failed ({resp.status_code})")
                return None
        except requests.RequestException as e:
            log.error(f"balldontlie request failed: {e}")
            return None

    def search_player(self, name: str) -> Optional[dict]:
        """Search for a player by name."""
        if name in self._player_cache:
            return self._player_cache[name]

        data = self._get("/players", params={"search": name, "per_page": 5})
        if not data or not data.get("data"):
            return None

        # Best match: exact last name or first+last
        players = data["data"]
        name_lower = name.lower()
        for p in players:
            full = f"{p.get('first_name', '')} {p.get('last_name', '')}".lower()
            if name_lower in full or full in name_lower:
                self._player_cache[name] = p
                return p

        # Return first result as fallback
        self._player_cache[name] = players[0]
        return players[0]

    def get_season_averages(self, player_id: int,
                            season: int = None) -> Optional[dict]:
        """Get season averages for a player."""
        if season is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1

        data = self._get("/season_averages", params={
            "player_ids[]": player_id,
            "season": season,
        })
        if not data or not data.get("data"):
            return None
        return data["data"][0] if data["data"] else None

    def get_player_game_logs(self, player_id: int, season: int = None,
                             last_n: int = 10) -> list[dict]:
        """Get recent game logs for a player."""
        if season is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1

        data = self._get("/stats", params={
            "player_ids[]": player_id,
            "seasons[]": season,
            "per_page": last_n,
            "sort": "-game.date",
        })
        if not data or not data.get("data"):
            return []
        return data["data"][:last_n]

    def get_player_stats(self, player_name: str) -> Optional[PlayerStats]:
        """Get full player stats (season averages + recent form).

        This is the main method used by the strategy. Returns a PlayerStats
        object with everything needed for the player prop model.
        """
        player = self.search_player(player_name)
        if not player:
            log.debug(f"Player not found: {player_name}")
            return None

        player_id = player["id"]
        team_info = player.get("team", {})
        team_name = team_info.get("full_name", "Unknown")

        # Season averages
        averages = self.get_season_averages(player_id)
        if not averages:
            log.debug(f"No season averages for {player_name}")
            return None

        # Recent game logs for std dev and form
        game_logs = self.get_player_game_logs(player_id, last_n=20)

        pts_list = [g.get("pts", 0) for g in game_logs if g.get("pts") is not None]
        reb_list = [g.get("reb", 0) for g in game_logs if g.get("reb") is not None]
        ast_list = [g.get("ast", 0) for g in game_logs if g.get("ast") is not None]

        import numpy as np

        pts_std = float(np.std(pts_list)) if len(pts_list) >= 5 else averages.get("pts", 15) * 0.3
        reb_std = float(np.std(reb_list)) if len(reb_list) >= 5 else averages.get("reb", 5) * 0.35
        ast_std = float(np.std(ast_list)) if len(ast_list) >= 5 else averages.get("ast", 4) * 0.35

        # Recent form (last 10)
        recent_pts = pts_list[:10]
        recent_reb = reb_list[:10]
        recent_ast = ast_list[:10]

        return PlayerStats(
            player_id=player_id,
            name=player_name,
            team=team_name,
            games_played=averages.get("games_played", 0),
            pts_avg=averages.get("pts", 0.0),
            reb_avg=averages.get("reb", 0.0),
            ast_avg=averages.get("ast", 0.0),
            min_avg=float(averages.get("min", "0").split(":")[0]) if isinstance(averages.get("min"), str) else float(averages.get("min", 0)),
            pts_std=round(pts_std, 2),
            reb_std=round(reb_std, 2),
            ast_std=round(ast_std, 2),
            recent_pts_avg=round(sum(recent_pts) / len(recent_pts), 2) if recent_pts else averages.get("pts", 0.0),
            recent_reb_avg=round(sum(recent_reb) / len(recent_reb), 2) if recent_reb else averages.get("reb", 0.0),
            recent_ast_avg=round(sum(recent_ast) / len(recent_ast), 2) if recent_ast else averages.get("ast", 0.0),
            recent_games=len(recent_pts),
        )

    def get_teams(self) -> list[dict]:
        """Fetch all NBA teams."""
        data = self._get("/teams", params={"per_page": 30})
        if not data:
            return []
        return data.get("data", [])

    def get_team_stats(self, team_name: str, season: int = None) -> Optional[TeamStats]:
        """Get team-level stats for game models.

        Since balldontlie doesn't directly expose team-level aggregates,
        we approximate from recent games.
        """
        if season is None:
            season = datetime.now().year if datetime.now().month >= 10 else datetime.now().year - 1

        # Find team
        teams = self.get_teams()
        team = None
        name_lower = team_name.lower()
        for t in teams:
            full = t.get("full_name", "").lower()
            city = t.get("city", "").lower()
            tname = t.get("name", "").lower()
            if name_lower in full or name_lower == city or name_lower == tname:
                team = t
                break

        if not team:
            log.debug(f"Team not found: {team_name}")
            return None

        team_id = team["id"]

        # Fetch recent games
        today = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        data = self._get("/games", params={
            "team_ids[]": team_id,
            "seasons[]": season,
            "start_date": start,
            "end_date": today,
            "per_page": 30,
        })

        if not data or not data.get("data"):
            # Return defaults
            return TeamStats(
                team_id=team_id,
                name=team.get("full_name", team_name),
                abbreviation=team.get("abbreviation", ""),
                wins=0, losses=0, win_pct=0.5,
                avg_points_for=110.0, avg_points_against=110.0,
                point_differential=0.0,
                home_record="0-0", away_record="0-0",
                pace=100.0,
            )

        games = data["data"]
        wins = 0
        losses = 0
        pts_for = []
        pts_against = []
        home_w, home_l = 0, 0
        away_w, away_l = 0, 0

        for g in games:
            home = g.get("home_team", {})
            visitor = g.get("visitor_team", {})
            home_score = g.get("home_team_score", 0)
            visitor_score = g.get("visitor_team_score", 0)

            if not home_score or not visitor_score:
                continue

            is_home = home.get("id") == team_id
            our_score = home_score if is_home else visitor_score
            opp_score = visitor_score if is_home else home_score

            pts_for.append(our_score)
            pts_against.append(opp_score)

            if our_score > opp_score:
                wins += 1
                if is_home:
                    home_w += 1
                else:
                    away_w += 1
            else:
                losses += 1
                if is_home:
                    home_l += 1
                else:
                    away_l += 1

        total = wins + losses
        avg_for = sum(pts_for) / len(pts_for) if pts_for else 110.0
        avg_against = sum(pts_against) / len(pts_against) if pts_against else 110.0

        return TeamStats(
            team_id=team_id,
            name=team.get("full_name", team_name),
            abbreviation=team.get("abbreviation", ""),
            wins=wins,
            losses=losses,
            win_pct=round(wins / total, 3) if total > 0 else 0.5,
            avg_points_for=round(avg_for, 1),
            avg_points_against=round(avg_against, 1),
            point_differential=round(avg_for - avg_against, 1),
            home_record=f"{home_w}-{home_l}",
            away_record=f"{away_w}-{away_l}",
            pace=round((avg_for + avg_against) / 2.0 * 100.0 / 110.0, 1),
        )


# ---------------------------------------------------------------------------
# NCAA Stats Client (public data)
# ---------------------------------------------------------------------------

class NCAStatsClient:
    """Client for college basketball stats from public sources.

    Uses simplified team ratings since free college basketball data is
    harder to obtain. Falls back to reasonable defaults.
    """

    # Pre-loaded team ratings (KenPom-style adjusted efficiency margin)
    # Updated periodically — these are approximate 2024-25 rankings.
    # Positive = good team, negative = bad team.
    DEFAULT_RATINGS: dict[str, float] = {
        "gonzaga": 28.0, "houston": 27.5, "duke": 26.0, "uconn": 25.5,
        "purdue": 25.0, "tennessee": 24.5, "kansas": 24.0, "auburn": 23.5,
        "alabama": 23.0, "iowa state": 22.5, "marquette": 22.0,
        "arizona": 21.5, "north carolina": 21.0, "creighton": 20.5,
        "baylor": 20.0, "kentucky": 19.5, "michigan state": 19.0,
        "san diego state": 18.5, "illinois": 18.0, "texas": 17.5,
        "michigan": 17.0, "villanova": 16.5, "wisconsin": 16.0,
        "ohio state": 15.5, "florida": 15.0, "colorado": 14.5,
        "virginia": 14.0, "indiana": 13.5, "pitt": 13.0,
        "st johns": 12.5,
    }

    def __init__(self):
        self._ratings = dict(self.DEFAULT_RATINGS)

    def get_team_rating(self, team_name: str) -> float:
        """Get adjusted efficiency margin for a team.

        Returns a rating where 0 is average, positive is above average.
        Falls back to 0.0 (average) for unknown teams.
        """
        name_lower = team_name.lower().strip()
        # Direct match
        if name_lower in self._ratings:
            return self._ratings[name_lower]
        # Partial match
        for key, rating in self._ratings.items():
            if key in name_lower or name_lower in key:
                return rating
        return 0.0  # Unknown team = average

    def estimate_win_probability(self, team_a: str, team_b: str,
                                  home_team: str = "") -> float:
        """Estimate win probability for team_a vs team_b.

        Uses a simple logistic model based on efficiency margin difference.
        Home court advantage is ~3.5 points in college basketball.
        """
        from scipy.stats import norm

        rating_a = self.get_team_rating(team_a)
        rating_b = self.get_team_rating(team_b)

        # Adjust for home court
        hca = 0.0
        if home_team:
            home_lower = home_team.lower()
            if home_lower in team_a.lower():
                hca = 3.5
            elif home_lower in team_b.lower():
                hca = -3.5

        # Expected point differential
        expected_diff = rating_a - rating_b + hca

        # Convert to win probability using normal CDF
        # Standard deviation of game outcomes is ~11 points in college
        game_std = 11.0
        win_prob = float(norm.cdf(expected_diff / game_std))

        return round(max(0.01, min(0.99, win_prob)), 4)


# ---------------------------------------------------------------------------
# MLB Stats Client (free MLB Stats API)
# ---------------------------------------------------------------------------

@dataclass
class MLBPlayerStats:
    """Season stats for an MLB player (pitcher or batter)."""
    player_id: int
    name: str
    team: str
    position: str  # "P" for pitcher, "IF"/"OF"/"C"/"DH" for batter
    games_played: int
    # Pitcher stats
    strikeouts_per_game: float
    strikeouts_std: float
    era: float
    whip: float
    innings_per_start: float
    # Batter stats
    hits_avg: float  # hits per game
    hits_std: float
    hrs_avg: float   # home runs per game
    hrs_std: float
    rbis_avg: float
    rbis_std: float
    runs_avg: float
    runs_std: float
    batting_avg: float
    # Recent form
    recent_k_avg: float  # last 5 starts for pitchers
    recent_hits_avg: float  # last 10 games for batters


@dataclass
class MLBTeamStats:
    """Team-level stats for MLB game/total models."""
    team_id: int
    name: str
    abbreviation: str
    wins: int
    losses: int
    win_pct: float
    runs_scored_per_game: float
    runs_allowed_per_game: float
    run_differential: float
    home_record: str
    away_record: str


class MLBStatsClient:
    """Client for the free MLB Stats API (statsapi.mlb.com).

    No API key required. Provides pitcher/batter stats and team records
    for MLB player prop and game winner models.
    """

    BASE_URL = "https://statsapi.mlb.com/api/v1"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CashFlow/1.0"})
        self._player_cache: dict = {}
        self._team_cache: dict = {}

    @property
    def available(self) -> bool:
        return True  # Free API, no key needed

    def _get(self, path: str, params: dict = None) -> Optional[dict]:
        """Make a GET request to MLB Stats API."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}{path}",
                params=params,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                return resp.json()
            else:
                log.debug(f"MLB API {path} failed ({resp.status_code})")
                return None
        except requests.RequestException as e:
            log.error(f"MLB API request failed: {e}")
            return None

    def search_player(self, name: str) -> Optional[dict]:
        """Search for an MLB player by name."""
        if name in self._player_cache:
            return self._player_cache[name]

        data = self._get("/people/search", params={
            "names": name,
            "sportId": 1,  # MLB
            "hydrate": "currentTeam",
        })
        if not data or not data.get("people"):
            return None

        # Best match
        people = data["people"]
        name_lower = name.lower()
        for p in people:
            full = p.get("fullName", "").lower()
            if name_lower in full or full in name_lower:
                self._player_cache[name] = p
                return p

        self._player_cache[name] = people[0]
        return people[0]

    def get_player_stats(self, player_name: str,
                          season: int = None) -> Optional[MLBPlayerStats]:
        """Get full player stats for MLB prop models."""
        player = self.search_player(player_name)
        if not player:
            log.debug(f"MLB player not found: {player_name}")
            return None

        player_id = player["id"]
        position = player.get("primaryPosition", {}).get("abbreviation", "")
        team_name = player.get("currentTeam", {}).get("name", "Unknown")

        if season is None:
            season = datetime.now().year

        # Fetch season stats
        data = self._get(f"/people/{player_id}/stats", params={
            "stats": "season",
            "season": season,
            "group": "pitching" if position == "P" else "hitting",
        })

        if not data or not data.get("stats"):
            return None

        splits = data["stats"][0].get("splits", [])
        if not splits:
            return None

        s = splits[0].get("stat", {})
        games = int(s.get("gamesPlayed", 0) or 0)
        if games == 0:
            return None

        # Fetch game logs for std devs and recent form
        log_data = self._get(f"/people/{player_id}/stats", params={
            "stats": "gameLog",
            "season": season,
            "group": "pitching" if position == "P" else "hitting",
        })

        game_logs = []
        if log_data and log_data.get("stats"):
            game_logs = log_data["stats"][0].get("splits", [])

        if position == "P":
            return self._build_pitcher_stats(
                player_id, player_name, team_name, position,
                games, s, game_logs,
            )
        else:
            return self._build_batter_stats(
                player_id, player_name, team_name, position,
                games, s, game_logs,
            )

    def _build_pitcher_stats(self, pid, name, team, pos, games, s,
                              game_logs) -> MLBPlayerStats:
        """Build MLBPlayerStats for a pitcher."""
        import numpy as np

        total_k = int(s.get("strikeOuts", 0) or 0)
        games_started = int(s.get("gamesStarted", 0) or 0)
        innings = float(s.get("inningsPitched", "0") or 0)
        era = float(s.get("era", "4.50") or 4.50)
        whip = float(s.get("whip", "1.30") or 1.30)

        starts = max(games_started, 1)
        k_per_game = total_k / starts
        ip_per_start = innings / starts if starts > 0 else 5.5

        # Strikeout std from game logs
        k_list = []
        for gl in game_logs:
            stat = gl.get("stat", {})
            k = int(stat.get("strikeOuts", 0) or 0)
            gs = int(stat.get("gamesStarted", 0) or 0)
            if gs > 0:  # Only count starts
                k_list.append(k)

        k_std = float(np.std(k_list)) if len(k_list) >= 5 else k_per_game * 0.35
        recent_k = k_list[:5]
        recent_k_avg = sum(recent_k) / len(recent_k) if recent_k else k_per_game

        return MLBPlayerStats(
            player_id=pid, name=name, team=team, position=pos,
            games_played=games,
            strikeouts_per_game=round(k_per_game, 2),
            strikeouts_std=round(max(1.5, k_std), 2),
            era=round(era, 2), whip=round(whip, 2),
            innings_per_start=round(ip_per_start, 1),
            hits_avg=0, hits_std=0, hrs_avg=0, hrs_std=0,
            rbis_avg=0, rbis_std=0, runs_avg=0, runs_std=0,
            batting_avg=0,
            recent_k_avg=round(recent_k_avg, 2),
            recent_hits_avg=0,
        )

    def _build_batter_stats(self, pid, name, team, pos, games, s,
                             game_logs) -> MLBPlayerStats:
        """Build MLBPlayerStats for a batter."""
        import numpy as np

        hits = int(s.get("hits", 0) or 0)
        hrs = int(s.get("homeRuns", 0) or 0)
        rbis = int(s.get("rbi", 0) or 0)
        runs = int(s.get("runs", 0) or 0)
        avg = float(s.get("avg", ".250") or .250)

        hits_pg = hits / games
        hrs_pg = hrs / games
        rbis_pg = rbis / games
        runs_pg = runs / games

        # Std devs from game logs
        h_list, hr_list, rbi_list, r_list = [], [], [], []
        for gl in game_logs:
            stat = gl.get("stat", {})
            h_list.append(int(stat.get("hits", 0) or 0))
            hr_list.append(int(stat.get("homeRuns", 0) or 0))
            rbi_list.append(int(stat.get("rbi", 0) or 0))
            r_list.append(int(stat.get("runs", 0) or 0))

        hits_std = float(np.std(h_list)) if len(h_list) >= 10 else hits_pg * 0.6
        hrs_std = float(np.std(hr_list)) if len(hr_list) >= 10 else max(0.3, hrs_pg * 0.8)
        rbis_std = float(np.std(rbi_list)) if len(rbi_list) >= 10 else max(0.5, rbis_pg * 0.7)
        runs_std = float(np.std(r_list)) if len(r_list) >= 10 else max(0.4, runs_pg * 0.7)

        recent_h = h_list[:10]
        recent_hits_avg = sum(recent_h) / len(recent_h) if recent_h else hits_pg

        return MLBPlayerStats(
            player_id=pid, name=name, team=team, position=pos,
            games_played=games,
            strikeouts_per_game=0, strikeouts_std=0,
            era=0, whip=0, innings_per_start=0,
            hits_avg=round(hits_pg, 2), hits_std=round(max(0.5, hits_std), 2),
            hrs_avg=round(hrs_pg, 3), hrs_std=round(max(0.2, hrs_std), 2),
            rbis_avg=round(rbis_pg, 2), rbis_std=round(max(0.4, rbis_std), 2),
            runs_avg=round(runs_pg, 2), runs_std=round(max(0.3, runs_std), 2),
            batting_avg=round(avg, 3),
            recent_k_avg=0,
            recent_hits_avg=round(recent_hits_avg, 2),
        )

    def get_team_stats(self, team_name: str,
                        season: int = None) -> Optional[MLBTeamStats]:
        """Get MLB team stats for game winner / run total models."""
        if season is None:
            season = datetime.now().year

        # Fetch standings
        data = self._get("/standings", params={
            "leagueId": "103,104",  # AL + NL
            "season": season,
            "hydrate": "team",
        })

        if not data or not data.get("records"):
            return None

        name_lower = team_name.lower()
        for record in data["records"]:
            for entry in record.get("teamRecords", []):
                team = entry.get("team", {})
                full = team.get("name", "").lower()
                short = team.get("teamName", "").lower()
                if name_lower in full or name_lower == short:
                    wins = int(entry.get("wins", 0))
                    losses = int(entry.get("losses", 0))
                    total = wins + losses
                    rs = float(entry.get("runsScored", 0) or 0)
                    ra = float(entry.get("runsAllowed", 0) or 0)
                    rs_pg = rs / total if total > 0 else 4.5
                    ra_pg = ra / total if total > 0 else 4.5

                    home_rec = entry.get("records", {}).get("splitRecords", [])
                    home_str = "0-0"
                    away_str = "0-0"
                    for rec in home_rec:
                        if rec.get("type") == "home":
                            home_str = f"{rec.get('wins', 0)}-{rec.get('losses', 0)}"
                        elif rec.get("type") == "away":
                            away_str = f"{rec.get('wins', 0)}-{rec.get('losses', 0)}"

                    return MLBTeamStats(
                        team_id=team.get("id", 0),
                        name=team.get("name", team_name),
                        abbreviation=team.get("abbreviation", ""),
                        wins=wins, losses=losses,
                        win_pct=round(wins / total, 3) if total > 0 else 0.5,
                        runs_scored_per_game=round(rs_pg, 2),
                        runs_allowed_per_game=round(ra_pg, 2),
                        run_differential=round(rs_pg - ra_pg, 2),
                        home_record=home_str,
                        away_record=away_str,
                    )

        log.debug(f"MLB team not found: {team_name}")
        return None


# ---------------------------------------------------------------------------
# Simulated data for paper trading without API keys
# ---------------------------------------------------------------------------

def generate_simulated_sports_contracts(num_contracts: int = 15) -> list[SportsContract]:
    """Generate simulated sports contracts for paper trading.

    Used when Kalshi API credentials are unavailable. Creates realistic
    NBA player props, game winners, and over/unders with reasonable prices.
    """
    import random
    from scipy.stats import norm

    today = datetime.now().strftime("%Y-%m-%d")
    contracts = []

    # Simulated player props
    players = [
        ("Donovan Mitchell", "Cleveland Cavaliers", 25.0, 5.5),
        ("Jayson Tatum", "Boston Celtics", 27.5, 6.0),
        ("Luka Doncic", "Dallas Mavericks", 28.0, 6.5),
        ("Shai Gilgeous-Alexander", "OKC Thunder", 31.0, 5.8),
        ("Anthony Edwards", "Minnesota Timberwolves", 26.0, 5.5),
        ("James Harden", "LA Clippers", 18.5, 5.0),
        ("LeBron James", "Los Angeles Lakers", 25.5, 5.2),
        ("Stephen Curry", "Golden State Warriors", 26.5, 6.0),
    ]

    for i, (name, team, avg, std) in enumerate(players):
        # Pick a threshold near the average
        threshold = round(avg / 5) * 5
        if random.random() > 0.5:
            threshold += 5

        model_prob = 1.0 - float(norm.cdf(threshold, loc=avg, scale=std))
        noise = random.gauss(0, 0.06)
        market_price = max(0.05, min(0.95, model_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-NBA-PROP-{i}",
            title=f"yes {name}: {threshold}+",
            contract_type="player_prop",
            sport="nba",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(50, 500),
            status="open",
            expiration=f"{today} 23:59",
            team_a=team, team_b="",
            player_name=name, stat_type="points",
            threshold=float(threshold),
            spread=0.0, total=0.0,
            home_team="", away_team="",
            event_ticker=f"SIM-EVENT-{i}",
        ))

    # Simulated game winners
    matchups = [
        ("Cleveland Cavaliers", "Detroit Pistons", 0.72),
        ("Boston Celtics", "Brooklyn Nets", 0.78),
        ("Denver Nuggets", "Phoenix Suns", 0.55),
        ("LA Clippers", "Sacramento Kings", 0.52),
    ]

    for i, (team_a, team_b, true_prob) in enumerate(matchups):
        noise = random.gauss(0, 0.05)
        market_price = max(0.05, min(0.95, true_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-NBA-WIN-{i}",
            title=f"yes {team_a}",
            contract_type="game_winner",
            sport="nba",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(100, 1000),
            status="open",
            expiration=f"{today} 23:59",
            team_a=team_a, team_b=team_b,
            player_name="", stat_type="",
            threshold=0.0, spread=0.0, total=0.0,
            home_team=team_a, away_team=team_b,
            event_ticker=f"SIM-GAME-{i}",
        ))

    # Simulated over/unders
    totals = [
        ("Cleveland vs Detroit", 215.5, 0.52),
        ("Boston vs Brooklyn", 222.5, 0.58),
        ("Denver vs Phoenix", 228.5, 0.48),
    ]

    for i, (matchup, total, true_prob) in enumerate(totals):
        noise = random.gauss(0, 0.05)
        market_price = max(0.05, min(0.95, true_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-NBA-TOTAL-{i}",
            title=f"Over {total} points scored",
            contract_type="over_under",
            sport="nba",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(80, 600),
            status="open",
            expiration=f"{today} 23:59",
            team_a="", team_b="",
            player_name="", stat_type="",
            threshold=0.0, spread=0.0, total=total,
            home_team="", away_team="",
            event_ticker=f"SIM-TOTAL-{i}",
        ))

    # Simulated MLB player props (pitcher strikeouts)
    pitchers = [
        ("Gerrit Cole", "New York Yankees", 7.5, 2.2),
        ("Spencer Strider", "Atlanta Braves", 9.0, 2.5),
        ("Corbin Burnes", "Baltimore Orioles", 7.0, 2.0),
        ("Zack Wheeler", "Philadelphia Phillies", 7.5, 2.1),
    ]

    for i, (name, team, avg_k, std_k) in enumerate(pitchers):
        threshold = round(avg_k - 0.5)
        if random.random() > 0.5:
            threshold += 1.5

        model_prob = 1.0 - float(norm.cdf(threshold, loc=avg_k, scale=std_k))
        noise = random.gauss(0, 0.06)
        market_price = max(0.05, min(0.95, model_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-MLB-KPROP-{i}",
            title=f"yes {name}: {threshold}+ strikeouts",
            contract_type="player_prop",
            sport="mlb",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(50, 500),
            status="open",
            expiration=f"{today} 23:59",
            team_a=team, team_b="",
            player_name=name, stat_type="strikeouts",
            threshold=float(threshold),
            spread=0.0, total=0.0,
            home_team="", away_team="",
            event_ticker=f"SIM-MLB-EVENT-{i}",
        ))

    # Simulated MLB game winners
    mlb_matchups = [
        ("New York Yankees", "Boston Red Sox", 0.55),
        ("Los Angeles Dodgers", "San Francisco Giants", 0.62),
        ("Atlanta Braves", "New York Mets", 0.58),
    ]

    for i, (team_a, team_b, true_prob) in enumerate(mlb_matchups):
        noise = random.gauss(0, 0.05)
        market_price = max(0.05, min(0.95, true_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-MLB-WIN-{i}",
            title=f"yes {team_a}",
            contract_type="game_winner",
            sport="mlb",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(100, 1000),
            status="open",
            expiration=f"{today} 23:59",
            team_a=team_a, team_b=team_b,
            player_name="", stat_type="",
            threshold=0.0, spread=0.0, total=0.0,
            home_team=team_a, away_team=team_b,
            event_ticker=f"SIM-MLB-GAME-{i}",
        ))

    # Simulated MLB run totals
    mlb_totals = [
        ("Yankees vs Red Sox", 8.5, 0.52),
        ("Dodgers vs Giants", 7.5, 0.55),
    ]

    for i, (matchup, total, true_prob) in enumerate(mlb_totals):
        noise = random.gauss(0, 0.05)
        market_price = max(0.05, min(0.95, true_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-MLB-TOTAL-{i}",
            title=f"Over {total} runs scored",
            contract_type="over_under",
            sport="mlb",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(80, 600),
            status="open",
            expiration=f"{today} 23:59",
            team_a="", team_b="",
            player_name="", stat_type="",
            threshold=0.0, spread=0.0, total=total,
            home_team="", away_team="",
            event_ticker=f"SIM-MLB-TOTAL-{i}",
        ))

    # Simulated PGA tournament winners
    pga_golfers = [
        ("Scottie Scheffler", "Xander Schauffele", 0.58),
        ("Rory McIlroy", "Collin Morikawa", 0.52),
        ("Jon Rahm", "Bryson DeChambeau", 0.50),
    ]

    for i, (p1, p2, true_prob) in enumerate(pga_golfers):
        noise = random.gauss(0, 0.05)
        market_price = max(0.05, min(0.95, true_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-PGA-WIN-{i}",
            title=f"yes {p1} beats {p2}",
            contract_type="game_winner",
            sport="pga",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(50, 300),
            status="open",
            expiration=f"{today} 23:59",
            team_a=p1, team_b=p2,
            player_name="", stat_type="",
            threshold=0.0, spread=0.0, total=0.0,
            home_team="", away_team="",
            event_ticker=f"SIM-PGA-{i}",
        ))

    # Simulated Tennis match winners
    tennis_matches = [
        ("Jannik Sinner", "Carlos Alcaraz", 0.52),
        ("Novak Djokovic", "Daniil Medvedev", 0.60),
    ]

    for i, (p1, p2, true_prob) in enumerate(tennis_matches):
        noise = random.gauss(0, 0.05)
        market_price = max(0.05, min(0.95, true_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-TENNIS-WIN-{i}",
            title=f"yes {p1} beats {p2}",
            contract_type="game_winner",
            sport="tennis",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(50, 300),
            status="open",
            expiration=f"{today} 23:59",
            team_a=p1, team_b=p2,
            player_name="", stat_type="",
            threshold=0.0, spread=0.0, total=0.0,
            home_team="", away_team="",
            event_ticker=f"SIM-TENNIS-{i}",
        ))

    # Simulated Soccer match winners
    soccer_matches = [
        ("Manchester City", "Liverpool", 0.48),
        ("Real Madrid", "Barcelona", 0.52),
    ]

    for i, (t1, t2, true_prob) in enumerate(soccer_matches):
        noise = random.gauss(0, 0.05)
        market_price = max(0.05, min(0.95, true_prob + noise))

        contracts.append(SportsContract(
            ticker=f"SIM-SOCCER-WIN-{i}",
            title=f"yes {t1} beats {t2}",
            contract_type="game_winner",
            sport="soccer",
            yes_price=round(market_price, 3),
            no_price=round(1.0 - market_price, 3),
            volume=random.randint(100, 800),
            status="open",
            expiration=f"{today} 23:59",
            team_a=t1, team_b=t2,
            player_name="", stat_type="",
            threshold=0.0, spread=0.0, total=0.0,
            home_team=t1, away_team=t2,
            event_ticker=f"SIM-SOCCER-{i}",
        ))

    return contracts
