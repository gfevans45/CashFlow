"""Polymarket data client for fetching markets, prices, and order books.

Uses two APIs:
  - Gamma API (gamma-api.polymarket.com): Market discovery, metadata, event info
  - CLOB API (clob.polymarket.com): Order books, prices, trading

No authentication needed for read-only data operations.
Trading requires a Polygon wallet with USDC.
"""
import requests
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"

# Default headers
HEADERS = {"User-Agent": "CashFlow/0.1.0"}


@dataclass
class PolyMarket:
    """Represents a single Polymarket binary market (YES/NO)."""
    condition_id: str
    question: str
    slug: str
    yes_price: float           # Current YES token price ($0.00-$1.00)
    no_price: float            # Current NO token price
    volume: float              # Total volume traded (USDC)
    volume_24h: float          # 24h volume
    liquidity: float           # Current liquidity
    end_date: Optional[str]    # When the market resolves
    category: str              # e.g., "politics", "sports", "crypto"
    active: bool
    closed: bool
    resolved: bool
    outcome: Optional[str] = None  # "Yes"/"No" if resolved
    description: str = ""
    tokens: list = field(default_factory=list)
    # Computed fields
    implied_prob: float = 0.0  # = yes_price
    spread: float = 0.0       # yes_price + no_price - 1.0 (market maker spread)


@dataclass
class OrderBookLevel:
    price: float
    size: float


@dataclass
class OrderBook:
    condition_id: str
    bids: list  # Buy orders (YES side)
    asks: list  # Sell orders (YES side)
    best_bid: float = 0.0
    best_ask: float = 0.0
    mid_price: float = 0.0
    spread: float = 0.0


class PolymarketClient:
    """Read-only client for Polymarket data."""

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _get(self, base: str, path: str, params: dict = None) -> dict:
        """Make a GET request with retry."""
        url = f"{base}{path}"
        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException:
                if attempt == 2:
                    raise
                time.sleep(1)
        return {}

    # --- Gamma API: Market Discovery ---

    def fetch_markets(
        self,
        limit: int = 100,
        offset: int = 0,
        active: bool = True,
        closed: bool = False,
        order: str = "volume24hr",
        ascending: bool = False,
    ) -> list:
        """Fetch markets from Gamma API.

        Returns list of PolyMarket objects.
        """
        params = {
            "limit": limit,
            "offset": offset,
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "order": order,
            "ascending": str(ascending).lower(),
        }

        data = self._get(GAMMA_API, "/markets", params)

        markets = []
        if isinstance(data, list):
            for m in data:
                market = self._parse_market(m)
                if market:
                    markets.append(market)

        return markets

    def fetch_all_active_markets(self, max_pages: int = 10) -> list:
        """Paginate through all active markets."""
        all_markets = []
        for page in range(max_pages):
            markets = self.fetch_markets(
                limit=100,
                offset=page * 100,
                active=True,
                closed=False,
            )
            if not markets:
                break
            all_markets.extend(markets)
            time.sleep(0.3)  # Rate limit
        return all_markets

    def fetch_market_by_slug(self, slug: str) -> Optional[PolyMarket]:
        """Fetch a specific market by its URL slug."""
        data = self._get(GAMMA_API, f"/markets/{slug}")
        if data:
            return self._parse_market(data)
        return None

    def search_markets(self, query: str, limit: int = 50) -> list:
        """Search markets by keyword."""
        params = {"_q": query, "limit": limit, "active": "true"}
        data = self._get(GAMMA_API, "/markets", params)
        markets = []
        if isinstance(data, list):
            for m in data:
                market = self._parse_market(m)
                if market:
                    markets.append(market)
        return markets

    # --- CLOB API: Prices & Order Books ---

    def fetch_price(self, token_id: str) -> dict:
        """Fetch current price for a token from CLOB."""
        data = self._get(CLOB_API, "/price", {"token_id": token_id})
        return data

    def fetch_order_book(self, token_id: str) -> Optional[OrderBook]:
        """Fetch order book for a token."""
        data = self._get(CLOB_API, "/book", {"token_id": token_id})
        if not data:
            return None

        bids = [
            OrderBookLevel(float(b["price"]), float(b["size"]))
            for b in data.get("bids", [])
        ]
        asks = [
            OrderBookLevel(float(a["price"]), float(a["size"]))
            for a in data.get("asks", [])
        ]

        best_bid = bids[0].price if bids else 0.0
        best_ask = asks[0].price if asks else 1.0
        mid = (best_bid + best_ask) / 2

        return OrderBook(
            condition_id=token_id,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid_price=mid,
            spread=best_ask - best_bid,
        )

    # --- Filters for Strategy ---

    def find_longshot_opportunities(
        self,
        max_yes_price: float = 0.10,
        min_volume: float = 1000,
        min_liquidity: float = 500,
    ) -> list:
        """Find markets where YES tokens are cheap (potential longshots).

        These are markets trading below $0.10 that might be undervalued.
        The strategy will then apply its own model to estimate true probability.
        """
        all_markets = self.fetch_all_active_markets()

        longshots = []
        for m in all_markets:
            if (
                m.yes_price <= max_yes_price
                and m.yes_price > 0.01  # Filter out dead markets
                and m.volume >= min_volume
                and m.liquidity >= min_liquidity
                and not m.resolved
                and not m.closed
            ):
                longshots.append(m)

        # Sort by volume (most liquid first)
        longshots.sort(key=lambda x: x.volume, reverse=True)
        return longshots

    def find_high_spread_markets(self, min_spread: float = 0.05) -> list:
        """Find markets with wide spreads (market making opportunities)."""
        all_markets = self.fetch_all_active_markets()
        wide_spread = [m for m in all_markets if m.spread >= min_spread and m.volume > 5000]
        wide_spread.sort(key=lambda x: x.spread, reverse=True)
        return wide_spread

    # --- Helpers ---

    def _parse_market(self, data: dict) -> Optional[PolyMarket]:
        """Parse raw API response into PolyMarket dataclass."""
        try:
            yes_price = float(data.get("outcomePrices", data.get("yes_price", "[0.5,0.5]")).__class__ == str
                and eval(data.get("outcomePrices", "[0.5,0.5]"))[0]
                or data.get("outcomePrices", [0.5, 0.5])[0]
                if isinstance(data.get("outcomePrices"), (list, str)) else 0.5)
        except Exception:
            yes_price = 0.5

        try:
            # Handle various response formats
            if "outcomePrices" in data:
                prices = data["outcomePrices"]
                if isinstance(prices, str):
                    import json
                    prices = json.loads(prices)
                if isinstance(prices, list) and len(prices) >= 2:
                    yes_price = float(prices[0])
                    no_price = float(prices[1])
                else:
                    yes_price = 0.5
                    no_price = 0.5
            elif "bestAsk" in data:
                yes_price = float(data.get("bestAsk", 0.5))
                no_price = 1.0 - yes_price
            else:
                yes_price = 0.5
                no_price = 0.5
        except Exception:
            yes_price = 0.5
            no_price = 0.5

        tokens = []
        if "clobTokenIds" in data:
            raw = data["clobTokenIds"]
            if isinstance(raw, str):
                import json
                try:
                    tokens = json.loads(raw)
                except Exception:
                    tokens = []
            elif isinstance(raw, list):
                tokens = raw

        market = PolyMarket(
            condition_id=data.get("conditionId", data.get("condition_id", "")),
            question=data.get("question", ""),
            slug=data.get("slug", ""),
            yes_price=yes_price,
            no_price=no_price,
            volume=float(data.get("volume", 0) or 0),
            volume_24h=float(data.get("volume24hr", 0) or 0),
            liquidity=float(data.get("liquidity", 0) or 0),
            end_date=data.get("endDate", data.get("end_date_iso")),
            category=data.get("groupItemTitle", data.get("category", "")),
            active=data.get("active", True),
            closed=data.get("closed", False),
            resolved=data.get("resolved", False),
            outcome=data.get("outcome"),
            description=data.get("description", "")[:500],
            tokens=tokens,
        )
        market.implied_prob = market.yes_price
        market.spread = market.yes_price + market.no_price - 1.0
        return market
