"""Weather data feeds for Kalshi temperature contract trading.

Provides two clients:
  - NWSClient: Fetches next-day high temperature forecasts from the
    National Weather Service API (free, no auth, requires User-Agent).
  - KalshiWeatherClient: Wraps the existing KalshiClient to fetch and
    parse temperature-related weather contracts from Kalshi.

The NWS API flow is: lat/lon -> /points -> gridpoint URL -> /forecast.
Gridpoint lookups are cached since they never change for a given lat/lon.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import requests

from cashflow.data.arbitrage_feeds import KalshiClient
from cashflow.utils.config import get_kalshi_credentials

log = logging.getLogger("cashflow.weather")

# NWS requires a descriptive User-Agent header
NWS_USER_AGENT = "(CashFlow Weather Bot, cashflow-bot@users.noreply.github.com)"

# Cache file for NWS gridpoint lookups
GRIDPOINT_CACHE_FILE = Path("data/nws_gridpoints.json")


@dataclass
class NWSForecast:
    """A parsed NWS forecast for a specific city and date."""
    city: str
    date: str                  # YYYY-MM-DD
    forecast_high: float       # Fahrenheit
    forecast_low: float
    short_forecast: str
    fetched_at: str


@dataclass
class KalshiWeatherMarket:
    """A parsed Kalshi weather/temperature contract."""
    ticker: str
    title: str
    city: str                  # City code (NYC, CHI, AUS, etc.)
    date: str                  # Settlement date YYYY-MM-DD
    bracket_type: str          # "above_X", "below_X", "between_X_Y"
    lower_bound: float         # Temperature threshold lower (F)
    upper_bound: float         # Temperature threshold upper (F)
    yes_price: float           # Current YES price (0.0-1.0)
    no_price: float            # Current NO price (0.0-1.0)
    volume: int
    status: str
    raw: dict


class NWSClient:
    """Client for the National Weather Service API.

    The NWS API is free, requires no authentication, and is the actual
    data source that Kalshi uses for contract settlement. This gives us
    a direct informational edge: we read the same source the market settles on.

    API docs: https://www.weather.gov/documentation/services-web-api
    """

    BASE_URL = "https://api.weather.gov"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": NWS_USER_AGENT,
            "Accept": "application/geo+json",
        })
        self._gridpoint_cache: dict = {}
        self._load_gridpoint_cache()

    def _load_gridpoint_cache(self):
        """Load cached gridpoint lookups from disk."""
        if GRIDPOINT_CACHE_FILE.exists():
            try:
                with open(GRIDPOINT_CACHE_FILE) as f:
                    self._gridpoint_cache = json.load(f)
                log.debug(f"Loaded {len(self._gridpoint_cache)} cached gridpoints")
            except (json.JSONDecodeError, IOError) as e:
                log.warning(f"Failed to load gridpoint cache: {e}")
                self._gridpoint_cache = {}

    def _save_gridpoint_cache(self):
        """Persist gridpoint cache to disk."""
        GRIDPOINT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(GRIDPOINT_CACHE_FILE, "w") as f:
                json.dump(self._gridpoint_cache, f, indent=2)
        except IOError as e:
            log.warning(f"Failed to save gridpoint cache: {e}")

    def get_gridpoint(self, lat: float, lon: float) -> Optional[dict]:
        """Look up the NWS gridpoint for a lat/lon pair.

        Returns dict with 'office', 'gridX', 'gridY', 'forecast_url'.
        Results are cached since gridpoints never change.
        """
        cache_key = f"{lat:.4f},{lon:.4f}"
        if cache_key in self._gridpoint_cache:
            return self._gridpoint_cache[cache_key]

        try:
            resp = self.session.get(
                f"{self.BASE_URL}/points/{lat:.4f},{lon:.4f}",
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                log.error(f"NWS /points failed ({resp.status_code}): {resp.text[:200]}")
                return None

            data = resp.json()
            props = data.get("properties", {})
            result = {
                "office": props.get("gridId", ""),
                "gridX": props.get("gridX"),
                "gridY": props.get("gridY"),
                "forecast_url": props.get("forecast", ""),
                "forecast_hourly_url": props.get("forecastHourly", ""),
            }

            self._gridpoint_cache[cache_key] = result
            self._save_gridpoint_cache()
            log.debug(f"Cached gridpoint for {cache_key}: {result['office']} "
                      f"({result['gridX']},{result['gridY']})")
            return result

        except requests.RequestException as e:
            log.error(f"NWS gridpoint lookup failed for ({lat}, {lon}): {e}")
            return None

    def get_forecast(self, lat: float, lon: float) -> list[NWSForecast]:
        """Get the 7-day forecast for a location.

        Returns a list of NWSForecast objects, one per forecast period.
        The first daytime period is typically today or tomorrow depending
        on time of day.
        """
        gridpoint = self.get_gridpoint(lat, lon)
        if not gridpoint or not gridpoint.get("forecast_url"):
            return []

        try:
            resp = self.session.get(
                gridpoint["forecast_url"],
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                log.error(f"NWS forecast failed ({resp.status_code}): {resp.text[:200]}")
                return []

            data = resp.json()
            periods = data.get("properties", {}).get("periods", [])
            forecasts = []

            for period in periods:
                # We only care about daytime periods (which have high temps)
                if not period.get("isDaytime", True):
                    continue

                # Parse the start time to get the date
                start_time = period.get("startTime", "")
                try:
                    dt = datetime.fromisoformat(start_time)
                    date_str = dt.strftime("%Y-%m-%d")
                except (ValueError, TypeError):
                    continue

                temp = period.get("temperature")
                if temp is None:
                    continue

                # NWS returns temperature in the unit specified
                unit = period.get("temperatureUnit", "F")
                temp_f = float(temp)
                if unit == "C":
                    temp_f = temp_f * 9.0 / 5.0 + 32.0

                forecasts.append(NWSForecast(
                    city="",  # Caller fills this in
                    date=date_str,
                    forecast_high=temp_f,
                    forecast_low=0.0,  # Not available in basic forecast
                    short_forecast=period.get("shortForecast", ""),
                    fetched_at=datetime.now().isoformat(),
                ))

            return forecasts

        except requests.RequestException as e:
            log.error(f"NWS forecast fetch failed: {e}")
            return []

    def get_forecast_high(self, lat: float, lon: float, target_date: str) -> Optional[float]:
        """Get the forecast high temperature for a specific date.

        Args:
            lat: Latitude
            lon: Longitude
            target_date: Date string YYYY-MM-DD

        Returns:
            Forecast high in Fahrenheit, or None if not available.
        """
        forecasts = self.get_forecast(lat, lon)
        for fc in forecasts:
            if fc.date == target_date:
                return fc.forecast_high

        if forecasts:
            log.warning(f"Target date {target_date} not found in forecast. "
                        f"Available dates: {[f.date for f in forecasts]}")
        return None


# City name patterns for parsing Kalshi contract titles
CITY_PATTERNS = {
    "NYC": [r"new york", r"nyc", r"manhattan", r"central park"],
    "CHI": [r"chicago", r"chi", r"o'?hare"],
    "MIA": [r"miami"],
    "AUS": [r"austin"],
    "LAX": [r"los angeles", r"la\b", r"lax"],
}


def _parse_city_from_title(title: str) -> str:
    """Extract city code from a Kalshi contract title."""
    title_lower = title.lower()
    for code, patterns in CITY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, title_lower):
                return code
    return ""


def _parse_temperature_bracket(title: str, subtitle: str = "") -> tuple:
    """Parse temperature bracket from Kalshi contract title/subtitle.

    Returns (bracket_type, lower_bound, upper_bound) or (None, 0, 0).

    Kalshi temperature contracts have titles like:
      "Will the high temperature in NYC be above 85F?"
      "High temp in Chicago between 70F and 75F"
      "NYC high temperature 80F or above"
    """
    text = f"{title} {subtitle}".lower()

    # Pattern: "above X" or "X or above" or "at least X" or ">= X"
    m = re.search(r"(?:above|at least|>=?)\s*(\d+)\s*[°f]?", text)
    if m:
        threshold = float(m.group(1))
        return f"above_{int(threshold)}", threshold, 200.0

    m = re.search(r"(\d+)\s*[°f]?\s*(?:or above|or higher|\+)", text)
    if m:
        threshold = float(m.group(1))
        return f"above_{int(threshold)}", threshold, 200.0

    # Pattern: "below X" or "under X" or "X or below"
    m = re.search(r"(?:below|under|<=?)\s*(\d+)\s*[°f]?", text)
    if m:
        threshold = float(m.group(1))
        return f"below_{int(threshold)}", -100.0, threshold

    m = re.search(r"(\d+)\s*[°f]?\s*(?:or below|or lower|or less)", text)
    if m:
        threshold = float(m.group(1))
        return f"below_{int(threshold)}", -100.0, threshold

    # Pattern: "between X and Y" or "X to Y" or "X-Y"
    m = re.search(r"(?:between\s+)?(\d+)\s*[°f]?\s*(?:and|to|-)\s*(\d+)\s*[°f]?", text)
    if m:
        low = float(m.group(1))
        high = float(m.group(2))
        if low > high:
            low, high = high, low
        return f"between_{int(low)}_{int(high)}", low, high

    return None, 0.0, 0.0


class KalshiWeatherClient:
    """Kalshi client specialized for weather/temperature contracts.

    Wraps the existing KalshiClient from arbitrage_feeds and adds:
      - Filtering for temperature markets
      - Parsing contract details (city, threshold, bracket type)
      - Authentication for price fetching

    Kalshi API v2: https://api.elections.kalshi.com/trade-api/v2
    Auth via KALSHI_EMAIL and KALSHI_PASSWORD env vars.
    """

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "CashFlow/1.0"})
        self._token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._authenticated = False

    @property
    def has_credentials(self) -> bool:
        """Check if Kalshi credentials are available."""
        creds = get_kalshi_credentials()
        return bool(creds.get("email") and creds.get("password"))

    def authenticate(self) -> bool:
        """Authenticate with Kalshi API using email/password.

        Returns True if authentication succeeded.
        """
        if self._authenticated and self._token_expiry and datetime.now() < self._token_expiry:
            return True

        creds = get_kalshi_credentials()
        email = creds.get("email", "")
        password = creds.get("password", "")

        if not email or not password:
            log.warning("No Kalshi credentials found (KALSHI_EMAIL / KALSHI_PASSWORD)")
            return False

        try:
            resp = self.session.post(
                f"{self.BASE_URL}/login",
                json={"email": email, "password": password},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                log.error(f"Kalshi auth failed ({resp.status_code}): {resp.text[:200]}")
                return False

            data = resp.json()
            self._token = data.get("token", "")
            if self._token:
                self.session.headers["Authorization"] = f"Bearer {self._token}"
                # Token typically valid for ~24 hours; refresh at 20 hours
                self._token_expiry = datetime.now() + timedelta(hours=20)
                self._authenticated = True
                log.info("Kalshi authentication successful")
                return True
            else:
                log.error("Kalshi auth response missing token")
                return False

        except requests.RequestException as e:
            log.error(f"Kalshi auth request failed: {e}")
            return False

    def get_weather_events(self) -> list:
        """Fetch weather-related events from Kalshi.

        Kalshi weather events typically have tickers like:
          HIGHNY-YYYY-MM-DD (NYC high temp)
          HIGHCHI-YYYY-MM-DD (Chicago high temp)
          etc.
        """
        try:
            # Search for temperature/weather events
            params = {
                "limit": 100,
                "status": "open",
            }
            resp = self.session.get(
                f"{self.BASE_URL}/events",
                params=params,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                log.error(f"Kalshi events failed ({resp.status_code}): {resp.text[:200]}")
                return []

            data = resp.json()
            events = data.get("events", [])

            # Filter for weather/temperature events
            weather_events = []
            weather_keywords = ["temp", "high", "weather", "degree", "fahrenheit"]
            weather_ticker_prefixes = ["HIGH", "TEMP", "WEATHER"]

            for event in events:
                ticker = event.get("event_ticker", "").upper()
                title = event.get("title", "").lower()
                category = event.get("category", "").lower()

                is_weather = (
                    category in ("weather", "climate", "temperature")
                    or any(kw in title for kw in weather_keywords)
                    or any(ticker.startswith(p) for p in weather_ticker_prefixes)
                )

                if is_weather:
                    weather_events.append(event)

            log.debug(f"Found {len(weather_events)} weather events out of {len(events)} total")
            return weather_events

        except requests.RequestException as e:
            log.error(f"Kalshi events fetch failed: {e}")
            return []

    def get_temperature_markets(self, event_ticker: str = None) -> list[KalshiWeatherMarket]:
        """Fetch temperature contract markets from Kalshi.

        If event_ticker is provided, fetches markets for that specific event.
        Otherwise, scans all weather events for temperature markets.

        Returns list of parsed KalshiWeatherMarket objects.
        """
        raw_markets = []

        if event_ticker:
            raw_markets = self._fetch_markets(event_ticker=event_ticker)
        else:
            # Fetch from all weather events
            events = self.get_weather_events()
            for event in events:
                ticker = event.get("event_ticker", "")
                if ticker:
                    markets = self._fetch_markets(event_ticker=ticker)
                    raw_markets.extend(markets)
                    time.sleep(0.2)  # Rate limit

        # Parse into structured objects
        parsed = []
        for m in raw_markets:
            parsed_market = self._parse_weather_market(m)
            if parsed_market:
                parsed.append(parsed_market)

        log.info(f"Found {len(parsed)} temperature contracts")
        return parsed

    def _fetch_markets(self, event_ticker: str = None) -> list:
        """Fetch raw market data from Kalshi API."""
        try:
            params = {"limit": 200, "status": "open"}
            if event_ticker:
                params["event_ticker"] = event_ticker

            resp = self.session.get(
                f"{self.BASE_URL}/markets",
                params=params,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                log.debug(f"Kalshi markets fetch failed ({resp.status_code})")
                return []

            return resp.json().get("markets", [])

        except requests.RequestException as e:
            log.error(f"Kalshi markets fetch failed: {e}")
            return []

    def _parse_weather_market(self, data: dict) -> Optional[KalshiWeatherMarket]:
        """Parse a raw Kalshi market into a KalshiWeatherMarket."""
        title = data.get("title", "") or data.get("subtitle", "")
        subtitle = data.get("subtitle", "")
        ticker = data.get("ticker", "")

        # Parse city
        city = _parse_city_from_title(f"{title} {subtitle} {ticker}")
        if not city:
            # Try to extract from ticker (e.g., HIGHNY -> NYC)
            ticker_upper = ticker.upper()
            ticker_city_map = {
                "NY": "NYC", "CHI": "CHI", "MIA": "MIA",
                "AUS": "AUS", "LA": "LAX", "LAX": "LAX",
            }
            for suffix, code in ticker_city_map.items():
                if suffix in ticker_upper:
                    city = code
                    break

        # Parse temperature bracket
        bracket_type, lower_bound, upper_bound = _parse_temperature_bracket(title, subtitle)
        if bracket_type is None:
            return None

        # Parse date from close_time or expiration_time
        date_str = ""
        for time_field in ["close_time", "expiration_time", "expected_expiration_time"]:
            ts = data.get(time_field, "")
            if ts:
                try:
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                    date_str = dt.strftime("%Y-%m-%d")
                    break
                except (ValueError, TypeError):
                    continue

        # Parse prices (Kalshi returns cents 0-100, we want 0.0-1.0)
        yes_price = float(data.get("yes_price", 0) or 0) / 100.0
        no_price = float(data.get("no_price", 0) or 0) / 100.0

        # If yes/no prices look like they're already 0-1, don't divide
        if data.get("yes_price", 0) and float(data.get("yes_price", 0)) <= 1.0:
            yes_price = float(data.get("yes_price", 0))
            no_price = float(data.get("no_price", 0) or 0)

        # Fallback: use last_price or yes_bid/yes_ask midpoint
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

        return KalshiWeatherMarket(
            ticker=ticker,
            title=title,
            city=city,
            date=date_str,
            bracket_type=bracket_type,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            yes_price=round(yes_price, 4),
            no_price=round(no_price, 4),
            volume=int(data.get("volume", 0) or 0),
            status=data.get("status", ""),
            raw=data,
        )

    def get_market_price(self, ticker: str) -> Optional[float]:
        """Get the current YES price for a specific contract ticker.

        Returns price as 0.0-1.0, or None if unavailable.
        """
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/markets/{ticker}",
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None

            data = resp.json().get("market", {})
            yes_price = float(data.get("yes_price", 0) or 0)
            if yes_price > 1:
                yes_price /= 100.0
            return round(yes_price, 4) if yes_price > 0 else None

        except (requests.RequestException, ValueError):
            return None
