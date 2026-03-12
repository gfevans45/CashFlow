"""Data fetching for prediction market strategies.

Uses free public data sources:
- FRED (Federal Reserve Economic Data) for economic indicators
- NOAA for weather data
- Yahoo Finance for market benchmarks
"""
import requests
import pandas as pd
from datetime import datetime, timedelta


# FRED API (free, 120 req/min, get key at https://fred.stlouisfed.org/docs/api/api_key.html)
FRED_BASE = "https://api.stlouisfed.org/fred"

# Key economic series we can use to predict Kalshi events
FRED_SERIES = {
    "cpi": "CPIAUCSL",           # CPI (monthly)
    "unemployment": "UNRATE",     # Unemployment rate
    "fed_funds": "FEDFUNDS",     # Fed funds rate
    "gdp": "GDP",                # GDP (quarterly)
    "pce": "PCEPI",              # PCE price index
    "nonfarm_payrolls": "PAYEMS", # Nonfarm payrolls
    "initial_claims": "ICSA",    # Weekly initial jobless claims
    "retail_sales": "RSAFS",     # Retail sales
    "housing_starts": "HOUST",   # Housing starts
    "sp500": "SP500",            # S&P 500
}


def fetch_fred_series(
    series_id: str,
    api_key: str = None,
    start_date: str = None,
    end_date: str = None,
) -> pd.DataFrame:
    """Fetch economic data from FRED."""
    if api_key is None:
        # Without API key, we use yfinance as fallback for market data
        return pd.DataFrame()

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
    }
    if start_date:
        params["observation_start"] = start_date
    if end_date:
        params["observation_end"] = end_date

    resp = requests.get(f"{FRED_BASE}/series/observations", params=params, timeout=10)
    if resp.status_code != 200:
        return pd.DataFrame()

    data = resp.json().get("observations", [])
    df = pd.DataFrame(data)
    if df.empty:
        return df

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df[["date", "value"]].dropna()
    df = df.set_index("date")
    return df


def build_economic_snapshot(api_key: str = None) -> dict:
    """Build a snapshot of current economic indicators for event prediction."""
    snapshot = {}
    for name, series_id in FRED_SERIES.items():
        df = fetch_fred_series(
            series_id,
            api_key=api_key,
            start_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
        )
        if not df.empty:
            snapshot[name] = {
                "latest": float(df["value"].iloc[-1]),
                "prev": float(df["value"].iloc[-2]) if len(df) > 1 else None,
                "mean_12m": float(df["value"].mean()),
                "std_12m": float(df["value"].std()),
                "trend": float(df["value"].iloc[-1] - df["value"].iloc[0]),
            }
    return snapshot


def fetch_sp500_range_data(lookback_days: int = 365) -> pd.DataFrame:
    """Fetch S&P 500 data for predicting daily/weekly range events."""
    from cashflow.data.market_data import _yahoo_download

    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    df = _yahoo_download("SPY", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))

    if df.empty:
        return df

    # Calculate daily returns and ranges
    df["daily_return"] = df["close"].pct_change()
    df["daily_range_pct"] = (df["high"] - df["low"]) / df["open"] * 100
    df["weekly_return"] = df["close"].pct_change(5)

    # Rolling statistics for distribution modeling
    df["return_mean_20d"] = df["daily_return"].rolling(20).mean()
    df["return_std_20d"] = df["daily_return"].rolling(20).std()
    df["vol_20d"] = df["daily_return"].rolling(20).std() * (252 ** 0.5)

    return df.dropna()


def simulate_kalshi_events(sp500_data: pd.DataFrame) -> pd.DataFrame:
    """Simulate Kalshi-style S&P 500 range events for backtesting.

    Creates simulated event contracts like:
    - "S&P 500 to close above X today" (Yes/No)
    - "S&P 500 daily move > 1%"
    """
    events = []

    for i in range(20, len(sp500_data)):
        row = sp500_data.iloc[i]
        prev = sp500_data.iloc[i - 1]

        # Event: "S&P 500 closes higher today"
        actual_up = row["close"] > prev["close"]
        # Use rolling stats to estimate probability
        hist_up_rate = (sp500_data["daily_return"].iloc[i-20:i] > 0).mean()

        events.append({
            "date": sp500_data.index[i],
            "event_type": "sp500_up_today",
            "model_prob": hist_up_rate,
            "actual_outcome": actual_up,
        })

        # Event: "S&P 500 moves more than 1% today"
        actual_big_move = abs(row["daily_return"]) > 0.01
        hist_big_move_rate = (abs(sp500_data["daily_return"].iloc[i-20:i]) > 0.01).mean()

        events.append({
            "date": sp500_data.index[i],
            "event_type": "sp500_big_move",
            "model_prob": hist_big_move_rate,
            "actual_outcome": actual_big_move,
        })

        # Event: "S&P 500 closes within 0.5% of open"
        actual_flat = abs(row["daily_return"]) < 0.005
        hist_flat_rate = (abs(sp500_data["daily_return"].iloc[i-20:i]) < 0.005).mean()

        events.append({
            "date": sp500_data.index[i],
            "event_type": "sp500_flat_day",
            "model_prob": hist_flat_rate,
            "actual_outcome": actual_flat,
        })

    return pd.DataFrame(events)
