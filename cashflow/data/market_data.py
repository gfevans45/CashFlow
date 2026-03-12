"""Market data fetching using Yahoo Finance (via requests, no external deps)."""
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
from io import StringIO


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate EMA."""
    return series.ewm(span=period, adjust=False).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD, signal, and histogram."""
    ema_fast = _ema(series, fast)
    ema_slow = _ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands."""
    middle = series.rolling(window=period).mean()
    rolling_std = series.rolling(window=period).std()
    upper = middle + std_dev * rolling_std
    lower = middle - std_dev * rolling_std
    pct = (series - lower) / (upper - lower)
    return upper, middle, lower, pct


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def _yahoo_download(symbol: str, start_date: str, end_date: str, interval: str = "1d") -> pd.DataFrame:
    """Download data from Yahoo Finance using the v8 chart API."""
    start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": start_ts,
        "period2": end_ts,
        "interval": interval,
        "includePrePost": "false",
    }
    headers = {"User-Agent": "Mozilla/5.0"}

    resp = requests.get(url, params=params, headers=headers, timeout=15)
    if resp.status_code != 200:
        return pd.DataFrame()

    data = resp.json()
    result = data.get("chart", {}).get("result")
    if not result:
        return pd.DataFrame()

    result = result[0]
    timestamps = result.get("timestamp", [])
    quote = result.get("indicators", {}).get("quote", [{}])[0]

    if not timestamps:
        return pd.DataFrame()

    df = pd.DataFrame({
        "open": quote.get("open"),
        "high": quote.get("high"),
        "low": quote.get("low"),
        "close": quote.get("close"),
        "volume": quote.get("volume"),
    }, index=pd.to_datetime(timestamps, unit="s"))

    df.index.name = "Date"
    return df.dropna()


def fetch_stock_data(
    symbol: str,
    period: str = "90d",
    interval: str = "1h",
) -> pd.DataFrame:
    """Fetch OHLCV data from Yahoo Finance."""
    # Convert period to date range
    period_map = {"30d": 30, "60d": 60, "90d": 90, "180d": 180, "1y": 365, "2y": 730}
    days = period_map.get(period, 90)
    end = datetime.now()
    start = end - timedelta(days=days)

    df = _yahoo_download(
        symbol,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        interval=interval,
    )

    if df.empty:
        return df

    return add_technical_indicators(df)


def fetch_multi_symbol_data(
    symbols: list,
    period: str = "90d",
    interval: str = "1h",
) -> dict:
    """Fetch data for multiple symbols."""
    data = {}
    for symbol in symbols:
        df = fetch_stock_data(symbol, period, interval)
        if not df.empty:
            data[symbol] = df
        time.sleep(0.3)  # Rate limit
    return data


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators used by our strategy."""
    if len(df) < 30:
        return df

    df["rsi"] = _rsi(df["close"], 14)
    df["bb_upper"], df["bb_middle"], df["bb_lower"], df["bb_pct"] = _bollinger_bands(df["close"], 20, 2.0)
    df["ema_fast"] = _ema(df["close"], 8)
    df["ema_slow"] = _ema(df["close"], 21)
    df["macd"], df["macd_signal"], df["macd_hist"] = _macd(df["close"])
    df["volume_sma"] = df["volume"].rolling(window=20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]
    df["atr"] = _atr(df["high"], df["low"], df["close"], 14)
    df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()

    return df.dropna()


def fetch_daily_for_backtest(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch daily data for backtesting over a date range."""
    df = _yahoo_download(symbol, start_date, end_date, interval="1d")

    if df.empty:
        return df

    return add_technical_indicators(df)
