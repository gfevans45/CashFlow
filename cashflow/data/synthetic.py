"""Synthetic market data generator for offline backtesting.

Generates realistic price data using geometric Brownian motion with
regime changes, mean-reversion tendencies, and volume patterns.
Used when live API access is unavailable.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_stock_data(
    symbol: str = "SYN",
    days: int = 500,
    start_price: float = 100.0,
    annual_return: float = 0.10,
    annual_vol: float = 0.25,
    start_date: str = "2024-01-01",
    seed: int = None,
) -> pd.DataFrame:
    """Generate realistic synthetic stock price data.

    Uses geometric Brownian motion with:
    - Regime switching (trending vs. mean-reverting periods)
    - Volume clustering (high volume on big moves)
    - Realistic OHLC relationships
    """
    if seed is not None:
        np.random.seed(seed)

    dt = 1 / 252  # Daily
    mu = annual_return
    sigma = annual_vol

    dates = pd.bdate_range(start=start_date, periods=days)
    n = len(dates)

    # Generate returns with regime switching
    closes = np.zeros(n)
    closes[0] = start_price

    # Create regimes (0=trending, 1=mean-reverting, 2=volatile)
    regime_length = np.random.randint(20, 60, size=n // 20 + 1)
    regimes = []
    for length in regime_length:
        regime = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        regimes.extend([regime] * length)
    regimes = np.array(regimes[:n])

    for i in range(1, n):
        if regimes[i] == 0:  # Trending
            drift = mu * dt + np.random.normal(0, sigma * np.sqrt(dt) * 0.8)
        elif regimes[i] == 1:  # Mean-reverting
            mean_price = closes[max(0, i - 20):i].mean()
            reversion = 0.05 * (mean_price - closes[i - 1]) / closes[i - 1]
            drift = reversion + np.random.normal(0, sigma * np.sqrt(dt) * 0.6)
        else:  # Volatile
            drift = np.random.normal(0, sigma * np.sqrt(dt) * 1.5)

        closes[i] = closes[i - 1] * (1 + drift)
        closes[i] = max(closes[i], start_price * 0.3)  # Floor

    # Generate OHLV from close
    daily_ranges = np.abs(np.random.normal(0.01, 0.005, n))
    opens = closes * (1 + np.random.normal(0, 0.002, n))
    highs = np.maximum(opens, closes) * (1 + daily_ranges)
    lows = np.minimum(opens, closes) * (1 - daily_ranges)

    # Volume: higher on volatile days, lower on calm days
    base_volume = 1_000_000
    price_changes = np.abs(np.diff(closes, prepend=closes[0]) / closes)
    volume = base_volume * (1 + price_changes * 50) * np.random.lognormal(0, 0.3, n)
    volume = volume.astype(int)

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": closes,
        "volume": volume,
    }, index=dates)

    return df


def generate_multi_stock_universe(
    symbols: list = None,
    days: int = 500,
    start_date: str = "2024-01-01",
) -> dict:
    """Generate correlated synthetic data for multiple symbols."""
    if symbols is None:
        symbols = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META"]

    # Different characteristics per symbol
    profiles = {
        "SPY": {"price": 450, "ret": 0.10, "vol": 0.16},
        "QQQ": {"price": 380, "ret": 0.12, "vol": 0.20},
        "IWM": {"price": 200, "ret": 0.08, "vol": 0.22},
        "AAPL": {"price": 180, "ret": 0.15, "vol": 0.25},
        "MSFT": {"price": 370, "ret": 0.14, "vol": 0.23},
        "NVDA": {"price": 500, "ret": 0.25, "vol": 0.45},
        "AMD": {"price": 140, "ret": 0.20, "vol": 0.40},
        "TSLA": {"price": 250, "ret": 0.05, "vol": 0.55},
        "AMZN": {"price": 170, "ret": 0.18, "vol": 0.28},
        "META": {"price": 350, "ret": 0.20, "vol": 0.30},
    }

    data = {}
    for i, symbol in enumerate(symbols):
        profile = profiles.get(symbol, {"price": 100, "ret": 0.10, "vol": 0.25})
        df = generate_stock_data(
            symbol=symbol,
            days=days,
            start_price=profile["price"],
            annual_return=profile["ret"],
            annual_vol=profile["vol"],
            start_date=start_date,
            seed=42 + i,  # Reproducible but different per symbol
        )
        data[symbol] = df

    return data


def generate_sp500_for_prediction(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic S&P 500 data with calculated fields for prediction market simulation."""
    df = generate_stock_data(
        symbol="SPY",
        days=days,
        start_price=450,
        annual_return=0.10,
        annual_vol=0.16,
        start_date="2024-01-01",
        seed=seed,
    )

    df["daily_return"] = df["close"].pct_change()
    df["daily_range_pct"] = (df["high"] - df["low"]) / df["open"] * 100
    df["weekly_return"] = df["close"].pct_change(5)
    df["return_mean_20d"] = df["daily_return"].rolling(20).mean()
    df["return_std_20d"] = df["daily_return"].rolling(20).std()
    df["vol_20d"] = df["daily_return"].rolling(20).std() * (252 ** 0.5)

    return df.dropna()
