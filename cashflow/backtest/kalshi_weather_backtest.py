"""Backtest for Kalshi Weather Temperature Strategy.

Core Thesis:
  NWS forecasts are highly accurate for next-day high temperatures
  (within 2-3°F typically). Kalshi weather contracts settle based on
  the NWS Daily Climate Report. When Kalshi's market price diverges
  from what the NWS forecast implies, we have edge.

  Our edge: We use the same data source (NWS) that settles the contract.
  Retail traders on Kalshi trade on gut feeling or consumer weather apps.
  We trade on the actual settlement source.

Markets:
  - NYC, Chicago, Miami, Austin, LA
  - Daily high temperature bracket contracts
  - Settle next morning based on NWS report

Data:
  - Historical actuals: NOAA Climate Data Online API
  - Forecast accuracy: Simulated from known NWS accuracy stats
    (mean error ~1.5°F, std ~2.5°F for 1-day forecasts)
"""
import pandas as pd
import numpy as np
import requests
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from cashflow.utils.risk import max_drawdown, sharpe_ratio, sortino_ratio


# --- NWS Forecast Accuracy Parameters ---
# Based on published NWS verification stats:
# 1-day high temp forecast: mean absolute error ~2.0°F, bias ~0.5°F
NWS_FORECAST_MAE = 2.0      # Mean absolute error in °F
NWS_FORECAST_BIAS = 0.5     # Slight warm bias
NWS_FORECAST_STD = 2.5      # Std dev of forecast error

# Kalshi market noise — how much retail traders deviate from fair value
MARKET_NOISE_STD = 0.08     # 8% noise in contract pricing


# --- Data Classes ---

@dataclass
class WeatherContract:
    """A simulated Kalshi temperature bracket contract."""
    contract_id: str
    city: str
    date: object
    bracket_type: str        # "above_X", "below_X", "between_X_Y"
    lower_bound: float       # Temperature lower bound (°F)
    upper_bound: float       # Temperature upper bound (°F)
    market_price: float      # Kalshi market price ($0-$1)
    model_prob: float        # Our NWS-based probability estimate
    edge: float              # model_prob - market_price
    actual_high: float = 0.0
    outcome: Optional[bool] = None


@dataclass
class WeatherTrade:
    """A trade on a weather contract."""
    contract: WeatherContract
    position: str            # "yes" or "no"
    entry_price: float
    stake: float
    num_contracts: int
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0


@dataclass
class CityConfig:
    """Configuration for a city's weather market."""
    name: str
    station_id: str          # NOAA station ID
    lat: float
    lon: float
    avg_high_summer: float   # Average summer high
    avg_high_winter: float   # Average winter high
    temp_std: float          # Typical daily variability


# --- City Configurations ---

CITIES = {
    "NYC": CityConfig("New York City", "USW00094728", 40.7128, -74.0060,
                       avg_high_summer=84, avg_high_winter=39, temp_std=8),
    "CHI": CityConfig("Chicago", "USW00094846", 41.8781, -87.6298,
                       avg_high_summer=83, avg_high_winter=32, temp_std=10),
    "MIA": CityConfig("Miami", "USW00012839", 25.7617, -80.1918,
                       avg_high_summer=91, avg_high_winter=77, temp_std=4),
    "AUS": CityConfig("Austin", "USW00013904", 30.2672, -97.7431,
                       avg_high_summer=96, avg_high_winter=62, temp_std=9),
    "LAX": CityConfig("Los Angeles", "USW00023174", 34.0522, -118.2437,
                       avg_high_summer=84, avg_high_winter=68, temp_std=6),
}


# --- Historical Data ---

def fetch_noaa_historical(station_id: str, start_date: str, end_date: str,
                          token: str = None) -> pd.DataFrame:
    """Fetch historical daily high temperatures from NOAA CDO API.

    If no token, generates realistic synthetic data based on city climatology.
    """
    if token:
        return _fetch_noaa_api(station_id, start_date, end_date, token)
    return pd.DataFrame()  # Will use synthetic


def generate_synthetic_temperatures(city: CityConfig, start_date: str,
                                     end_date: str, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic daily high temperatures.

    Uses seasonal sinusoidal pattern + random variation calibrated
    to each city's climatology.
    """
    np.random.seed(seed)
    dates = pd.date_range(start_date, end_date, freq="D")

    temps = []
    for date in dates:
        # Day of year -> seasonal component
        doy = date.timetuple().tm_yday
        # Sinusoidal seasonal pattern (peak around day 200 = mid-July)
        seasonal = np.sin(2 * np.pi * (doy - 100) / 365)
        # Scale between winter and summer averages
        avg = (city.avg_high_summer + city.avg_high_winter) / 2
        amplitude = (city.avg_high_summer - city.avg_high_winter) / 2
        base_temp = avg + amplitude * seasonal

        # Add daily variation
        daily_noise = np.random.normal(0, city.temp_std * 0.5)
        # Add occasional weather systems (autocorrelated multi-day events)
        temps.append(base_temp + daily_noise)

    # Add autocorrelation (weather persists across days)
    temps = np.array(temps)
    for i in range(1, len(temps)):
        temps[i] = 0.6 * temps[i] + 0.4 * temps[i - 1] + np.random.normal(0, 1)

    df = pd.DataFrame({
        "date": dates,
        "actual_high": np.round(temps, 1),
    }).set_index("date")

    return df


def simulate_nws_forecast(actual_high: float, seed: int = None) -> float:
    """Simulate an NWS forecast for a given actual high temperature.

    NWS 1-day forecasts have known accuracy characteristics:
    - Mean absolute error: ~2.0°F
    - Slight warm bias: ~0.5°F
    - Standard deviation: ~2.5°F
    """
    if seed is not None:
        np.random.seed(seed)

    # Forecast = actual + bias + noise
    error = np.random.normal(NWS_FORECAST_BIAS, NWS_FORECAST_STD)
    forecast = actual_high + error
    return round(forecast, 1)


# --- Contract Generation ---

def generate_temperature_brackets(forecast_high: float, city: CityConfig):
    """Generate Kalshi-style temperature bracket contracts.

    Kalshi typically offers:
    - "High temp above X°F" at several thresholds
    - "High temp below X°F" at several thresholds
    - Sometimes range brackets

    Thresholds are typically at round numbers (5°F increments)
    near the expected high.
    """
    # Round forecast to nearest 5
    center = round(forecast_high / 5) * 5

    contracts = []

    # "Above X" contracts at center-10, center-5, center, center+5, center+10
    for offset in [-10, -5, 0, 5, 10]:
        threshold = center + offset
        contracts.append({
            "type": f"above_{threshold}",
            "lower": threshold,
            "upper": 200,  # Effectively infinite
            "threshold": threshold,
        })

    # "Between X and Y" bracket contracts
    for offset in [-10, -5, 0, 5]:
        low = center + offset
        high = low + 5
        contracts.append({
            "type": f"between_{low}_{high}",
            "lower": low,
            "upper": high,
            "threshold": (low + high) / 2,
        })

    return contracts


def price_weather_contract(contract_info: dict, market_forecast: float,
                           city: CityConfig, noise_std: float = MARKET_NOISE_STD):
    """Price a weather contract from the market's perspective.

    The market uses a noisy estimate of the temperature distribution.
    Market participants may use consumer weather apps, gut feeling, etc.
    """
    from scipy.stats import norm

    # Market's estimate of high temp distribution
    # Uses a wider std than NWS (less accurate)
    market_std = city.temp_std * 0.6  # Market thinks temps are this variable

    lower = contract_info["lower"]
    upper = contract_info["upper"]

    if upper >= 200:  # "Above X" contract
        prob = 1 - norm.cdf(lower, loc=market_forecast, scale=market_std)
    elif lower <= -100:  # "Below X" contract
        prob = norm.cdf(upper, loc=market_forecast, scale=market_std)
    else:  # "Between X and Y"
        prob = norm.cdf(upper, loc=market_forecast, scale=market_std) - \
               norm.cdf(lower, loc=market_forecast, scale=market_std)

    # Add market noise (retail mispricing)
    market_price = prob + np.random.normal(0, noise_std)
    market_price = np.clip(market_price, 0.03, 0.97)

    return round(market_price, 3)


def model_weather_probability(contract_info: dict, nws_forecast: float):
    """Our model's probability using NWS forecast.

    We trust NWS accuracy stats: forecast error ~ N(0.5, 2.5)
    So actual high ~ N(nws_forecast - bias, NWS_FORECAST_STD)
    """
    from scipy.stats import norm

    # Our model: actual high is distributed around the NWS forecast
    # with known error characteristics
    model_mean = nws_forecast - NWS_FORECAST_BIAS  # Correct for known warm bias
    model_std = NWS_FORECAST_STD

    lower = contract_info["lower"]
    upper = contract_info["upper"]

    if upper >= 200:  # "Above X"
        prob = 1 - norm.cdf(lower, loc=model_mean, scale=model_std)
    elif lower <= -100:  # "Below X"
        prob = norm.cdf(upper, loc=model_mean, scale=model_std)
    else:  # "Between X and Y"
        prob = norm.cdf(upper, loc=model_mean, scale=model_std) - \
               norm.cdf(lower, loc=model_mean, scale=model_std)

    return round(prob, 4)


# --- Strategy ---

class WeatherStrategy:
    """NWS-based weather trading strategy for Kalshi."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.min_edge = config.get("min_edge", 0.05)        # 5% minimum edge
        self.kelly_fraction = config.get("kelly_fraction", 0.25)
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_positions = config.get("max_positions", 5)
        self.min_stake = config.get("min_stake", 1.0)
        self.max_stake = config.get("max_stake", 5.0)
        self.daily_loss_limit_pct = config.get("daily_loss_limit_pct", 0.05)
        # Prefer contracts near $0.50 for 1:1 R:R
        self.preferred_price_low = 0.40
        self.preferred_price_high = 0.60
        self.acceptable_price_low = 0.25
        self.acceptable_price_high = 0.75

    def evaluate_contracts(self, contracts: list, capital: float,
                           daily_pnl: float, open_count: int) -> list:
        """Evaluate weather contracts and return trades to take."""
        daily_loss_limit = capital * self.daily_loss_limit_pct
        if daily_pnl <= -daily_loss_limit:
            return []

        trades = []
        available_slots = self.max_positions - open_count

        scored = []
        for c in contracts:
            # Check for YES edge
            yes_edge = c.model_prob - c.market_price
            # Check for NO edge
            no_edge = (1 - c.model_prob) - (1 - c.market_price)

            best_edge = max(yes_edge, no_edge)
            position = "yes" if yes_edge >= no_edge else "no"
            effective_price = c.market_price if position == "yes" else (1 - c.market_price)

            if best_edge < self.min_edge:
                continue

            in_preferred = self.preferred_price_low <= effective_price <= self.preferred_price_high
            in_acceptable = self.acceptable_price_low <= effective_price <= self.acceptable_price_high

            if not in_acceptable:
                continue

            scored.append((best_edge, c, position, effective_price))

        scored.sort(key=lambda x: -x[0])

        for edge, contract, position, eff_price in scored[:available_slots]:
            if len(trades) >= available_slots:
                break

            stake = self._kelly_size(edge, eff_price, capital)
            if stake < self.min_stake:
                continue

            num_contracts = max(1, int(stake / eff_price))
            actual_stake = round(num_contracts * eff_price, 2)

            if daily_pnl - actual_stake <= -daily_loss_limit:
                continue

            trade = WeatherTrade(
                contract=contract,
                position=position,
                entry_price=eff_price,
                stake=actual_stake,
                num_contracts=num_contracts,
            )
            trades.append(trade)

        return trades

    def _kelly_size(self, edge: float, price: float, capital: float) -> float:
        """Quarter-Kelly sizing."""
        if price <= 0 or price >= 1:
            return 0.0

        kelly = edge / (1 - price)
        stake = capital * kelly * self.kelly_fraction
        max_allowed = min(capital * self.max_position_pct, self.max_stake)
        return round(min(stake, max_allowed), 2)

    def settle_trade(self, trade: WeatherTrade, actual_high: float) -> float:
        """Settle a trade based on actual high temperature."""
        contract = trade.contract

        # Determine outcome
        if contract.upper_bound >= 200:  # "Above X"
            outcome = actual_high >= contract.lower_bound
        elif contract.lower_bound <= -100:  # "Below X"
            outcome = actual_high <= contract.upper_bound
        else:  # "Between X and Y"
            outcome = contract.lower_bound <= actual_high < contract.upper_bound

        contract.outcome = outcome

        # Calculate P&L
        if trade.position == "yes":
            if outcome:
                trade.pnl = round(trade.num_contracts * (1.0 - trade.entry_price), 2)
            else:
                trade.pnl = round(-trade.stake, 2)
        else:  # "no" position
            if not outcome:
                trade.pnl = round(trade.num_contracts * (1.0 - trade.entry_price), 2)
            else:
                trade.pnl = round(-trade.stake, 2)

        trade.exit_reason = "settlement"
        return trade.pnl


# --- Backtester ---

class WeatherBacktester:
    """Backtest the weather trading strategy."""

    def __init__(self, capital: float = 50.0, config: dict = None):
        self.initial_capital = capital
        self.config = config or {}
        self.strategy = WeatherStrategy(self.config)
        self.cities_to_trade = self.config.get("cities", ["NYC", "CHI", "AUS"])

    def run(self, lookback_days: int = 90):
        """Run the full backtest."""
        print("=" * 60)
        print("  Kalshi Weather Strategy Backtest")
        print("  NWS Forecast Edge | Temperature Contracts")
        print("=" * 60)

        # Generate synthetic temperature data for each city
        end_date = datetime.now()
        # Extra days for warmup
        start_date = end_date - timedelta(days=lookback_days + 30)

        city_data = {}
        for city_code in self.cities_to_trade:
            city = CITIES[city_code]
            print(f"Generating temperature data for {city.name}...")
            df = generate_synthetic_temperatures(
                city,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                seed=hash(city_code) % (2**31),
            )
            city_data[city_code] = df
            print(f"  {len(df)} days, avg high: {df['actual_high'].mean():.1f}°F")

        # Run backtest
        capital = self.initial_capital
        all_trades = []
        equity_curve = [capital]
        daily_results = []
        np.random.seed(42)

        # Trading days (skip first 7 for forecast warmup)
        all_dates = city_data[self.cities_to_trade[0]].index[7:]
        # Limit to lookback_days
        all_dates = all_dates[-lookback_days:]

        print(f"\nBacktesting {len(all_dates)} days across {len(self.cities_to_trade)} cities")
        print(f"Starting capital: ${capital:.2f}")
        print("-" * 60)

        for date in all_dates:
            daily_pnl = 0.0
            daily_trades = []
            open_count = 0

            for city_code in self.cities_to_trade:
                city = CITIES[city_code]
                df = city_data[city_code]

                if date not in df.index:
                    continue

                actual_high = df.loc[date, "actual_high"]

                # Simulate NWS forecast (made morning of)
                nws_forecast = simulate_nws_forecast(
                    actual_high,
                    seed=hash(f"{city_code}_{date}") % (2**31)
                )

                # Market's forecast (noisier — represents retail consensus)
                market_forecast = actual_high + np.random.normal(0, city.temp_std * 0.4)

                # Generate contracts
                contract_infos = generate_temperature_brackets(market_forecast, city)

                contracts = []
                for j, info in enumerate(contract_infos):
                    mkt_price = price_weather_contract(info, market_forecast, city)
                    model_prob = model_weather_probability(info, nws_forecast)

                    contract = WeatherContract(
                        contract_id=f"{city_code}_{date.strftime('%Y%m%d')}_{j}",
                        city=city_code,
                        date=date,
                        bracket_type=info["type"],
                        lower_bound=info["lower"],
                        upper_bound=info["upper"],
                        market_price=mkt_price,
                        model_prob=model_prob,
                        edge=round(model_prob - mkt_price, 4),
                        actual_high=actual_high,
                    )
                    contracts.append(contract)

                # Evaluate and trade
                trades = self.strategy.evaluate_contracts(
                    contracts, capital, daily_pnl, open_count
                )

                for trade in trades:
                    pnl = self.strategy.settle_trade(trade, actual_high)
                    daily_pnl += pnl
                    capital += pnl
                    daily_trades.append(trade)
                    open_count += 1

                    # Check daily loss limit
                    if daily_pnl <= -(capital * self.strategy.daily_loss_limit_pct):
                        break

            all_trades.extend(daily_trades)
            equity_curve.append(round(capital, 2))

            daily_results.append({
                "date": date,
                "trades": len(daily_trades),
                "pnl": round(daily_pnl, 2),
                "capital": round(capital, 2),
            })

        return self._compute_results(all_trades, daily_results, equity_curve)

    def _compute_results(self, all_trades, daily_results, equity_curve):
        """Compute and display results."""
        if not all_trades:
            return {"total_trades": 0, "message": "No trades generated"}

        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in all_trades)

        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                daily_returns.append(
                    (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                )

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
        win_rate = len(wins) / len(all_trades)

        # By city
        city_stats = {}
        for city_code in self.cities_to_trade:
            ct = [t for t in all_trades if t.contract.city == city_code]
            cw = [t for t in ct if t.pnl > 0]
            city_stats[city_code] = {
                "trades": len(ct),
                "wins": len(cw),
                "pnl": round(sum(t.pnl for t in ct), 2),
                "win_rate": round(len(cw) / len(ct) * 100, 1) if ct else 0,
            }

        # By position type
        yes_trades = [t for t in all_trades if t.position == "yes"]
        no_trades = [t for t in all_trades if t.position == "no"]

        # By contract type
        above_trades = [t for t in all_trades if "above" in t.contract.bracket_type]
        between_trades = [t for t in all_trades if "between" in t.contract.bracket_type]

        # Profitable days
        trading_days = len([d for d in daily_results if d["trades"] > 0])
        profitable_days = len([d for d in daily_results if d["pnl"] > 0])

        # R:R
        rr = round(avg_win / avg_loss, 2) if avg_loss > 0 else float("inf")

        # Average edge on trades taken
        avg_edge = np.mean([abs(t.contract.edge) for t in all_trades])

        results = {
            "strategy": "Kalshi Weather: NWS Forecast Edge",
            "initial_capital": self.initial_capital,
            "final_capital": round(equity_curve[-1], 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(
                (equity_curve[-1] / self.initial_capital - 1) * 100, 2
            ),
            "total_trades": len(all_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "risk_reward": rr,
            "avg_edge_pct": round(avg_edge * 100, 2),
            "avg_daily_return_pct": round(
                np.mean(daily_returns) * 100, 3
            ) if daily_returns else 0,
            "avg_daily_pnl": round(
                total_pnl / trading_days, 2
            ) if trading_days > 0 else 0,
            "trading_days": trading_days,
            "profitable_days": profitable_days,
            "profitable_days_pct": round(
                profitable_days / trading_days * 100, 1
            ) if trading_days > 0 else 0,
            "trades_per_day": round(
                len(all_trades) / trading_days, 1
            ) if trading_days > 0 else 0,
            "max_drawdown_pct": round(
                max_drawdown(equity_curve) * 100, 2
            ) if len(equity_curve) > 1 else 0,
            "sharpe_ratio": round(
                sharpe_ratio(daily_returns), 2
            ) if daily_returns else 0,
            "sortino_ratio": round(
                sortino_ratio(daily_returns), 2
            ) if daily_returns else 0,
            "city_stats": city_stats,
            "equity_curve": equity_curve,
        }

        self._print_results(results, yes_trades, no_trades, above_trades, between_trades)
        return results

    def _print_results(self, r, yes_trades, no_trades, above_trades, between_trades):
        """Pretty-print results."""
        print(f"\n{'='*60}")
        print(f"  {r['strategy']}")
        print(f"  90-Day Backtest Results")
        print(f"{'='*60}")

        print(f"\n--- PERFORMANCE ---")
        print(f"  Initial Capital:    ${r['initial_capital']:.2f}")
        print(f"  Final Capital:      ${r['final_capital']:.2f}")
        print(f"  Total P&L:          ${r['total_pnl']:.2f}")
        print(f"  Total Return:       {r['total_return_pct']}%")
        print(f"  Avg Daily Return:   {r['avg_daily_return_pct']}%")
        print(f"  Avg Daily P&L:      ${r['avg_daily_pnl']:.2f}")

        print(f"\n--- TRADES ---")
        print(f"  Total Trades:       {r['total_trades']}")
        print(f"  Trades/Day:         {r['trades_per_day']}")
        print(f"  Win Rate:           {r['win_rate_pct']}%")
        print(f"  Avg Win:            ${r['avg_win']:.2f}")
        print(f"  Avg Loss:           ${r['avg_loss']:.2f}")
        print(f"  Risk:Reward:        1:{r['risk_reward']}")
        print(f"  Avg Edge:           {r['avg_edge_pct']}%")

        print(f"\n--- RISK ---")
        print(f"  Max Drawdown:       {r['max_drawdown_pct']}%")
        print(f"  Sharpe Ratio:       {r['sharpe_ratio']}")
        print(f"  Sortino Ratio:      {r['sortino_ratio']}")
        print(f"  Profitable Days:    {r['profitable_days']}/{r['trading_days']} ({r['profitable_days_pct']}%)")

        print(f"\n--- BY CITY ---")
        for code, stats in r["city_stats"].items():
            city_name = CITIES[code].name
            print(f"  {city_name}: {stats['trades']} trades, "
                  f"{stats['win_rate']}% win rate, ${stats['pnl']:.2f} P&L")

        print(f"\n--- BY POSITION ---")
        yw = len([t for t in yes_trades if t.pnl > 0])
        nw = len([t for t in no_trades if t.pnl > 0])
        print(f"  YES: {len(yes_trades)} trades, "
              f"{yw}/{len(yes_trades)} wins, "
              f"${sum(t.pnl for t in yes_trades):.2f} P&L")
        print(f"  NO:  {len(no_trades)} trades, "
              f"{nw}/{len(no_trades)} wins, "
              f"${sum(t.pnl for t in no_trades):.2f} P&L")

        print(f"\n--- BY CONTRACT TYPE ---")
        aw = len([t for t in above_trades if t.pnl > 0])
        bw = len([t for t in between_trades if t.pnl > 0])
        print(f"  Above threshold: {len(above_trades)} trades, "
              f"{aw} wins, ${sum(t.pnl for t in above_trades):.2f} P&L")
        print(f"  Between bracket: {len(between_trades)} trades, "
              f"{bw} wins, ${sum(t.pnl for t in between_trades):.2f} P&L")

        print(f"\n{'='*60}")
