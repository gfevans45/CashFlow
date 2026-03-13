#!/usr/bin/env python3
"""CashFlow - Live/Paper Weather Trading Bot for Kalshi.

Trades Kalshi temperature contracts using NWS forecast data as the
primary signal. The NWS is the same data source Kalshi uses to settle
contracts, giving us a direct informational edge over retail traders
who rely on consumer weather apps or gut feeling.

Daily schedule (all times Eastern):
  09:15 - Pull NWS forecasts for target cities
  09:30 - Scan Kalshi weather markets, compare model prob vs market price
  09:30 - Place paper/live trades where edge > 5%
  Next AM - Check settlement, log P&L

Usage:
    python run_live_weather.py                    # Paper trade (default)
    python run_live_weather.py --live             # Live trading
    python run_live_weather.py --once             # Single run then exit
    python run_live_weather.py --capital 100      # Custom starting capital
    python run_live_weather.py --cities NYC CHI   # Trade specific cities
"""

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from cashflow.backtest.kalshi_weather_backtest import (
    CITIES,
    CityConfig,
    WeatherContract,
    WeatherStrategy,
    WeatherTrade,
    model_weather_probability,
)
from cashflow.data.weather_feeds import (
    KalshiWeatherClient,
    KalshiWeatherMarket,
    NWSClient,
    NWSForecast,
)
from cashflow.utils.config import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "weather.log"),
    ],
)
log = logging.getLogger("cashflow.weather")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_FILE = Path("data/weather_state.json")
ET = ZoneInfo("America/New_York")

# Schedule times (Eastern)
FORECAST_HOUR = 9
FORECAST_MINUTE = 15
SCAN_HOUR = 9
SCAN_MINUTE = 30

# Default cities to trade
DEFAULT_CITIES = ["NYC", "CHI", "AUS"]

# Paper-mode simulated price parameters
SIM_PRICE_NOISE = 0.06  # Noise around model prob for simulated prices


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

@dataclass
class DailyRecord:
    """Record of one day's trading activity."""
    date: str
    forecasts: dict          # city -> forecast_high
    trades_placed: int
    trades_settled: int
    daily_pnl: float
    capital_after: float


def save_state(state: dict):
    """Persist bot state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_saved"] = datetime.now(ET).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.debug(f"State saved to {STATE_FILE}")


def load_state() -> dict:
    """Load bot state from disk, or return empty state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load state: {e}. Starting fresh.")
    return {}


def init_state(capital: float, cities: list) -> dict:
    """Initialize a fresh bot state."""
    return {
        "capital": capital,
        "initial_capital": capital,
        "cities": cities,
        "open_trades": [],
        "settled_trades": [],
        "daily_records": [],
        "total_pnl": 0.0,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "daily_pnl": 0.0,
        "last_trade_date": "",
        "last_forecast_date": "",
        "forecasts": {},        # city -> {date, high, fetched_at}
        "started_at": datetime.now(ET).isoformat(),
    }


# ---------------------------------------------------------------------------
# Core bot logic
# ---------------------------------------------------------------------------

class WeatherTradingBot:
    """Live/paper trading bot for Kalshi weather contracts.

    Pulls NWS forecasts, compares against Kalshi contract prices (real or
    simulated), and places trades where our NWS-based model shows edge.
    """

    def __init__(self, state: dict, paper_mode: bool = True):
        self.state = state
        self.paper_mode = paper_mode
        self.capital = state["capital"]
        self.cities = state["cities"]

        # Strategy (reuse backtest config with live-appropriate sizing)
        self.strategy = WeatherStrategy({
            "min_edge": 0.05,
            "kelly_fraction": 0.25,
            "max_position_pct": 0.10,
            "max_positions": 5,
            "min_stake": 1.0,
            "max_stake": 5.0,
            "daily_loss_limit_pct": 0.05,
        })

        # Data clients
        self.nws = NWSClient()
        self.kalshi: Optional[KalshiWeatherClient] = None
        self._kalshi_available = False

        # Try to initialize Kalshi client
        self._init_kalshi()

    def _init_kalshi(self):
        """Initialize Kalshi client if credentials are available."""
        try:
            self.kalshi = KalshiWeatherClient()
            if self.kalshi.has_credentials:
                if self.kalshi.authenticate():
                    self._kalshi_available = True
                    log.info("Kalshi API connected and authenticated")
                else:
                    log.warning("Kalshi credentials found but authentication failed")
            else:
                log.info("No Kalshi credentials found - using simulated prices")
        except Exception as e:
            log.warning(f"Failed to initialize Kalshi client: {e}")

    # --- Forecasts ---

    def fetch_forecasts(self) -> dict:
        """Fetch NWS forecasts for all target cities.

        Returns dict of city_code -> forecast_high (F).
        """
        today = datetime.now(ET)
        tomorrow = today + timedelta(days=1)
        target_date = tomorrow.strftime("%Y-%m-%d")

        log.info(f"Fetching NWS forecasts for {target_date} ({len(self.cities)} cities)")

        forecasts = {}
        for city_code in self.cities:
            city = CITIES.get(city_code)
            if not city:
                log.warning(f"Unknown city code: {city_code}")
                continue

            try:
                high = self.nws.get_forecast_high(city.lat, city.lon, target_date)
                if high is not None:
                    forecasts[city_code] = {
                        "date": target_date,
                        "high": high,
                        "fetched_at": datetime.now(ET).isoformat(),
                    }
                    log.info(f"  {city.name}: forecast high {high:.0f}F for {target_date}")
                else:
                    # Try today's date as fallback (NWS may not have tomorrow yet early AM)
                    today_str = today.strftime("%Y-%m-%d")
                    high = self.nws.get_forecast_high(city.lat, city.lon, today_str)
                    if high is not None:
                        forecasts[city_code] = {
                            "date": today_str,
                            "high": high,
                            "fetched_at": datetime.now(ET).isoformat(),
                        }
                        log.info(f"  {city.name}: forecast high {high:.0f}F for {today_str} (today fallback)")
                    else:
                        log.warning(f"  {city.name}: no forecast available")

                time.sleep(0.5)  # Be nice to NWS API

            except Exception as e:
                log.error(f"  {city.name}: forecast fetch failed: {e}")

        self.state["forecasts"] = forecasts
        self.state["last_forecast_date"] = datetime.now(ET).strftime("%Y-%m-%d")
        return forecasts

    # --- Market scanning ---

    def scan_markets(self, forecasts: dict) -> list:
        """Scan Kalshi markets and build WeatherContract objects.

        If Kalshi API is available, fetch real contract prices.
        Otherwise, generate simulated contracts from NWS forecast.

        Returns list of WeatherContract objects with model probabilities.
        """
        if self._kalshi_available:
            return self._scan_real_markets(forecasts)
        else:
            return self._scan_simulated_markets(forecasts)

    def _scan_real_markets(self, forecasts: dict) -> list:
        """Fetch real Kalshi weather contracts and compute model probs."""
        log.info("Scanning Kalshi for weather contracts (live API)")

        contracts = []
        try:
            kalshi_markets = self.kalshi.get_temperature_markets()
        except Exception as e:
            log.error(f"Failed to fetch Kalshi markets: {e}")
            log.info("Falling back to simulated prices")
            return self._scan_simulated_markets(forecasts)

        for km in kalshi_markets:
            # Match to a city we're trading
            if km.city not in forecasts:
                continue

            forecast_data = forecasts[km.city]
            nws_high = forecast_data["high"]

            # Build contract_info dict for model_weather_probability
            contract_info = {
                "type": km.bracket_type,
                "lower": km.lower_bound,
                "upper": km.upper_bound,
            }

            model_prob = model_weather_probability(contract_info, nws_high)
            market_price = km.yes_price

            if market_price <= 0:
                continue

            contract = WeatherContract(
                contract_id=km.ticker,
                city=km.city,
                date=km.date or forecast_data["date"],
                bracket_type=km.bracket_type,
                lower_bound=km.lower_bound,
                upper_bound=km.upper_bound,
                market_price=market_price,
                model_prob=model_prob,
                edge=round(model_prob - market_price, 4),
            )
            contracts.append(contract)

        log.info(f"  Matched {len(contracts)} contracts across target cities")
        return contracts

    def _scan_simulated_markets(self, forecasts: dict) -> list:
        """Generate simulated contracts when Kalshi API is unavailable.

        Uses the NWS forecast to create realistic bracket contracts with
        noisy simulated market prices (as if retail traders set them).
        """
        log.info("Generating simulated weather contracts (no Kalshi API)")
        if not self._kalshi_available:
            log.info("  (would check Kalshi prices if credentials were available)")

        contracts = []

        for city_code, forecast_data in forecasts.items():
            nws_high = forecast_data["high"]
            date_str = forecast_data["date"]
            city = CITIES.get(city_code)
            if not city:
                continue

            # Generate bracket contracts around the forecast
            center = round(nws_high / 5) * 5

            bracket_configs = []
            # "Above X" contracts
            for offset in [-10, -5, 0, 5, 10]:
                threshold = center + offset
                bracket_configs.append({
                    "type": f"above_{int(threshold)}",
                    "lower": float(threshold),
                    "upper": 200.0,
                })
            # "Between X and Y" contracts
            for offset in [-10, -5, 0, 5]:
                low = center + offset
                high = low + 5
                bracket_configs.append({
                    "type": f"between_{int(low)}_{int(high)}",
                    "lower": float(low),
                    "upper": float(high),
                })

            for i, info in enumerate(bracket_configs):
                model_prob = model_weather_probability(info, nws_high)

                # Simulate a market price with some noise
                # (retail traders are less accurate than NWS)
                noise = random.gauss(0, SIM_PRICE_NOISE)
                market_price = max(0.03, min(0.97, model_prob + noise))
                market_price = round(market_price, 3)

                contract = WeatherContract(
                    contract_id=f"SIM-{city_code}-{date_str}-{i}",
                    city=city_code,
                    date=date_str,
                    bracket_type=info["type"],
                    lower_bound=info["lower"],
                    upper_bound=info["upper"],
                    market_price=market_price,
                    model_prob=model_prob,
                    edge=round(model_prob - market_price, 4),
                )
                contracts.append(contract)

        log.info(f"  Generated {len(contracts)} simulated contracts "
                 f"across {len(forecasts)} cities")
        return contracts

    # --- Trading ---

    def evaluate_and_trade(self, contracts: list) -> list:
        """Evaluate contracts and place trades (paper or live).

        Returns list of WeatherTrade objects for trades placed.
        """
        open_count = len(self.state.get("open_trades", []))
        daily_pnl = self.state.get("daily_pnl", 0.0)

        trades = self.strategy.evaluate_contracts(
            contracts, self.capital, daily_pnl, open_count,
        )

        if not trades:
            log.info("No trades meet edge/sizing criteria")
            return []

        mode_tag = "PAPER" if self.paper_mode else "LIVE"
        placed = []

        for trade in trades:
            c = trade.contract
            log.info(
                f"  [{mode_tag}] {trade.position.upper()} {c.city} {c.bracket_type} "
                f"| Price: ${trade.entry_price:.3f} | Model: {c.model_prob:.3f} "
                f"| Edge: {c.edge:+.3f} | Stake: ${trade.stake:.2f} "
                f"| Contracts: {trade.num_contracts}"
            )

            if not self.paper_mode and self._kalshi_available:
                log.info(f"    [LIVE] Would submit order to Kalshi (execution pending)")
                # TODO: Implement Kalshi order submission
                # self.kalshi.place_order(c.contract_id, trade.position, trade.num_contracts, ...)

            # Track the trade
            trade_record = {
                "contract_id": c.contract_id,
                "city": c.city,
                "date": str(c.date),
                "bracket_type": c.bracket_type,
                "lower_bound": c.lower_bound,
                "upper_bound": c.upper_bound,
                "market_price": c.market_price,
                "model_prob": c.model_prob,
                "edge": c.edge,
                "position": trade.position,
                "entry_price": trade.entry_price,
                "stake": trade.stake,
                "num_contracts": trade.num_contracts,
                "placed_at": datetime.now(ET).isoformat(),
                "settled": False,
                "pnl": 0.0,
            }

            self.state.setdefault("open_trades", []).append(trade_record)
            self.capital -= trade.stake
            self.state["capital"] = round(self.capital, 2)
            self.state["total_trades"] = self.state.get("total_trades", 0) + 1
            placed.append(trade)

        log.info(f"Placed {len(placed)} trades | Capital remaining: ${self.capital:.2f}")
        return placed

    # --- Settlement ---

    def check_settlements(self) -> float:
        """Check and settle any trades from previous days.

        In paper mode, we fetch the NWS actual high temperature to settle.
        For simulated contracts, we use the NWS forecast as a proxy for
        actual (since we can't get real historical actuals easily).

        Returns total settlement P&L.
        """
        open_trades = self.state.get("open_trades", [])
        if not open_trades:
            return 0.0

        today = datetime.now(ET).strftime("%Y-%m-%d")
        settlement_pnl = 0.0
        still_open = []
        newly_settled = []

        for trade_record in open_trades:
            trade_date = trade_record.get("date", "")

            # Only settle trades from previous days
            if trade_date >= today:
                still_open.append(trade_record)
                continue

            # For paper mode, we need to determine the actual outcome.
            # Attempt to get actual high from NWS observation (or use forecast as proxy).
            city_code = trade_record.get("city", "")
            city = CITIES.get(city_code)
            actual_high = None

            if city:
                # Try NWS forecast for that date as a proxy for actual
                # (In production with real Kalshi, settlement is automatic)
                try:
                    actual_high = self.nws.get_forecast_high(
                        city.lat, city.lon, trade_date
                    )
                except Exception:
                    pass

            if actual_high is None:
                # Use the model's forecast as best estimate
                forecast_data = self.state.get("forecasts", {}).get(city_code, {})
                if forecast_data.get("date") == trade_date:
                    actual_high = forecast_data.get("high")

            if actual_high is None:
                # Can't settle yet - keep open
                log.warning(f"Cannot settle {trade_record['contract_id']}: no actual temp")
                still_open.append(trade_record)
                continue

            # Reconstruct WeatherContract and WeatherTrade for settlement
            contract = WeatherContract(
                contract_id=trade_record["contract_id"],
                city=city_code,
                date=trade_date,
                bracket_type=trade_record["bracket_type"],
                lower_bound=trade_record["lower_bound"],
                upper_bound=trade_record["upper_bound"],
                market_price=trade_record["market_price"],
                model_prob=trade_record["model_prob"],
                edge=trade_record["edge"],
                actual_high=actual_high,
            )
            trade = WeatherTrade(
                contract=contract,
                position=trade_record["position"],
                entry_price=trade_record["entry_price"],
                stake=trade_record["stake"],
                num_contracts=trade_record["num_contracts"],
            )

            pnl = self.strategy.settle_trade(trade, actual_high)
            settlement_pnl += pnl
            self.capital += pnl + trade.stake  # Return stake + P&L
            self.state["capital"] = round(self.capital, 2)

            outcome_str = "WIN" if pnl > 0 else "LOSS"
            log.info(
                f"  SETTLED [{outcome_str}]: {city_code} {trade_record['bracket_type']} "
                f"| Actual: {actual_high:.0f}F | P&L: ${pnl:+.2f}"
            )

            trade_record["settled"] = True
            trade_record["pnl"] = pnl
            trade_record["actual_high"] = actual_high
            trade_record["settled_at"] = datetime.now(ET).isoformat()
            newly_settled.append(trade_record)

            if pnl > 0:
                self.state["total_wins"] = self.state.get("total_wins", 0) + 1
            else:
                self.state["total_losses"] = self.state.get("total_losses", 0) + 1

        self.state["open_trades"] = still_open
        self.state.setdefault("settled_trades", []).extend(newly_settled)
        self.state["total_pnl"] = round(
            self.state.get("total_pnl", 0.0) + settlement_pnl, 2
        )

        if newly_settled:
            log.info(f"Settled {len(newly_settled)} trades | "
                     f"Settlement P&L: ${settlement_pnl:+.2f} | "
                     f"Capital: ${self.capital:.2f}")

        return settlement_pnl

    # --- Daily run ---

    def run_daily_cycle(self):
        """Execute one full daily trading cycle.

        1. Check settlements from previous trades
        2. Fetch NWS forecasts
        3. Scan Kalshi markets (or simulated)
        4. Evaluate and place trades
        5. Save state
        """
        mode = "PAPER" if self.paper_mode else "LIVE"
        today = datetime.now(ET).strftime("%Y-%m-%d")

        log.info("=" * 60)
        log.info(f"  Weather Trading Bot - Daily Cycle [{mode}]")
        log.info(f"  Date: {today}")
        log.info(f"  Capital: ${self.capital:.2f}")
        log.info(f"  Open trades: {len(self.state.get('open_trades', []))}")
        log.info(f"  Cities: {', '.join(self.cities)}")
        log.info("=" * 60)

        # Reset daily P&L
        self.state["daily_pnl"] = 0.0

        # Step 1: Settle previous trades
        settlement_pnl = self.check_settlements()
        self.state["daily_pnl"] += settlement_pnl

        # Step 2: Fetch forecasts
        forecasts = self.fetch_forecasts()
        if not forecasts:
            log.warning("No forecasts available - skipping trading")
            save_state(self.state)
            return

        # Step 3: Scan markets
        contracts = self.scan_markets(forecasts)
        if not contracts:
            log.warning("No contracts found - skipping trading")
            save_state(self.state)
            return

        # Log contract summary
        for c in sorted(contracts, key=lambda x: abs(x.edge), reverse=True)[:10]:
            log.debug(
                f"  {c.city} {c.bracket_type}: market=${c.market_price:.3f} "
                f"model={c.model_prob:.3f} edge={c.edge:+.3f}"
            )

        # Step 4: Evaluate and trade
        trades = self.evaluate_and_trade(contracts)

        # Step 5: Record daily result
        daily_record = {
            "date": today,
            "forecasts": {
                city: fc.get("high", 0) for city, fc in forecasts.items()
            },
            "trades_placed": len(trades),
            "settlement_pnl": round(settlement_pnl, 2),
            "capital": round(self.capital, 2),
        }
        self.state.setdefault("daily_records", []).append(daily_record)
        self.state["last_trade_date"] = today

        # Save state
        save_state(self.state)

        # Print summary
        self._print_daily_summary(trades, settlement_pnl, forecasts)

    def _print_daily_summary(self, trades: list, settlement_pnl: float, forecasts: dict):
        """Print end-of-cycle summary."""
        total_pnl = self.state.get("total_pnl", 0.0)
        total_trades = self.state.get("total_trades", 0)
        total_wins = self.state.get("total_wins", 0)
        total_losses = self.state.get("total_losses", 0)
        win_rate = (total_wins / (total_wins + total_losses) * 100) if (total_wins + total_losses) > 0 else 0

        log.info("-" * 60)
        log.info("  DAILY SUMMARY")
        log.info("-" * 60)
        log.info(f"  Forecasts fetched: {len(forecasts)} cities")
        log.info(f"  Trades placed: {len(trades)}")
        log.info(f"  Settlement P&L: ${settlement_pnl:+.2f}")
        log.info(f"  Open trades: {len(self.state.get('open_trades', []))}")
        log.info(f"  Capital: ${self.capital:.2f}")
        log.info(f"  Total P&L: ${total_pnl:+.2f}")
        log.info(f"  Total trades: {total_trades} ({total_wins}W/{total_losses}L, {win_rate:.0f}%)")
        log.info(f"  State: {STATE_FILE}")
        log.info("-" * 60)


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def _time_until_next_run(target_hour: int, target_minute: int) -> float:
    """Calculate seconds until the next target time (Eastern)."""
    now = datetime.now(ET)
    target = now.replace(hour=target_hour, minute=target_minute, second=0, microsecond=0)

    if now >= target:
        # Already past today's target time, schedule for tomorrow
        target += timedelta(days=1)

    # Skip weekends (Kalshi weather markets don't always run weekends)
    while target.weekday() >= 5:  # Saturday=5, Sunday=6
        target += timedelta(days=1)

    delta = (target - now).total_seconds()
    return max(0, delta)


def _is_trading_time() -> bool:
    """Check if we're within the daily trading window."""
    now = datetime.now(ET)
    # Trading window: 9:00 AM - 10:00 AM ET
    return now.hour == 9 or (now.hour == 10 and now.minute == 0)


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_running = True


def _signal_handler(sig, frame):
    global _running
    log.info("Shutdown signal received. Saving state and exiting...")
    _running = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _running

    parser = argparse.ArgumentParser(
        description="CashFlow Weather Trading Bot for Kalshi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_live_weather.py                    # Paper trade, default settings
  python run_live_weather.py --once             # Single run and exit
  python run_live_weather.py --live             # Live trading (requires Kalshi creds)
  python run_live_weather.py --capital 100      # Start with $100
  python run_live_weather.py --cities NYC CHI   # Trade only NYC and Chicago
  python run_live_weather.py --reset            # Clear state, start fresh
        """,
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading (default: paper mode)",
    )
    parser.add_argument(
        "--capital", type=float, default=50.0,
        help="Starting capital in dollars (default: $50)",
    )
    parser.add_argument(
        "--cities", nargs="+", default=DEFAULT_CITIES,
        help=f"City codes to trade (default: {' '.join(DEFAULT_CITIES)}). "
             f"Available: {', '.join(CITIES.keys())}",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle and exit (for testing / cron)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear saved state and start fresh",
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    # Validate cities
    for city in args.cities:
        if city not in CITIES:
            log.error(f"Unknown city code: {city}. Available: {', '.join(CITIES.keys())}")
            sys.exit(1)

    mode = "LIVE" if args.live else "PAPER"
    log.info("=" * 60)
    log.info(f"  CashFlow Weather Trading Bot - {mode} MODE")
    log.info(f"  Capital: ${args.capital:.2f}")
    log.info(f"  Cities: {', '.join(args.cities)}")
    if args.once:
        log.info("  Mode: single run")
    else:
        log.info(f"  Schedule: daily at {SCAN_HOUR}:{SCAN_MINUTE:02d} ET")
    log.info("=" * 60)

    if args.live:
        log.warning("LIVE TRADING ENABLED - Real money at risk!")
        log.warning("Kalshi order execution is not yet implemented - trades will be logged.")

    # State management
    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log.info("State reset.")

    state = load_state()
    if not state or args.reset:
        state = init_state(args.capital, args.cities)
        log.info("Initialized fresh state")
    else:
        # Merge cities from args (may have changed)
        state["cities"] = args.cities
        log.info(f"Restored state: ${state['capital']:.2f} capital, "
                 f"{len(state.get('open_trades', []))} open trades, "
                 f"${state.get('total_pnl', 0):.2f} total P&L")

    # Create bot
    bot = WeatherTradingBot(state, paper_mode=not args.live)

    # Handle graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.once:
        # Single run mode
        bot.run_daily_cycle()
        _print_session_summary(bot)
        return

    # Long-running service mode
    log.info(f"Starting daily schedule loop. Press Ctrl+C to stop.")

    while _running:
        try:
            # Calculate time until next trading window
            wait_seconds = _time_until_next_run(SCAN_HOUR, SCAN_MINUTE)

            if wait_seconds > 60:
                next_run = datetime.now(ET) + timedelta(seconds=wait_seconds)
                log.info(f"Next run: {next_run.strftime('%Y-%m-%d %H:%M ET')} "
                         f"(in {wait_seconds/3600:.1f} hours)")

            # Sleep in short increments for responsive shutdown
            slept = 0
            while slept < wait_seconds and _running:
                chunk = min(30, wait_seconds - slept)
                time.sleep(chunk)
                slept += chunk

            if not _running:
                break

            # Run the daily cycle
            bot.run_daily_cycle()

        except Exception as e:
            log.error(f"Daily cycle failed: {e}", exc_info=True)
            # Wait 5 minutes before retrying on error
            for _ in range(30):
                if not _running:
                    break
                time.sleep(10)

    # Final save
    save_state(bot.state)
    _print_session_summary(bot)
    log.info("Shutdown complete.")


def _print_session_summary(bot: WeatherTradingBot):
    """Print session summary on exit."""
    state = bot.state
    total_trades = state.get("total_trades", 0)
    total_wins = state.get("total_wins", 0)
    total_losses = state.get("total_losses", 0)
    total_pnl = state.get("total_pnl", 0.0)

    log.info("")
    log.info("=" * 60)
    log.info("  SESSION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Capital: ${bot.capital:.2f} (started: ${state.get('initial_capital', 50):.2f})")
    log.info(f"  Total P&L: ${total_pnl:+.2f}")
    log.info(f"  Total trades: {total_trades}")
    if total_wins + total_losses > 0:
        wr = total_wins / (total_wins + total_losses) * 100
        log.info(f"  Win rate: {total_wins}/{total_wins + total_losses} ({wr:.0f}%)")
    log.info(f"  Open trades: {len(state.get('open_trades', []))}")
    log.info(f"  State saved to: {STATE_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
