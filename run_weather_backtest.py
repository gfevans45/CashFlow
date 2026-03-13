#!/usr/bin/env python3
"""Run the Kalshi Weather Strategy backtest.

Usage:
    python run_weather_backtest.py              # Default 90-day backtest
    python run_weather_backtest.py --days 60    # Custom lookback
    python run_weather_backtest.py --capital 50 # Custom starting capital
"""
import argparse
import sys

from cashflow.backtest.kalshi_weather_backtest import WeatherBacktester


def main():
    parser = argparse.ArgumentParser(description="Kalshi Weather Strategy Backtest")
    parser.add_argument("--days", type=int, default=90, help="Lookback period in days")
    parser.add_argument("--capital", type=float, default=50.0, help="Starting capital")
    args = parser.parse_args()

    config = {
        "min_edge": 0.05,           # 5% edge required
        "kelly_fraction": 0.25,     # Quarter-Kelly
        "max_position_pct": 0.10,   # 10% of capital max per trade
        "max_positions": 5,
        "min_stake": 1.0,
        "max_stake": 5.0,
        "daily_loss_limit_pct": 0.05,
        "cities": ["NYC", "CHI", "AUS"],  # 3 cities = more opportunities
    }

    backtester = WeatherBacktester(capital=args.capital, config=config)
    results = backtester.run(lookback_days=args.days)

    if results.get("message"):
        print(f"\nBacktest failed: {results['message']}")
        sys.exit(1)

    # Print verdict
    print("\n--- VERDICT ---")
    wr = results.get("win_rate_pct", 0)
    dd = results.get("max_drawdown_pct", 100)
    sharpe = results.get("sharpe_ratio", 0)
    daily_ret = results.get("avg_daily_return_pct", 0)
    tpd = results.get("trades_per_day", 0)

    checks = []
    checks.append(("Win rate > 60%", wr > 60))
    checks.append(("Max drawdown < 15%", dd < 15))
    checks.append(("Sharpe > 1.5", sharpe > 1.5))
    checks.append(("Avg daily return > 0.5%", daily_ret > 0.5))
    checks.append(("Trades/day 2-5", 1.5 <= tpd <= 6))
    checks.append(("Risk:Reward ~ 1:1", 0.7 <= results.get("risk_reward", 0) <= 1.5))

    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")

    print(f"\n  {passed}/{len(checks)} targets met")

    if passed >= 5:
        print("  -> Strategy looks viable for paper trading!")
    elif passed >= 4:
        print("  -> Strategy needs minor tuning before paper trading.")
    elif passed >= 3:
        print("  -> Strategy shows promise but needs work.")
    else:
        print("  -> Strategy needs significant rework.")


if __name__ == "__main__":
    main()
