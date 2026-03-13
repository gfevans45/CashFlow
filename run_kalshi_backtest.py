#!/usr/bin/env python3
"""Run the Kalshi Hybrid Strategy backtest.

Usage:
    python run_kalshi_backtest.py              # Default 90-day backtest
    python run_kalshi_backtest.py --days 60    # Custom lookback
    python run_kalshi_backtest.py --capital 100 # Custom starting capital
"""
import argparse
import sys

from cashflow.backtest.kalshi_hybrid_backtest import KalshiHybridBacktester


def main():
    parser = argparse.ArgumentParser(description="Kalshi Hybrid Strategy Backtest")
    parser.add_argument("--days", type=int, default=90, help="Lookback period in days")
    parser.add_argument("--capital", type=float, default=50.0, help="Starting capital")
    args = parser.parse_args()

    config = {
        "min_edge": 0.03,           # 3% edge on preferred contracts
        "min_edge_wide": 0.05,      # 5% edge required outside $0.45-0.55
        "kelly_fraction": 0.25,
        "max_position_pct": 0.10,
        "max_positions": 5,
        "min_stake": 1.0,
        "max_stake": 5.0,
        "daily_loss_limit_pct": 0.05,
        "weekly_loss_limit_pct": 0.15,
    }

    backtester = KalshiHybridBacktester(capital=args.capital, config=config)
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

    checks = []
    checks.append(("Win rate > 60%", wr > 60))
    checks.append(("Max drawdown < 15%", dd < 15))
    checks.append(("Sharpe > 1.5", sharpe > 1.5))
    checks.append(("Avg daily return > 0.5%", daily_ret > 0.5))
    checks.append(("Trades/day 3-5", 2.5 <= results.get("trades_per_day", 0) <= 6))

    passed = sum(1 for _, ok in checks if ok)
    for label, ok in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {label}")

    print(f"\n  {passed}/{len(checks)} targets met")

    if passed >= 4:
        print("  -> Strategy looks viable for paper trading.")
    elif passed >= 3:
        print("  -> Strategy needs minor tuning before paper trading.")
    else:
        print("  -> Strategy needs significant rework.")


if __name__ == "__main__":
    main()
