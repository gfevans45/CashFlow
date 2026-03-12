#!/usr/bin/env python3
"""CashFlow - Run strategy comparison backtest.

Usage:
    python run_backtest.py                # Run full comparison (all 3 strategies)
    python run_backtest.py --stock        # Stock strategy only
    python run_backtest.py --prediction   # Kalshi prediction market only
    python run_backtest.py --polymarket   # Polymarket longshot hunter only
    python run_backtest.py --plot         # Generate equity curve charts
"""
import sys
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from cashflow.simulation.runner import run_comparison_backtest
from cashflow.backtest.engine import StockBacktester, PredictionBacktester, print_results
from cashflow.backtest.polymarket_backtest import (
    PolymarketLongshotBacktester,
    print_longshot_results,
)
from cashflow.utils.config import load_config


def plot_all_strategies(stock_r, pred_r, poly_r, save_path="data/equity_curves.png"):
    """Generate equity curve comparison chart for all strategies."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CashFlow Strategy Comparison", fontsize=14, fontweight="bold")

    # Stock equity curve
    if stock_r and "equity_curve" in stock_r:
        ax = axes[0][0]
        eq = stock_r["equity_curve"]
        ax.plot(eq, color="blue", linewidth=1.5)
        ax.axhline(y=eq[0], color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Stock Strategy ({stock_r['total_return_pct']}%)")
        ax.set_ylabel("Portfolio Value ($)")
        ax.fill_between(range(len(eq)), eq[0], eq, alpha=0.1, color="blue")

    # Prediction market equity curve
    if pred_r and "equity_curve" in pred_r:
        ax = axes[0][1]
        eq = pred_r["equity_curve"]
        ax.plot(eq, color="green", linewidth=1.5)
        ax.axhline(y=eq[0], color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Kalshi-style ({pred_r['total_return_pct']}%)")
        ax.set_ylabel("Portfolio Value ($)")
        ax.fill_between(range(len(eq)), eq[0], eq, alpha=0.1, color="green")

    # Polymarket longshot equity curve
    if poly_r and "equity_curve" in poly_r:
        ax = axes[1][0]
        eq = poly_r["equity_curve"]
        ax.plot(eq, color="orange", linewidth=1.5)
        ax.axhline(y=eq[0], color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Polymarket Longshots ({poly_r['total_return_pct']}%)")
        ax.set_ylabel("Portfolio Value ($)")
        ax.set_xlabel("Event #")
        ax.fill_between(range(len(eq)), eq[0], eq, alpha=0.1, color="orange")

        # Annotate wins with markers
        if "positions" in poly_r:
            wins_idx = [i+1 for i, p in enumerate(poly_r["positions"]) if p.pnl > 0]
            for idx in wins_idx:
                if idx < len(eq):
                    ax.plot(idx, eq[idx], "^", color="green", markersize=4, alpha=0.7)

    # Combined portfolio
    ax = axes[1][1]
    all_results = [(r, c) for r, c in [
        (stock_r, "blue"), (pred_r, "green"), (poly_r, "orange")
    ] if r and "equity_curve" in r]

    if all_results:
        max_len = max(len(r["equity_curve"]) for r, _ in all_results)
        combined = []
        for i in range(max_len):
            total = 0
            for r, _ in all_results:
                eq = r["equity_curve"]
                total += eq[min(i, len(eq) - 1)]
            combined.append(total)

        ax.plot(combined, color="purple", linewidth=2)
        initial = sum(r["initial_capital"] for r, _ in all_results)
        ax.axhline(y=initial, color="gray", linestyle="--", alpha=0.5)
        total_ret = (combined[-1] / initial - 1) * 100
        ax.set_title(f"Combined Portfolio ({total_ret:.1f}%)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Time (events/bars)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {save_path}")
    plt.close()


def run_polymarket_backtest(config):
    """Run the Polymarket longshot strategy backtest."""
    capital = config["general"]["initial_capital"]
    # Allocate 1/3 of capital to Polymarket
    poly_capital = capital / 3

    print(f"\n[Polymarket Longshot] Running backtest with ${poly_capital:.2f}...")
    bt = PolymarketLongshotBacktester(config, initial_capital=poly_capital)
    results = bt.run(n_events=500, days=365, seed=42, model_accuracy=0.70)
    print_longshot_results(results)
    return results


def main():
    parser = argparse.ArgumentParser(description="CashFlow Strategy Backtest")
    parser.add_argument("--stock", action="store_true", help="Run stock strategy only")
    parser.add_argument("--prediction", action="store_true", help="Run Kalshi prediction market only")
    parser.add_argument("--polymarket", action="store_true", help="Run Polymarket longshot only")
    parser.add_argument("--plot", action="store_true", help="Generate charts")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    capital = config["general"]["initial_capital"]

    if args.stock:
        bt = StockBacktester(config, initial_capital=capital)
        results = bt.run(
            start_date=config["backtest"]["start_date"],
            end_date=config["backtest"]["end_date"],
        )
        print_results(results)
    elif args.prediction:
        bt = PredictionBacktester(config, initial_capital=capital)
        results = bt.run()
        print_results(results)
    elif args.polymarket:
        results = run_polymarket_backtest(config)
        if args.plot and results:
            plot_all_strategies({}, {}, results)
    else:
        # Run all three strategies with equal allocation
        third = capital / 3
        print("=" * 60)
        print("  CashFlow - Full Strategy Comparison")
        print(f"  Capital: ${capital:.2f} (${third:.2f} per strategy)")
        print("=" * 60)

        # Stock
        print(f"\n[1/3] Stock Mean Reversion...")
        stock_bt = StockBacktester(config, initial_capital=third)
        stock_r = stock_bt.run(
            start_date=config["backtest"]["start_date"],
            end_date=config["backtest"]["end_date"],
        )
        print_results(stock_r)

        # Kalshi-style prediction market
        print(f"\n[2/3] Kalshi Prediction Market...")
        pred_bt = PredictionBacktester(config, initial_capital=third)
        pred_r = pred_bt.run()
        print_results(pred_r)

        # Polymarket longshots
        print(f"\n[3/3] Polymarket Longshot Hunter...")
        poly_bt = PolymarketLongshotBacktester(config, initial_capital=third)
        poly_r = poly_bt.run(n_events=500, days=365, seed=42, model_accuracy=0.70)
        print_longshot_results(poly_r)

        # Head-to-head comparison
        print(f"\n{'='*60}")
        print("  HEAD-TO-HEAD COMPARISON")
        print(f"{'='*60}")

        from tabulate import tabulate
        comparison = []
        for label, r in [("Stock", stock_r), ("Kalshi", pred_r), ("Polymarket", poly_r)]:
            if r and "total_return_pct" in r:
                comparison.append({
                    "Strategy": label,
                    "Return": f"{r['total_return_pct']}%",
                    "Trades": r.get("total_trades", 0),
                    "Win Rate": f"{r.get('win_rate_pct', 0)}%",
                    "Sharpe": r.get("sharpe_ratio", 0),
                    "Max DD": f"{r.get('max_drawdown_pct', 0)}%",
                    "Final $": f"${r['final_capital']:,.2f}",
                })
        if comparison:
            print(tabulate(comparison, headers="keys", tablefmt="grid"))

            combined = sum(r["final_capital"] for _, r in
                          [("s", stock_r), ("p", pred_r), ("m", poly_r)]
                          if r and "final_capital" in r)
            print(f"\n  Combined: ${capital:.2f} -> ${combined:,.2f} ({(combined/capital-1)*100:.1f}%)")

            # Winner
            all_r = [("Stock", stock_r), ("Kalshi", pred_r), ("Polymarket", poly_r)]
            best = max(all_r, key=lambda x: x[1].get("total_return_pct", -999) if x[1] else -999)
            print(f"  WINNER: {best[0]} ({best[1]['total_return_pct']}%)")

        if args.plot:
            plot_all_strategies(stock_r, pred_r, poly_r)

        # Save results
        import json
        results_path = Path("data")
        results_path.mkdir(exist_ok=True)
        summary = {
            "timestamp": str(pd.Timestamp.now()) if 'pd' in dir() else str(datetime.now()),
            "initial_capital": capital,
        }
        for label, r in [("stock", stock_r), ("kalshi", pred_r), ("polymarket", poly_r)]:
            if r:
                summary[label] = {k: v for k, v in r.items()
                                  if k not in ("equity_curve", "trades", "contracts", "positions")}
        with open(results_path / "backtest_results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime
    main()
