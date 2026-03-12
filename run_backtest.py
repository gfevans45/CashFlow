#!/usr/bin/env python3
"""CashFlow - Run strategy comparison backtest.

Usage:
    python run_backtest.py              # Run full comparison
    python run_backtest.py --stock      # Stock strategy only
    python run_backtest.py --prediction # Prediction market only
    python run_backtest.py --plot       # Generate equity curve charts
"""
import sys
import argparse
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

from cashflow.simulation.runner import run_comparison_backtest
from cashflow.backtest.engine import StockBacktester, PredictionBacktester, print_results
from cashflow.utils.config import load_config


def plot_equity_curves(stock_results: dict, pred_results: dict, save_path: str = "data/equity_curves.png"):
    """Generate equity curve comparison chart."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CashFlow Strategy Comparison", fontsize=14, fontweight="bold")

    # Stock equity curve
    if stock_results and "equity_curve" in stock_results:
        ax = axes[0][0]
        eq = stock_results["equity_curve"]
        ax.plot(eq, color="blue", linewidth=1.5)
        ax.axhline(y=eq[0], color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Stock Strategy (Return: {stock_results['total_return_pct']}%)")
        ax.set_ylabel("Portfolio Value ($)")
        ax.fill_between(range(len(eq)), eq[0], eq, alpha=0.1, color="blue")

    # Prediction market equity curve
    if pred_results and "equity_curve" in pred_results:
        ax = axes[0][1]
        eq = pred_results["equity_curve"]
        ax.plot(eq, color="green", linewidth=1.5)
        ax.axhline(y=eq[0], color="gray", linestyle="--", alpha=0.5)
        ax.set_title(f"Prediction Market (Return: {pred_results['total_return_pct']}%)")
        ax.set_ylabel("Portfolio Value ($)")
        ax.fill_between(range(len(eq)), eq[0], eq, alpha=0.1, color="green")

    # Trade P&L distribution (stock)
    if stock_results and "trades" in stock_results:
        ax = axes[1][0]
        pnls = [t.pnl for t in stock_results["trades"]]
        colors = ["green" if p > 0 else "red" for p in pnls]
        ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.7)
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_title("Stock Trade P&L")
        ax.set_ylabel("P&L ($)")
        ax.set_xlabel("Trade #")

    # Combined equity
    ax = axes[1][1]
    if stock_results and pred_results and "equity_curve" in stock_results and "equity_curve" in pred_results:
        seq = stock_results["equity_curve"]
        peq = pred_results["equity_curve"]
        # Combine by normalizing to same length
        max_len = max(len(seq), len(peq))
        combined = []
        for i in range(max_len):
            s = seq[min(i, len(seq) - 1)]
            p = peq[min(i, len(peq) - 1)]
            combined.append(s + p)
        ax.plot(combined, color="purple", linewidth=2)
        ax.axhline(y=combined[0], color="gray", linestyle="--", alpha=0.5)
        initial = stock_results["initial_capital"] + pred_results["initial_capital"]
        total_ret = (combined[-1] / initial - 1) * 100
        ax.set_title(f"Combined Portfolio (Return: {total_ret:.2f}%)")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Time (bars)")

    plt.tight_layout()
    Path(save_path).parent.mkdir(exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="CashFlow Strategy Backtest")
    parser.add_argument("--stock", action="store_true", help="Run stock strategy only")
    parser.add_argument("--prediction", action="store_true", help="Run prediction market only")
    parser.add_argument("--plot", action="store_true", help="Generate charts")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.stock:
        bt = StockBacktester(config, initial_capital=config["general"]["initial_capital"])
        results = bt.run(
            start_date=config["backtest"]["start_date"],
            end_date=config["backtest"]["end_date"],
        )
        print_results(results)
        if args.plot and results:
            plot_equity_curves(results, {})
    elif args.prediction:
        bt = PredictionBacktester(config, initial_capital=config["general"]["initial_capital"])
        results = bt.run()
        print_results(results)
        if args.plot and results:
            plot_equity_curves({}, results)
    else:
        stock_results, pred_results = run_comparison_backtest(args.config)
        if args.plot:
            plot_equity_curves(stock_results, pred_results)


if __name__ == "__main__":
    main()
