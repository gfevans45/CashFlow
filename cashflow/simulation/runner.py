"""Paper trading simulation runner.

Runs both strategies in parallel paper-trading mode and tracks
real-time performance for ROI comparison.
"""
import json
import time
from datetime import datetime
from pathlib import Path

from cashflow.backtest.engine import StockBacktester, PredictionBacktester, print_results
from cashflow.utils.config import load_config


def run_comparison_backtest(config_path: str = None):
    """Run both strategies through backtesting and compare ROI."""
    config = load_config(config_path)

    initial_capital = config["general"]["initial_capital"]

    print("=" * 60)
    print("  CashFlow - Strategy Comparison Backtest")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    # --- Stock Strategy Backtest ---
    print("\n[1/2] Running Stock Mean Reversion Backtest...")
    stock_bt = StockBacktester(config, initial_capital=initial_capital / 2)
    stock_results = stock_bt.run(
        start_date=config["backtest"]["start_date"],
        end_date=config["backtest"]["end_date"],
    )
    print_results(stock_results, "Stock Strategy")

    # --- Prediction Market Backtest ---
    print("\n[2/2] Running Prediction Market Backtest...")
    pred_bt = PredictionBacktester(config, initial_capital=initial_capital / 2)
    pred_results = pred_bt.run(lookback_days=365)
    print_results(pred_results, "Prediction Market")

    # --- Comparison ---
    print("\n" + "=" * 60)
    print("  HEAD-TO-HEAD COMPARISON")
    print("=" * 60)

    comparison = []
    for label, r in [("Stock", stock_results), ("Prediction", pred_results)]:
        if r and "total_return_pct" in r:
            comparison.append({
                "Strategy": label,
                "Return": f"{r['total_return_pct']}%",
                "Trades": r["total_trades"],
                "Win Rate": f"{r['win_rate_pct']}%",
                "Sharpe": r["sharpe_ratio"],
                "Max DD": f"{r['max_drawdown_pct']}%",
                "Final $": f"${r['final_capital']:,.2f}",
            })

    if comparison:
        from tabulate import tabulate
        print(tabulate(comparison, headers="keys", tablefmt="grid"))

        # Combined performance
        combined_final = sum(
            r.get("final_capital", 0)
            for r in [stock_results, pred_results]
            if r and "final_capital" in r
        )
        combined_return = (combined_final / initial_capital - 1) * 100

        print(f"\n  Combined Portfolio:")
        print(f"    Initial: ${initial_capital:,.2f}")
        print(f"    Final:   ${combined_final:,.2f}")
        print(f"    Return:  {combined_return:.2f}%")
        print(f"    Monthly: ~{combined_return / 12:.2f}% (if annualized)")

        # Determine winner
        stock_ret = stock_results.get("total_return_pct", 0)
        pred_ret = pred_results.get("total_return_pct", 0)
        if stock_ret > pred_ret:
            print(f"\n  >> WINNER: Stock Strategy (+{stock_ret}%)")
            print(f"     Recommendation: Allocate more capital to stocks")
        elif pred_ret > stock_ret:
            print(f"\n  >> WINNER: Prediction Market (+{pred_ret}%)")
            print(f"     Recommendation: Allocate more capital to prediction markets")
        else:
            print(f"\n  >> TIE: Both strategies returned {stock_ret}%")

    # Save results
    results_path = Path(__file__).parent.parent.parent / "data"
    results_path.mkdir(exist_ok=True)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "initial_capital": initial_capital,
        "stock_results": {k: v for k, v in stock_results.items() if k not in ("equity_curve", "trades")} if stock_results else {},
        "prediction_results": {k: v for k, v in pred_results.items() if k not in ("equity_curve", "contracts")} if pred_results else {},
    }

    with open(results_path / "backtest_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to data/backtest_results.json")

    return stock_results, pred_results


if __name__ == "__main__":
    run_comparison_backtest()
