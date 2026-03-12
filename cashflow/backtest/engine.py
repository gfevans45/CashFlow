"""Backtesting engine for both stock and prediction market strategies."""
import pandas as pd
import numpy as np
from datetime import datetime
from tabulate import tabulate

from cashflow.strategies.stock_mean_reversion import (
    StockMeanReversionStrategy,
    StrategyState,
    Signal,
)
from cashflow.strategies.prediction_market import (
    PredictionMarketStrategy,
    PredictionState,
    EventContract,
)
from cashflow.data.market_data import fetch_daily_for_backtest, add_technical_indicators
from cashflow.data.prediction_data import fetch_sp500_range_data, simulate_kalshi_events
from cashflow.data.synthetic import generate_multi_stock_universe, generate_sp500_for_prediction
from cashflow.utils.risk import max_drawdown, sharpe_ratio, sortino_ratio


class StockBacktester:
    """Backtest the stock mean-reversion strategy."""

    def __init__(self, config: dict, initial_capital: float = 500.0):
        self.config = config
        self.initial_capital = initial_capital
        self.strategy = StockMeanReversionStrategy(config.get("stock_strategy", {}))
        self.slippage_pct = config.get("backtest", {}).get("slippage_pct", 0.05) / 100

    def run(
        self,
        symbols: list = None,
        start_date: str = "2024-01-01",
        end_date: str = "2025-12-31",
    ) -> dict:
        """Run backtest over historical data."""
        if symbols is None:
            symbols = self.config.get("stock_strategy", {}).get(
                "symbols", ["SPY", "QQQ"]
            )

        state = StrategyState(capital=self.initial_capital, max_positions=self.strategy.max_positions)
        state.equity_curve.append(self.initial_capital)

        # Fetch data for all symbols (fallback to synthetic if API unavailable)
        print(f"Fetching data for {len(symbols)} symbols...")
        all_data = {}
        try:
            for symbol in symbols:
                df = fetch_daily_for_backtest(symbol, start_date, end_date)
                if not df.empty:
                    all_data[symbol] = df
                    print(f"  {symbol}: {len(df)} bars (live)")
        except Exception as e:
            print(f"  Live data unavailable ({type(e).__name__}), using synthetic data...")
            all_data = {}

        if not all_data:
            print("  Generating synthetic market data for backtesting...")
            synthetic = generate_multi_stock_universe(symbols, days=500, start_date=start_date)
            for symbol, df in synthetic.items():
                df = add_technical_indicators(df)
                if not df.empty:
                    all_data[symbol] = df
                    print(f"  {symbol}: {len(df)} bars (synthetic)")

        # Get union of all dates
        all_dates = sorted(set().union(*(df.index for df in all_data.values())))
        print(f"\nBacktesting from {all_dates[0].date()} to {all_dates[-1].date()}")
        print(f"Total bars: {len(all_dates)}")
        print("-" * 60)

        # Iterate through each bar
        for i, date in enumerate(all_dates):
            for symbol, df in all_data.items():
                if date not in df.index:
                    continue

                # Get data up to current bar (no lookahead)
                historical = df.loc[:date]
                if len(historical) < 30:
                    continue

                signal = self.strategy.generate_signal(historical, symbol, state)
                if signal != Signal.HOLD:
                    current_bar = historical.iloc[-1]
                    # Apply slippage
                    self.strategy.execute_signal(signal, symbol, current_bar, state)

            # Record equity
            portfolio_value = state.capital
            for sym, trade in state.positions.items():
                if sym in all_data and date in all_data[sym].index:
                    current_price = all_data[sym].loc[date, "close"]
                    if trade.direction == "long":
                        portfolio_value += trade.shares * current_price
                    else:
                        portfolio_value += trade.shares * (2 * trade.entry_price - current_price)
            state.equity_curve.append(portfolio_value)

        # Close any remaining positions at last price
        for symbol in list(state.positions.keys()):
            if symbol in all_data:
                last_bar = all_data[symbol].iloc[-1]
                trade = state.positions[symbol]
                if trade.direction == "long":
                    exit_signal = Signal.EXIT_LONG
                else:
                    exit_signal = Signal.EXIT_SHORT
                self.strategy.execute_signal(exit_signal, symbol, last_bar, state)

        return self._compute_results(state)

    def _compute_results(self, state: StrategyState) -> dict:
        """Compute performance metrics from backtest results."""
        equity = state.equity_curve
        trades = state.closed_trades

        if not trades:
            return {
                "total_return": 0,
                "total_trades": 0,
                "message": "No trades generated",
            }

        # Returns
        returns = []
        for i in range(1, len(equity)):
            if equity[i - 1] > 0:
                returns.append((equity[i] - equity[i - 1]) / equity[i - 1])

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_pnl = sum(t.pnl for t in trades)
        win_rate = len(wins) / len(trades) if trades else 0
        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([t.pnl for t in losses]) if losses else 0
        profit_factor = abs(sum(t.pnl for t in wins) / sum(t.pnl for t in losses)) if losses and sum(t.pnl for t in losses) != 0 else float("inf")

        results = {
            "strategy": "Stock Mean Reversion + Momentum",
            "initial_capital": self.initial_capital,
            "final_capital": round(equity[-1], 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((equity[-1] / self.initial_capital - 1) * 100, 2),
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(profit_factor, 2),
            "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
            "sharpe_ratio": round(sharpe_ratio(returns), 2) if returns else 0,
            "sortino_ratio": round(sortino_ratio(returns), 2) if returns else 0,
            "equity_curve": equity,
            "trades": trades,
        }

        # Print exit reason breakdown
        exit_reasons = {}
        for t in trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1
        results["exit_reasons"] = exit_reasons

        return results


class PredictionBacktester:
    """Backtest the prediction market strategy."""

    def __init__(self, config: dict, initial_capital: float = 500.0):
        self.config = config
        self.initial_capital = initial_capital
        self.strategy = PredictionMarketStrategy(config.get("prediction_market", {}))

    def run(self, lookback_days: int = 365) -> dict:
        """Run backtest using simulated Kalshi events from S&P 500 data."""
        print("Fetching S&P 500 data for prediction market simulation...")
        try:
            sp500_data = fetch_sp500_range_data(lookback_days=lookback_days)
        except Exception as e:
            print(f"  Live data unavailable ({type(e).__name__}), using synthetic data...")
            sp500_data = pd.DataFrame()

        if sp500_data.empty:
            print("  Generating synthetic S&P 500 data...")
            sp500_data = generate_sp500_for_prediction(days=lookback_days)

        print(f"  Got {len(sp500_data)} days of data")

        # Generate simulated events
        events_df = simulate_kalshi_events(sp500_data)
        print(f"  Generated {len(events_df)} simulated events")
        print("-" * 60)

        state = PredictionState(capital=self.initial_capital, max_open_positions=5)
        state.equity_curve.append(self.initial_capital)

        # Group events by date
        for date, day_events in events_df.groupby("date"):
            # Get historical data up to this date for model
            hist = sp500_data.loc[:date]

            for _, event_row in day_events.iterrows():
                event_type = event_row["event_type"]
                actual = event_row["actual_outcome"]

                # Our model estimates probability from history
                model_prob = self.strategy.generate_model_probability(
                    event_type, hist, lookback=20
                )

                # Simulate what the market price would be
                market_price = self.strategy.simulate_market_price(
                    event_row["model_prob"], noise=0.08
                )

                # Evaluate if we should trade
                contract = self.strategy.evaluate_event(
                    event_type, model_prob, market_price, state
                )

                if contract:
                    contract.date = date
                    state.open_positions.append(contract)
                    state.capital -= contract.stake

                    # Immediately settle (these are daily events)
                    self.strategy.settle_contract(contract, actual, state)

            state.equity_curve.append(state.capital)

        return self._compute_results(state)

    def _compute_results(self, state: PredictionState) -> dict:
        """Compute performance metrics."""
        equity = state.equity_curve
        contracts = state.settled_positions

        if not contracts:
            return {"total_return": 0, "total_trades": 0, "message": "No trades"}

        returns = []
        for i in range(1, len(equity)):
            if equity[i - 1] > 0:
                returns.append((equity[i] - equity[i - 1]) / equity[i - 1])

        wins = [c for c in contracts if c.pnl > 0]
        losses = [c for c in contracts if c.pnl <= 0]
        total_pnl = sum(c.pnl for c in contracts)

        # Breakdown by event type
        by_type = {}
        for c in contracts:
            if c.event_type not in by_type:
                by_type[c.event_type] = {"trades": 0, "wins": 0, "pnl": 0}
            by_type[c.event_type]["trades"] += 1
            if c.pnl > 0:
                by_type[c.event_type]["wins"] += 1
            by_type[c.event_type]["pnl"] += c.pnl

        win_rate = len(wins) / len(contracts) if contracts else 0
        avg_win = np.mean([c.pnl for c in wins]) if wins else 0
        avg_loss = np.mean([c.pnl for c in losses]) if losses else 0

        return {
            "strategy": "Prediction Market (Kalshi-style)",
            "initial_capital": self.initial_capital,
            "final_capital": round(equity[-1], 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round((equity[-1] / self.initial_capital - 1) * 100, 2),
            "total_trades": len(contracts),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "max_drawdown_pct": round(max_drawdown(equity) * 100, 2) if len(equity) > 1 else 0,
            "sharpe_ratio": round(sharpe_ratio(returns), 2) if returns else 0,
            "sortino_ratio": round(sortino_ratio(returns), 2) if returns else 0,
            "equity_curve": equity,
            "by_event_type": by_type,
            "contracts": contracts,
        }


def print_results(results: dict, title: str = ""):
    """Pretty-print backtest results."""
    if not results or "message" in results:
        print(f"\n{title}: {results.get('message', 'No results')}")
        return

    print(f"\n{'='*60}")
    print(f"  {results.get('strategy', title)}")
    print(f"{'='*60}")

    metrics = [
        ["Initial Capital", f"${results['initial_capital']:,.2f}"],
        ["Final Capital", f"${results['final_capital']:,.2f}"],
        ["Total P&L", f"${results['total_pnl']:,.2f}"],
        ["Total Return", f"{results['total_return_pct']}%"],
        ["Total Trades", results["total_trades"]],
        ["Win Rate", f"{results['win_rate_pct']}%"],
        ["Avg Win", f"${results['avg_win']:.2f}"],
        ["Avg Loss", f"${results['avg_loss']:.2f}"],
        ["Max Drawdown", f"{results['max_drawdown_pct']}%"],
        ["Sharpe Ratio", results["sharpe_ratio"]],
        ["Sortino Ratio", results["sortino_ratio"]],
    ]

    if "profit_factor" in results:
        metrics.append(["Profit Factor", results["profit_factor"]])

    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid"))

    if "exit_reasons" in results:
        print(f"\nExit Reasons:")
        for reason, count in results["exit_reasons"].items():
            print(f"  {reason}: {count}")

    if "by_event_type" in results:
        print(f"\nPerformance by Event Type:")
        rows = []
        for etype, stats in results["by_event_type"].items():
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            rows.append([etype, stats["trades"], f"{wr:.1f}%", f"${stats['pnl']:.2f}"])
        print(tabulate(rows, headers=["Event Type", "Trades", "Win Rate", "P&L"], tablefmt="grid"))
