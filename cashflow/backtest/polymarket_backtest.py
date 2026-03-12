"""Backtester for the Polymarket Longshot strategy.

Since we can't access historical Polymarket data in this environment,
we simulate realistic longshot event markets with known outcomes.

The simulation models:
  - Events across multiple categories (politics, crypto, sports, economics)
  - Realistic market prices (most longshots DO lose)
  - True probabilities that sometimes differ from market prices
  - Favorite-longshot bias (markets systematically underprice some longshots)
  - Time decay (prices drift as events approach resolution)
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from tabulate import tabulate

from cashflow.strategies.polymarket_longshot import (
    PolymarketLongshotStrategy,
    LongshotPortfolio,
    LongshotPosition,
)
from cashflow.data.arbitrage_feeds import (
    ArbitrageEngine,
    simulate_arbitrage_feeds,
    ExternalProbability,
)
from cashflow.utils.risk import max_drawdown, sharpe_ratio, sortino_ratio


# Event categories with their characteristics
EVENT_CATEGORIES = {
    "politics": {
        "frequency": 0.15,  # How often events in this category appear
        "base_longshot_rate": 0.12,  # Base rate of longshots actually winning
        "bias_factor": 1.3,  # How much the market underprices (>1 = underpriced)
    },
    "crypto": {
        "frequency": 0.25,
        "base_longshot_rate": 0.15,  # Crypto is volatile, more upsets
        "bias_factor": 1.4,
    },
    "sports": {
        "frequency": 0.20,
        "base_longshot_rate": 0.10,
        "bias_factor": 1.2,
    },
    "economics": {
        "frequency": 0.15,
        "base_longshot_rate": 0.18,  # Economic surprises happen
        "bias_factor": 1.5,
    },
    "entertainment": {
        "frequency": 0.10,
        "base_longshot_rate": 0.08,
        "bias_factor": 1.1,
    },
    "weather": {
        "frequency": 0.15,
        "base_longshot_rate": 0.14,
        "bias_factor": 1.3,
    },
}


def generate_simulated_longshot_events(
    n_events: int = 500,
    days: int = 365,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a realistic set of longshot event markets.

    Each event has:
      - A market price (what Polymarket shows, $0.01-$0.10)
      - A true probability (slightly higher due to favorite-longshot bias)
      - A category
      - An outcome (True/False)
      - Volume and liquidity
    """
    np.random.seed(seed)

    events = []
    start_date = datetime(2024, 1, 1)

    for i in range(n_events):
        # Pick a category weighted by frequency
        cats = list(EVENT_CATEGORIES.keys())
        weights = [EVENT_CATEGORIES[c]["frequency"] for c in cats]
        weights = [w / sum(weights) for w in weights]
        category = np.random.choice(cats, p=weights)
        cat_info = EVENT_CATEGORIES[category]

        # Generate market price ($0.01 to $0.10)
        market_price = np.random.uniform(0.02, 0.10)

        # True probability is higher than market price (favorite-longshot bias)
        # The bias factor represents how much the market undervalues longshots
        true_prob = min(market_price * cat_info["bias_factor"], 0.35)

        # Add some noise to true probability
        true_prob += np.random.normal(0, 0.03)
        true_prob = np.clip(true_prob, 0.01, 0.40)

        # Determine outcome based on true probability
        outcome = np.random.random() < true_prob

        # Random day within the range
        event_day = start_date + timedelta(days=np.random.randint(0, days))

        # Simulate volume/liquidity (correlated with how "interesting" the market is)
        base_volume = np.random.lognormal(8, 1.5)  # Median ~$3000
        liquidity = base_volume * np.random.uniform(0.1, 0.5)

        events.append({
            "event_id": f"evt_{i:04d}",
            "category": category,
            "question": f"Simulated {category} event #{i}",
            "market_price": round(market_price, 4),
            "true_prob": round(true_prob, 4),
            "outcome": outcome,
            "date": event_day,
            "volume": round(base_volume, 2),
            "liquidity": round(liquidity, 2),
            "days_to_resolution": np.random.randint(1, 90),
        })

    df = pd.DataFrame(events)
    df = df.sort_values("date").reset_index(drop=True)
    return df


class PolymarketLongshotBacktester:
    """Backtest the longshot strategy on simulated Polymarket events."""

    def __init__(self, config: dict = None, initial_capital: float = 250.0):
        config = config or {}
        self.initial_capital = initial_capital
        self.strategy = PolymarketLongshotStrategy(config.get("polymarket_longshot", {}))

    def run(
        self,
        n_events: int = 500,
        days: int = 365,
        seed: int = 42,
        model_accuracy: float = 0.70,
        use_arbitrage: bool = True,
    ) -> dict:
        """Run the backtest.

        Args:
            n_events: Number of simulated events
            days: Time span in days
            seed: Random seed for reproducibility
            model_accuracy: How accurate our model is at estimating true_prob.
                           0.70 means our model estimate correlates ~70% with truth.
            use_arbitrage: Whether to use cross-platform ensemble probabilities.
        """
        print("Generating simulated Polymarket longshot events...")
        events_df = generate_simulated_longshot_events(n_events, days, seed)
        print(f"  Generated {len(events_df)} events across {days} days")
        print(f"  Categories: {dict(events_df['category'].value_counts())}")
        print(f"  Avg market price: ${events_df['market_price'].mean():.4f}")
        print(f"  Actual win rate of longshots: {events_df['outcome'].mean():.1%}")
        print(f"  Model accuracy: {model_accuracy:.0%}")

        # Generate cross-platform probability feeds for ensemble
        arb_engine = ArbitrageEngine()
        arb_feeds = {}
        if use_arbitrage:
            print("  Generating cross-platform probability feeds...")
            sim_feeds = simulate_arbitrage_feeds(n_events=n_events, seed=seed + 200)
            for feed in sim_feeds:
                arb_feeds[feed["event_id"]] = feed["external_probs"]
            print(f"  Loaded {len(arb_feeds)} cross-platform feeds (ensemble mode)")
        else:
            print("  Single-model mode (no cross-platform data)")

        print("-" * 60)

        portfolio = LongshotPortfolio(
            capital=self.initial_capital,
            max_positions=self.strategy.max_positions,
        )
        portfolio.equity_curve.append(self.initial_capital)

        np.random.seed(seed + 100)  # Different seed for model noise

        trades_evaluated = 0
        trades_taken = 0
        ensemble_boosts = 0  # Track how often ensemble changed the decision

        # Process events chronologically
        for idx, (_, event) in enumerate(events_df.iterrows()):
            # Simulate our internal model's probability estimate
            noise = np.random.normal(0, (1 - model_accuracy) * 0.15)
            internal_model_prob = event["true_prob"] + noise
            internal_model_prob = np.clip(internal_model_prob, 0.01, 0.95)

            # Compute ensemble probability if arbitrage feeds available
            if use_arbitrage and idx in arb_feeds:
                external_probs = arb_feeds[idx]
                ensemble_prob, confidence, source_count = (
                    arb_engine.compute_ensemble_probability(
                        external_probs, internal_model_prob
                    )
                )
                # Use ensemble if we have enough sources and confidence
                if source_count >= 2 and confidence >= 0.3:
                    if abs(ensemble_prob - internal_model_prob) > 0.02:
                        ensemble_boosts += 1
                    model_prob = ensemble_prob
                else:
                    model_prob = internal_model_prob
            else:
                model_prob = internal_model_prob

            trades_evaluated += 1

            # Evaluate the market
            position = self.strategy.evaluate_market(
                market_id=event["event_id"],
                question=event["question"],
                category=event["category"],
                market_price=event["market_price"],
                model_prob=model_prob,
                portfolio=portfolio,
            )

            if position:
                self.strategy.open_position(position, portfolio)
                trades_taken += 1

                # Simulate price drift before resolution
                days_held = min(event["days_to_resolution"], 30)

                # Check for early exits (price changes)
                # Simulate intermediate price movement
                if days_held > 3:
                    # Price can drift toward true value
                    drift = (event["true_prob"] - event["market_price"]) * 0.3
                    new_price = event["market_price"] + drift + np.random.normal(0, 0.02)
                    new_price = np.clip(new_price, 0.005, 0.95)

                    prices = {position.market_id: new_price}
                    exits = self.strategy.check_early_exits(portfolio, prices)
                    for pos, price, reason in exits:
                        self.strategy.close_position(
                            pos, outcome=None, portfolio=portfolio,
                            exit_price=price, reason=reason,
                        )

                # If still open, settle at resolution
                if position.is_open:
                    self.strategy.close_position(
                        position,
                        outcome=event["outcome"],
                        portfolio=portfolio,
                        reason="settled",
                    )

            # Record equity at each event
            portfolio.equity_curve.append(portfolio.capital)

        # Compute results
        print(f"\nEvaluated {trades_evaluated} events, took {trades_taken} positions")
        if use_arbitrage:
            print(f"  Ensemble overrode internal model {ensemble_boosts} times")
        return self._compute_results(portfolio, events_df)

    def _compute_results(self, portfolio: LongshotPortfolio, events_df: pd.DataFrame) -> dict:
        """Compute comprehensive backtest results."""
        summary = self.strategy.portfolio_summary(portfolio)
        equity = portfolio.equity_curve

        # Returns series
        returns = []
        for i in range(1, len(equity)):
            if equity[i - 1] > 0:
                returns.append((equity[i] - equity[i - 1]) / equity[i - 1])

        # Category breakdown
        cat_stats = {}
        for pos in portfolio.closed:
            cat = pos.category
            if cat not in cat_stats:
                cat_stats[cat] = {"trades": 0, "wins": 0, "pnl": 0, "invested": 0}
            cat_stats[cat]["trades"] += 1
            if pos.pnl > 0:
                cat_stats[cat]["wins"] += 1
            cat_stats[cat]["pnl"] += pos.pnl
            cat_stats[cat]["invested"] += pos.total_cost

        # Price tier analysis
        tier_stats = {}
        for pos in portfolio.closed:
            if pos.entry_price <= 0.03:
                tier = "$0.01-$0.03"
            elif pos.entry_price <= 0.05:
                tier = "$0.03-$0.05"
            elif pos.entry_price <= 0.07:
                tier = "$0.05-$0.07"
            else:
                tier = "$0.07-$0.10"

            if tier not in tier_stats:
                tier_stats[tier] = {"trades": 0, "wins": 0, "pnl": 0}
            tier_stats[tier]["trades"] += 1
            if pos.pnl > 0:
                tier_stats[tier]["wins"] += 1
            tier_stats[tier]["pnl"] += pos.pnl

        results = {
            "strategy": "Polymarket Longshot Hunter",
            "initial_capital": self.initial_capital,
            "final_capital": round(equity[-1], 2),
            "total_pnl": round(equity[-1] - self.initial_capital, 2),
            "total_return_pct": round((equity[-1] / self.initial_capital - 1) * 100, 2),
            **summary,
            "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
            "sharpe_ratio": round(sharpe_ratio(returns), 2) if returns else 0,
            "sortino_ratio": round(min(sortino_ratio(returns), 99.99), 2) if returns else 0,
            "equity_curve": equity,
            "category_stats": cat_stats,
            "price_tier_stats": tier_stats,
            "positions": portfolio.closed,
        }

        return results


def print_longshot_results(results: dict):
    """Pretty-print the longshot backtest results."""
    if not results or results.get("total_trades", 0) == 0:
        print("No trades generated.")
        return

    print(f"\n{'='*60}")
    print(f"  {results.get('strategy', 'Polymarket Longshot Hunter')}")
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
        ["Largest Win", f"${results['largest_win']:.2f}"],
        ["Largest Loss", f"${results['largest_loss']:.2f}"],
        ["ROI on Capital Risked", f"{results['roi_pct']}%"],
        ["Max Drawdown", f"{results['max_drawdown_pct']}%"],
        ["Sharpe Ratio", results["sharpe_ratio"]],
        ["Sortino Ratio", results["sortino_ratio"]],
        ["Avg Entry Price", f"${results['avg_entry_price']:.4f}"],
        ["Avg Edge", f"{results['avg_edge']:.2%}"],
        ["Avg Kelly Fraction", f"{results['avg_kelly']:.4f}"],
    ]
    print(tabulate(metrics, headers=["Metric", "Value"], tablefmt="grid"))

    # Category breakdown
    if results.get("category_stats"):
        print(f"\nPerformance by Category:")
        cat_rows = []
        for cat, stats in sorted(results["category_stats"].items()):
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            roi = stats["pnl"] / stats["invested"] * 100 if stats["invested"] > 0 else 0
            cat_rows.append([
                cat,
                stats["trades"],
                f"{wr:.1f}%",
                f"${stats['pnl']:.2f}",
                f"{roi:.1f}%",
            ])
        print(tabulate(cat_rows, headers=["Category", "Trades", "Win Rate", "P&L", "ROI"], tablefmt="grid"))

    # Price tier breakdown
    if results.get("price_tier_stats"):
        print(f"\nPerformance by Entry Price Tier:")
        tier_rows = []
        for tier, stats in sorted(results["price_tier_stats"].items()):
            wr = stats["wins"] / stats["trades"] * 100 if stats["trades"] > 0 else 0
            tier_rows.append([tier, stats["trades"], f"{wr:.1f}%", f"${stats['pnl']:.2f}"])
        print(tabulate(tier_rows, headers=["Price Tier", "Trades", "Win Rate", "P&L"], tablefmt="grid"))
