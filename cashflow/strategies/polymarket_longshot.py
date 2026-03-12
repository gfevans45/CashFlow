"""Polymarket Longshot Hunter Strategy.

Core Thesis:
  Binary event markets systematically misprice low-probability events.
  "Favorite-longshot bias" is well-documented in academic literature:
  markets tend to OVERPRICE favorites and UNDERPRICE longshots slightly,
  but sometimes longshots get pushed TOO low by retail sentiment.

  We exploit this by:
    1. Scanning for YES tokens trading below $0.10 (implied <10% prob)
    2. Running our probability model to estimate TRUE probability
    3. If model_prob >= 15% but market says <10%, we have massive edge
    4. Size with Kelly Criterion (fractional) given the asymmetric payoff
    5. Portfolio of 10-20 uncorrelated longshots = diversified high-EV book

  Payoff Math (why this works):
    - Buy YES at $0.05, model says 15% chance
    - If YES wins: payout = $1.00, profit = $0.95 per contract (19:1)
    - If YES loses: loss = $0.05 per contract
    - EV = 0.15 * $0.95 - 0.85 * $0.05 = $0.1425 - $0.0425 = +$0.10
    - That's +200% expected return per contract!
    - Even at 10% accuracy on a diversified book, we profit.

  Risk Management:
    - Fractional Kelly (1/4 Kelly) prevents ruin
    - Max 5% of bankroll on any single event
    - Diversify across uncorrelated event categories
    - Hard floor: never bet more than we can afford to lose entirely
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


@dataclass
class LongshotPosition:
    """A position in a Polymarket longshot contract."""
    market_id: str
    question: str
    category: str
    entry_price: float         # What we paid per YES token ($0.01-$0.10)
    model_prob: float          # Our model's estimated true probability
    num_contracts: float       # Number of YES tokens purchased
    total_cost: float          # Total USDC spent
    max_payout: float          # num_contracts * $1.00 if YES wins
    expected_value: float      # model_prob * max_payout - total_cost
    kelly_fraction: float      # Kelly sizing used
    edge: float                # model_prob - entry_price
    entry_date: object = None
    exit_price: float = 0.0
    exit_date: object = None
    outcome: Optional[bool] = None  # True=YES won, False=NO won
    pnl: float = 0.0
    is_open: bool = True
    exit_reason: str = ""


@dataclass
class LongshotPortfolio:
    """Portfolio state for the longshot strategy."""
    capital: float = 250.0     # Starting allocation
    positions: list = field(default_factory=list)   # Open positions
    closed: list = field(default_factory=list)       # Settled positions
    equity_curve: list = field(default_factory=list)
    total_invested: float = 0.0  # Currently locked in positions
    max_positions: int = 20
    # Category tracking for diversification
    category_exposure: dict = field(default_factory=dict)

    @property
    def available_capital(self) -> float:
        return self.capital - self.total_invested

    @property
    def portfolio_value(self) -> float:
        """Total value = cash + marked-to-market positions."""
        position_value = sum(p.num_contracts * p.entry_price for p in self.positions)
        return self.capital - self.total_invested + position_value + self.total_invested


class PolymarketLongshotStrategy:
    """Longshot hunter: buy cheap YES tokens with positive expected value."""

    def __init__(self, config: dict = None):
        config = config or {}
        # Entry filters
        self.max_entry_price = config.get("max_entry_price", 0.10)
        self.min_entry_price = config.get("min_entry_price", 0.01)
        self.min_model_prob = config.get("min_model_prob", 0.15)
        self.min_edge = config.get("min_edge", 0.05)  # 5% minimum edge

        # Kelly sizing
        self.kelly_fraction = config.get("kelly_fraction", 0.25)  # Quarter-Kelly
        self.max_bet_pct = config.get("max_bet_pct", 5.0)  # Max 5% of bankroll
        self.min_bet_usd = config.get("min_bet_usd", 1.0)
        self.max_bet_usd = config.get("max_bet_usd", 25.0)

        # Diversification
        self.max_positions = config.get("max_positions", 20)
        self.max_per_category = config.get("max_per_category", 4)

        # Exit rules
        self.take_profit_multiplier = config.get("take_profit_multiplier", 3.0)
        self.stop_loss_pct = config.get("stop_loss_pct", 50.0)  # Sell if drops 50% from entry

    def kelly_for_binary(self, model_prob: float, market_price: float) -> float:
        """Kelly Criterion for binary contracts.

        For a binary event:
          - Cost to buy: market_price per contract
          - Win payout: $1.00 per contract
          - Odds (b) = (1 - market_price) / market_price = net profit per $1 risked

        Kelly fraction f* = (p * b - q) / b
          where p = model_prob, q = 1-p, b = odds

        We use fractional Kelly (1/4) for safety.
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        p = model_prob
        q = 1.0 - p
        b = (1.0 - market_price) / market_price  # Decimal odds - 1

        kelly = (p * b - q) / b
        if kelly <= 0:
            return 0.0

        return kelly * self.kelly_fraction

    def expected_value(self, model_prob: float, market_price: float, stake: float) -> float:
        """Calculate expected value of a position.

        EV = prob_win * profit_if_win - prob_lose * loss_if_lose
        """
        num_contracts = stake / market_price
        profit_if_win = num_contracts * (1.0 - market_price)
        loss_if_lose = stake
        return model_prob * profit_if_win - (1.0 - model_prob) * loss_if_lose

    def evaluate_market(
        self,
        market_id: str,
        question: str,
        category: str,
        market_price: float,
        model_prob: float,
        portfolio: LongshotPortfolio,
    ) -> Optional[LongshotPosition]:
        """Evaluate a single market for entry.

        Returns a LongshotPosition if the market meets all criteria.
        """
        # --- Filter Gates ---

        # Gate 1: Price range
        if market_price < self.min_entry_price or market_price > self.max_entry_price:
            return None

        # Gate 2: Model probability
        if model_prob < self.min_model_prob:
            return None

        # Gate 3: Minimum edge
        edge = model_prob - market_price
        if edge < self.min_edge:
            return None

        # Gate 4: Portfolio limits
        if len(portfolio.positions) >= self.max_positions:
            return None

        # Gate 5: Category diversification
        cat_count = portfolio.category_exposure.get(category, 0)
        if cat_count >= self.max_per_category:
            return None

        # Gate 6: Available capital
        if portfolio.available_capital < self.min_bet_usd:
            return None

        # --- Position Sizing via Kelly ---

        kelly_frac = self.kelly_for_binary(model_prob, market_price)
        if kelly_frac <= 0:
            return None

        # Calculate stake
        kelly_dollars = portfolio.capital * kelly_frac
        max_dollars = portfolio.capital * (self.max_bet_pct / 100.0)

        stake = min(
            kelly_dollars,
            max_dollars,
            self.max_bet_usd,
            portfolio.available_capital,
        )

        if stake < self.min_bet_usd:
            return None

        stake = round(stake, 2)
        num_contracts = stake / market_price
        max_payout = num_contracts * 1.0
        ev = self.expected_value(model_prob, market_price, stake)

        return LongshotPosition(
            market_id=market_id,
            question=question,
            category=category,
            entry_price=market_price,
            model_prob=model_prob,
            num_contracts=round(num_contracts, 2),
            total_cost=stake,
            max_payout=round(max_payout, 2),
            expected_value=round(ev, 4),
            kelly_fraction=round(kelly_frac, 4),
            edge=round(edge, 4),
            entry_date=datetime.now(),
        )

    def open_position(self, position: LongshotPosition, portfolio: LongshotPortfolio):
        """Add a position to the portfolio."""
        portfolio.positions.append(position)
        portfolio.total_invested += position.total_cost
        cat = position.category
        portfolio.category_exposure[cat] = portfolio.category_exposure.get(cat, 0) + 1

    def close_position(
        self,
        position: LongshotPosition,
        outcome: bool,
        portfolio: LongshotPortfolio,
        exit_price: float = None,
        reason: str = "settled",
    ):
        """Close/settle a position.

        For settlement: outcome=True means YES won (we get $1/contract).
        For early exit: use exit_price to calculate P&L from selling tokens.
        """
        position.is_open = False
        position.outcome = outcome
        position.exit_date = datetime.now()
        position.exit_reason = reason

        if reason == "settled":
            if outcome:
                # YES won - each contract pays $1.00
                payout = position.num_contracts * 1.0
                position.pnl = payout - position.total_cost
                position.exit_price = 1.0
            else:
                # NO won - our YES tokens are worthless
                position.pnl = -position.total_cost
                position.exit_price = 0.0
        elif exit_price is not None:
            # Early exit (sold tokens before resolution)
            revenue = position.num_contracts * exit_price
            position.pnl = revenue - position.total_cost
            position.exit_price = exit_price

        # Update portfolio: return cost + pnl (on win: cost+profit, on loss: cost+(-cost)=0)
        portfolio.total_invested -= position.total_cost
        portfolio.capital += position.pnl  # Only add the P&L (cost was never deducted from capital)
        if position in portfolio.positions:
            portfolio.positions.remove(position)
        portfolio.closed.append(position)

        cat = position.category
        if cat in portfolio.category_exposure:
            portfolio.category_exposure[cat] = max(0, portfolio.category_exposure[cat] - 1)

    def check_early_exits(
        self,
        portfolio: LongshotPortfolio,
        current_prices: dict,
    ) -> list:
        """Check if any positions should be exited early.

        Exit rules:
          1. Take profit: price rises to 3x entry (sell the tokens)
          2. Stop loss: price drops 50% from entry
        """
        exits = []
        for pos in list(portfolio.positions):
            current = current_prices.get(pos.market_id, pos.entry_price)

            # Take profit: price tripled
            if current >= pos.entry_price * self.take_profit_multiplier:
                exits.append((pos, current, "take_profit"))

            # Stop loss: price halved
            elif current <= pos.entry_price * (1 - self.stop_loss_pct / 100):
                exits.append((pos, current, "stop_loss"))

        return exits

    def portfolio_summary(self, portfolio: LongshotPortfolio) -> dict:
        """Generate portfolio summary statistics."""
        closed = portfolio.closed
        if not closed:
            return {
                "total_trades": 0,
                "message": "No closed positions yet",
                "capital": portfolio.capital,
                "open_positions": len(portfolio.positions),
                "invested": portfolio.total_invested,
            }

        wins = [p for p in closed if p.pnl > 0]
        losses = [p for p in closed if p.pnl <= 0]
        total_pnl = sum(p.pnl for p in closed)
        total_risked = sum(p.total_cost for p in closed)

        # Win rate (for settled positions)
        settled = [p for p in closed if p.exit_reason == "settled"]
        settled_wins = [p for p in settled if p.outcome]

        return {
            "total_trades": len(closed),
            "open_positions": len(portfolio.positions),
            "capital": round(portfolio.capital, 2),
            "invested": round(portfolio.total_invested, 2),
            "total_pnl": round(total_pnl, 2),
            "total_risked": round(total_risked, 2),
            "roi_pct": round(total_pnl / total_risked * 100, 2) if total_risked > 0 else 0,
            "win_count": len(wins),
            "loss_count": len(losses),
            "win_rate_pct": round(len(wins) / len(closed) * 100, 2),
            "avg_win": round(np.mean([p.pnl for p in wins]), 2) if wins else 0,
            "avg_loss": round(np.mean([p.pnl for p in losses]), 2) if losses else 0,
            "largest_win": round(max(p.pnl for p in wins), 2) if wins else 0,
            "largest_loss": round(min(p.pnl for p in losses), 2) if losses else 0,
            "settled_count": len(settled),
            "settled_win_rate": round(len(settled_wins) / len(settled) * 100, 2) if settled else 0,
            "avg_entry_price": round(np.mean([p.entry_price for p in closed]), 4),
            "avg_edge": round(np.mean([p.edge for p in closed]), 4),
            "avg_kelly": round(np.mean([p.kelly_fraction for p in closed]), 4),
            "categories": dict(portfolio.category_exposure),
        }
