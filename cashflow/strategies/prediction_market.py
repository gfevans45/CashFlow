"""Prediction Market Strategy for Kalshi-style Event Contracts.

Strategy Logic:
  We model the probability of binary events using historical data and
  compare our model's probability against the market price.

  EDGE DETECTION:
    1. Model probability from historical data (e.g., "S&P 500 up today" ~53%)
    2. Market price for the contract (e.g., $0.48 = 48% implied prob)
    3. If model_prob - market_prob > min_edge (5%), we have a trade
    4. Size using fractional Kelly Criterion

  EVENT CATEGORIES:
    - S&P 500 daily direction/range (use historical volatility + trend)
    - Economic data surprises (CPI, jobs - use consensus vs. trend)
    - Weather events (temperature ranges - use NOAA data)

  This can be backtested using simulated events from historical S&P data.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EventContract:
    event_id: str
    event_type: str
    description: str
    market_price: float  # Market implied probability (0.0 to 1.0)
    model_prob: float    # Our model's estimated probability
    date: object
    outcome: Optional[bool] = None  # True if event occurred
    position: Optional[str] = None  # "yes" or "no"
    stake: float = 0.0
    pnl: float = 0.0
    settled: bool = False


@dataclass
class PredictionState:
    capital: float = 500.0
    open_positions: list = field(default_factory=list)
    settled_positions: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    max_open_positions: int = 5


class PredictionMarketStrategy:
    """Event-driven prediction market strategy."""

    def __init__(self, config: dict):
        self.config = config
        self.min_edge_pct = config.get("min_edge_pct", 5.0) / 100
        self.kelly_fraction = config.get("kelly_fraction", 0.25)
        self.max_position_size = config.get("max_position_size", 25.0)
        self.max_open = config.get("max_open_positions", 5)

    def kelly_size(self, model_prob: float, market_price: float) -> float:
        """Calculate Kelly-optimal stake for a binary contract.

        For a binary bet at price p with true probability q:
          Kelly fraction = (q * (1/p - 1) - (1-q)) / (1/p - 1)
                        = (q - p) / (1 - p)
        """
        if market_price <= 0 or market_price >= 1:
            return 0.0

        # For "yes" position
        edge = model_prob - market_price
        if edge <= 0:
            return 0.0

        odds = (1.0 / market_price) - 1.0
        kelly = (model_prob * odds - (1 - model_prob)) / odds

        # Apply fractional Kelly
        return max(0.0, kelly * self.kelly_fraction)

    def evaluate_event(
        self,
        event_type: str,
        model_prob: float,
        market_price: float,
        state: PredictionState,
    ) -> Optional[EventContract]:
        """Evaluate whether to take a position on an event.

        Returns an EventContract with position details if we find edge.
        """
        if len(state.open_positions) >= self.max_open:
            return None

        # Check for YES edge
        yes_edge = model_prob - market_price
        # Check for NO edge
        no_edge = (1 - model_prob) - (1 - market_price)

        position = None
        edge = 0.0

        if yes_edge >= self.min_edge_pct:
            position = "yes"
            edge = yes_edge
            kelly = self.kelly_size(model_prob, market_price)
        elif no_edge >= self.min_edge_pct:
            position = "no"
            edge = no_edge
            # For NO: we're buying at (1 - market_price) with prob (1 - model_prob)
            kelly = self.kelly_size(1 - model_prob, 1 - market_price)
        else:
            return None

        if kelly <= 0:
            return None

        # Calculate stake
        stake = min(
            state.capital * kelly,
            self.max_position_size,
            state.capital * 0.1,  # Never more than 10% on one event
        )

        if stake < 1.0:  # Minimum $1 stake
            return None

        return EventContract(
            event_id=f"{event_type}_{pd.Timestamp.now().strftime('%Y%m%d%H%M')}",
            event_type=event_type,
            description=f"{event_type} | edge={edge:.1%} | kelly={kelly:.3f}",
            market_price=market_price,
            model_prob=model_prob,
            date=pd.Timestamp.now(),
            position=position,
            stake=round(stake, 2),
        )

    def settle_contract(
        self,
        contract: EventContract,
        outcome: bool,
        state: PredictionState,
    ) -> float:
        """Settle an event contract and calculate P&L.

        For binary contracts:
          - YES position: pay market_price, receive $1 if outcome=True
          - NO position: pay (1-market_price), receive $1 if outcome=False
        """
        contract.outcome = outcome
        contract.settled = True

        # Stake was already deducted from capital when position was opened.
        # On win: return stake + profit. On loss: stake is already gone.
        if contract.position == "yes":
            if outcome:
                payout = contract.stake / contract.market_price
                contract.pnl = payout - contract.stake
                state.capital += payout  # Return full payout
            else:
                contract.pnl = -contract.stake
                # Stake already deducted, nothing returned
        elif contract.position == "no":
            if not outcome:
                payout = contract.stake / (1 - contract.market_price)
                contract.pnl = payout - contract.stake
                state.capital += payout
            else:
                contract.pnl = -contract.stake
        state.settled_positions.append(contract)

        if contract in state.open_positions:
            state.open_positions.remove(contract)

        return contract.pnl

    def generate_model_probability(
        self,
        event_type: str,
        historical_data: pd.DataFrame,
        lookback: int = 20,
    ) -> float:
        """Generate a model probability for an event type using historical data.

        This is a simple rolling-window frequency estimator.
        In production, this would use ML models with more features.
        """
        if len(historical_data) < lookback:
            return 0.5  # No edge if insufficient data

        recent = historical_data.tail(lookback)

        if event_type == "sp500_up_today":
            return float((recent["daily_return"] > 0).mean())

        elif event_type == "sp500_big_move":
            return float((abs(recent["daily_return"]) > 0.01).mean())

        elif event_type == "sp500_flat_day":
            return float((abs(recent["daily_return"]) < 0.005).mean())

        elif event_type == "sp500_above_ma":
            if "ema_slow" in recent.columns:
                return float((recent["close"] > recent["ema_slow"]).mean())

        return 0.5

    def simulate_market_price(self, true_prob: float, noise: float = 0.08) -> float:
        """Simulate a market price with some noise around the true probability.

        In real markets, the price is set by supply/demand and may deviate
        from the true probability, creating opportunities.
        """
        market = true_prob + np.random.normal(0, noise)
        return np.clip(market, 0.05, 0.95)
