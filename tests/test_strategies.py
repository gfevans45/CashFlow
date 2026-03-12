"""Unit tests for strategy logic."""
import pytest
import pandas as pd
import numpy as np

from cashflow.strategies.stock_mean_reversion import (
    StockMeanReversionStrategy,
    StrategyState,
    Signal,
)
from cashflow.strategies.prediction_market import (
    PredictionMarketStrategy,
    PredictionState,
)
from cashflow.utils.risk import (
    kelly_criterion,
    half_kelly,
    calculate_position_size,
    max_drawdown,
    sharpe_ratio,
)


# --- Risk Utils Tests ---

class TestRiskUtils:
    def test_kelly_criterion_positive_edge(self):
        # 60% win rate, 1:1 payoff
        k = kelly_criterion(0.6, 1.0, 1.0)
        assert 0.15 < k < 0.25  # Should be ~0.20

    def test_kelly_criterion_no_edge(self):
        k = kelly_criterion(0.5, 1.0, 1.0)
        assert k == 0.0

    def test_kelly_criterion_negative_edge(self):
        k = kelly_criterion(0.4, 1.0, 1.0)
        assert k == 0.0

    def test_half_kelly(self):
        full = kelly_criterion(0.6, 1.0, 1.0)
        half = half_kelly(0.6, 1.0, 1.0)
        assert abs(half - full / 2) < 0.001

    def test_position_size_fixed(self):
        size = calculate_position_size(
            capital=500, risk_per_trade_pct=2.0,
            entry_price=100, stop_loss_price=98, method="fixed"
        )
        # Risk $10, stop at $2 away = 5 shares
        assert abs(size - 5.0) < 0.01

    def test_position_size_max_affordable(self):
        size = calculate_position_size(
            capital=100, risk_per_trade_pct=50.0,
            entry_price=100, stop_loss_price=50, method="fixed"
        )
        # Can only afford 1 share
        assert size == 1.0

    def test_max_drawdown(self):
        equity = [100, 110, 105, 90, 95, 100]
        dd = max_drawdown(equity)
        assert abs(dd - (110 - 90) / 110) < 0.001

    def test_sharpe_ratio_flat(self):
        returns = [0.0] * 100
        assert sharpe_ratio(returns) == 0.0


# --- Stock Strategy Tests ---

class TestStockStrategy:
    @pytest.fixture
    def strategy(self):
        return StockMeanReversionStrategy({
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "volume_threshold": 1.5,
            "stop_loss_pct": 2.0,
            "take_profit_pct": 4.0,
            "max_positions": 3,
        })

    @pytest.fixture
    def state(self):
        return StrategyState(capital=500.0, max_positions=3)

    def _make_bar(self, **kwargs):
        defaults = {
            "open": 100, "high": 101, "low": 99, "close": 100,
            "volume": 1000000, "rsi": 50, "bb_upper": 102, "bb_middle": 100,
            "bb_lower": 98, "bb_pct": 0.5, "ema_fast": 100.5, "ema_slow": 100,
            "macd": 0.1, "macd_signal": 0.05, "macd_hist": 0.05,
            "volume_sma": 800000, "volume_ratio": 1.25, "atr": 1.5, "vwap": 100,
        }
        defaults.update(kwargs)
        return pd.Series(defaults)

    def test_no_signal_normal_conditions(self, strategy, state):
        bars = [self._make_bar(), self._make_bar()]
        df = pd.DataFrame(bars)
        signal = strategy.generate_signal(df, "SPY", state)
        assert signal == Signal.HOLD

    def test_long_signal_oversold(self, strategy, state):
        bar1 = self._make_bar(rsi=35, macd_hist=-0.1)
        bar2 = self._make_bar(
            rsi=28, close=97.5, bb_lower=98,
            ema_fast=100.5, ema_slow=100,
            macd_hist=0.05, volume_ratio=1.8,
        )
        df = pd.DataFrame([bar1, bar2])
        signal = strategy.generate_signal(df, "SPY", state)
        assert signal == Signal.LONG

    def test_max_positions_respected(self, strategy, state):
        state.positions = {"A": None, "B": None, "C": None}
        bar1 = self._make_bar()
        bar2 = self._make_bar(rsi=25, close=97, bb_lower=98, volume_ratio=2.0)
        df = pd.DataFrame([bar1, bar2])
        signal = strategy.generate_signal(df, "SPY", state)
        assert signal == Signal.HOLD

    def test_execute_long(self, strategy, state):
        bar = self._make_bar(close=100, atr=1.5)
        bar.name = pd.Timestamp("2024-01-01")
        trade = strategy.execute_signal(Signal.LONG, "SPY", bar, state)
        assert trade is not None
        assert trade.direction == "long"
        assert trade.shares > 0
        assert state.capital < 500.0
        assert "SPY" in state.positions


# --- Prediction Market Tests ---

class TestPredictionStrategy:
    @pytest.fixture
    def strategy(self):
        return PredictionMarketStrategy({
            "min_edge_pct": 5.0,
            "kelly_fraction": 0.25,
            "max_position_size": 25.0,
            "max_open_positions": 5,
        })

    @pytest.fixture
    def state(self):
        return PredictionState(capital=500.0)

    def test_no_trade_without_edge(self, strategy, state):
        contract = strategy.evaluate_event("test", 0.50, 0.50, state)
        assert contract is None

    def test_yes_trade_with_edge(self, strategy, state):
        # Model says 65% but market says 50%
        contract = strategy.evaluate_event("test", 0.65, 0.50, state)
        assert contract is not None
        assert contract.position == "yes"
        assert contract.stake > 0

    def test_no_trade_with_edge(self, strategy, state):
        # Model says 30% but market says 50% (edge on NO side)
        contract = strategy.evaluate_event("test", 0.30, 0.50, state)
        assert contract is not None
        assert contract.position == "no"

    def test_settle_yes_win(self, strategy, state):
        contract = strategy.evaluate_event("test", 0.70, 0.50, state)
        assert contract is not None
        initial_cap = state.capital
        state.open_positions.append(contract)
        state.capital -= contract.stake
        pnl = strategy.settle_contract(contract, True, state)
        assert pnl > 0
        assert contract.settled

    def test_settle_yes_loss(self, strategy, state):
        contract = strategy.evaluate_event("test", 0.70, 0.50, state)
        assert contract is not None
        state.open_positions.append(contract)
        state.capital -= contract.stake
        pnl = strategy.settle_contract(contract, False, state)
        assert pnl < 0

    def test_kelly_sizing(self, strategy):
        size = strategy.kelly_size(0.7, 0.5)
        assert size > 0
        # Should be fractional Kelly (0.25x)
        assert size < 0.25

    def test_max_positions_respected(self, strategy, state):
        state.open_positions = [None] * 5
        contract = strategy.evaluate_event("test", 0.80, 0.50, state)
        assert contract is None
