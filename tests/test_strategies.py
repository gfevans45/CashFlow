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
from cashflow.strategies.polymarket_longshot import (
    PolymarketLongshotStrategy,
    LongshotPortfolio,
)
from cashflow.data.arbitrage_feeds import (
    ArbitrageEngine,
    ExternalProbability,
    simulate_arbitrage_feeds,
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


# --- Polymarket Longshot Strategy Tests ---

class TestPolymarketLongshot:
    @pytest.fixture
    def strategy(self):
        return PolymarketLongshotStrategy({
            "max_entry_price": 0.10,
            "min_entry_price": 0.01,
            "min_model_prob": 0.15,
            "min_edge": 0.05,
            "kelly_fraction": 0.25,
            "max_bet_pct": 5.0,
            "min_bet_usd": 1.0,
            "max_bet_usd": 25.0,
            "max_positions": 20,
            "max_per_category": 4,
        })

    @pytest.fixture
    def portfolio(self):
        return LongshotPortfolio(capital=250.0)

    def test_no_entry_price_too_high(self, strategy, portfolio):
        pos = strategy.evaluate_market("m1", "Test?", "crypto", 0.15, 0.25, portfolio)
        assert pos is None  # Price > max_entry_price

    def test_no_entry_model_prob_too_low(self, strategy, portfolio):
        pos = strategy.evaluate_market("m1", "Test?", "crypto", 0.05, 0.08, portfolio)
        assert pos is None  # Model prob < min_model_prob

    def test_no_entry_insufficient_edge(self, strategy, portfolio):
        pos = strategy.evaluate_market("m1", "Test?", "crypto", 0.08, 0.12, portfolio)
        assert pos is None  # Edge = 4% < 5% min

    def test_entry_with_edge(self, strategy, portfolio):
        # Market at $0.05, model says 20% = 15% edge
        pos = strategy.evaluate_market("m1", "Test?", "crypto", 0.05, 0.20, portfolio)
        assert pos is not None
        assert pos.entry_price == 0.05
        assert pos.model_prob == 0.20
        assert pos.edge == 0.15
        assert pos.total_cost >= 1.0  # At least minimum bet
        assert pos.total_cost <= 25.0  # At most maximum bet
        assert pos.num_contracts > 0
        assert pos.max_payout > pos.total_cost  # Potential upside

    def test_kelly_sizing_longshot(self, strategy):
        # Longshot: buy at $0.05 with 20% true prob
        kelly = strategy.kelly_for_binary(0.20, 0.05)
        assert kelly > 0
        # Full Kelly would be high for longshots, but we use 1/4
        assert kelly < 0.10  # Fractional Kelly keeps it reasonable

    def test_kelly_no_edge(self, strategy):
        kelly = strategy.kelly_for_binary(0.05, 0.10)
        assert kelly == 0.0  # No edge = no bet

    def test_expected_value_positive(self, strategy):
        # Buy at $0.05, true prob 20%, stake $10
        ev = strategy.expected_value(0.20, 0.05, 10.0)
        # EV = 0.20 * (200 * 0.95) - 0.80 * 10 = 0.20*190 - 8 = 38-8 = 30
        assert ev > 0

    def test_expected_value_negative(self, strategy):
        ev = strategy.expected_value(0.03, 0.10, 10.0)
        assert ev < 0  # True prob lower than market = negative EV

    def test_settle_win(self, strategy, portfolio):
        pos = strategy.evaluate_market("m1", "Test?", "crypto", 0.05, 0.20, portfolio)
        assert pos is not None
        strategy.open_position(pos, portfolio)
        initial_invested = portfolio.total_invested

        strategy.close_position(pos, outcome=True, portfolio=portfolio, reason="settled")
        assert pos.pnl > 0
        assert pos.exit_price == 1.0
        assert not pos.is_open
        # Portfolio should have gotten the payout
        assert portfolio.capital > 250.0

    def test_settle_loss(self, strategy, portfolio):
        pos = strategy.evaluate_market("m1", "Test?", "crypto", 0.05, 0.20, portfolio)
        assert pos is not None
        cost = pos.total_cost
        strategy.open_position(pos, portfolio)
        assert portfolio.total_invested == cost

        strategy.close_position(pos, outcome=False, portfolio=portfolio, reason="settled")
        assert pos.pnl == -cost
        assert pos.exit_price == 0.0
        assert portfolio.capital < 250.0  # Lost the stake
        assert portfolio.total_invested == 0  # Position closed

    def test_category_diversification(self, strategy, portfolio):
        # Fill up crypto category (max 4)
        for i in range(4):
            pos = strategy.evaluate_market(f"m{i}", "Test?", "crypto", 0.05, 0.25, portfolio)
            if pos:
                strategy.open_position(pos, portfolio)

        # 5th crypto position should be rejected
        pos = strategy.evaluate_market("m99", "Test?", "crypto", 0.05, 0.25, portfolio)
        assert pos is None

        # But a different category should still work
        pos = strategy.evaluate_market("m100", "Test?", "sports", 0.05, 0.25, portfolio)
        assert pos is not None

    def test_max_positions_limit(self, strategy, portfolio):
        strategy.max_positions = 3
        for i in range(3):
            pos = strategy.evaluate_market(f"m{i}", "Test?", f"cat{i}", 0.05, 0.25, portfolio)
            if pos:
                strategy.open_position(pos, portfolio)

        pos = strategy.evaluate_market("m99", "Test?", "other", 0.05, 0.25, portfolio)
        assert pos is None

    def test_early_exit_take_profit(self, strategy, portfolio):
        pos = strategy.evaluate_market("m1", "Test?", "crypto", 0.05, 0.20, portfolio)
        assert pos is not None
        strategy.open_position(pos, portfolio)

        # Price more than tripled (use 0.16 to avoid float precision on 0.05*3.0)
        exits = strategy.check_early_exits(portfolio, {"m1": 0.16})
        assert len(exits) == 1
        assert exits[0][2] == "take_profit"

    def test_payoff_math(self, strategy, portfolio):
        """Verify the core thesis: buy at $0.05, win pays 19:1."""
        pos = strategy.evaluate_market("m1", "Test?", "crypto", 0.05, 0.20, portfolio)
        assert pos is not None
        contracts = pos.num_contracts
        cost = pos.total_cost
        # If YES wins, each contract pays $1.00
        payout = contracts * 1.0
        profit = payout - cost
        # Profit should be ~19x the cost (buy at $0.05 = 20 contracts per $1)
        assert profit / cost > 15  # At least 15:1 payoff ratio


# --- Arbitrage Engine Tests ---

class TestArbitrageEngine:
    @pytest.fixture
    def engine(self):
        return ArbitrageEngine()

    def _make_probs(self, probs_by_source):
        """Helper to create ExternalProbability list from dict."""
        return [
            ExternalProbability(
                source=source,
                question="Test question",
                probability=prob,
                volume=5000.0,
                confidence=ArbitrageEngine.SOURCE_WEIGHTS.get(source, 0.1),
            )
            for source, prob in probs_by_source.items()
        ]

    def test_ensemble_single_source(self, engine):
        probs = self._make_probs({"metaculus": 0.20})
        ensemble, confidence, count = engine.compute_ensemble_probability(probs)
        assert abs(ensemble - 0.20) < 0.01
        assert count == 1

    def test_ensemble_multiple_sources(self, engine):
        probs = self._make_probs({
            "metaculus": 0.20,
            "kalshi": 0.18,
            "manifold": 0.22,
        })
        ensemble, confidence, count = engine.compute_ensemble_probability(probs)
        assert 0.15 < ensemble < 0.25  # Weighted average in range
        assert count == 3
        assert confidence > 0

    def test_ensemble_with_internal_model(self, engine):
        probs = self._make_probs({"metaculus": 0.25, "kalshi": 0.20})
        ensemble, _, count = engine.compute_ensemble_probability(probs, internal_model_prob=0.30)
        # Internal model is 0.30, sources are 0.20-0.25, ensemble should be between
        assert 0.18 < ensemble < 0.30
        assert count == 2  # source_count doesn't include internal model

    def test_ensemble_empty_sources(self, engine):
        ensemble, confidence, count = engine.compute_ensemble_probability([])
        assert ensemble == 0.5
        assert confidence == 0.0
        assert count == 0

    def test_ensemble_internal_only(self, engine):
        ensemble, _, count = engine.compute_ensemble_probability([], internal_model_prob=0.15)
        assert abs(ensemble - 0.15) < 0.01
        assert count == 0

    def test_source_weights_exist(self):
        expected = {"metaculus", "kalshi", "predictit", "manifold", "model"}
        assert set(ArbitrageEngine.SOURCE_WEIGHTS.keys()) == expected

    def test_metaculus_highest_weight(self):
        weights = ArbitrageEngine.SOURCE_WEIGHTS
        assert weights["metaculus"] >= max(
            weights["kalshi"], weights["predictit"], weights["manifold"]
        )

    def test_volume_boost(self, engine):
        # High volume should increase weight
        high_vol = [ExternalProbability(
            source="kalshi", question="Test", probability=0.30,
            volume=50000.0, confidence=0.85,
        )]
        low_vol = [ExternalProbability(
            source="kalshi", question="Test", probability=0.30,
            volume=100.0, confidence=0.85,
        )]
        # Both should give same probability (single source)
        e1, _, _ = engine.compute_ensemble_probability(high_vol, internal_model_prob=0.10)
        e2, _, _ = engine.compute_ensemble_probability(low_vol, internal_model_prob=0.10)
        # High volume source gets more weight, so ensemble closer to 0.30
        assert e1 > e2

    def test_question_similarity(self, engine):
        sim = engine._question_similarity(
            "Will Bitcoin hit 100k by 2025?",
            "Bitcoin price above 100k in 2025",
        )
        assert sim > 0.3  # Should find overlap

    def test_question_similarity_no_overlap(self, engine):
        sim = engine._question_similarity(
            "Will it rain tomorrow?",
            "Bitcoin price prediction",
        )
        assert sim < 0.2

    def test_simulate_arbitrage_feeds(self):
        feeds = simulate_arbitrage_feeds(n_events=50, seed=123)
        assert len(feeds) == 50
        for feed in feeds:
            assert "event_id" in feed
            assert "true_prob" in feed
            assert "external_probs" in feed
            assert len(feed["external_probs"]) >= 2
            for prob in feed["external_probs"]:
                assert isinstance(prob, ExternalProbability)
                assert 0.0 < prob.probability < 1.0

    def test_extract_search_terms(self, engine):
        terms = engine._extract_search_terms("Will Bitcoin hit $100k by 2025?")
        assert len(terms) >= 1
        assert any("bitcoin" in t.lower() for t in terms)
