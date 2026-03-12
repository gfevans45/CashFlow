"""Mean Reversion + Momentum Confirmation Strategy for Stocks.

Strategy Logic:
  ENTRY (Long):
    1. RSI < 30 (oversold)
    2. Price touches or crosses below lower Bollinger Band
    3. Momentum confirmation: EMA(8) > EMA(21) or MACD histogram turning positive
    4. Volume > 1.5x 20-day average (institutional interest)

  ENTRY (Short - only in paper trading):
    1. RSI > 70 (overbought)
    2. Price touches or crosses above upper Bollinger Band
    3. EMA(8) < EMA(21) or MACD histogram turning negative
    4. Volume > 1.5x average

  EXIT:
    - Stop loss: 2% below entry (adjustable via ATR)
    - Take profit: 4% above entry (2:1 reward-to-risk)
    - Trailing stop: Move stop to breakeven at 1.5% profit
    - Time stop: Exit after 5 bars if neither target hit

This strategy works well on hourly bars for liquid stocks/ETFs.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Signal(Enum):
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"


@dataclass
class Trade:
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    entry_time: object
    shares: float
    stop_loss: float
    take_profit: float
    trailing_stop: float = 0.0
    bars_held: int = 0
    pnl: float = 0.0
    exit_price: float = 0.0
    exit_time: object = None
    exit_reason: str = ""
    is_open: bool = True


@dataclass
class StrategyState:
    capital: float = 500.0
    positions: dict = field(default_factory=dict)  # symbol -> Trade
    closed_trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)
    max_positions: int = 3


class StockMeanReversionStrategy:
    """Mean reversion with momentum confirmation."""

    def __init__(self, config: dict):
        self.config = config
        self.rsi_oversold = config.get("rsi_oversold", 30)
        self.rsi_overbought = config.get("rsi_overbought", 70)
        self.volume_threshold = config.get("volume_threshold", 1.5)
        self.stop_loss_pct = config.get("stop_loss_pct", 2.0) / 100
        self.take_profit_pct = config.get("take_profit_pct", 4.0) / 100
        self.max_positions = config.get("max_positions", 3)
        self.max_bars_held = config.get("max_bars_held", 5)

    def generate_signal(self, df: pd.DataFrame, symbol: str, state: StrategyState) -> Signal:
        """Analyze current bar and generate a trading signal."""
        if len(df) < 2:
            return Signal.HOLD

        current = df.iloc[-1]
        prev = df.iloc[-2]

        # Check if we already have a position in this symbol
        if symbol in state.positions:
            return self._check_exit(current, prev, state.positions[symbol])

        # Don't open new positions if at max
        if len(state.positions) >= self.max_positions:
            return Signal.HOLD

        # --- LONG ENTRY CONDITIONS ---
        conditions_long = [
            current["rsi"] < self.rsi_oversold,
            current["close"] <= current["bb_lower"],
            (current["ema_fast"] > current["ema_slow"]) or (current["macd_hist"] > prev["macd_hist"]),
            current["volume_ratio"] > self.volume_threshold,
        ]

        if sum(conditions_long) >= 3:  # Need at least 3 of 4 conditions
            return Signal.LONG

        # --- SHORT ENTRY CONDITIONS ---
        conditions_short = [
            current["rsi"] > self.rsi_overbought,
            current["close"] >= current["bb_upper"],
            (current["ema_fast"] < current["ema_slow"]) or (current["macd_hist"] < prev["macd_hist"]),
            current["volume_ratio"] > self.volume_threshold,
        ]

        if sum(conditions_short) >= 3:
            return Signal.SHORT

        return Signal.HOLD

    def _check_exit(self, current: pd.Series, prev: pd.Series, trade: Trade) -> Signal:
        """Check if an existing position should be exited."""
        trade.bars_held += 1
        price = current["close"]

        if trade.direction == "long":
            # Stop loss
            if price <= trade.stop_loss:
                return Signal.EXIT_LONG
            # Take profit
            if price >= trade.take_profit:
                return Signal.EXIT_LONG
            # Trailing stop: move to breakeven at 1.5% profit
            if price >= trade.entry_price * 1.015 and trade.trailing_stop < trade.entry_price:
                trade.trailing_stop = trade.entry_price
            if trade.trailing_stop > 0 and price <= trade.trailing_stop:
                return Signal.EXIT_LONG
            # Time stop
            if trade.bars_held >= self.max_bars_held:
                return Signal.EXIT_LONG

        elif trade.direction == "short":
            if price >= trade.stop_loss:
                return Signal.EXIT_SHORT
            if price <= trade.take_profit:
                return Signal.EXIT_SHORT
            if price <= trade.entry_price * 0.985 and trade.trailing_stop > trade.entry_price:
                trade.trailing_stop = trade.entry_price
            if trade.trailing_stop > 0 and price >= trade.trailing_stop:
                return Signal.EXIT_SHORT
            if trade.bars_held >= self.max_bars_held:
                return Signal.EXIT_SHORT

        return Signal.HOLD

    def execute_signal(
        self,
        signal: Signal,
        symbol: str,
        current_bar: pd.Series,
        state: StrategyState,
    ) -> Optional[Trade]:
        """Execute a signal and update state."""
        price = current_bar["close"]
        atr = current_bar.get("atr", price * 0.02)
        timestamp = current_bar.name

        if signal == Signal.LONG:
            stop = price * (1 - self.stop_loss_pct)
            tp = price * (1 + self.take_profit_pct)

            # Position sizing: risk 2% of capital
            risk_per_share = price - stop
            if risk_per_share <= 0:
                return None
            max_risk = state.capital * 0.02
            shares = min(max_risk / risk_per_share, state.capital / price)
            shares = round(shares, 4)  # Alpaca supports fractional

            if shares * price > state.capital or shares <= 0:
                return None

            trade = Trade(
                symbol=symbol,
                direction="long",
                entry_price=price,
                entry_time=timestamp,
                shares=shares,
                stop_loss=stop,
                take_profit=tp,
            )
            state.positions[symbol] = trade
            state.capital -= shares * price
            return trade

        elif signal == Signal.SHORT:
            stop = price * (1 + self.stop_loss_pct)
            tp = price * (1 - self.take_profit_pct)

            risk_per_share = stop - price
            if risk_per_share <= 0:
                return None
            max_risk = state.capital * 0.02
            shares = min(max_risk / risk_per_share, state.capital / price)
            shares = round(shares, 4)

            if shares * price > state.capital or shares <= 0:
                return None

            trade = Trade(
                symbol=symbol,
                direction="short",
                entry_price=price,
                entry_time=timestamp,
                shares=shares,
                stop_loss=stop,
                take_profit=tp,
            )
            state.positions[symbol] = trade
            state.capital -= shares * price  # margin reservation
            return trade

        elif signal in (Signal.EXIT_LONG, Signal.EXIT_SHORT):
            if symbol not in state.positions:
                return None
            trade = state.positions[symbol]
            trade.exit_price = price
            trade.exit_time = timestamp
            trade.is_open = False

            if trade.direction == "long":
                trade.pnl = (price - trade.entry_price) * trade.shares
                trade.exit_reason = self._exit_reason(trade, price)
            else:
                trade.pnl = (trade.entry_price - price) * trade.shares
                trade.exit_reason = self._exit_reason(trade, price)

            state.capital += (trade.shares * trade.entry_price) + trade.pnl
            state.closed_trades.append(trade)
            del state.positions[symbol]
            return trade

        return None

    def _exit_reason(self, trade: Trade, price: float) -> str:
        if trade.direction == "long":
            if price <= trade.stop_loss:
                return "stop_loss"
            if price >= trade.take_profit:
                return "take_profit"
            if trade.trailing_stop > 0 and price <= trade.trailing_stop:
                return "trailing_stop"
        else:
            if price >= trade.stop_loss:
                return "stop_loss"
            if price <= trade.take_profit:
                return "take_profit"
            if trade.trailing_stop > 0 and price >= trade.trailing_stop:
                return "trailing_stop"
        if trade.bars_held >= self.max_bars_held:
            return "time_stop"
        return "manual"
