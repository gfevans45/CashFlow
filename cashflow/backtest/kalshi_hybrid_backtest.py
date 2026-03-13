"""Backtest for Kalshi Hybrid Vol Fade + Directional Tilt Strategy.

Simulates trading S&P 500 daily range contracts (weekdays) and
Bitcoin range contracts (overnight/weekends) on Kalshi over 90 days.

Uses 5-min candle data for S&P exit monitoring and 15-min for BTC.
Generates simulated Kalshi bracket contracts from actual price data.
"""
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from cashflow.data.market_data import _yahoo_download, _rsi, _ema, _bollinger_bands
from cashflow.utils.risk import max_drawdown, sharpe_ratio, sortino_ratio


# --- Data Classes ---

@dataclass
class BracketContract:
    """A simulated Kalshi bracket contract."""
    contract_id: str
    asset: str              # "SPY" or "BTC"
    date: object
    lower_bound: float      # e.g., 5200
    upper_bound: float      # e.g., 5225
    market_price: float     # Contract price ($0.00-$1.00)
    model_prob: float       # Our model's estimated probability
    edge: float             # model_prob - market_price
    direction: str          # "bracket", "up", "down"
    outcome: Optional[bool] = None
    settled: bool = False


@dataclass
class Trade:
    """A completed or open trade."""
    contract: BracketContract
    entry_price: float      # What we paid
    stake: float            # Dollar amount
    num_contracts: int
    entry_time: object = None
    exit_time: object = None
    exit_price: float = 0.0
    exit_reason: str = ""   # "take_profit", "stop_loss", "expiration"
    pnl: float = 0.0
    is_open: bool = True


@dataclass
class DayResult:
    """Results for a single trading day."""
    date: object
    asset: str
    trades: list = field(default_factory=list)
    pnl: float = 0.0
    starting_capital: float = 0.0
    ending_capital: float = 0.0
    daily_loss_limit_hit: bool = False


# --- Contract Simulation ---

def generate_sp500_brackets(open_price: float, daily_vol: float, n_brackets: int = 5):
    """Generate Kalshi-style bracket contracts from S&P 500 price data.

    Kalshi typically offers ~5 brackets per day for S&P 500:
    - 1 center bracket (highest prob, ~40-50%)
    - 2 adjacent brackets (~15-25% each)
    - 2 tail brackets (~5-10% each)

    The center bracket spans roughly 1 standard deviation.
    """
    daily_std = daily_vol * open_price
    if daily_std < 1:
        daily_std = open_price * 0.008  # Floor at ~0.8% daily move

    brackets = [
        # Far below: < -1.5σ  (~7%)
        (round(open_price - 4.0 * daily_std, 2), round(open_price - 1.5 * daily_std, 2)),
        # Below: -1.5σ to -0.5σ  (~24%)
        (round(open_price - 1.5 * daily_std, 2), round(open_price - 0.5 * daily_std, 2)),
        # Center: -0.5σ to +0.5σ  (~38%)
        (round(open_price - 0.5 * daily_std, 2), round(open_price + 0.5 * daily_std, 2)),
        # Above: +0.5σ to +1.5σ  (~24%)
        (round(open_price + 0.5 * daily_std, 2), round(open_price + 1.5 * daily_std, 2)),
        # Far above: > +1.5σ  (~7%)
        (round(open_price + 1.5 * daily_std, 2), round(open_price + 4.0 * daily_std, 2)),
    ]

    # Directional contracts (above/below open) — ~50% each
    brackets.append(
        (round(open_price, 2), round(open_price + 5 * daily_std, 2))  # "Closes higher"
    )
    brackets.append(
        (round(open_price - 5 * daily_std, 2), round(open_price, 2))  # "Closes lower"
    )

    return brackets


def generate_btc_brackets(open_price: float, daily_vol: float, n_brackets: int = 5):
    """Generate Kalshi-style bracket contracts for Bitcoin.

    BTC is more volatile, so brackets are wider.
    """
    daily_std = daily_vol * open_price
    if daily_std < 100:
        daily_std = open_price * 0.025  # Floor at ~2.5% daily move

    brackets = [
        (round(open_price - 4.0 * daily_std, 2), round(open_price - 1.5 * daily_std, 2)),
        (round(open_price - 1.5 * daily_std, 2), round(open_price - 0.5 * daily_std, 2)),
        (round(open_price - 0.5 * daily_std, 2), round(open_price + 0.5 * daily_std, 2)),
        (round(open_price + 0.5 * daily_std, 2), round(open_price + 1.5 * daily_std, 2)),
        (round(open_price + 1.5 * daily_std, 2), round(open_price + 4.0 * daily_std, 2)),
    ]

    # Directional
    brackets.append(
        (round(open_price, 2), round(open_price + 5 * daily_std, 2))
    )
    brackets.append(
        (round(open_price - 5 * daily_std, 2), round(open_price, 2))
    )

    return brackets


def price_bracket_contract(lower: float, upper: float, open_price: float,
                           implied_vol: float, noise: float = 0.06):
    """Price a bracket contract using a normal distribution + noise.

    This simulates what Kalshi's market would price the bracket at.
    Uses implied vol (which is typically higher than realized vol) and
    adds noise to simulate market inefficiency (our edge source).

    The vol premium + noise creates the edge our model can exploit:
    market uses inflated vol -> overprices tails, underprices center.
    """
    from scipy.stats import norm

    daily_std = open_price * implied_vol
    if daily_std == 0:
        return 0.5

    z_lower = (lower - open_price) / daily_std
    z_upper = (upper - open_price) / daily_std
    true_prob = norm.cdf(z_upper) - norm.cdf(z_lower)

    # Add market noise (mispricing) — bigger noise on center brackets
    # because more retail flow pushes them around
    if true_prob > 0.25:
        noise_scale = noise * 1.2  # Center brackets have more noise
    else:
        noise_scale = noise

    market_price = true_prob + np.random.normal(0, noise_scale)
    market_price = np.clip(market_price, 0.02, 0.98)

    return round(market_price, 3)


def model_bracket_probability(lower: float, upper: float, open_price: float,
                              realized_vol: float):
    """Our model's probability estimate using realized volatility."""
    from scipy.stats import norm

    daily_std = open_price * realized_vol
    if daily_std == 0:
        return 0.5

    z_lower = (lower - open_price) / daily_std
    z_upper = (upper - open_price) / daily_std
    prob = norm.cdf(z_upper) - norm.cdf(z_lower)

    return round(prob, 4)


# --- Technical Indicators for Directional Tilt ---

def compute_directional_signal(daily_data: pd.DataFrame, lookback: int = 20):
    """Compute directional signal from technical indicators.

    Returns a value from -1 (strong bearish) to +1 (strong bullish).
    Uses RSI, EMA crossover, and Bollinger Bands.
    """
    if len(daily_data) < lookback + 5:
        return 0.0

    close = daily_data["close"]

    signals = []

    # RSI signal
    rsi = _rsi(close, 14)
    current_rsi = rsi.iloc[-1]
    if current_rsi < 30:
        signals.append(1.0)   # Oversold = bullish
    elif current_rsi > 70:
        signals.append(-1.0)  # Overbought = bearish
    elif current_rsi < 45:
        signals.append(0.3)
    elif current_rsi > 55:
        signals.append(-0.3)
    else:
        signals.append(0.0)

    # EMA crossover signal
    ema_fast = _ema(close, 8)
    ema_slow = _ema(close, 21)
    ema_diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / close.iloc[-1]
    if ema_diff > 0.005:
        signals.append(1.0)
    elif ema_diff < -0.005:
        signals.append(-1.0)
    else:
        signals.append(ema_diff / 0.005)  # Proportional

    # Bollinger Band signal
    bb_upper, bb_middle, bb_lower, bb_pct = _bollinger_bands(close, 20, 2.0)
    current_bb_pct = bb_pct.iloc[-1]
    if current_bb_pct < 0.0:
        signals.append(1.0)   # Below lower band = bullish
    elif current_bb_pct > 1.0:
        signals.append(-1.0)  # Above upper band = bearish
    else:
        signals.append(0.5 - current_bb_pct)  # Proportional

    # Average signals — need 2+ aligned for a strong signal
    avg_signal = np.mean(signals)
    aligned = sum(1 for s in signals if np.sign(s) == np.sign(avg_signal))

    if aligned >= 2 and abs(avg_signal) > 0.3:
        return round(np.clip(avg_signal, -1, 1), 3)
    return 0.0


# --- Strategy Logic ---

class KalshiHybridStrategy:
    """Hybrid Vol Fade + Directional Tilt strategy for Kalshi contracts."""

    def __init__(self, config: dict = None):
        config = config or {}
        self.min_edge = config.get("min_edge", 0.05)
        self.min_edge_wide = config.get("min_edge_wide", 0.08)
        self.kelly_fraction = config.get("kelly_fraction", 0.25)
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_positions = config.get("max_positions", 5)
        self.min_stake = config.get("min_stake", 1.0)
        self.max_stake = config.get("max_stake", 5.0)
        self.daily_loss_limit_pct = config.get("daily_loss_limit_pct", 0.05)
        self.weekly_loss_limit_pct = config.get("weekly_loss_limit_pct", 0.15)
        # Contract price range for natural 1:1
        self.preferred_price_low = 0.40
        self.preferred_price_high = 0.60
        self.acceptable_price_low = 0.30
        self.acceptable_price_high = 0.70
        # Exit levels
        self.take_profit_offset = 0.25
        self.stop_loss_offset = 0.25

    def evaluate_contracts(self, contracts: list, capital: float,
                           open_positions: int, directional_signal: float,
                           daily_pnl: float):
        """Evaluate a set of bracket contracts and return trades to take.

        Args:
            contracts: List of BracketContract
            capital: Current capital
            open_positions: Number of currently open positions
            directional_signal: -1 to +1 directional bias
            daily_pnl: Running P&L for today (for loss limit check)

        Returns:
            List of Trade objects to execute
        """
        # Check daily loss limit
        daily_loss_limit = capital * self.daily_loss_limit_pct
        if daily_pnl <= -daily_loss_limit:
            return []

        trades = []
        available_slots = self.max_positions - open_positions

        # Sort by edge (best opportunities first)
        scored = []
        for c in contracts:
            if c.edge < self.min_edge:
                continue

            # Prefer contracts in the $0.45-$0.55 range
            in_preferred = self.preferred_price_low <= c.market_price <= self.preferred_price_high
            in_acceptable = self.acceptable_price_low <= c.market_price <= self.acceptable_price_high

            if not in_acceptable:
                continue
            if not in_preferred and c.edge < self.min_edge_wide:
                continue

            # Apply directional tilt scoring
            direction_bonus = 0.0
            if directional_signal > 0.3 and c.direction == "up":
                direction_bonus = 0.02
            elif directional_signal < -0.3 and c.direction == "down":
                direction_bonus = 0.02
            elif abs(directional_signal) > 0.3 and c.direction == "bracket":
                # For brackets, tilt toward brackets above/below center
                direction_bonus = 0.01

            score = c.edge + direction_bonus
            scored.append((score, c))

        scored.sort(key=lambda x: -x[0])

        for score, contract in scored[:available_slots]:
            if len(trades) >= available_slots:
                break

            # Kelly sizing
            stake = self._kelly_size(contract, capital)
            if stake < self.min_stake:
                continue

            num_contracts = max(1, int(stake / contract.market_price))
            actual_stake = num_contracts * contract.market_price

            # Check if this would breach daily loss limit
            if daily_pnl - actual_stake <= -daily_loss_limit:
                continue

            trade = Trade(
                contract=contract,
                entry_price=contract.market_price,
                stake=round(actual_stake, 2),
                num_contracts=num_contracts,
            )
            trades.append(trade)

        return trades

    def _kelly_size(self, contract: BracketContract, capital: float) -> float:
        """Quarter-Kelly sizing for a binary contract."""
        p = contract.model_prob
        q = contract.market_price

        if q <= 0 or q >= 1:
            return 0.0

        edge = p - q
        if edge <= 0:
            return 0.0

        kelly = edge / (1 - q)
        stake = capital * kelly * self.kelly_fraction
        max_allowed = capital * self.max_position_pct
        stake = min(stake, max_allowed, self.max_stake)

        return round(stake, 2)

    def check_exit(self, trade: Trade, current_price: float):
        """Check if a trade should be exited early.

        Returns (should_exit, exit_reason) tuple.
        """
        tp_price = trade.entry_price + self.take_profit_offset
        sl_price = trade.entry_price - self.stop_loss_offset

        # Clamp to valid range
        tp_price = min(tp_price, 0.98)
        sl_price = max(sl_price, 0.02)

        if current_price >= tp_price:
            return True, "take_profit"
        if current_price <= sl_price:
            return True, "stop_loss"
        return False, ""


# --- Intraday Price Simulation ---

def simulate_intraday_path(open_price: float, close_price: float,
                           high: float, low: float, n_steps: int,
                           seed: int = None):
    """Simulate a realistic intraday price path between open and close.

    Uses a Brownian bridge constrained by the actual OHLC data.
    Returns array of n_steps prices.
    """
    if seed is not None:
        np.random.seed(seed)

    prices = np.zeros(n_steps)
    prices[0] = open_price
    prices[-1] = close_price

    # Brownian bridge
    vol = (high - low) / open_price  # Intraday range as vol proxy
    dt = 1.0 / n_steps

    for i in range(1, n_steps - 1):
        t = i / n_steps
        remaining = 1 - t
        # Bridge: drift toward close, with constrained noise
        drift = (close_price - prices[i - 1]) / (remaining * n_steps)
        noise = np.random.normal(0, vol * open_price * np.sqrt(dt) * 0.5)
        prices[i] = prices[i - 1] + drift + noise
        # Constrain to actual high/low
        prices[i] = np.clip(prices[i], low * 0.999, high * 1.001)

    return prices


def simulate_contract_price_path(entry_price: float, will_win: bool,
                                 n_steps: int, seed: int = None):
    """Simulate how a binary contract price moves intraday.

    Contract prices drift toward 1.0 (if winning) or 0.0 (if losing)
    as the day progresses and the outcome becomes clearer.
    """
    if seed is not None:
        np.random.seed(seed)

    final_price = 0.95 if will_win else 0.05
    prices = np.zeros(n_steps)
    prices[0] = entry_price

    for i in range(1, n_steps):
        t = i / n_steps
        # Gradually reveal information
        info_weight = t ** 1.5  # Accelerating toward end of day
        target = entry_price * (1 - info_weight) + final_price * info_weight
        noise = np.random.normal(0, 0.03 * (1 - t))  # Less noise as day progresses
        prices[i] = target + noise
        prices[i] = np.clip(prices[i], 0.01, 0.99)

    return prices


# --- Backtest Runner ---

class KalshiHybridBacktester:
    """Backtester for the Kalshi Hybrid strategy."""

    def __init__(self, capital: float = 50.0, config: dict = None):
        self.initial_capital = capital
        self.config = config or {}
        self.strategy = KalshiHybridStrategy(self.config)
        # Monitoring intervals
        self.sp500_monitor_mins = 5
        self.btc_monitor_mins = 15
        # S&P trading hours: 390 mins (9:30-4:00)
        self.sp500_steps = 390 // self.sp500_monitor_mins  # 78 steps
        # BTC overnight: ~900 mins (6pm-9am next day)
        self.btc_steps = 900 // self.btc_monitor_mins  # 60 steps

    def run(self, lookback_days: int = 90):
        """Run the full backtest."""
        print("=" * 60)
        print("  Kalshi Hybrid Strategy Backtest")
        print("  Vol Fade + Directional Tilt | 1:1 R:R")
        print("=" * 60)

        # Fetch data
        spy_daily, spy_5min = self._fetch_spy_data(lookback_days)
        btc_daily, btc_5min = self._fetch_btc_data(lookback_days)

        if spy_daily.empty:
            print("ERROR: Could not fetch S&P 500 data.")
            return {}

        print(f"\nSPY daily bars: {len(spy_daily)}")
        print(f"SPY 5-min bars: {len(spy_5min)}")
        print(f"BTC daily bars: {len(btc_daily)}")
        print(f"BTC 5-min bars: {len(btc_5min)}")

        # Run backtest
        capital = self.initial_capital
        all_trades = []
        day_results = []
        equity_curve = [capital]
        weekly_pnl = 0.0
        week_start_capital = capital

        np.random.seed(42)

        trading_days = spy_daily.index[21:]  # Skip first 21 days for indicator warmup

        for i, date in enumerate(trading_days):
            # --- S&P 500 (Weekday) ---
            if date.weekday() < 5:  # Mon-Fri
                day_result = self._trade_day_sp500(
                    date, spy_daily, spy_5min, capital, weekly_pnl
                )
                capital = day_result.ending_capital
                all_trades.extend(day_result.trades)
                day_results.append(day_result)

            # --- Bitcoin (Every day, overnight) ---
            if not btc_daily.empty and date in btc_daily.index:
                btc_result = self._trade_day_btc(
                    date, btc_daily, btc_5min, capital, weekly_pnl
                )
                capital = btc_result.ending_capital
                all_trades.extend(btc_result.trades)
                day_results.append(btc_result)

            equity_curve.append(capital)

            # Weekly reset check (Friday)
            if date.weekday() == 4:
                weekly_pnl = 0.0
                week_start_capital = capital
            else:
                weekly_pnl = capital - week_start_capital

        return self._compute_results(
            all_trades, day_results, equity_curve
        )

    def _trade_day_sp500(self, date, daily_data, intraday_data,
                         capital, weekly_pnl):
        """Simulate one day of S&P 500 trading."""
        result = DayResult(
            date=date, asset="SPY",
            starting_capital=capital
        )

        hist = daily_data.loc[:date]
        if len(hist) < 22:
            result.ending_capital = capital
            return result

        today = daily_data.loc[date]
        open_price = today["open"]
        close_price = today["close"]
        high = today["high"]
        low = today["low"]

        # Calculate volatilities
        returns_20d = hist["close"].pct_change().tail(20)
        realized_vol = returns_20d.std()
        implied_vol = realized_vol * 1.25  # Markets typically overprice vol (vol risk premium)

        # Directional signal
        dir_signal = compute_directional_signal(hist)

        # Generate bracket contracts
        brackets = generate_sp500_brackets(open_price, realized_vol)

        contracts = []
        for j, (lower, upper) in enumerate(brackets):
            mkt_price = price_bracket_contract(
                lower, upper, open_price, implied_vol, noise=0.04
            )
            model_prob = model_bracket_probability(
                lower, upper, open_price, realized_vol
            )

            # Determine if this is up/down/bracket relative to open
            mid = (lower + upper) / 2
            if mid > open_price + open_price * realized_vol * 0.3:
                direction = "up"
            elif mid < open_price - open_price * realized_vol * 0.3:
                direction = "down"
            else:
                direction = "bracket"

            # Did the close actually land in this bracket?
            outcome = lower <= close_price < upper

            contract = BracketContract(
                contract_id=f"SPY_{date.strftime('%Y%m%d')}_{j}",
                asset="SPY",
                date=date,
                lower_bound=lower,
                upper_bound=upper,
                market_price=mkt_price,
                model_prob=model_prob,
                edge=round(model_prob - mkt_price, 4),
                direction=direction,
                outcome=outcome,
            )
            contracts.append(contract)

        # Evaluate and place trades
        daily_pnl = 0.0
        trades = self.strategy.evaluate_contracts(
            contracts, capital, 0, dir_signal, daily_pnl
        )

        # Simulate intraday monitoring with 5-min steps
        for trade in trades:
            trade.entry_time = date
            will_win = trade.contract.outcome

            # Simulate contract price path
            price_path = simulate_contract_price_path(
                trade.entry_price, will_win, self.sp500_steps,
                seed=hash(trade.contract.contract_id) % (2**31)
            )

            # Check exits at each 5-min step
            exited = False
            for step, price in enumerate(price_path):
                should_exit, reason = self.strategy.check_exit(trade, price)
                if should_exit:
                    trade.exit_price = price
                    trade.exit_reason = reason
                    trade.pnl = round(
                        trade.num_contracts * (price - trade.entry_price), 2
                    )
                    trade.is_open = False
                    exited = True
                    break

            # If no early exit, settle at expiration
            if not exited:
                if will_win:
                    trade.exit_price = 1.0
                    trade.pnl = round(
                        trade.num_contracts * (1.0 - trade.entry_price), 2
                    )
                else:
                    trade.exit_price = 0.0
                    trade.pnl = round(
                        trade.num_contracts * (0.0 - trade.entry_price), 2
                    )
                trade.exit_reason = "expiration"
                trade.is_open = False

            daily_pnl += trade.pnl
            capital += trade.pnl

            # Check daily loss limit mid-day
            if daily_pnl <= -(capital * self.strategy.daily_loss_limit_pct):
                result.daily_loss_limit_hit = True
                break

        result.trades = trades
        result.pnl = round(daily_pnl, 2)
        result.ending_capital = round(capital, 2)
        return result

    def _trade_day_btc(self, date, daily_data, intraday_data,
                       capital, weekly_pnl):
        """Simulate one overnight/weekend BTC trading session."""
        result = DayResult(
            date=date, asset="BTC",
            starting_capital=capital
        )

        hist = daily_data.loc[:date]
        if len(hist) < 22:
            result.ending_capital = capital
            return result

        today = daily_data.loc[date]
        open_price = today["open"]
        close_price = today["close"]
        high = today["high"]
        low = today["low"]

        returns_20d = hist["close"].pct_change().tail(20)
        realized_vol = returns_20d.std()
        implied_vol = realized_vol * 1.30  # BTC implied vol premium is higher

        dir_signal = compute_directional_signal(hist)

        brackets = generate_btc_brackets(open_price, realized_vol)

        contracts = []
        for j, (lower, upper) in enumerate(brackets):
            mkt_price = price_bracket_contract(
                lower, upper, open_price, implied_vol, noise=0.05
            )
            model_prob = model_bracket_probability(
                lower, upper, open_price, realized_vol
            )

            mid = (lower + upper) / 2
            if mid > open_price + open_price * realized_vol * 0.3:
                direction = "up"
            elif mid < open_price - open_price * realized_vol * 0.3:
                direction = "down"
            else:
                direction = "bracket"

            outcome = lower <= close_price < upper

            contract = BracketContract(
                contract_id=f"BTC_{date.strftime('%Y%m%d')}_{j}",
                asset="BTC",
                date=date,
                lower_bound=lower,
                upper_bound=upper,
                market_price=mkt_price,
                model_prob=model_prob,
                edge=round(model_prob - mkt_price, 4),
                direction=direction,
                outcome=outcome,
            )
            contracts.append(contract)

        daily_pnl = 0.0
        trades = self.strategy.evaluate_contracts(
            contracts, capital, 0, dir_signal, daily_pnl
        )

        for trade in trades:
            trade.entry_time = date
            will_win = trade.contract.outcome

            price_path = simulate_contract_price_path(
                trade.entry_price, will_win, self.btc_steps,
                seed=hash(trade.contract.contract_id) % (2**31)
            )

            exited = False
            for step, price in enumerate(price_path):
                should_exit, reason = self.strategy.check_exit(trade, price)
                if should_exit:
                    trade.exit_price = price
                    trade.exit_reason = reason
                    trade.pnl = round(
                        trade.num_contracts * (price - trade.entry_price), 2
                    )
                    trade.is_open = False
                    exited = True
                    break

            if not exited:
                if will_win:
                    trade.exit_price = 1.0
                    trade.pnl = round(
                        trade.num_contracts * (1.0 - trade.entry_price), 2
                    )
                else:
                    trade.exit_price = 0.0
                    trade.pnl = round(
                        trade.num_contracts * (0.0 - trade.entry_price), 2
                    )
                trade.exit_reason = "expiration"
                trade.is_open = False

            daily_pnl += trade.pnl
            capital += trade.pnl

            if daily_pnl <= -(capital * self.strategy.daily_loss_limit_pct):
                result.daily_loss_limit_hit = True
                break

        result.trades = trades
        result.pnl = round(daily_pnl, 2)
        result.ending_capital = round(capital, 2)
        return result

    def _fetch_spy_data(self, lookback_days):
        """Fetch SPY daily and 5-min data."""
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 30)  # Extra for indicator warmup

        print("Fetching SPY daily data...")
        daily = _yahoo_download("SPY", start.strftime("%Y-%m-%d"),
                                end.strftime("%Y-%m-%d"), interval="1d")

        print("Fetching SPY 5-min data (last 60 days)...")
        start_5m = end - timedelta(days=59)
        intraday = _yahoo_download("SPY", start_5m.strftime("%Y-%m-%d"),
                                   end.strftime("%Y-%m-%d"), interval="5m")

        return daily, intraday

    def _fetch_btc_data(self, lookback_days):
        """Fetch BTC daily and 5-min data."""
        end = datetime.now()
        start = end - timedelta(days=lookback_days + 30)

        print("Fetching BTC daily data...")
        daily = _yahoo_download("BTC-USD", start.strftime("%Y-%m-%d"),
                                end.strftime("%Y-%m-%d"), interval="1d")

        print("Fetching BTC 5-min data (last 60 days)...")
        start_5m = end - timedelta(days=59)
        intraday = _yahoo_download("BTC-USD", start_5m.strftime("%Y-%m-%d"),
                                   end.strftime("%Y-%m-%d"), interval="5m")

        return daily, intraday

    def _compute_results(self, all_trades, day_results, equity_curve):
        """Compute and print backtest results."""
        if not all_trades:
            return {"total_trades": 0, "message": "No trades generated"}

        wins = [t for t in all_trades if t.pnl > 0]
        losses = [t for t in all_trades if t.pnl <= 0]
        total_pnl = sum(t.pnl for t in all_trades)

        # Daily returns
        daily_returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                daily_returns.append(
                    (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                )

        avg_win = np.mean([t.pnl for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t.pnl) for t in losses]) if losses else 0
        win_rate = len(wins) / len(all_trades) if all_trades else 0

        # By asset
        spy_trades = [t for t in all_trades if t.contract.asset == "SPY"]
        btc_trades = [t for t in all_trades if t.contract.asset == "BTC"]

        # By exit reason
        exit_reasons = {}
        for t in all_trades:
            exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

        # Trading days and daily stats
        trading_days = len(set(r.date for r in day_results))
        profitable_days = len(set(
            r.date for r in day_results if r.pnl > 0
        ))
        loss_limit_days = sum(1 for r in day_results if r.daily_loss_limit_hit)

        # Average daily return
        avg_daily_return = np.mean(daily_returns) if daily_returns else 0
        avg_daily_pnl = total_pnl / trading_days if trading_days > 0 else 0

        # R:R ratio
        rr_ratio = round(avg_win / avg_loss, 2) if avg_loss > 0 else float("inf")

        results = {
            "strategy": "Kalshi Hybrid: Vol Fade + Directional Tilt",
            "initial_capital": self.initial_capital,
            "final_capital": round(equity_curve[-1], 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(
                (equity_curve[-1] / self.initial_capital - 1) * 100, 2
            ),
            "total_trades": len(all_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win": round(avg_win, 2),
            "avg_loss": round(avg_loss, 2),
            "risk_reward_ratio": rr_ratio,
            "avg_daily_return_pct": round(avg_daily_return * 100, 3),
            "avg_daily_pnl": round(avg_daily_pnl, 2),
            "trading_days": trading_days,
            "profitable_days": profitable_days,
            "profitable_days_pct": round(
                profitable_days / trading_days * 100, 1
            ) if trading_days > 0 else 0,
            "loss_limit_days": loss_limit_days,
            "max_drawdown_pct": round(
                max_drawdown(equity_curve) * 100, 2
            ) if len(equity_curve) > 1 else 0,
            "sharpe_ratio": round(
                sharpe_ratio(daily_returns), 2
            ) if daily_returns else 0,
            "sortino_ratio": round(
                sortino_ratio(daily_returns), 2
            ) if daily_returns else 0,
            "trades_per_day": round(
                len(all_trades) / trading_days, 1
            ) if trading_days > 0 else 0,
            "exit_reasons": exit_reasons,
            "equity_curve": equity_curve,
            "spy_trades": len(spy_trades),
            "spy_wins": len([t for t in spy_trades if t.pnl > 0]),
            "spy_pnl": round(sum(t.pnl for t in spy_trades), 2),
            "btc_trades": len(btc_trades),
            "btc_wins": len([t for t in btc_trades if t.pnl > 0]),
            "btc_pnl": round(sum(t.pnl for t in btc_trades), 2),
            "all_trades": all_trades,
            "day_results": day_results,
        }

        self._print_results(results)
        return results

    def _print_results(self, r):
        """Pretty-print backtest results."""
        print(f"\n{'='*60}")
        print(f"  {r['strategy']}")
        print(f"  90-Day Backtest Results")
        print(f"{'='*60}")

        print(f"\n--- PERFORMANCE ---")
        print(f"  Initial Capital:    ${r['initial_capital']:.2f}")
        print(f"  Final Capital:      ${r['final_capital']:.2f}")
        print(f"  Total P&L:          ${r['total_pnl']:.2f}")
        print(f"  Total Return:       {r['total_return_pct']}%")
        print(f"  Avg Daily Return:   {r['avg_daily_return_pct']}%")
        print(f"  Avg Daily P&L:      ${r['avg_daily_pnl']:.2f}")

        print(f"\n--- TRADES ---")
        print(f"  Total Trades:       {r['total_trades']}")
        print(f"  Trades/Day:         {r['trades_per_day']}")
        print(f"  Win Rate:           {r['win_rate_pct']}%")
        print(f"  Avg Win:            ${r['avg_win']:.2f}")
        print(f"  Avg Loss:           ${r['avg_loss']:.2f}")
        print(f"  Risk:Reward:        1:{r['risk_reward_ratio']}")

        print(f"\n--- RISK ---")
        print(f"  Max Drawdown:       {r['max_drawdown_pct']}%")
        print(f"  Sharpe Ratio:       {r['sharpe_ratio']}")
        print(f"  Sortino Ratio:      {r['sortino_ratio']}")
        print(f"  Profitable Days:    {r['profitable_days']}/{r['trading_days']} ({r['profitable_days_pct']}%)")
        print(f"  Loss Limit Days:    {r['loss_limit_days']}")

        print(f"\n--- BY ASSET ---")
        print(f"  SPY: {r['spy_trades']} trades, {r['spy_wins']} wins, ${r['spy_pnl']:.2f} P&L")
        print(f"  BTC: {r['btc_trades']} trades, {r['btc_wins']} wins, ${r['btc_pnl']:.2f} P&L")

        print(f"\n--- EXIT REASONS ---")
        for reason, count in r["exit_reasons"].items():
            pct = count / r["total_trades"] * 100
            print(f"  {reason}: {count} ({pct:.1f}%)")

        print(f"\n{'='*60}")
