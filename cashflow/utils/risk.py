"""Position sizing and risk management utilities."""
import numpy as np


def kelly_criterion(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Calculate Kelly Criterion fraction.

    Returns the optimal fraction of bankroll to bet.
    """
    if avg_loss == 0:
        return 0.0
    b = avg_win / abs(avg_loss)  # odds ratio
    p = win_rate
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0.0, kelly)


def half_kelly(win_rate: float, avg_win: float, avg_loss: float) -> float:
    """Half-Kelly for more conservative sizing."""
    return kelly_criterion(win_rate, avg_win, avg_loss) * 0.5


def calculate_position_size(
    capital: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
    method: str = "fixed",
    kelly_params: dict = None,
) -> float:
    """Calculate number of shares/contracts to trade.

    Args:
        capital: Total available capital
        risk_per_trade_pct: Max % of capital to risk
        entry_price: Entry price
        stop_loss_price: Stop loss price
        method: 'fixed', 'kelly', or 'equal_weight'
        kelly_params: dict with win_rate, avg_win, avg_loss for kelly method

    Returns:
        Number of shares (can be fractional for Alpaca)
    """
    max_risk_dollars = capital * (risk_per_trade_pct / 100.0)
    risk_per_share = abs(entry_price - stop_loss_price)

    if risk_per_share == 0:
        return 0.0

    if method == "kelly" and kelly_params:
        kelly_frac = half_kelly(
            kelly_params["win_rate"],
            kelly_params["avg_win"],
            kelly_params["avg_loss"],
        )
        kelly_dollars = capital * kelly_frac
        risk_dollars = min(kelly_dollars, max_risk_dollars)
    else:
        risk_dollars = max_risk_dollars

    shares = risk_dollars / risk_per_share
    # Ensure we don't exceed what we can afford
    max_affordable = capital / entry_price
    return min(shares, max_affordable)


def max_drawdown(equity_curve: list) -> float:
    """Calculate maximum drawdown from an equity curve."""
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def sharpe_ratio(returns: list, risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    returns_arr = np.array(returns)
    excess = returns_arr - (risk_free_rate / 252)
    if np.std(excess) == 0:
        return 0.0
    return np.mean(excess) / np.std(excess) * np.sqrt(252)


def sortino_ratio(returns: list, risk_free_rate: float = 0.05) -> float:
    """Calculate annualized Sortino ratio (downside deviation only)."""
    if len(returns) < 2:
        return 0.0
    returns_arr = np.array(returns)
    excess = returns_arr - (risk_free_rate / 252)
    downside = excess[excess < 0]
    if len(downside) == 0 or np.std(downside) == 0:
        return 0.0
    return np.mean(excess) / np.std(downside) * np.sqrt(252)
