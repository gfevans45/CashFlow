# Kalshi Hybrid Vol Fade + Directional Strategy

## Overview

Automated trading strategy for Kalshi binary event contracts targeting S&P 500 daily range/direction contracts (weekdays) and Bitcoin range contracts (overnight/weekends). Designed for consistent small gains with strict risk management.

## Strategy Profile

- **Style**: Grinder — high win rate, small consistent gains
- **Automation**: Fully automated, no manual approval
- **Capital**: $50 starting
- **Target daily return**: 0.5-1.5%
- **Target win rate**: >60%
- **R:R**: 1:1 enforced via contract selection + active exits

## Markets

| Market | Hours (ET) | Scan Frequency | Exit Monitoring |
|--------|-----------|----------------|-----------------|
| S&P 500 daily contracts | 9:15 AM - 4:00 PM | At open | Every 5 min |
| Bitcoin daily/overnight | 6:00 PM - 8:55 AM + weekends | At 6 PM / 8 AM weekends | Every 15 min |

## Entry Rules

### Signal 1: Volatility Fade (core)
- Calculate 20-day realized volatility from historical price data
- Compare against Kalshi implied volatility (derived from bracket contract prices)
- When implied vol > realized vol: market is overpricing wide brackets
- Buy the narrow/center bracket (higher probability of landing)

### Signal 2: Directional Tilt (boost)
- RSI (14-period): oversold (<30) = bullish tilt, overbought (>70) = bearish tilt
- EMA crossover (8/21): fast > slow = bullish, fast < slow = bearish
- Bollinger Band position: below lower band = bullish, above upper = bearish
- When 2+ indicators align, bias bracket selection in trend direction

### Contract Selection
- Prefer contracts priced $0.45-$0.55 (natural 1:1 R:R)
- Accept contracts $0.35-$0.65 if model edge >8%
- Minimum 5% model edge to enter any trade

## Exit Rules

### Active Management (1:1 enforcement)
- At entry, calculate symmetric exit levels:
  - Take profit: entry price + $0.25
  - Stop loss: entry price - $0.25
- Monitor prices at scan frequency (5 min S&P, 15 min BTC)
- Exit via limit order when trigger hit

### Expiration
- If neither exit trigger hits, contract settles at expiration
- For $0.45-0.55 contracts, expiration is approximately 1:1

## Position Sizing

- **Method**: Quarter-Kelly, capped
- **Max per trade**: $5 (10% of capital)
- **Min per trade**: $1
- **Max open positions**: 5
- **Anti-martingale**: sizes scale with capital automatically

### Kelly Formula for Binary Contracts
```
edge = model_prob - market_price
kelly = edge / (1 - market_price)
stake = capital * kelly * 0.25  # quarter-Kelly
stake = clamp(stake, $1, min($5, capital * 0.10))
```

## Risk Management

| Rule | Value |
|------|-------|
| Daily loss limit | $2.50 (5% of capital) |
| Weekly loss limit | $7.50 (15% of capital) |
| Max positions | 5 concurrent |
| Max per trade | 10% of capital |
| Circuit breaker | Stop trading for day if daily limit hit |
| Weekly recovery | Reduce sizing by 50% if weekly limit hit |

## Daily Schedule

| Time (ET) | Action |
|-----------|--------|
| 9:15 AM | Pull S&P data, calculate indicators, scan Kalshi |
| 9:30 AM | Place S&P trades (2-3 brackets + optional directional) |
| 9:35 AM - 3:55 PM | Monitor positions every 5 min |
| 4:00 PM | S&P contracts settle, log results |
| 6:00 PM | Scan BTC overnight contracts, place trades |
| 6:05 PM - 8:55 AM | Monitor BTC every 15 min |
| Sat 8:00 AM | Scan BTC weekend contracts |
| Sun 8:00 AM | Scan BTC weekend contracts |

## Backtest Plan

- **Period**: Last 90 days
- **S&P data**: 5-min candles from Yahoo Finance (60 days real, 30 days simulated intraday)
- **BTC data**: 5-min candles for overnight/weekend simulation
- **Simulated Kalshi contracts**: Generated from actual price ranges with realistic bracket spacing

### Backtest Metrics

| Metric | Target |
|--------|--------|
| Win rate | >60% |
| Average R:R | ~1:1 |
| Daily return | 0.5-1.5% |
| Max drawdown | <15% |
| Sharpe ratio | >1.5 |
| Trades per day | 3-5 |
| Daily loss limit hits | <2x per month |

### Backtest Comparisons
1. Vol Fade alone vs Hybrid (directional tilt value)
2. $0.45-0.55 only vs wider range with active exits
3. Quarter-Kelly vs Half-Kelly vs fixed sizing
4. 5-min vs 15-min exit monitoring

## Infrastructure

- **Server**: DigitalOcean $4/mo droplet (512MB RAM, 1 vCPU)
- **Deployment**: systemd service, same as current setup
- **Dependencies**: Python 3, requests, pandas, numpy
- **API**: Kalshi REST API (free account for market data, auth for trading)

## Future Enhancements (not in scope)
- Daily email/SMS summary
- Adaptive vol model (GARCH instead of rolling window)
- Additional event categories (economics, weather)
- Web dashboard for monitoring
