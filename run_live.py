#!/usr/bin/env python3
"""CashFlow - Live/Paper Trading Runner.

Runs the Polymarket longshot scanner on a schedule.
In paper mode (default), it logs what it WOULD trade.
In live mode, it executes trades via the Polymarket CLOB API.

Usage:
    python run_live.py                    # Paper trade (default)
    python run_live.py --live             # Live trading (requires wallet)
    python run_live.py --interval 10      # Scan every 10 minutes
    python run_live.py --capital 50       # Start with $50
    python run_live.py --once             # Single scan then exit
"""
import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from cashflow.data.polymarket_client import PolymarketClient
from cashflow.data.arbitrage_feeds import ArbitrageEngine, ExternalProbability
from cashflow.strategies.polymarket_longshot import (
    PolymarketLongshotStrategy,
    LongshotPortfolio,
    LongshotPosition,
)
from cashflow.utils.config import load_config

# --- Logging ---

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "cashflow.log"),
    ],
)
log = logging.getLogger("cashflow")


# --- State Persistence ---

STATE_FILE = Path("data/portfolio_state.json")


def save_state(portfolio: LongshotPortfolio, scan_count: int):
    """Persist portfolio state to disk."""
    state = {
        "capital": portfolio.capital,
        "total_invested": portfolio.total_invested,
        "scan_count": scan_count,
        "last_updated": datetime.now().isoformat(),
        "open_positions": [
            {
                "market_id": p.market_id,
                "question": p.question,
                "category": p.category,
                "entry_price": p.entry_price,
                "model_prob": p.model_prob,
                "num_contracts": p.num_contracts,
                "total_cost": p.total_cost,
                "max_payout": p.max_payout,
                "expected_value": p.expected_value,
                "kelly_fraction": p.kelly_fraction,
                "edge": p.edge,
                "entry_date": str(p.entry_date) if p.entry_date else None,
            }
            for p in portfolio.positions
        ],
        "closed_count": len(portfolio.closed),
        "total_pnl": round(sum(p.pnl for p in portfolio.closed), 2),
        "equity_curve": portfolio.equity_curve[-100:],  # Last 100 points
    }
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_state(config: dict, capital: float) -> tuple:
    """Load portfolio state from disk, or create fresh."""
    poly_config = config.get("polymarket_longshot", {})
    strategy = PolymarketLongshotStrategy(poly_config)

    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                state = json.load(f)
            portfolio = LongshotPortfolio(
                capital=state["capital"],
                max_positions=strategy.max_positions,
                total_invested=state["total_invested"],
            )
            portfolio.equity_curve = state.get("equity_curve", [state["capital"]])
            # Restore open positions
            for pos_data in state.get("open_positions", []):
                pos = LongshotPosition(**pos_data)
                portfolio.positions.append(pos)
                cat = pos.category
                portfolio.category_exposure[cat] = portfolio.category_exposure.get(cat, 0) + 1
            scan_count = state.get("scan_count", 0)
            log.info(f"Restored state: ${portfolio.capital:.2f} capital, "
                     f"{len(portfolio.positions)} open positions, "
                     f"${state.get('total_pnl', 0):.2f} total P&L")
            return strategy, portfolio, scan_count
        except Exception as e:
            log.warning(f"Failed to load state: {e}. Starting fresh.")

    portfolio = LongshotPortfolio(capital=capital, max_positions=strategy.max_positions)
    portfolio.equity_curve.append(capital)
    return strategy, portfolio, 0


# --- Scanner ---

class LiveScanner:
    """Scans Polymarket for longshot opportunities using cross-platform signals."""

    def __init__(self, strategy, portfolio, arb_engine, paper_mode=True):
        self.strategy = strategy
        self.portfolio = portfolio
        self.arb_engine = arb_engine
        self.paper_mode = paper_mode
        self.poly_client = PolymarketClient()
        self.scan_count = 0
        self.opportunities_found = 0
        self.trades_taken = 0

    def scan(self):
        """Run one scan cycle."""
        self.scan_count += 1
        mode = "PAPER" if self.paper_mode else "LIVE"
        log.info(f"--- Scan #{self.scan_count} [{mode}] ---")
        log.info(f"Capital: ${self.portfolio.capital:.2f} | "
                 f"Invested: ${self.portfolio.total_invested:.2f} | "
                 f"Open: {len(self.portfolio.positions)}/{self.strategy.max_positions}")

        # Step 1: Find cheap YES tokens on Polymarket
        try:
            longshots = self.poly_client.find_longshot_opportunities(
                max_yes_price=self.strategy.max_entry_price,
                min_volume=1000,
                min_liquidity=500,
            )
            log.info(f"Found {len(longshots)} longshot markets on Polymarket")
        except Exception as e:
            log.error(f"Failed to fetch Polymarket data: {e}")
            longshots = []

        if not longshots:
            log.info("No longshot opportunities found this scan.")
            self.portfolio.equity_curve.append(self.portfolio.capital)
            return

        # Step 2: Check open positions for exits
        self._check_exits(longshots)

        # Step 3: Evaluate new opportunities
        new_trades = 0
        for market in longshots:
            if len(self.portfolio.positions) >= self.strategy.max_positions:
                break

            # Skip if we already have a position in this market
            if any(p.market_id == market.condition_id for p in self.portfolio.positions):
                continue

            # Get cross-platform probability estimate
            model_prob = self._get_ensemble_probability(market)
            if model_prob is None:
                continue

            # Evaluate through strategy gates
            position = self.strategy.evaluate_market(
                market_id=market.condition_id,
                question=market.question,
                category=market.category,
                market_price=market.yes_price,
                model_prob=model_prob,
                portfolio=self.portfolio,
            )

            if position:
                self.opportunities_found += 1
                log.info(
                    f"  OPPORTUNITY: {market.question[:60]}... | "
                    f"Price: ${market.yes_price:.3f} | "
                    f"Model: {model_prob:.1%} | "
                    f"Edge: {position.edge:.1%} | "
                    f"Stake: ${position.total_cost:.2f} | "
                    f"Max Payout: ${position.max_payout:.2f}"
                )

                if self.paper_mode:
                    log.info(f"  [PAPER] Would open position (not executing)")
                    # Still track it in paper mode for P&L tracking
                    self.strategy.open_position(position, self.portfolio)
                    new_trades += 1
                else:
                    # TODO: Execute via Polymarket CLOB API
                    # For now, log the trade that would be made
                    log.info(f"  [LIVE] Executing trade... (CLOB integration pending)")
                    self.strategy.open_position(position, self.portfolio)
                    new_trades += 1

        self.trades_taken += new_trades
        self.portfolio.equity_curve.append(self.portfolio.capital)

        log.info(f"Scan #{self.scan_count} complete: {new_trades} new trades | "
                 f"Total open: {len(self.portfolio.positions)} | "
                 f"Capital: ${self.portfolio.capital:.2f}")

    def _get_ensemble_probability(self, market) -> float:
        """Get ensemble probability from cross-platform sources."""
        try:
            external_probs = self.arb_engine.find_matching_probabilities(market.question)
            if external_probs:
                ensemble, confidence, count = self.arb_engine.compute_ensemble_probability(
                    external_probs
                )
                if count >= 2 and confidence >= 0.3:
                    log.debug(f"  Ensemble: {ensemble:.3f} (conf={confidence:.2f}, sources={count})")
                    return ensemble

            # Fall back to Polymarket price as a baseline (no trade if no edge)
            return None
        except Exception as e:
            log.debug(f"  Arbitrage lookup failed: {e}")
            return None

    def _check_exits(self, markets):
        """Check open positions against current prices for early exits."""
        if not self.portfolio.positions:
            return

        # Build price map from current market data
        price_map = {}
        for market in markets:
            price_map[market.condition_id] = market.yes_price

        exits = self.strategy.check_early_exits(self.portfolio, price_map)
        for pos, price, reason in exits:
            log.info(
                f"  EXIT [{reason}]: {pos.question[:50]}... | "
                f"Entry: ${pos.entry_price:.3f} -> Current: ${price:.3f} | "
                f"P&L: ${pos.num_contracts * price - pos.total_cost:.2f}"
            )
            if self.paper_mode:
                log.info(f"  [PAPER] Would close position")
            self.strategy.close_position(
                pos, outcome=None, portfolio=self.portfolio,
                exit_price=price, reason=reason,
            )


# --- Main Loop ---

_running = True


def _signal_handler(sig, frame):
    global _running
    log.info("Shutdown signal received. Saving state and exiting...")
    _running = False


def main():
    global _running

    parser = argparse.ArgumentParser(description="CashFlow Live/Paper Trading")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: paper)")
    parser.add_argument("--capital", type=float, default=50.0, help="Starting capital (default: $50)")
    parser.add_argument("--interval", type=int, default=15, help="Scan interval in minutes (default: 15)")
    parser.add_argument("--once", action="store_true", help="Run single scan and exit")
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override capital
    if args.capital:
        config["general"]["initial_capital"] = args.capital

    mode = "LIVE" if args.live else "PAPER"
    log.info(f"{'='*50}")
    log.info(f"  CashFlow Trading Bot - {mode} MODE")
    log.info(f"  Capital: ${args.capital:.2f}")
    log.info(f"  Scan interval: {args.interval} min")
    log.info(f"{'='*50}")

    if args.live:
        log.warning("LIVE TRADING ENABLED - Real money at risk!")
        log.warning("CLOB execution is not yet implemented - trades will be logged only.")

    # Load or create state
    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log.info("State reset.")

    strategy, portfolio, scan_count = load_state(config, args.capital)
    arb_engine = ArbitrageEngine(config.get("polymarket_longshot", {}))

    scanner = LiveScanner(
        strategy=strategy,
        portfolio=portfolio,
        arb_engine=arb_engine,
        paper_mode=not args.live,
    )
    scanner.scan_count = scan_count

    # Handle graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.once:
        scanner.scan()
        save_state(portfolio, scanner.scan_count)
        _print_summary(scanner, portfolio)
        return

    # Main loop
    log.info(f"Starting scan loop (every {args.interval} min). Press Ctrl+C to stop.")
    while _running:
        try:
            scanner.scan()
            save_state(portfolio, scanner.scan_count)
        except Exception as e:
            log.error(f"Scan failed: {e}", exc_info=True)

        if not _running:
            break

        log.info(f"Next scan in {args.interval} minutes...")
        # Sleep in 10s increments so we can catch shutdown signals
        for _ in range(args.interval * 6):
            if not _running:
                break
            time.sleep(10)

    # Final save
    save_state(portfolio, scanner.scan_count)
    _print_summary(scanner, portfolio)
    log.info("Shutdown complete.")


def _print_summary(scanner, portfolio):
    """Print session summary."""
    log.info(f"\n{'='*50}")
    log.info(f"  SESSION SUMMARY")
    log.info(f"{'='*50}")
    log.info(f"  Scans completed: {scanner.scan_count}")
    log.info(f"  Opportunities found: {scanner.opportunities_found}")
    log.info(f"  Trades taken: {scanner.trades_taken}")
    log.info(f"  Open positions: {len(portfolio.positions)}")
    log.info(f"  Closed positions: {len(portfolio.closed)}")
    if portfolio.closed:
        total_pnl = sum(p.pnl for p in portfolio.closed)
        wins = sum(1 for p in portfolio.closed if p.pnl > 0)
        log.info(f"  Total P&L: ${total_pnl:.2f}")
        log.info(f"  Win rate: {wins}/{len(portfolio.closed)}")
    log.info(f"  Capital: ${portfolio.capital:.2f}")
    log.info(f"  State saved to: {STATE_FILE}")


if __name__ == "__main__":
    main()
