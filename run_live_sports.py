#!/usr/bin/env python3
"""CashFlow - Live/Paper Sports Trading Bot for Kalshi.

Trades Kalshi sports contracts (NBA player props, game winners,
spreads, over/unders) using statistical models backed by free
NBA data from balldontlie.io.

Daily schedule (all times Eastern):
  10:00 - Scan Kalshi sports markets, fetch player/team stats
  10:00 - Place paper/live trades where model shows >= 5% edge
  Next AM - Check settlements, log P&L

Usage:
    python run_live_sports.py                    # Paper trade (default)
    python run_live_sports.py --live             # Live trading
    python run_live_sports.py --once             # Single run then exit
    python run_live_sports.py --capital 100      # Custom starting capital
    python run_live_sports.py --sport nba        # Only trade NBA contracts
    python run_live_sports.py --reset            # Clear state, start fresh
"""

import argparse
import json
import logging
import random
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

from cashflow.data.sports_feeds import (
    KalshiSportsClient,
    MLBStatsClient,
    NBAStatsClient,
    NCAStatsClient,
    SportsContract,
    generate_simulated_sports_contracts,
)
from cashflow.strategies.sports_strategy import (
    SportsStrategy,
    SportsTrade,
)
from cashflow.utils.config import load_config

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "sports.log"),
    ],
)
log = logging.getLogger("cashflow.sports")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATE_FILE = Path("data/sports_state.json")
ET = ZoneInfo("America/New_York")

# Scan interval (minutes) — sports odds change frequently
SCAN_INTERVAL_MINUTES = 2

# Active hours (Eastern) — only scan when games are listed
ACTIVE_START_HOUR = 10  # 10 AM ET
ACTIVE_END_HOUR = 24    # Midnight ET (11:59 PM)

# Supported sports
SUPPORTED_SPORTS = ["nba", "ncaa", "mlb", "pga", "tennis", "soccer"]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def save_state(state: dict):
    """Persist bot state to disk."""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    state["last_saved"] = datetime.now(ET).isoformat()
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)
    log.debug(f"State saved to {STATE_FILE}")


def load_state() -> dict:
    """Load bot state from disk, or return empty state."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            log.warning(f"Failed to load state: {e}. Starting fresh.")
    return {}


def init_state(capital: float, sport: str) -> dict:
    """Initialize a fresh bot state."""
    return {
        "capital": capital,
        "initial_capital": capital,
        "sport_filter": sport,
        "open_trades": [],
        "settled_trades": [],
        "daily_records": [],
        "total_pnl": 0.0,
        "total_trades": 0,
        "total_wins": 0,
        "total_losses": 0,
        "daily_pnl": 0.0,
        "last_trade_date": "",
        "started_at": datetime.now(ET).isoformat(),
    }


# ---------------------------------------------------------------------------
# Core bot logic
# ---------------------------------------------------------------------------

class SportsTradingBot:
    """Live/paper trading bot for Kalshi sports contracts.

    Fetches sports contracts from Kalshi (or simulates them), computes
    model probabilities using player/team stats, and places trades
    where the model shows sufficient edge over market prices.
    """

    def __init__(self, state: dict, paper_mode: bool = True):
        self.state = state
        self.paper_mode = paper_mode
        self.capital = state["capital"]
        self.sport_filter = state.get("sport_filter", "")

        # Strategy (sized for $100 capital)
        self.strategy = SportsStrategy({
            "min_edge": 0.05,
            "kelly_fraction": 0.25,
            "max_position_pct": 0.10,
            "max_positions": 5,
            "min_stake": 1.0,
            "max_stake": 10.0,
            "daily_loss_limit_pct": 0.05,
        })

        # Data clients
        self.nba_stats = NBAStatsClient()
        self.ncaa_stats = NCAStatsClient()
        self.mlb_stats = MLBStatsClient()
        self.kalshi: Optional[KalshiSportsClient] = None
        self._kalshi_available = False

        self._init_kalshi()

    def _init_kalshi(self):
        """Initialize Kalshi client if credentials are available."""
        try:
            self.kalshi = KalshiSportsClient()
            if self.kalshi.has_credentials:
                if self.kalshi.authenticate():
                    self._kalshi_available = True
                    log.info("Kalshi API connected and authenticated")
                else:
                    log.warning("Kalshi credentials found but auth failed")
            else:
                log.info("No Kalshi credentials — using simulated contracts")
        except Exception as e:
            log.warning(f"Failed to initialize Kalshi client: {e}")

    # --- Market scanning ---

    def scan_markets(self) -> list[SportsContract]:
        """Scan for sports contracts from Kalshi or generate simulated ones.

        Returns list of SportsContract objects.
        """
        if self._kalshi_available:
            return self._scan_real_markets()
        else:
            return self._scan_simulated_markets()

    def _scan_real_markets(self) -> list[SportsContract]:
        """Fetch real Kalshi sports contracts."""
        log.info("Scanning Kalshi for sports contracts (live API)")
        try:
            contracts = self.kalshi.get_sports_contracts(
                sport_filter=self.sport_filter or None,
            )
            log.info(f"  Found {len(contracts)} contracts from Kalshi")
            return contracts
        except Exception as e:
            log.error(f"Failed to fetch Kalshi sports markets: {e}")
            log.info("Falling back to simulated contracts")
            return self._scan_simulated_markets()

    def _scan_simulated_markets(self) -> list[SportsContract]:
        """Generate simulated contracts when Kalshi API is unavailable."""
        log.info("Generating simulated sports contracts (no Kalshi API)")
        contracts = generate_simulated_sports_contracts(num_contracts=15)

        if self.sport_filter:
            contracts = [c for c in contracts
                         if c.sport == self.sport_filter]

        log.info(f"  Generated {len(contracts)} simulated contracts")
        return contracts

    # --- Probability estimation ---

    def compute_probabilities(self, contracts: list[SportsContract]
                               ) -> list[tuple[SportsContract, float]]:
        """Compute model probabilities using a two-pass approach.

        Pass 1 (instant, no API calls):
          Compute base probabilities for ALL contracts using fallback
          models. This uses threshold-based estimates for player props
          and neutral priors for game outcomes.

        Pass 2 (targeted API calls):
          Screen for NBA contracts showing >= 3% base edge. Only fetch
          real stats from balldontlie for those few candidates (~5-15).
          Recompute their probabilities with actual player/team data.

        This keeps API usage minimal (typically < 30 calls per cycle)
        while still validating trades with real stats before execution.

        Returns list of (contract, model_probability) tuples.
        """
        # ── Pass 1: Base probabilities (no API calls) ──
        base_results = []
        for contract in contracts:
            try:
                prob = self._compute_single_probability(
                    contract, stats_available=False,
                )
                base_results.append((contract, prob))
            except Exception as e:
                log.debug(f"Base prob failed for {contract.ticker}: {e}")
                base_results.append((contract, 0.5))

        # ── Screen: Find NBA candidates with base edge >= 3% ──
        SCREEN_EDGE = 0.03  # Lower than trade threshold (5%) to catch near-misses
        nba_candidates = []
        final_results = []

        for contract, base_prob in base_results:
            if contract.sport != "nba":
                final_results.append((contract, base_prob))
                continue

            # Check if base model shows potential edge
            market = contract.yes_price
            if market <= 0 or market >= 1:
                final_results.append((contract, base_prob))
                continue

            yes_edge = base_prob - market
            no_edge = market - base_prob
            best_edge = max(yes_edge, no_edge)

            if best_edge >= SCREEN_EDGE and 0.20 <= market <= 0.80:
                nba_candidates.append((contract, base_prob))
            else:
                final_results.append((contract, base_prob))

        log.info(f"  Pass 1 complete: {len(base_results)} base probs computed, "
                 f"{len(nba_candidates)} NBA candidates with >= {SCREEN_EDGE:.0%} base edge")

        if not nba_candidates or not self.nba_stats.available:
            # No candidates or no API — use base probs as-is
            final_results.extend(nba_candidates)
            return final_results

        # ── Pass 2: Validate candidates with real stats ──
        # Collect unique players/teams from candidates only
        unique_players = set()
        unique_teams = set()
        for contract, _ in nba_candidates:
            if contract.contract_type == "player_prop" and contract.player_name:
                unique_players.add(contract.player_name)
            if contract.team_a:
                unique_teams.add(contract.team_a)
            if contract.team_b:
                unique_teams.add(contract.team_b)

        log.info(f"  Pass 2: Fetching stats for {len(unique_players)} players, "
                 f"{len(unique_teams)} teams (from {len(nba_candidates)} candidates)")

        # Fetch with circuit breaker
        consecutive_failures = 0
        fetched = 0
        api_ok = True

        for name in unique_players:
            if consecutive_failures >= 2:
                log.warning(f"  Circuit breaker: API down after {fetched} lookups. "
                            f"Using base probs for remaining candidates.")
                api_ok = False
                break
            try:
                result = self.nba_stats.get_player_stats(name)
                if result:
                    consecutive_failures = 0
                    fetched += 1
                else:
                    consecutive_failures += 1
            except Exception:
                consecutive_failures += 1

        if api_ok:
            for name in unique_teams:
                if consecutive_failures >= 2:
                    log.warning(f"  Circuit breaker: API down for team stats.")
                    api_ok = False
                    break
                try:
                    result = self.nba_stats.get_team_stats(name)
                    if result:
                        consecutive_failures = 0
                        fetched += 1
                    else:
                        consecutive_failures += 1
                except Exception:
                    consecutive_failures += 1

        log.info(f"  Pass 2 complete: {fetched} stats fetched")

        # Recompute probabilities for candidates with real stats
        for contract, base_prob in nba_candidates:
            try:
                validated_prob = self._compute_single_probability(
                    contract, stats_available=api_ok,
                )
                final_results.append((contract, validated_prob))
            except Exception as e:
                log.debug(f"Validated prob failed for {contract.ticker}: {e}")
                final_results.append((contract, base_prob))

        return final_results

    def _compute_single_probability(self, contract: SportsContract,
                                     stats_available: bool) -> float:
        """Compute model probability for one contract."""
        player_stats = None
        team_a_stats = None
        team_b_stats = None
        mlb_player_stats = None
        mlb_team_a_stats = None
        mlb_team_b_stats = None

        sport = contract.sport

        if sport == "mlb":
            # MLB-specific stats
            if contract.contract_type == "player_prop" and contract.player_name:
                try:
                    mlb_player_stats = self.mlb_stats.get_player_stats(
                        contract.player_name,
                    )
                    if mlb_player_stats:
                        log.debug(f"  MLB stats for {contract.player_name}: "
                                  f"K/g={mlb_player_stats.strikeouts_per_game}, "
                                  f"H/g={mlb_player_stats.hits_avg}")
                except Exception as e:
                    log.debug(f"  MLB stats lookup failed for "
                              f"{contract.player_name}: {e}")

            elif contract.contract_type in ("game_winner", "over_under"):
                if contract.team_a:
                    try:
                        mlb_team_a_stats = self.mlb_stats.get_team_stats(
                            contract.team_a,
                        )
                    except Exception as e:
                        log.debug(f"  MLB team stats failed for {contract.team_a}: {e}")
                if contract.team_b:
                    try:
                        mlb_team_b_stats = self.mlb_stats.get_team_stats(
                            contract.team_b,
                        )
                    except Exception as e:
                        log.debug(f"  MLB team stats failed for {contract.team_b}: {e}")

        elif sport in ("nba", "ncaa"):
            # NBA/NCAA stats (original logic)
            if contract.contract_type == "player_prop" and contract.player_name:
                if stats_available:
                    try:
                        player_stats = self.nba_stats.get_player_stats(
                            contract.player_name,
                        )
                        if player_stats:
                            log.debug(f"  Stats for {contract.player_name}: "
                                      f"{player_stats.pts_avg} ppg "
                                      f"(std={player_stats.pts_std})")
                    except Exception as e:
                        log.debug(f"  Stats lookup failed for "
                                  f"{contract.player_name}: {e}")

            elif contract.contract_type in ("game_winner", "spread", "over_under"):
                if stats_available and contract.team_a:
                    try:
                        team_a_stats = self.nba_stats.get_team_stats(
                            contract.team_a,
                        )
                    except Exception as e:
                        log.debug(f"  Team stats failed for {contract.team_a}: {e}")

                if stats_available and contract.team_b:
                    try:
                        team_b_stats = self.nba_stats.get_team_stats(
                            contract.team_b,
                        )
                    except Exception as e:
                        log.debug(f"  Team stats failed for {contract.team_b}: {e}")

        return self.strategy.estimate_contract_probability(
            contract,
            player_stats=player_stats,
            team_a_stats=team_a_stats,
            team_b_stats=team_b_stats,
            ncaa_client=self.ncaa_stats if sport == "ncaa" else None,
            mlb_player_stats=mlb_player_stats,
            mlb_team_a_stats=mlb_team_a_stats,
            mlb_team_b_stats=mlb_team_b_stats,
        )

    # --- Trading ---

    def evaluate_and_trade(self, contracts_with_probs: list[tuple]
                            ) -> list[SportsTrade]:
        """Evaluate contracts and place trades (paper or live).

        Returns list of SportsTrade objects for trades placed.
        """
        open_count = len(self.state.get("open_trades", []))
        daily_pnl = self.state.get("daily_pnl", 0.0)

        trades = self.strategy.evaluate_contracts(
            contracts_with_probs, self.capital, daily_pnl, open_count,
        )

        if not trades:
            log.info("No trades meet edge/sizing criteria")
            return []

        mode_tag = "PAPER" if self.paper_mode else "LIVE"
        placed = []

        for trade in trades:
            log.info(
                f"  [{mode_tag}] {trade.position.upper()} "
                f"{trade.contract_type} | {trade.title[:50]} "
                f"| Price: ${trade.entry_price:.3f} "
                f"| Model: {trade.model_prob:.3f} "
                f"| Edge: {trade.edge:+.3f} "
                f"| Stake: ${trade.stake:.2f} "
                f"| Qty: {trade.num_contracts}"
            )

            order_id = None
            order_status = "paper"
            fill_count = trade.num_contracts  # paper assumes full fill

            if not self.paper_mode and self._kalshi_available:
                price_cents = int(trade.entry_price * 100)
                result = self.kalshi.place_order(
                    trade.ticker, trade.position,
                    trade.num_contracts, price_cents,
                )
                if result:
                    order = result.get("order", {})
                    order_id = order.get("order_id")
                    order_status = order.get("status", "unknown")
                    fill_count = int(float(order.get("fill_count_fp", "0")))
                    log.info(f"    [LIVE] Order {order_id}: status={order_status}, "
                             f"filled={fill_count}/{trade.num_contracts}")
                else:
                    log.warning(f"    [LIVE] Order submission failed — skipping trade")
                    continue  # Don't track or deduct capital for failed orders

            # Track the trade
            trade_record = {
                "ticker": trade.ticker,
                "title": trade.title,
                "contract_type": trade.contract_type,
                "sport": trade.sport,
                "position": trade.position,
                "entry_price": trade.entry_price,
                "model_prob": trade.model_prob,
                "market_price": trade.market_price,
                "edge": trade.edge,
                "stake": trade.stake,
                "num_contracts": trade.num_contracts,
                "placed_at": datetime.now(ET).isoformat(),
                "settled": False,
                "pnl": 0.0,
                "order_id": order_id,
                "order_status": order_status,
                "fill_count": fill_count,
            }

            self.state.setdefault("open_trades", []).append(trade_record)

            # Only deduct capital for filled contracts, not resting ones
            if order_status == "executed":
                # Fully filled — deduct full stake
                self.capital -= trade.stake
            elif order_status == "resting" and fill_count > 0:
                # Partially filled — deduct proportional stake
                filled_stake = trade.stake * (fill_count / trade.num_contracts)
                self.capital -= filled_stake
            elif order_status == "paper":
                # Paper mode — deduct full stake
                self.capital -= trade.stake
            # else: resting with 0 fills — no capital deducted yet

            self.state["capital"] = round(self.capital, 2)
            self.state["total_trades"] = self.state.get("total_trades", 0) + 1
            placed.append(trade)

        log.info(f"Placed {len(placed)} trades | "
                 f"Capital remaining: ${self.capital:.2f}")
        return placed

    # --- Settlement ---

    def check_settlements(self) -> float:
        """Check and settle trades.

        Live mode: Query Kalshi positions for actual settlement results.
        Paper mode: Simulate outcomes using model probability (coin flip).

        Returns total settlement P&L.
        """
        open_trades = self.state.get("open_trades", [])
        if not open_trades:
            return 0.0

        today = datetime.now(ET).strftime("%Y-%m-%d")
        settlement_pnl = 0.0
        still_open = []
        newly_settled = []

        # In live mode, fetch positions from Kalshi to check settlements
        kalshi_positions = {}
        if not self.paper_mode and self._kalshi_available:
            positions = self.kalshi.get_positions()
            for pos in positions:
                ticker = pos.get("ticker", pos.get("market_ticker", ""))
                if ticker:
                    kalshi_positions[ticker] = pos

        for trade_record in open_trades:
            placed_at = trade_record.get("placed_at", "")
            trade_date = placed_at[:10] if placed_at else ""
            fill_count = trade_record.get("fill_count", trade_record.get("num_contracts", 0))

            # Skip unfilled orders (resting with 0 fills)
            if trade_record.get("order_status") == "resting" and fill_count == 0:
                still_open.append(trade_record)
                continue

            # Live mode: check Kalshi for settlement
            if not self.paper_mode and self._kalshi_available:
                ticker = trade_record["ticker"]
                position_data = kalshi_positions.get(ticker)

                if position_data:
                    # Check if market has settled
                    market_result = position_data.get("market_result", "")
                    if market_result in ("yes", "no", "all_no", "all_yes"):
                        outcome = market_result in ("yes", "all_yes")
                        trade = self._make_trade(trade_record, fill_count)
                        pnl = self.strategy.settle_trade(trade, outcome)
                        settlement_pnl += pnl
                        self.capital += pnl + trade.stake
                        self.state["capital"] = round(self.capital, 2)

                        outcome_str = "WIN" if pnl > 0 else "LOSS"
                        log.info(
                            f"  SETTLED [{outcome_str}]: {trade_record['title'][:40]} "
                            f"| {trade_record['position'].upper()} "
                            f"| P&L: ${pnl:+.2f} (Kalshi: {market_result})"
                        )

                        trade_record["settled"] = True
                        trade_record["pnl"] = pnl
                        trade_record["outcome"] = outcome
                        trade_record["settled_at"] = datetime.now(ET).isoformat()
                        newly_settled.append(trade_record)

                        if pnl > 0:
                            self.state["total_wins"] = self.state.get("total_wins", 0) + 1
                        else:
                            self.state["total_losses"] = self.state.get("total_losses", 0) + 1
                        continue

                # Not settled yet — keep open
                still_open.append(trade_record)
                continue

            # Paper mode: only settle trades from previous days
            if trade_date >= today:
                still_open.append(trade_record)
                continue

            model_prob = trade_record.get("model_prob", 0.5)
            outcome = random.random() < model_prob
            trade = self._make_trade(trade_record, fill_count)
            pnl = self.strategy.settle_trade(trade, outcome)
            settlement_pnl += pnl
            self.capital += pnl + trade.stake
            self.state["capital"] = round(self.capital, 2)

            outcome_str = "WIN" if pnl > 0 else "LOSS"
            log.info(
                f"  SETTLED [{outcome_str}]: {trade_record['title'][:40]} "
                f"| {trade_record['position'].upper()} "
                f"| P&L: ${pnl:+.2f} (simulated)"
            )

            trade_record["settled"] = True
            trade_record["pnl"] = pnl
            trade_record["outcome"] = outcome
            trade_record["settled_at"] = datetime.now(ET).isoformat()
            newly_settled.append(trade_record)

            if pnl > 0:
                self.state["total_wins"] = self.state.get("total_wins", 0) + 1
            else:
                self.state["total_losses"] = self.state.get("total_losses", 0) + 1

        self.state["open_trades"] = still_open
        self.state.setdefault("settled_trades", []).extend(newly_settled)
        self.state["total_pnl"] = round(
            self.state.get("total_pnl", 0.0) + settlement_pnl, 2,
        )

        if newly_settled:
            log.info(f"Settled {len(newly_settled)} trades | "
                     f"Settlement P&L: ${settlement_pnl:+.2f} | "
                     f"Capital: ${self.capital:.2f}")

        return settlement_pnl

    def _make_trade(self, trade_record: dict, fill_count: int = None) -> SportsTrade:
        """Reconstruct a SportsTrade from a trade record."""
        num = fill_count if fill_count else trade_record["num_contracts"]
        entry = trade_record["entry_price"]
        stake = num * entry  # Recalculate based on actual fills
        return SportsTrade(
            ticker=trade_record["ticker"],
            title=trade_record["title"],
            contract_type=trade_record["contract_type"],
            sport=trade_record["sport"],
            position=trade_record["position"],
            entry_price=entry,
            model_prob=trade_record.get("model_prob", 0.5),
            market_price=trade_record["market_price"],
            edge=trade_record["edge"],
            stake=stake,
            num_contracts=num,
        )

    # --- Order management ---

    def manage_orders(self, contracts: list = None):
        """Check resting orders and cancel if edge has deteriorated.

        For each resting order:
        - Fetch current market price from scanned contracts
        - Recompute edge: model_prob vs current market
        - Cancel if edge < min_edge threshold (5%)
        - Reclaim capital for cancelled/unfilled orders
        - Update fill status for partially/fully filled orders
        """
        if self.paper_mode or not self._kalshi_available:
            return

        open_trades = self.state.get("open_trades", [])
        resting = [t for t in open_trades if t.get("order_status") == "resting"]
        if not resting:
            return

        log.info(f"Checking {len(resting)} resting orders...")

        # Build ticker -> current market price lookup from scanned contracts
        market_prices = {}
        if contracts:
            for c in contracts:
                market_prices[c.ticker] = c.yes_price

        # Fetch current order status from Kalshi
        kalshi_orders = self.kalshi.get_orders(status="resting")
        kalshi_order_map = {o["order_id"]: o for o in kalshi_orders}

        min_edge = self.strategy.min_edge
        cancelled = 0
        updated = 0

        for trade_record in open_trades:
            order_id = trade_record.get("order_id")
            if not order_id or trade_record.get("order_status") != "resting":
                continue

            # Check if order is still resting on Kalshi
            kalshi_order = kalshi_order_map.get(order_id)
            if not kalshi_order:
                # Order no longer resting — it filled or was already cancelled
                # Fetch all orders to find its final status
                fill_count = trade_record.get("fill_count", 0)
                trade_record["order_status"] = "executed" if fill_count > 0 else "canceled"
                if trade_record["order_status"] == "executed" and fill_count > 0:
                    # Deduct capital for fills we haven't accounted for
                    filled_stake = trade_record["stake"] * (fill_count / trade_record["num_contracts"])
                    self.capital -= filled_stake
                    self.state["capital"] = round(self.capital, 2)
                log.info(f"  Order {order_id[:12]}... no longer resting → {trade_record['order_status']}")
                updated += 1
                continue

            # Update fill count from Kalshi
            new_fill_count = int(float(kalshi_order.get("fill_count_fp", "0")))
            old_fill_count = trade_record.get("fill_count", 0)
            if new_fill_count > old_fill_count:
                # New fills since last check — deduct capital for new fills
                new_fills = new_fill_count - old_fill_count
                per_contract_cost = trade_record["stake"] / trade_record["num_contracts"]
                self.capital -= new_fills * per_contract_cost
                self.state["capital"] = round(self.capital, 2)
                trade_record["fill_count"] = new_fill_count
                log.info(f"  Order {order_id[:12]}... filled {new_fills} more "
                         f"({new_fill_count}/{trade_record['num_contracts']} total)")
                updated += 1

            # Re-evaluate edge using current market price
            ticker = trade_record["ticker"]
            current_price = market_prices.get(ticker)
            if current_price is None:
                # Can't find current price — leave order alone
                continue

            model_prob = trade_record["model_prob"]
            position = trade_record["position"]

            if position == "yes":
                current_edge = model_prob - current_price
            else:  # "no"
                current_edge = current_price - model_prob

            if current_edge < min_edge:
                # Edge gone — cancel unfilled portion
                unfilled = trade_record["num_contracts"] - new_fill_count
                if unfilled > 0 and self.kalshi.cancel_order(order_id):
                    trade_record["order_status"] = "canceled"
                    log.info(f"  CANCELLED: {trade_record['title'][:40]} | "
                             f"edge={current_edge:+.3f} < {min_edge:.3f} | "
                             f"cancelled {unfilled} unfilled contracts")
                    cancelled += 1
                elif unfilled == 0:
                    # Fully filled, just update status
                    trade_record["order_status"] = "executed"
                    updated += 1

        # Remove cancelled orders with 0 fills from open_trades
        still_open = []
        for t in open_trades:
            if t.get("order_status") == "canceled" and t.get("fill_count", 0) == 0:
                # Fully cancelled, no fills — drop from tracking
                self.state["total_trades"] = max(0, self.state.get("total_trades", 0) - 1)
                continue
            elif t.get("order_status") == "canceled" and t.get("fill_count", 0) > 0:
                # Partially filled then cancelled — keep as executed with reduced size
                t["order_status"] = "executed"
                t["num_contracts"] = t["fill_count"]
                t["stake"] = t["stake"] * (t["fill_count"] / t.get("num_contracts", t["fill_count"]))
                still_open.append(t)
            else:
                still_open.append(t)

        self.state["open_trades"] = still_open
        save_state(self.state)

        if cancelled or updated:
            log.info(f"Order management: {cancelled} cancelled, {updated} updated | "
                     f"Capital: ${self.capital:.2f}")

    # --- Daily run ---

    def run_daily_cycle(self):
        """Execute one full daily trading cycle.

        1. Check settlements from previous trades
        1.5. Manage resting orders (cancel if edge gone)
        2. Scan Kalshi sports markets (or simulated)
        3. Fetch player/team stats and compute model probabilities
        4. Evaluate and place trades
        5. Save state
        """
        mode = "PAPER" if self.paper_mode else "LIVE"
        today = datetime.now(ET).strftime("%Y-%m-%d")

        log.info("=" * 60)
        log.info(f"  Sports Trading Bot - Daily Cycle [{mode}]")
        log.info(f"  Date: {today}")
        log.info(f"  Capital: ${self.capital:.2f}")
        log.info(f"  Open trades: {len(self.state.get('open_trades', []))}")
        log.info(f"  Sport filter: {self.sport_filter or 'all'}")
        log.info(f"  Stats API: {'available' if self.nba_stats.available else 'unavailable (using fallbacks)'}")
        log.info("=" * 60)

        # Reset daily P&L
        self.state["daily_pnl"] = 0.0

        # Step 1: Settle previous trades
        settlement_pnl = self.check_settlements()
        self.state["daily_pnl"] += settlement_pnl

        # Step 2: Scan markets
        contracts = self.scan_markets()
        if not contracts:
            log.warning("No sports contracts found — skipping trading")
            save_state(self.state)
            return

        # Step 2.5: Manage resting orders (cancel if edge gone)
        self.manage_orders(contracts)

        # Step 3: Compute probabilities
        log.info(f"Computing model probabilities for {len(contracts)} contracts...")
        contracts_with_probs = self.compute_probabilities(contracts)

        # Log top opportunities
        sorted_by_edge = sorted(
            contracts_with_probs,
            key=lambda x: abs(x[1] - x[0].yes_price),
            reverse=True,
        )
        for contract, prob in sorted_by_edge[:8]:
            edge = prob - contract.yes_price
            log.info(
                f"  {contract.contract_type:12s} | "
                f"{contract.title[:40]:40s} | "
                f"mkt=${contract.yes_price:.3f} "
                f"model={prob:.3f} "
                f"edge={edge:+.3f}"
            )

        # Step 4: Evaluate and trade
        trades = self.evaluate_and_trade(contracts_with_probs)

        # Step 5: Record daily result
        daily_record = {
            "date": today,
            "contracts_scanned": len(contracts),
            "trades_placed": len(trades),
            "settlement_pnl": round(settlement_pnl, 2),
            "capital": round(self.capital, 2),
        }
        self.state.setdefault("daily_records", []).append(daily_record)
        self.state["last_trade_date"] = today

        # Save state
        save_state(self.state)

        # Print summary
        self._print_daily_summary(trades, settlement_pnl, contracts)

    def _print_daily_summary(self, trades: list, settlement_pnl: float,
                              contracts: list):
        """Print end-of-cycle summary."""
        total_pnl = self.state.get("total_pnl", 0.0)
        total_trades = self.state.get("total_trades", 0)
        total_wins = self.state.get("total_wins", 0)
        total_losses = self.state.get("total_losses", 0)
        win_rate = (total_wins / (total_wins + total_losses) * 100
                    if (total_wins + total_losses) > 0 else 0)

        log.info("-" * 60)
        log.info("  DAILY SUMMARY")
        log.info("-" * 60)
        log.info(f"  Contracts scanned: {len(contracts)}")
        log.info(f"  Trades placed: {len(trades)}")
        log.info(f"  Settlement P&L: ${settlement_pnl:+.2f}")
        log.info(f"  Open trades: {len(self.state.get('open_trades', []))}")
        log.info(f"  Capital: ${self.capital:.2f}")
        log.info(f"  Total P&L: ${total_pnl:+.2f}")
        log.info(f"  Total trades: {total_trades} "
                 f"({total_wins}W/{total_losses}L, {win_rate:.0f}%)")
        log.info(f"  State: {STATE_FILE}")
        log.info("-" * 60)


# ---------------------------------------------------------------------------
# Scheduling helpers
# ---------------------------------------------------------------------------

def _is_active_hours() -> bool:
    """Check if we're within active scanning hours (Eastern)."""
    now = datetime.now(ET)
    end = ACTIVE_END_HOUR if ACTIVE_END_HOUR <= 23 else 24
    return ACTIVE_START_HOUR <= now.hour < end


def _seconds_until_active() -> float:
    """Seconds until the next active window opens."""
    now = datetime.now(ET)
    if now.hour < ACTIVE_START_HOUR:
        target = now.replace(hour=ACTIVE_START_HOUR, minute=0,
                             second=0, microsecond=0)
    else:
        # Past end — next active is tomorrow morning
        tomorrow = now + timedelta(days=1)
        target = tomorrow.replace(hour=ACTIVE_START_HOUR, minute=0,
                                  second=0, microsecond=0)
    return max(0, (target - now).total_seconds())


# ---------------------------------------------------------------------------
# Signal handling
# ---------------------------------------------------------------------------

_running = True


def _signal_handler(sig, frame):
    global _running
    log.info("Shutdown signal received. Saving state and exiting...")
    _running = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global _running

    parser = argparse.ArgumentParser(
        description="CashFlow Sports Trading Bot for Kalshi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_live_sports.py                    # Paper trade, default settings
  python run_live_sports.py --once             # Single run and exit
  python run_live_sports.py --live             # Live trading (requires Kalshi creds)
  python run_live_sports.py --capital 100      # Start with $100
  python run_live_sports.py --sport nba        # Only trade NBA contracts
  python run_live_sports.py --reset            # Clear state, start fresh
        """,
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading (default: paper mode)",
    )
    parser.add_argument(
        "--capital", type=float, default=100.0,
        help="Starting capital in dollars (default: $100)",
    )
    parser.add_argument(
        "--sport", type=str, default="",
        choices=["", "nba", "ncaa", "nfl", "mlb", "pga", "tennis", "soccer"],
        help="Filter to a specific sport (default: all)",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Run a single cycle and exit (for testing / cron)",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear saved state and start fresh",
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to config YAML file",
    )
    args = parser.parse_args()

    mode = "LIVE" if args.live else "PAPER"
    log.info("=" * 60)
    log.info(f"  CashFlow Sports Trading Bot - {mode} MODE")
    log.info(f"  Capital: ${args.capital:.2f}")
    log.info(f"  Sport: {args.sport or 'all'}")
    if args.once:
        log.info("  Mode: single run")
    else:
        log.info(f"  Schedule: every {SCAN_INTERVAL_MINUTES} min, "
                 f"{ACTIVE_START_HOUR}:00-{ACTIVE_END_HOUR}:00 ET")
    log.info("=" * 60)

    if args.live:
        log.warning("LIVE TRADING ENABLED - Real money at risk!")

    # State management
    if args.reset and STATE_FILE.exists():
        STATE_FILE.unlink()
        log.info("State reset.")

    state = load_state()
    if not state or args.reset:
        state = init_state(args.capital, args.sport)
        log.info("Initialized fresh state")
    else:
        state["sport_filter"] = args.sport
        log.info(f"Restored state: ${state['capital']:.2f} capital, "
                 f"{len(state.get('open_trades', []))} open trades, "
                 f"${state.get('total_pnl', 0):.2f} total P&L")

    # Create bot
    bot = SportsTradingBot(state, paper_mode=not args.live)

    # Handle graceful shutdown
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    if args.once:
        bot.run_daily_cycle()
        _print_session_summary(bot)
        return

    # Long-running service mode — scan every N minutes during active hours
    log.info(f"Starting scan loop: every {SCAN_INTERVAL_MINUTES} min, "
             f"{ACTIVE_START_HOUR}:00-{ACTIVE_END_HOUR}:00 ET. Ctrl+C to stop.")

    scan_interval_sec = SCAN_INTERVAL_MINUTES * 60

    while _running:
        try:
            # Wait until active hours if outside window
            if not _is_active_hours():
                wait = _seconds_until_active()
                resume_time = datetime.now(ET) + timedelta(seconds=wait)
                log.info(f"Outside active hours. Sleeping until "
                         f"{resume_time.strftime('%Y-%m-%d %H:%M ET')} "
                         f"({wait / 3600:.1f} hours)")
                slept = 0
                while slept < wait and _running:
                    chunk = min(30, wait - slept)
                    time.sleep(chunk)
                    slept += chunk
                if not _running:
                    break
                continue  # Re-check active hours

            # Run a scan cycle
            bot.run_daily_cycle()

            if not _running:
                break

            # Sleep until next scan
            next_scan = datetime.now(ET) + timedelta(seconds=scan_interval_sec)
            log.info(f"Next scan: {next_scan.strftime('%H:%M ET')} "
                     f"(in {SCAN_INTERVAL_MINUTES} min)")

            slept = 0
            while slept < scan_interval_sec and _running:
                chunk = min(30, scan_interval_sec - slept)
                time.sleep(chunk)
                slept += chunk

        except Exception as e:
            log.error(f"Scan cycle failed: {e}", exc_info=True)
            # Short backoff before retrying
            for _ in range(6):
                if not _running:
                    break
                time.sleep(10)

    # Final save
    save_state(bot.state)
    _print_session_summary(bot)
    log.info("Shutdown complete.")


def _print_session_summary(bot: SportsTradingBot):
    """Print session summary on exit."""
    state = bot.state
    total_trades = state.get("total_trades", 0)
    total_wins = state.get("total_wins", 0)
    total_losses = state.get("total_losses", 0)
    total_pnl = state.get("total_pnl", 0.0)

    log.info("")
    log.info("=" * 60)
    log.info("  SESSION SUMMARY")
    log.info("=" * 60)
    log.info(f"  Capital: ${bot.capital:.2f} "
             f"(started: ${state.get('initial_capital', 50):.2f})")
    log.info(f"  Total P&L: ${total_pnl:+.2f}")
    log.info(f"  Total trades: {total_trades}")
    if total_wins + total_losses > 0:
        wr = total_wins / (total_wins + total_losses) * 100
        log.info(f"  Win rate: {total_wins}/{total_wins + total_losses} "
                 f"({wr:.0f}%)")
    log.info(f"  Open trades: {len(state.get('open_trades', []))}")
    log.info(f"  State saved to: {STATE_FILE}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
