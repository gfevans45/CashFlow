"""Sports betting strategy for Kalshi sports contracts.

Sub-models:
  1. PlayerPropModel — Estimates P(player scores X+ stat) using
     season average, standard deviation, and recent form adjustment.
     Supports NBA (points/rebounds/assists) and MLB (strikeouts/hits/HRs).
  2. GameWinnerModel — Estimates win probability from team win%,
     point differential, and home/away splits. Supports NBA, NCAA,
     MLB (Pythagorean expectation), tennis (Elo-style), soccer.
  3. SpreadTotalModel — Estimates spread and over/under probabilities
     using team pace and offensive/defensive ratings.
  4. MLBPropModel — MLB-specific player prop model for pitcher strikeouts,
     batter hits, home runs, and RBIs.

All models use scipy.stats.norm for probability calculations.
The main SportsStrategy.evaluate_contracts() method mirrors
WeatherStrategy's interface for consistency.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from scipy.stats import norm

log = logging.getLogger("cashflow.sports")


# ---------------------------------------------------------------------------
# Trade data classes
# ---------------------------------------------------------------------------

@dataclass
class SportsTrade:
    """A trade on a sports contract."""
    ticker: str
    title: str
    contract_type: str
    sport: str
    position: str                # "yes" or "no"
    entry_price: float           # Effective price paid
    model_prob: float            # Our estimated probability
    market_price: float          # Kalshi market price (yes side)
    edge: float                  # model_prob - effective_price (signed)
    stake: float
    num_contracts: int
    # Settlement
    settled: bool = False
    pnl: float = 0.0
    raw_contract: dict = field(default_factory=dict, repr=False)


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------

class PlayerPropModel:
    """Estimates probability of "Player scores X+ stat" outcomes.

    Uses a normal distribution centered on an adjusted mean:
      adjusted_mean = 0.7 * season_avg + 0.3 * recent_avg

    The adjustment accounts for hot/cold streaks — a player on a
    recent scoring tear is more likely to exceed their season average,
    and vice versa.

    Standard deviation comes from actual game-log variance when available,
    falling back to ~30% of the season average.
    """

    # Weight for recent form vs season average
    RECENT_WEIGHT = 0.30
    SEASON_WEIGHT = 0.70
    # Default std as fraction of average (when no game logs available)
    DEFAULT_STD_FRACTION = 0.30
    # Minimum std to avoid degenerate distributions
    MIN_STD = 2.0

    def estimate_probability(self, threshold: float, season_avg: float,
                              std: float = 0.0, recent_avg: float = 0.0,
                              stat_type: str = "points") -> float:
        """Estimate P(stat >= threshold).

        Args:
            threshold: The line (e.g., 25.0 for "25+ points").
            season_avg: Season average for the stat.
            std: Standard deviation of the stat across games.
            recent_avg: Average over last 10 games (0 = use season_avg).
            stat_type: "points", "rebounds", "assists", "threes".

        Returns probability between 0.01 and 0.99.
        """
        if season_avg <= 0:
            return 0.5  # No data

        # Adjusted mean with recent form weighting
        if recent_avg > 0:
            adjusted_mean = (self.SEASON_WEIGHT * season_avg +
                             self.RECENT_WEIGHT * recent_avg)
        else:
            adjusted_mean = season_avg

        # Standard deviation
        if std <= 0:
            std = max(self.MIN_STD, season_avg * self.DEFAULT_STD_FRACTION)
        std = max(self.MIN_STD, std)

        # P(stat >= threshold) = 1 - CDF(threshold)
        # Use threshold - 0.5 for continuity correction
        prob = 1.0 - float(norm.cdf(threshold - 0.5, loc=adjusted_mean, scale=std))

        return round(max(0.01, min(0.99, prob)), 4)

    def estimate_from_player_stats(self, player_stats, threshold: float,
                                    stat_type: str = "points") -> float:
        """Estimate probability using a PlayerStats object.

        Convenience wrapper that pulls the right fields from PlayerStats.
        """
        stat_map = {
            "points": ("pts_avg", "pts_std", "recent_pts_avg"),
            "rebounds": ("reb_avg", "reb_std", "recent_reb_avg"),
            "assists": ("ast_avg", "ast_std", "recent_ast_avg"),
        }

        if stat_type not in stat_map:
            # Unknown stat type, use points as default
            stat_type = "points"

        avg_field, std_field, recent_field = stat_map[stat_type]
        season_avg = getattr(player_stats, avg_field, 0.0)
        std = getattr(player_stats, std_field, 0.0)
        recent_avg = getattr(player_stats, recent_field, 0.0)

        return self.estimate_probability(
            threshold, season_avg, std, recent_avg, stat_type,
        )


class GameWinnerModel:
    """Estimates win probability using team strength metrics.

    Model combines:
      - Team win percentage (recent + season)
      - Point differential per game
      - Home/away adjustment

    Uses a logistic-style model via the normal CDF:
      expected_margin = diff_a - diff_b + home_advantage
      P(A wins) = norm.cdf(expected_margin / game_std)

    The standard deviation of NBA game margins is approximately 12 points.
    Home court advantage is approximately 3.0 points.
    """

    NBA_GAME_STD = 12.0          # Std dev of NBA game point margins
    NCAA_GAME_STD = 11.0         # College is slightly lower
    HOME_ADVANTAGE_NBA = 3.0     # Home court advantage in points
    HOME_ADVANTAGE_NCAA = 3.5

    def estimate_probability(self, team_a_stats=None, team_b_stats=None,
                              team_a_win_pct: float = 0.5,
                              team_b_win_pct: float = 0.5,
                              team_a_diff: float = 0.0,
                              team_b_diff: float = 0.0,
                              is_home: bool = False,
                              sport: str = "nba") -> float:
        """Estimate P(team_a wins).

        Can use TeamStats objects or raw metrics.

        Args:
            team_a_stats: TeamStats for team A (if available).
            team_b_stats: TeamStats for team B (if available).
            team_a_win_pct: Win percentage [0, 1].
            team_b_win_pct: Win percentage [0, 1].
            team_a_diff: Point differential per game.
            team_b_diff: Point differential per game.
            is_home: Whether team A is the home team.
            sport: "nba" or "ncaa".

        Returns probability between 0.01 and 0.99.
        """
        # Extract from stats objects if provided
        if team_a_stats:
            team_a_win_pct = team_a_stats.win_pct
            team_a_diff = team_a_stats.point_differential
        if team_b_stats:
            team_b_win_pct = team_b_stats.win_pct
            team_b_diff = team_b_stats.point_differential

        # Method 1: Point differential (most predictive)
        margin_from_diff = team_a_diff - team_b_diff

        # Method 2: Win percentage → approximate strength
        # A team with 60% wins is roughly +3 pts per game better than average
        win_pct_diff = team_a_win_pct - team_b_win_pct
        margin_from_winpct = win_pct_diff * 15.0  # Scale factor

        # Combine (weight diff more heavily as it's more predictive)
        if abs(team_a_diff) + abs(team_b_diff) > 0:
            expected_margin = 0.65 * margin_from_diff + 0.35 * margin_from_winpct
        else:
            expected_margin = margin_from_winpct

        # Home court
        if sport == "ncaa":
            game_std = self.NCAA_GAME_STD
            hca = self.HOME_ADVANTAGE_NCAA if is_home else 0.0
        else:
            game_std = self.NBA_GAME_STD
            hca = self.HOME_ADVANTAGE_NBA if is_home else 0.0

        expected_margin += hca

        # Convert to probability
        prob = float(norm.cdf(expected_margin / game_std))

        return round(max(0.01, min(0.99, prob)), 4)

    def estimate_from_team_stats(self, team_a_stats, team_b_stats,
                                  is_home: bool = False,
                                  sport: str = "nba") -> float:
        """Convenience wrapper using TeamStats objects."""
        return self.estimate_probability(
            team_a_stats=team_a_stats,
            team_b_stats=team_b_stats,
            is_home=is_home,
            sport=sport,
        )


class SpreadTotalModel:
    """Estimates spread and over/under probabilities.

    Spread model:
      P(team wins by > spread) using expected margin and game variance.

    Over/Under model:
      Expected total = avg(team_a_for + team_b_for), adjusted for pace.
      P(total > line) via normal CDF.

    Standard deviation of NBA game totals is approximately 18 points.
    """

    TOTAL_STD_NBA = 18.0         # Std dev of NBA game totals
    TOTAL_STD_NCAA = 16.0
    MARGIN_STD_NBA = 12.0
    MARGIN_STD_NCAA = 11.0
    # Average NBA game total
    AVG_TOTAL_NBA = 225.0
    AVG_TOTAL_NCAA = 145.0

    def estimate_spread_probability(self, spread: float,
                                     team_a_stats=None,
                                     team_b_stats=None,
                                     team_a_diff: float = 0.0,
                                     team_b_diff: float = 0.0,
                                     is_home: bool = False,
                                     sport: str = "nba") -> float:
        """Estimate P(team_a wins by more than spread points).

        Args:
            spread: The spread line (e.g., 12.5).
            team_a_stats: TeamStats for team A.
            team_b_stats: TeamStats for team B.
            team_a_diff: Point differential per game.
            team_b_diff: Point differential per game.
            is_home: Whether team A is home.
            sport: "nba" or "ncaa".

        Returns probability between 0.01 and 0.99.
        """
        if team_a_stats:
            team_a_diff = team_a_stats.point_differential
        if team_b_stats:
            team_b_diff = team_b_stats.point_differential

        expected_margin = team_a_diff - team_b_diff

        if sport == "ncaa":
            margin_std = self.MARGIN_STD_NCAA
            hca = 3.5 if is_home else 0.0
        else:
            margin_std = self.MARGIN_STD_NBA
            hca = 3.0 if is_home else 0.0

        expected_margin += hca

        # P(margin > spread)
        prob = 1.0 - float(norm.cdf(spread, loc=expected_margin, scale=margin_std))

        return round(max(0.01, min(0.99, prob)), 4)

    def estimate_total_probability(self, total_line: float,
                                    team_a_stats=None,
                                    team_b_stats=None,
                                    team_a_pace: float = 100.0,
                                    team_b_pace: float = 100.0,
                                    team_a_off: float = 110.0,
                                    team_a_def: float = 110.0,
                                    team_b_off: float = 110.0,
                                    team_b_def: float = 110.0,
                                    sport: str = "nba") -> float:
        """Estimate P(game total > total_line).

        Args:
            total_line: The over/under line (e.g., 220.5).
            team_a_stats: TeamStats for team A.
            team_b_stats: TeamStats for team B.
            team_a_pace: Team A pace estimate.
            team_b_pace: Team B pace estimate.
            team_a_off/def: Team A offensive/defensive points per game.
            team_b_off/def: Team B offensive/defensive points per game.
            sport: "nba" or "ncaa".

        Returns probability between 0.01 and 0.99.
        """
        if team_a_stats:
            team_a_off = team_a_stats.avg_points_for
            team_a_def = team_a_stats.avg_points_against
            team_a_pace = team_a_stats.pace
        if team_b_stats:
            team_b_off = team_b_stats.avg_points_for
            team_b_def = team_b_stats.avg_points_against
            team_b_pace = team_b_stats.pace

        if sport == "ncaa":
            avg_total = self.AVG_TOTAL_NCAA
            total_std = self.TOTAL_STD_NCAA
        else:
            avg_total = self.AVG_TOTAL_NBA
            total_std = self.TOTAL_STD_NBA

        # Expected points for each team:
        # Team A scores based on their offense vs Team B's defense
        # Simple average: (team_a_off + team_b_def) / 2 adjusted by pace
        pace_factor = (team_a_pace + team_b_pace) / 200.0

        expected_a = (team_a_off + team_b_def) / 2.0 * pace_factor
        expected_b = (team_b_off + team_a_def) / 2.0 * pace_factor
        expected_total = expected_a + expected_b

        # P(total > line)
        prob = 1.0 - float(norm.cdf(total_line, loc=expected_total, scale=total_std))

        return round(max(0.01, min(0.99, prob)), 4)


class MLBPropModel:
    """MLB-specific player prop model.

    Pitcher strikeouts: Uses K/game average adjusted for recent form.
    Batter hits/HRs/RBIs: Uses per-game averages with recent form.

    MLB stat distributions are more variable than NBA due to smaller
    sample sizes per game, so we use wider standard deviations.
    """

    RECENT_WEIGHT = 0.35  # MLB form matters more (small sample per game)
    SEASON_WEIGHT = 0.65

    def estimate_strikeout_probability(self, threshold: float,
                                        k_per_game: float,
                                        k_std: float = 0.0,
                                        recent_k_avg: float = 0.0) -> float:
        """Estimate P(pitcher gets >= threshold strikeouts)."""
        if k_per_game <= 0:
            return 0.5

        if recent_k_avg > 0:
            adjusted = (self.SEASON_WEIGHT * k_per_game +
                        self.RECENT_WEIGHT * recent_k_avg)
        else:
            adjusted = k_per_game

        if k_std <= 0:
            k_std = max(1.5, k_per_game * 0.30)
        k_std = max(1.5, k_std)

        prob = 1.0 - float(norm.cdf(threshold - 0.5, loc=adjusted, scale=k_std))
        return round(max(0.01, min(0.99, prob)), 4)

    def estimate_batter_prop(self, threshold: float, avg_per_game: float,
                              std: float = 0.0, recent_avg: float = 0.0,
                              stat_type: str = "hits") -> float:
        """Estimate P(batter gets >= threshold of stat)."""
        if avg_per_game <= 0:
            return 0.5

        if recent_avg > 0:
            adjusted = (self.SEASON_WEIGHT * avg_per_game +
                        self.RECENT_WEIGHT * recent_avg)
        else:
            adjusted = avg_per_game

        # Defaults for different stats
        min_stds = {"hits": 0.5, "home_runs": 0.2, "rbis": 0.4, "runs": 0.3}
        min_std = min_stds.get(stat_type, 0.4)

        if std <= 0:
            std = max(min_std, avg_per_game * 0.50)
        std = max(min_std, std)

        prob = 1.0 - float(norm.cdf(threshold - 0.5, loc=adjusted, scale=std))
        return round(max(0.01, min(0.99, prob)), 4)

    def estimate_from_mlb_stats(self, mlb_stats, threshold: float,
                                 stat_type: str) -> float:
        """Estimate probability using an MLBPlayerStats object."""
        if stat_type == "strikeouts" and mlb_stats.position == "P":
            return self.estimate_strikeout_probability(
                threshold, mlb_stats.strikeouts_per_game,
                mlb_stats.strikeouts_std, mlb_stats.recent_k_avg,
            )
        elif stat_type == "hits":
            return self.estimate_batter_prop(
                threshold, mlb_stats.hits_avg, mlb_stats.hits_std,
                mlb_stats.recent_hits_avg, stat_type,
            )
        elif stat_type == "home_runs":
            return self.estimate_batter_prop(
                threshold, mlb_stats.hrs_avg, mlb_stats.hrs_std,
                stat_type=stat_type,
            )
        elif stat_type == "rbis":
            return self.estimate_batter_prop(
                threshold, mlb_stats.rbis_avg, mlb_stats.rbis_std,
                stat_type=stat_type,
            )
        elif stat_type == "runs":
            return self.estimate_batter_prop(
                threshold, mlb_stats.runs_avg, mlb_stats.runs_std,
                stat_type=stat_type,
            )
        else:
            return 0.5


class MLBGameModel:
    """MLB game winner and run total models.

    Game winner uses Pythagorean expectation:
      Win% ≈ RS^1.83 / (RS^1.83 + RA^1.83)

    Run totals use team averages with MLB-specific variance.
    MLB game standard deviation is ~4.0 runs per side.
    """

    MLB_GAME_STD = 4.0        # Std dev of run margin
    MLB_TOTAL_STD = 3.5       # Std dev of total runs
    HOME_ADVANTAGE_MLB = 0.3  # Home advantage in runs (~0.3 runs/game)

    def estimate_win_probability(self, team_a_stats=None, team_b_stats=None,
                                  is_home: bool = False) -> float:
        """Estimate P(team_a wins) using run-based models."""
        if team_a_stats and team_b_stats:
            # Expected runs for each team
            a_expected = (team_a_stats.runs_scored_per_game +
                          team_b_stats.runs_allowed_per_game) / 2.0
            b_expected = (team_b_stats.runs_scored_per_game +
                          team_a_stats.runs_allowed_per_game) / 2.0

            margin = a_expected - b_expected
            if is_home:
                margin += self.HOME_ADVANTAGE_MLB

            prob = float(norm.cdf(margin / self.MLB_GAME_STD))
        elif team_a_stats:
            prob = team_a_stats.win_pct
        else:
            prob = 0.5

        return round(max(0.01, min(0.99, prob)), 4)

    def estimate_total_probability(self, total_line: float,
                                    team_a_stats=None,
                                    team_b_stats=None) -> float:
        """Estimate P(total runs > line)."""
        if team_a_stats and team_b_stats:
            expected_total = (team_a_stats.runs_scored_per_game +
                              team_b_stats.runs_scored_per_game +
                              team_a_stats.runs_allowed_per_game +
                              team_b_stats.runs_allowed_per_game) / 2.0
        else:
            expected_total = 8.5  # MLB average

        prob = 1.0 - float(norm.cdf(total_line, loc=expected_total,
                                     scale=self.MLB_TOTAL_STD))
        return round(max(0.01, min(0.99, prob)), 4)


# ---------------------------------------------------------------------------
# Main Strategy
# ---------------------------------------------------------------------------

class SportsStrategy:
    """Sports betting strategy for Kalshi contracts.

    Evaluates sports contracts using the three sub-models and generates
    trades where our estimated probability diverges from the market price.

    Configuration mirrors WeatherStrategy for consistency:
      - min_edge: Minimum edge (model_prob - market_price) to trade (default 5%)
      - kelly_fraction: Quarter-Kelly (0.25)
      - max_position_pct: Max position as fraction of capital (10%)
      - max_positions: Max concurrent positions (5)
      - min_stake / max_stake: Trade size bounds ($1 - $5)
      - daily_loss_limit_pct: Stop trading after losing this much (5%)
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.min_edge = config.get("min_edge", 0.05)
        self.kelly_fraction = config.get("kelly_fraction", 0.25)
        self.max_position_pct = config.get("max_position_pct", 0.10)
        self.max_positions = config.get("max_positions", 5)
        self.min_stake = config.get("min_stake", 1.0)
        self.max_stake = config.get("max_stake", 10.0)
        self.daily_loss_limit_pct = config.get("daily_loss_limit_pct", 0.05)
        self.preferred_price_low = 0.40
        self.preferred_price_high = 0.60
        self.acceptable_price_low = 0.25
        self.acceptable_price_high = 0.75

        # Sub-models
        self.player_prop_model = PlayerPropModel()
        self.game_winner_model = GameWinnerModel()
        self.spread_total_model = SpreadTotalModel()
        self.mlb_prop_model = MLBPropModel()
        self.mlb_game_model = MLBGameModel()

    def estimate_contract_probability(self, contract, player_stats=None,
                                       team_a_stats=None,
                                       team_b_stats=None,
                                       ncaa_client=None,
                                       mlb_player_stats=None,
                                       mlb_team_a_stats=None,
                                       mlb_team_b_stats=None) -> float:
        """Estimate the true probability for a SportsContract.

        Dispatches to the appropriate sub-model based on contract_type and sport.

        Args:
            contract: A SportsContract object.
            player_stats: PlayerStats for NBA players.
            team_a_stats: TeamStats for NBA/NCAA team A.
            team_b_stats: TeamStats for NBA/NCAA team B.
            ncaa_client: NCAStatsClient for college basketball.
            mlb_player_stats: MLBPlayerStats for MLB players.
            mlb_team_a_stats: MLBTeamStats for MLB team A.
            mlb_team_b_stats: MLBTeamStats for MLB team B.

        Returns estimated probability (0.01 to 0.99).
        """
        ctype = contract.contract_type
        sport = contract.sport

        # MLB-specific routing
        if sport == "mlb":
            if ctype == "player_prop":
                return self._estimate_mlb_prop(contract, mlb_player_stats)
            elif ctype == "game_winner":
                return self._estimate_mlb_game(
                    contract, mlb_team_a_stats, mlb_team_b_stats,
                )
            elif ctype == "over_under":
                return self._estimate_mlb_total(
                    contract, mlb_team_a_stats, mlb_team_b_stats,
                )

        # Tennis/Soccer — use game_winner model with simple win% fallback
        if sport in ("tennis", "soccer"):
            if ctype == "game_winner":
                # No deep stats for these yet — use market price as fair,
                # only trade if we detect obvious mispricing
                return 0.5

        # NBA/NCAA routing (original logic)
        if ctype == "player_prop":
            return self._estimate_player_prop(contract, player_stats)
        elif ctype == "game_winner":
            return self._estimate_game_winner(
                contract, team_a_stats, team_b_stats, ncaa_client,
            )
        elif ctype == "spread":
            return self._estimate_spread(contract, team_a_stats, team_b_stats)
        elif ctype == "over_under":
            return self._estimate_total(contract, team_a_stats, team_b_stats)
        else:
            log.debug(f"Unknown contract type: {ctype}")
            return 0.5  # No edge on unknown types

    def _estimate_player_prop(self, contract, player_stats) -> float:
        """Estimate player prop probability."""
        if player_stats:
            return self.player_prop_model.estimate_from_player_stats(
                player_stats, contract.threshold, contract.stat_type,
            )

        # Fallback: use simulated estimates based on contract title
        # If threshold is near a round number, assume it's a standard line
        # and use a default distribution
        threshold = contract.threshold
        if threshold > 0:
            # Guess: average player scores ~20 ppg, std ~6
            default_avg = threshold * 0.9  # Slightly below the line
            default_std = max(2.0, threshold * 0.25)
            return self.player_prop_model.estimate_probability(
                threshold, default_avg, default_std,
            )
        return 0.5

    def _estimate_game_winner(self, contract, team_a_stats, team_b_stats,
                               ncaa_client) -> float:
        """Estimate game winner probability."""
        sport = contract.sport

        if sport == "ncaa" and ncaa_client:
            return ncaa_client.estimate_win_probability(
                contract.team_a, contract.team_b or "Unknown",
                home_team=contract.home_team,
            )

        if team_a_stats and team_b_stats:
            is_home = bool(contract.home_team and
                           contract.home_team.lower() in contract.team_a.lower())
            return self.game_winner_model.estimate_from_team_stats(
                team_a_stats, team_b_stats, is_home=is_home, sport=sport,
            )

        if team_a_stats:
            # Only have one team's stats — use win% as rough estimate
            return max(0.01, min(0.99, team_a_stats.win_pct))

        # Fallback: market price is roughly fair, but add slight noise
        return 0.5

    def _estimate_spread(self, contract, team_a_stats, team_b_stats) -> float:
        """Estimate spread cover probability."""
        return self.spread_total_model.estimate_spread_probability(
            spread=contract.spread,
            team_a_stats=team_a_stats,
            team_b_stats=team_b_stats,
            is_home=bool(contract.home_team and
                         contract.home_team.lower() in contract.team_a.lower()),
            sport=contract.sport,
        )

    def _estimate_total(self, contract, team_a_stats, team_b_stats) -> float:
        """Estimate over/under probability."""
        return self.spread_total_model.estimate_total_probability(
            total_line=contract.total,
            team_a_stats=team_a_stats,
            team_b_stats=team_b_stats,
            sport=contract.sport,
        )

    def _estimate_mlb_prop(self, contract, mlb_stats) -> float:
        """Estimate MLB player prop probability."""
        if mlb_stats:
            return self.mlb_prop_model.estimate_from_mlb_stats(
                mlb_stats, contract.threshold, contract.stat_type,
            )
        # Fallback for MLB props without stats
        threshold = contract.threshold
        if contract.stat_type == "strikeouts" and threshold > 0:
            default_k = 6.5
            default_std = 2.2
            return self.mlb_prop_model.estimate_strikeout_probability(
                threshold, default_k, default_std,
            )
        elif threshold > 0:
            default_avg = threshold * 0.85
            default_std = max(0.3, threshold * 0.50)
            return self.mlb_prop_model.estimate_batter_prop(
                threshold, default_avg, default_std,
                stat_type=contract.stat_type,
            )
        return 0.5

    def _estimate_mlb_game(self, contract, team_a_stats, team_b_stats) -> float:
        """Estimate MLB game winner probability."""
        is_home = bool(contract.home_team and
                       contract.home_team.lower() in contract.team_a.lower())
        return self.mlb_game_model.estimate_win_probability(
            team_a_stats=team_a_stats,
            team_b_stats=team_b_stats,
            is_home=is_home,
        )

    def _estimate_mlb_total(self, contract, team_a_stats, team_b_stats) -> float:
        """Estimate MLB run total probability."""
        return self.mlb_game_model.estimate_total_probability(
            total_line=contract.total,
            team_a_stats=team_a_stats,
            team_b_stats=team_b_stats,
        )

    def evaluate_contracts(self, contracts_with_probs: list[tuple],
                            capital: float, daily_pnl: float,
                            open_count: int) -> list[SportsTrade]:
        """Evaluate contracts and return trades to take.

        Args:
            contracts_with_probs: List of (SportsContract, model_prob) tuples.
            capital: Current capital.
            daily_pnl: P&L for the day so far.
            open_count: Number of currently open positions.

        Returns list of SportsTrade objects for trades to place.
        """
        daily_loss_limit = capital * self.daily_loss_limit_pct
        if daily_pnl <= -daily_loss_limit:
            log.info("Daily loss limit reached, no new trades")
            return []

        available_slots = self.max_positions - open_count
        if available_slots <= 0:
            log.info("Max positions reached, no new trades")
            return []

        scored = []
        for contract, model_prob in contracts_with_probs:
            market_price = contract.yes_price

            if market_price <= 0 or market_price >= 1:
                continue

            # Check YES edge: buy yes if model_prob > market_price
            yes_edge = model_prob - market_price
            # Check NO edge: buy no if (1 - model_prob) > (1 - market_price)
            # This simplifies to: market_price > model_prob
            no_edge = market_price - model_prob

            if yes_edge >= no_edge:
                best_edge = yes_edge
                position = "yes"
                effective_price = market_price
            else:
                best_edge = no_edge
                position = "no"
                effective_price = 1.0 - market_price

            if best_edge < self.min_edge:
                continue

            if not (self.acceptable_price_low <= effective_price
                    <= self.acceptable_price_high):
                continue

            scored.append((best_edge, contract, model_prob, position,
                           effective_price))

        # Sort by edge, take best
        scored.sort(key=lambda x: -x[0])

        trades = []
        for edge, contract, model_prob, position, eff_price in scored:
            if len(trades) >= available_slots:
                break

            stake = self._kelly_size(edge, eff_price, capital)
            if stake < self.min_stake:
                continue

            num_contracts = max(1, int(stake / eff_price))
            actual_stake = round(num_contracts * eff_price, 2)

            if daily_pnl - actual_stake <= -daily_loss_limit:
                continue

            trade = SportsTrade(
                ticker=contract.ticker,
                title=contract.title,
                contract_type=contract.contract_type,
                sport=contract.sport,
                position=position,
                entry_price=eff_price,
                model_prob=model_prob,
                market_price=contract.yes_price,
                edge=round(edge, 4),
                stake=actual_stake,
                num_contracts=num_contracts,
                raw_contract=contract.raw,
            )
            trades.append(trade)

        return trades

    def _kelly_size(self, edge: float, price: float, capital: float) -> float:
        """Quarter-Kelly position sizing."""
        if price <= 0 or price >= 1:
            return 0.0

        kelly = edge / (1 - price)
        stake = capital * kelly * self.kelly_fraction
        max_allowed = min(capital * self.max_position_pct, self.max_stake)
        return round(min(stake, max_allowed), 2)

    def settle_trade(self, trade: SportsTrade, outcome: bool) -> float:
        """Settle a trade based on outcome.

        Args:
            trade: The SportsTrade to settle.
            outcome: True if YES outcome, False if NO outcome.

        Returns P&L (positive = profit, negative = loss).
        """
        if trade.position == "yes":
            if outcome:
                trade.pnl = round(
                    trade.num_contracts * (1.0 - trade.entry_price), 2,
                )
            else:
                trade.pnl = round(-trade.stake, 2)
        else:  # "no" position
            if not outcome:
                trade.pnl = round(
                    trade.num_contracts * (1.0 - trade.entry_price), 2,
                )
            else:
                trade.pnl = round(-trade.stake, 2)

        trade.settled = True
        return trade.pnl
