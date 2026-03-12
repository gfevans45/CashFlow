"""Cross-platform probability feeds for arbitrage detection.

Aggregates probability estimates from multiple prediction markets and
forecasting platforms to compare against Polymarket prices.

Free API sources:
  - Manifold Markets: Free, 500 req/min, no auth for reads
  - Metaculus: Free, best-calibrated forecasters
  - PredictIt: Free, no auth, US politics
  - Kalshi: Free market data (auth needed for trading)

The ensemble of these sources gives us a robust "fair probability"
estimate. When Polymarket diverges from this ensemble, we trade.
"""
import requests
import time
import json
import re
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime


HEADERS = {"User-Agent": "CashFlow/0.1.0"}


@dataclass
class ExternalProbability:
    """A probability estimate from an external source."""
    source: str           # "manifold", "metaculus", "predictit", "kalshi"
    question: str         # The question text
    probability: float    # 0.0 to 1.0
    volume: float         # Trading volume or # of forecasters
    url: str = ""
    last_updated: str = ""
    confidence: float = 1.0  # Weight for this source (0-1)


@dataclass
class ArbitrageSignal:
    """Signal when we detect cross-platform mispricing."""
    polymarket_question: str
    polymarket_price: float       # YES token price
    ensemble_probability: float   # Weighted average from all sources
    edge: float                   # ensemble_prob - polymarket_price
    sources: list                 # List of ExternalProbability
    confidence_score: float       # How confident we are in the signal (0-1)
    timestamp: str = ""


class ManifoldClient:
    """Client for Manifold Markets API (play money, but well-calibrated)."""

    BASE_URL = "https://api.manifold.markets/v0"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def search_markets(self, query: str, limit: int = 10) -> list:
        """Search Manifold markets by keyword."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/search-markets",
                params={"term": query, "limit": limit, "sort": "liquidity"},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            markets = resp.json()
            return [self._parse_market(m) for m in markets if m.get("probability") is not None]
        except requests.RequestException:
            return []

    def get_market(self, market_id: str) -> Optional[ExternalProbability]:
        """Get a specific market by ID."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/market/{market_id}",
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return None
            return self._parse_market(resp.json())
        except requests.RequestException:
            return None

    def get_markets(self, limit: int = 100) -> list:
        """Get recent active markets."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/markets",
                params={"limit": limit, "sort": "newest"},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            return [self._parse_market(m) for m in resp.json()
                    if m.get("probability") is not None and not m.get("isResolved")]
        except requests.RequestException:
            return []

    def _parse_market(self, data: dict) -> ExternalProbability:
        return ExternalProbability(
            source="manifold",
            question=data.get("question", ""),
            probability=float(data.get("probability", 0.5)),
            volume=float(data.get("volume", 0) or 0),
            url=data.get("url", ""),
            last_updated=data.get("lastUpdatedTime", ""),
            confidence=0.7,  # Play money = slightly less weight
        )


class MetaculusClient:
    """Client for Metaculus API (expert forecasters, best calibration)."""

    BASE_URL = "https://www.metaculus.com/api"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def search_questions(self, query: str, limit: int = 10) -> list:
        """Search Metaculus questions."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/questions/",
                params={
                    "search": query,
                    "limit": limit,
                    "status": "open",
                    "type": "forecast",
                    "order_by": "-activity",
                },
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = data.get("results", data) if isinstance(data, dict) else data
            if not isinstance(results, list):
                return []
            return [self._parse_question(q) for q in results
                    if self._get_probability(q) is not None]
        except requests.RequestException:
            return []

    def _get_probability(self, data: dict) -> Optional[float]:
        """Extract community probability from various Metaculus response formats."""
        # Try community_prediction first
        cp = data.get("community_prediction")
        if cp:
            if isinstance(cp, dict):
                return cp.get("full", {}).get("q2")  # Median
            elif isinstance(cp, (int, float)):
                return float(cp)

        # Try question.aggregations
        q = data.get("question", {})
        if isinstance(q, dict):
            agg = q.get("aggregations", {})
            recency = agg.get("recency_weighted", {})
            if recency:
                centers = recency.get("centers", [])
                if centers:
                    return centers[0]

        # Try my_forecasts or other fields
        forecast = data.get("my_forecasts", {})
        if isinstance(forecast, dict) and forecast.get("latest"):
            return forecast["latest"].get("forecast_values", [None])[0]

        return None

    def _parse_question(self, data: dict) -> ExternalProbability:
        prob = self._get_probability(data) or 0.5
        forecasters = data.get("number_of_forecasters", 0) or 0
        return ExternalProbability(
            source="metaculus",
            question=data.get("title", data.get("question_text", "")),
            probability=prob,
            volume=float(forecasters),
            url=f"https://www.metaculus.com/questions/{data.get('id', '')}",
            last_updated=data.get("last_activity_time", ""),
            confidence=0.9,  # Metaculus forecasters are well-calibrated
        )


class PredictItClient:
    """Client for PredictIt API (US politics, real money)."""

    BASE_URL = "https://www.predictit.org/api/marketdata"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def get_all_markets(self) -> list:
        """Fetch all PredictIt markets (single endpoint, no pagination)."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/all/",
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            markets = data.get("markets", [])
            results = []
            for market in markets:
                for contract in market.get("contracts", []):
                    if contract.get("status") == "Open":
                        results.append(self._parse_contract(market, contract))
            return results
        except requests.RequestException:
            return []

    def search_markets(self, query: str) -> list:
        """Search PredictIt markets by keyword (client-side filter)."""
        all_markets = self.get_all_markets()
        query_lower = query.lower()
        return [m for m in all_markets if query_lower in m.question.lower()]

    def _parse_contract(self, market: dict, contract: dict) -> ExternalProbability:
        # PredictIt prices are 0.01-0.99, directly = probability
        yes_price = contract.get("lastTradePrice") or contract.get("bestBuyYesCost") or 0.5
        return ExternalProbability(
            source="predictit",
            question=f"{market.get('name', '')} - {contract.get('name', '')}",
            probability=float(yes_price),
            volume=float(contract.get("totalSharesTraded", 0) or 0),
            url=market.get("url", ""),
            last_updated=contract.get("dateEnd", ""),
            confidence=0.75,  # Real money but 850-contract limit distorts prices
        )


class KalshiClient:
    """Client for Kalshi API (CFTC-regulated, free market data)."""

    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, timeout: int = 10):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def get_events(self, limit: int = 50, status: str = "open") -> list:
        """Fetch active Kalshi events."""
        try:
            resp = self.session.get(
                f"{self.BASE_URL}/events",
                params={"limit": limit, "status": status},
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            return data.get("events", [])
        except requests.RequestException:
            return []

    def get_markets(self, event_ticker: str = None, limit: int = 100) -> list:
        """Fetch Kalshi markets, optionally filtered by event."""
        try:
            params = {"limit": limit, "status": "open"}
            if event_ticker:
                params["event_ticker"] = event_ticker
            resp = self.session.get(
                f"{self.BASE_URL}/markets",
                params=params,
                timeout=self.timeout,
            )
            if resp.status_code != 200:
                return []
            data = resp.json()
            results = []
            for m in data.get("markets", []):
                results.append(self._parse_market(m))
            return results
        except requests.RequestException:
            return []

    def search_markets(self, query: str) -> list:
        """Search Kalshi markets (client-side filter)."""
        markets = self.get_markets(limit=200)
        query_lower = query.lower()
        return [m for m in markets if query_lower in m.question.lower()]

    def _parse_market(self, data: dict) -> ExternalProbability:
        # Kalshi yes_price is in cents (0-100), convert to 0-1
        yes_price = float(data.get("yes_price", 50)) / 100.0
        volume = float(data.get("volume", 0) or 0)
        return ExternalProbability(
            source="kalshi",
            question=data.get("title", data.get("subtitle", "")),
            probability=yes_price,
            volume=volume,
            url=f"https://kalshi.com/markets/{data.get('ticker', '')}",
            last_updated=data.get("last_price_time", ""),
            confidence=0.85,  # Real money, CFTC regulated
        )


class ArbitrageEngine:
    """Cross-platform probability aggregator and arbitrage detector.

    Fetches probabilities from multiple sources, computes a weighted
    ensemble estimate, and compares against Polymarket prices.
    """

    # Source weights for ensemble averaging
    SOURCE_WEIGHTS = {
        "metaculus": 0.30,    # Best calibrated
        "kalshi": 0.25,       # Real money, regulated
        "predictit": 0.20,    # Real money, political focus
        "manifold": 0.15,     # Play money but large community
        "model": 0.10,        # Our internal model
    }

    def __init__(self, config: dict = None):
        self.config = config or {}
        self.manifold = ManifoldClient()
        self.metaculus = MetaculusClient()
        self.predictit = PredictItClient()
        self.kalshi = KalshiClient()
        self.min_sources = self.config.get("min_sources", 2)
        self.min_confidence = self.config.get("min_confidence", 0.5)

    def find_matching_probabilities(self, question: str) -> list:
        """Search all platforms for markets matching a question.

        Returns list of ExternalProbability from all sources.
        """
        # Extract key terms from the question
        search_terms = self._extract_search_terms(question)
        all_probs = []

        for search_term in search_terms[:2]:  # Max 2 search attempts
            # Search all platforms in sequence (parallel would be better in production)
            for name, client in [
                ("manifold", self.manifold),
                ("metaculus", self.metaculus),
                ("kalshi", self.kalshi),
            ]:
                try:
                    if hasattr(client, "search_markets"):
                        results = client.search_markets(search_term)
                    elif hasattr(client, "search_questions"):
                        results = client.search_questions(search_term)
                    else:
                        continue

                    # Filter for relevance
                    for prob in results[:3]:  # Top 3 per source
                        similarity = self._question_similarity(question, prob.question)
                        if similarity > 0.3:
                            prob.confidence *= similarity
                            all_probs.append(prob)

                    time.sleep(0.2)  # Rate limit
                except Exception:
                    continue

        return all_probs

    def compute_ensemble_probability(
        self,
        external_probs: list,
        internal_model_prob: float = None,
    ) -> tuple:
        """Compute weighted ensemble probability from multiple sources.

        Returns (ensemble_prob, confidence_score, source_count).
        """
        if not external_probs and internal_model_prob is None:
            return 0.5, 0.0, 0

        weighted_sum = 0.0
        weight_total = 0.0

        for prob in external_probs:
            source_weight = self.SOURCE_WEIGHTS.get(prob.source, 0.1)
            adjusted_weight = source_weight * prob.confidence

            # Volume-based confidence boost (more volume = more reliable)
            if prob.volume > 10000:
                adjusted_weight *= 1.2
            elif prob.volume > 1000:
                adjusted_weight *= 1.1

            weighted_sum += prob.probability * adjusted_weight
            weight_total += adjusted_weight

        if internal_model_prob is not None:
            model_weight = self.SOURCE_WEIGHTS.get("model", 0.1)
            weighted_sum += internal_model_prob * model_weight
            weight_total += model_weight

        if weight_total == 0:
            return 0.5, 0.0, 0

        ensemble_prob = weighted_sum / weight_total
        # Confidence based on number of agreeing sources and total weight
        source_count = len(set(p.source for p in external_probs))
        confidence = min(1.0, weight_total * (source_count / 4))

        return ensemble_prob, confidence, source_count

    def detect_arbitrage(
        self,
        polymarket_question: str,
        polymarket_price: float,
        min_edge: float = 0.05,
        internal_model_prob: float = None,
    ) -> Optional[ArbitrageSignal]:
        """Detect arbitrage between Polymarket and other sources.

        Returns an ArbitrageSignal if significant mispricing is found.
        """
        # Fetch external probabilities
        external_probs = self.find_matching_probabilities(polymarket_question)

        # Compute ensemble
        ensemble_prob, confidence, source_count = self.compute_ensemble_probability(
            external_probs, internal_model_prob
        )

        # Check if we have enough sources for a signal
        if source_count < self.min_sources and internal_model_prob is None:
            return None

        if confidence < self.min_confidence:
            return None

        # Calculate edge
        edge = ensemble_prob - polymarket_price

        if edge < min_edge:
            return None

        return ArbitrageSignal(
            polymarket_question=polymarket_question,
            polymarket_price=polymarket_price,
            ensemble_probability=round(ensemble_prob, 4),
            edge=round(edge, 4),
            sources=external_probs,
            confidence_score=round(confidence, 4),
            timestamp=datetime.now().isoformat(),
        )

    def scan_polymarket_for_arbitrage(
        self,
        polymarket_markets: list,
        min_edge: float = 0.05,
    ) -> list:
        """Scan a list of Polymarket markets for arbitrage opportunities.

        Args:
            polymarket_markets: List of PolyMarket objects from polymarket_client
            min_edge: Minimum edge to flag

        Returns:
            List of ArbitrageSignal objects, sorted by edge (largest first)
        """
        signals = []

        for market in polymarket_markets:
            signal = self.detect_arbitrage(
                polymarket_question=market.question,
                polymarket_price=market.yes_price,
                min_edge=min_edge,
            )
            if signal:
                signals.append(signal)
            time.sleep(0.5)  # Be nice to external APIs

        signals.sort(key=lambda s: s.edge, reverse=True)
        return signals

    def _extract_search_terms(self, question: str) -> list:
        """Extract meaningful search terms from a Polymarket question."""
        # Remove common prediction market phrasing
        q = question.lower()
        for phrase in ["will ", "will the ", "will a ", "by ", "before ", "in 2025", "in 2026", "?"]:
            q = q.replace(phrase, " ")

        # Take the most meaningful 3-4 words
        words = [w for w in q.split() if len(w) > 2 and w not in {
            "the", "and", "for", "that", "this", "with", "from",
            "have", "has", "been", "was", "are", "were", "not",
        }]

        if len(words) <= 4:
            return [" ".join(words)]
        return [" ".join(words[:4]), " ".join(words[:3])]

    def _question_similarity(self, q1: str, q2: str) -> float:
        """Simple word-overlap similarity between two questions."""
        words1 = set(q1.lower().split())
        words2 = set(q2.lower().split())
        # Remove stop words
        stop = {"the", "a", "an", "is", "are", "was", "were", "will", "be", "to", "of", "in", "for", "on", "?", "by"}
        words1 -= stop
        words2 -= stop

        if not words1 or not words2:
            return 0.0

        overlap = words1 & words2
        return len(overlap) / max(len(words1), len(words2))


def simulate_arbitrage_feeds(
    n_events: int = 100,
    seed: int = 42,
) -> list:
    """Simulate cross-platform probability feeds for backtesting.

    Generates synthetic ExternalProbability objects that mimic
    what we'd get from Manifold, Metaculus, Kalshi, etc.
    """
    import numpy as np
    np.random.seed(seed)

    simulated = []
    sources = ["manifold", "metaculus", "kalshi", "predictit"]
    source_noise = {
        "metaculus": 0.03,    # Most accurate
        "kalshi": 0.05,       # Good but has some noise
        "predictit": 0.06,    # Slightly more noise (contract limits)
        "manifold": 0.08,     # Play money, more noise
    }

    for i in range(n_events):
        # True probability
        true_prob = np.random.beta(2, 8)  # Skewed toward low probs (longshots)
        true_prob = np.clip(true_prob, 0.01, 0.50)

        # Each source has its own estimate with source-specific noise
        event_probs = []
        available_sources = np.random.choice(
            sources, size=np.random.randint(2, 5), replace=False
        )

        for source in available_sources:
            noise = source_noise[source]
            estimate = true_prob + np.random.normal(0, noise)
            estimate = np.clip(estimate, 0.01, 0.95)

            event_probs.append(ExternalProbability(
                source=source,
                question=f"Simulated event #{i}",
                probability=round(estimate, 4),
                volume=float(np.random.lognormal(7, 1.5)),
                confidence=ArbitrageEngine.SOURCE_WEIGHTS.get(source, 0.1),
            ))

        simulated.append({
            "event_id": i,
            "true_prob": true_prob,
            "external_probs": event_probs,
        })

    return simulated
