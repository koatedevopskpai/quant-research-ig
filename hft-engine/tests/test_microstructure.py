"""
test_microstructure.py — Unit tests for microstructure analytics.
"""

from math import isclose

import pytest

from hft_engine.book.order_book import OrderBook, PriceLevel
from hft_engine.feeds.tick_consumer import MarketDataMessage
from hft_engine.microstructure.analyser import MicrostructureAnalyser, TradeSignResult


class TestMicrostructureAnalyser:
    """Test spread analytics, depth profiles, and trade classification."""

    # ------------------------------------------------------------------
    # Spread stats
    # ------------------------------------------------------------------

    def test_spread_stats_empty(self) -> None:
        analyser = MicrostructureAnalyser()
        stats = analyser.spread_stats()
        assert stats.count == 0
        assert stats.mean == 0.0

    def test_spread_stats_single_observation(self) -> None:
        analyser = MicrostructureAnalyser()
        msg = MarketDataMessage(
            epic="TEST", message_type="L1",
            bid_price=100.0, ask_price=100.10,
        )
        analyser.on_tick(msg)
        stats = analyser.spread_stats()
        assert stats.count == 1
        assert isclose(stats.mean, 0.10)
        assert isclose(stats.min, 0.10)
        assert isclose(stats.max, 0.10)

    def test_spread_stats_multiple(self) -> None:
        analyser = MicrostructureAnalyser()
        spreads = [(100.0, 100.10), (100.05, 100.12), (99.95, 100.05)]
        for bid, ask in spreads:
            msg = MarketDataMessage(
                epic="TEST", message_type="L1",
                bid_price=bid, ask_price=ask,
            )
            analyser.on_tick(msg)
        stats = analyser.spread_stats()
        assert stats.count == 3
        # Expected mean: (0.10 + 0.07 + 0.10) / 3 = 0.09
        assert isclose(stats.mean, 0.09, abs_tol=0.001)
        assert stats.min == 0.07

    def test_spread_stats_window(self) -> None:
        """Window should only keep the last N observations."""
        analyser = MicrostructureAnalyser(window_size=2)
        for i in range(5):
            msg = MarketDataMessage(
                epic="TEST", message_type="L1",
                bid_price=float(100 - i),
                ask_price=float(101 - i),
            )
            analyser.on_tick(msg)
        stats = analyser.spread_stats()
        assert stats.count == 2

    def test_spread_stats_ignores_non_l1(self) -> None:
        """TRADE and L2 messages should not affect spread stats."""
        analyser = MicrostructureAnalyser()
        msg = MarketDataMessage(epic="TEST", message_type="TRADE")
        analyser.on_tick(msg)
        assert analyser.spread_stats().count == 0

    # ------------------------------------------------------------------
    # Depth profile
    # ------------------------------------------------------------------

    def test_current_depth(self) -> None:
        book = OrderBook()
        book.apply_snapshot(
            bids=[PriceLevel(100.50, 5000), PriceLevel(100.40, 10000)],
            asks=[PriceLevel(100.60, 7000), PriceLevel(100.70, 8000)],
        )
        profile = MicrostructureAnalyser.current_depth(book)
        assert len(profile.bid_levels) == 2
        assert len(profile.ask_levels) == 2
        assert isclose(profile.total_bid_volume, 15000)
        assert isclose(profile.total_ask_volume, 15000)
        assert isclose(profile.imbalance, 0.0, abs_tol=0.001)

    def test_current_depth_imbalance(self) -> None:
        book = OrderBook()
        book.apply_snapshot(
            bids=[PriceLevel(100.50, 10000)],
            asks=[PriceLevel(100.60, 5000)],
        )
        profile = MicrostructureAnalyser.current_depth(book)
        assert profile.imbalance > 0  # more bid volume = buying pressure

    def test_current_depth_empty_book(self) -> None:
        book = OrderBook()
        profile = MicrostructureAnalyser.current_depth(book)
        assert len(profile.bid_levels) == 0
        assert profile.total_bid_volume == 0.0
        assert profile.weighted_mid == 0.0

    # ------------------------------------------------------------------
    # Trade sign classification
    # ------------------------------------------------------------------

    @pytest.fixture
    def book_with_l1(self) -> OrderBook:
        book = OrderBook()
        book.apply_snapshot(
            bids=[PriceLevel(100.40, 5000)],
            asks=[PriceLevel(100.60, 5000)],
        )
        return book

    def test_classify_aggressive_buy(self, book_with_l1: OrderBook) -> None:
        result = MicrostructureAnalyser.classify_trade(100.60, 1000, book_with_l1)
        assert result.side == "buy"
        assert result.confidence == 1.0

    def test_classify_aggressive_sell(self, book_with_l1: OrderBook) -> None:
        result = MicrostructureAnalyser.classify_trade(100.40, 1000, book_with_l1)
        assert result.side == "sell"
        assert result.confidence == 1.0

    def test_classify_above_ask_is_buy(self, book_with_l1: OrderBook) -> None:
        """A trade above the ask is definitely an aggressive buy."""
        result = MicrostructureAnalyser.classify_trade(100.65, 1000, book_with_l1)
        assert result.side == "buy"

    def test_classify_inside_spread_above_mid(self, book_with_l1: OrderBook) -> None:
        """Trade inside spread but above mid-price -> buy with low confidence."""
        result = MicrostructureAnalyser.classify_trade(100.52, 1000, book_with_l1)
        assert result.side == "buy"
        assert result.confidence == 0.5

    def test_classify_inside_spread_below_mid(self, book_with_l1: OrderBook) -> None:
        result = MicrostructureAnalyser.classify_trade(100.48, 1000, book_with_l1)
        assert result.side == "sell"
        assert result.confidence == 0.5

    def test_classify_empty_book(self) -> None:
        book = OrderBook()
        result = MicrostructureAnalyser.classify_trade(100.50, 1000, book)
        assert result.side == "neutral"
        assert result.confidence == 0.0


class TestMicrostructurePurePython:
    """
    Tests running exclusively through the pure-Python MicrostructureAnalyser path.
    These verify the Python fallback works when C++ is disabled.
    """

    @pytest.fixture
    def analyser(self) -> MicrostructureAnalyser:
        return MicrostructureAnalyser(window_size=50, use_cpp=False)

    def _make_l1(self, bid: float = 100.0, ask: float = 101.0,
                 bid_size: float = 5000, ask_size: float = 3000) -> MarketDataMessage:
        return MarketDataMessage(
            epic="TEST", message_type="L1",
            bid_price=bid, ask_price=ask,
            bid_size=bid_size, ask_size=ask_size,
        )

    def test_spread_stats_empty(self, analyser: MicrostructureAnalyser) -> None:
        stats = analyser.spread_stats()
        assert stats.count == 0

    def test_spread_stats_single(self, analyser: MicrostructureAnalyser) -> None:
        analyser.on_tick(self._make_l1(bid=100.0, ask=101.0))
        stats = analyser.spread_stats()
        assert stats.count == 1
        assert stats.mean == 1.0

    def test_spread_stats_multiple(self, analyser: MicrostructureAnalyser) -> None:
        for i in range(5):
            analyser.on_tick(self._make_l1(bid=100.0, ask=101.0 + i * 0.1))
        stats = analyser.spread_stats()
        assert stats.count == 5
        assert stats.mean > 1.0

    def test_spread_stats_window(self, analyser: MicrostructureAnalyser) -> None:
        for i in range(100):
            analyser.on_tick(self._make_l1(bid=100.0, ask=101.0))
        stats = analyser.spread_stats()
        assert stats.count == 50

    def test_ignores_non_l1(self, analyser: MicrostructureAnalyser) -> None:
        msg = MarketDataMessage(
            epic="TEST", message_type="TRADE",
            last_price=100.5, last_size=1000,
        )
        analyser.on_tick(msg)
        assert analyser.spread_stats().count == 0

    def test_current_depth(self) -> None:
        book = OrderBook(use_cpp=False)
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        profile = MicrostructureAnalyser.current_depth(book)
        assert profile.total_bid_volume == 5000
        assert profile.total_ask_volume == 3000
        assert profile.imbalance > 0

    def test_current_depth_empty_book(self) -> None:
        book = OrderBook(use_cpp=False)
        profile = MicrostructureAnalyser.current_depth(book)
        assert profile.total_bid_volume == 0
        assert profile.total_ask_volume == 0

    def test_classify_buy(self) -> None:
        book = OrderBook(use_cpp=False)
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        result = MicrostructureAnalyser.classify_trade(101.00, 1000, book)
        assert result.side == "buy"
        assert result.confidence == 1.0

    def test_classify_sell(self) -> None:
        book = OrderBook(use_cpp=False)
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        result = MicrostructureAnalyser.classify_trade(100.50, 1000, book)
        assert result.side == "sell"
        assert result.confidence == 1.0

    def test_classify_inside_spread_above_mid(self) -> None:
        book = OrderBook(use_cpp=False)
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        result = MicrostructureAnalyser.classify_trade(100.80, 1000, book)
        assert result.side == "buy"
        assert result.confidence == 0.5

    def test_classify_inside_spread_below_mid(self) -> None:
        book = OrderBook(use_cpp=False)
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        result = MicrostructureAnalyser.classify_trade(100.70, 1000, book)
        assert result.side == "sell"
        assert result.confidence == 0.5

    def test_classify_empty_book(self) -> None:
        book = OrderBook(use_cpp=False)
        result = MicrostructureAnalyser.classify_trade(100.0, 1000, book)
        assert result.side == "neutral"
        assert result.confidence == 0.0

    def test_classify_above_ask_is_buy(self) -> None:
        book = OrderBook(use_cpp=False)
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        result = MicrostructureAnalyser.classify_trade(101.50, 1000, book)
        assert result.side == "buy"
        assert result.confidence == 0.8
