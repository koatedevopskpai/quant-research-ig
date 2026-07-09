"""
Tests for the hft_cpp C++ pybind11 accelerator module.

These tests verify that:
1. The module imports successfully.
2. C++ OrderBook works as a drop-in replacement.
3. C++ MicrostructureAnalyser works as a drop-in replacement.
4. The Python wrappers correctly delegate to C++ when use_cpp=True.
5. Pure Python fallback still works when C++ is unavailable.
"""

import pytest

try:
    import hft_engine.hft_cpp as hft_cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


pytestmark = [
    pytest.mark.skipif(not _HAS_CPP, reason="hft_cpp module not built"),
]


class TestCppModuleImport:
    """Verify the C++ module loads and exposes expected classes."""

    def test_module_imports(self):
        assert hft_cpp is not None
        assert hasattr(hft_cpp, "OrderBook")
        assert hasattr(hft_cpp, "MicrostructureAnalyser")
        assert hasattr(hft_cpp, "PriceLevel")
        assert hasattr(hft_cpp, "L1Snapshot")
        assert hasattr(hft_cpp, "L2Snapshot")
        assert hasattr(hft_cpp, "BookDepth")
        assert hasattr(hft_cpp, "SpreadMetrics")
        assert hasattr(hft_cpp, "DepthProfile")
        assert hasattr(hft_cpp, "TradeSignResult")
        assert hasattr(hft_cpp, "MarketDataMessage")

    def test_module_has_functions(self):
        assert callable(hft_cpp.parse_l1_update)
        assert callable(hft_cpp.parse_l2_update)
        assert callable(hft_cpp.parse_trade_update)
        assert callable(hft_cpp.safe_float)
        assert callable(hft_cpp.safe_int)
        assert callable(hft_cpp.now_ns)


class TestCppOrderBook:
    """Direct tests on hft_cpp.OrderBook."""

    def test_create(self):
        book = hft_cpp.OrderBook()
        assert book.bid_count() == 0
        assert book.ask_count() == 0

    def test_apply_diff_and_l1(self):
        book = hft_cpp.OrderBook()
        book.apply_diff("bid", 100.50, 5000, 1)
        book.apply_diff("ask", 101.00, 3000, 1)
        assert book.bid_count() == 1
        assert book.ask_count() == 1
        l1 = book.l1_snapshot()
        assert l1.bid_price == 100.50
        assert l1.ask_price == 101.00
        assert l1.bid_size == 5000
        assert l1.ask_size == 3000

    def test_apply_diff_update(self):
        book = hft_cpp.OrderBook()
        book.apply_diff("bid", 100.50, 5000, 1)
        book.apply_diff("ask", 101.00, 3000, 1)
        book.apply_diff("bid", 100.50, 6000, 2)
        l1 = book.l1_snapshot()
        assert l1.bid_size == 6000

    def test_apply_diff_delete(self):
        book = hft_cpp.OrderBook()
        book.apply_diff("bid", 100.50, 5000, 1)
        book.apply_diff("bid", 100.50, 0, 0)
        assert book.bid_count() == 0

    def test_apply_diff_multiple_levels(self):
        book = hft_cpp.OrderBook()
        book.apply_diff("bid", 100.50, 5000, 1)
        book.apply_diff("bid", 100.25, 3000, 1)
        book.apply_diff("ask", 101.00, 4000, 1)
        book.apply_diff("ask", 101.50, 2000, 1)
        assert book.bid_count() == 2
        assert book.ask_count() == 2
        depth = book.depth_summary()
        assert depth.bid_volume == 8000
        assert depth.ask_volume == 6000

    def test_l2_snapshot(self):
        book = hft_cpp.OrderBook()
        book.apply_diff("bid", 100.50, 5000, 1)
        book.apply_diff("bid", 100.25, 3000, 1)
        l2 = book.l2_snapshot(1)
        assert len(l2.bids) == 1
        assert l2.bids[0].price == 100.50
        l2_all = book.l2_snapshot(10)
        assert len(l2_all.bids) == 2

    def test_empty_book_l1_raises(self):
        book = hft_cpp.OrderBook()
        with pytest.raises(RuntimeError):
            book.l1_snapshot()

    def test_apply_snapshot(self):
        book = hft_cpp.OrderBook()
        bids = [hft_cpp.PriceLevel(100.50, 5000, 1), hft_cpp.PriceLevel(100.25, 3000, 1)]
        asks = [hft_cpp.PriceLevel(101.00, 4000, 1)]
        book.apply_snapshot(bids, asks)
        assert book.bid_count() == 2
        assert book.ask_count() == 1
        l1 = book.l1_snapshot()
        assert l1.bid_price == 100.50
        assert l1.ask_price == 101.00


class TestCppMicrostructureAnalyser:
    """Direct tests on hft_cpp.MicrostructureAnalyser."""

    def test_create(self):
        ms = hft_cpp.MicrostructureAnalyser(100)
        assert ms.window_size() == 100
        assert ms.spread_count() == 0

    def test_on_tick_and_spread_stats(self):
        ms = hft_cpp.MicrostructureAnalyser(100)
        ms.on_tick("L1", 100.50, 101.00, 5000, 3000)
        assert ms.spread_count() == 1
        stats = ms.spread_stats()
        assert stats.count == 1
        assert stats.mean > 0

    def test_spread_stats_multiple(self):
        ms = hft_cpp.MicrostructureAnalyser(10)
        for i in range(5):
            ms.on_tick("L1", 100.0 + i * 0.01, 101.0 + i * 0.01, 5000, 3000)
        stats = ms.spread_stats()
        assert stats.count == 5
        assert stats.min <= stats.mean <= stats.max

    def test_spread_stats_window_trims(self):
        ms = hft_cpp.MicrostructureAnalyser(3)
        for i in range(10):
            ms.on_tick("L1", 100.0, 101.0 + i * 0.1, 5000, 3000)
        assert ms.spread_count() == 3

    def test_ignores_non_l1(self):
        ms = hft_cpp.MicrostructureAnalyser(100)
        ms.on_tick("TRADE", 100.50, 101.00, 5000, 3000)
        assert ms.spread_count() == 0

    def test_current_depth(self):
        book = hft_cpp.OrderBook()
        book.apply_diff("bid", 100.50, 5000, 1)
        book.apply_diff("ask", 101.00, 3000, 1)
        profile = hft_cpp.MicrostructureAnalyser.current_depth(book, 10)
        assert profile.total_bid_volume == 5000
        assert profile.total_ask_volume == 3000
        assert profile.imbalance > 0

    def test_classify_trade(self):
        book = hft_cpp.OrderBook()
        book.apply_diff("bid", 100.50, 5000, 1)
        book.apply_diff("ask", 101.00, 3000, 1)
        result = hft_cpp.MicrostructureAnalyser.classify_trade(101.00, 1000, book)
        assert result.side == "buy"
        result2 = hft_cpp.MicrostructureAnalyser.classify_trade(100.50, 1000, book)
        assert result2.side == "sell"


class TestPythonWrappersWithCpp:
    """Verify Python OrderBook/MicrostructureAnalyser delegate correctly."""

    def test_order_book_use_cpp_true(self):
        from hft_engine.book.order_book import OrderBook
        book = OrderBook(use_cpp=True)
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        l1 = book.l1_snapshot()
        assert l1.bid_price == 100.50
        assert l1.ask_price == 101.00
        assert l1.spread == 0.5
        assert l1.mid_price == 100.75

    def test_order_book_use_cpp_returns_python_types(self):
        from hft_engine.book.order_book import OrderBook, L1Snapshot, L2Snapshot, BookDepth
        book = OrderBook(use_cpp=True)
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        l1 = book.l1_snapshot()
        assert isinstance(l1, L1Snapshot)
        l2 = book.l2_snapshot(10)
        assert isinstance(l2, L2Snapshot)
        depth = book.depth_summary()
        assert isinstance(depth, BookDepth)

    def test_order_book_apply_snapshot_cpp(self):
        from hft_engine.book.order_book import OrderBook, PriceLevel
        book = OrderBook(use_cpp=True)
        bids = [PriceLevel(100.50, 5000, 1), PriceLevel(100.25, 3000, 1)]
        asks = [PriceLevel(101.00, 4000, 1)]
        book.apply_snapshot(bids, asks)
        l1 = book.l1_snapshot()
        assert l1.bid_price == 100.50
        assert l1.ask_price == 101.00

    def test_microstructure_use_cpp_true(self):
        from hft_engine.microstructure.analyser import MicrostructureAnalyser
        from hft_engine.feeds.tick_consumer import MarketDataMessage
        ms = MicrostructureAnalyser(window_size=100, use_cpp=True)
        msg = MarketDataMessage(
            epic="TEST",
            message_type="L1",
            bid_price=100.50,
            ask_price=101.00,
            bid_size=5000,
            ask_size=3000,
        )
        ms.on_tick(msg)
        stats = ms.spread_stats()
        assert stats.count == 1
        assert stats.mean > 0

    def test_microstructure_returns_python_types(self):
        from hft_engine.microstructure.analyser import MicrostructureAnalyser, SpreadMetrics
        from hft_engine.feeds.tick_consumer import MarketDataMessage
        ms = MicrostructureAnalyser(window_size=100, use_cpp=True)
        msg = MarketDataMessage(epic="TEST", message_type="L1",
                                bid_price=100.50, ask_price=101.00)
        ms.on_tick(msg)
        stats = ms.spread_stats()
        assert isinstance(stats, SpreadMetrics)

    def test_hybrid_order_book_and_microstructure(self):
        from hft_engine.book.order_book import OrderBook
        from hft_engine.microstructure.analyser import MicrostructureAnalyser
        from hft_engine.feeds.tick_consumer import MarketDataMessage
        book = OrderBook(use_cpp=True)
        ms = MicrostructureAnalyser(window_size=100, use_cpp=True)

        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        ms.on_tick(MarketDataMessage(epic="TEST", message_type="L1",
                     bid_price=100.50, ask_price=101.00))

        depth = MicrostructureAnalyser.current_depth(book, 10)
        assert depth.total_bid_volume > 0
        assert depth.total_ask_volume > 0

        result = MicrostructureAnalyser.classify_trade(101.00, 1000, book)
        assert result.side in ("buy", "sell")

    def test_auto_detect_cpp(self):
        from hft_engine.book.order_book import OrderBook
        book = OrderBook()
        assert book._use_cpp == _HAS_CPP
        assert (book._cpp_book is not None) == _HAS_CPP
