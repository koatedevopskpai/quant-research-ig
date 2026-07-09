"""
test_order_book.py — Unit tests for the L2 order book.
"""

from math import isclose

import pytest

from hft_engine.book.order_book import OrderBook, PriceLevel


class TestOrderBook:
    """Test the L2 order book data structure and diff application."""

    # ------------------------------------------------------------------
    # Fixtures
    # ------------------------------------------------------------------

    @pytest.fixture
    def populated_book(self) -> OrderBook:
        """Book with a known set of bid and ask levels."""
        book = OrderBook()
        book.apply_snapshot(
            bids=[
                PriceLevel(100.50, 5000),
                PriceLevel(100.40, 10000),
                PriceLevel(100.30, 15000),
            ],
            asks=[
                PriceLevel(100.60, 7000),
                PriceLevel(100.70, 8000),
                PriceLevel(100.80, 9000),
            ],
        )
        return book

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def test_apply_snapshot(self, populated_book: OrderBook) -> None:
        l2 = populated_book.l2_snapshot()
        assert len(l2.bids) == 3
        assert len(l2.asks) == 3
        # Bids should be sorted descending
        assert l2.bids[0].price == 100.50
        assert l2.bids[1].price == 100.40
        assert l2.bids[2].price == 100.30
        # Asks should be sorted ascending
        assert l2.asks[0].price == 100.60
        assert l2.asks[1].price == 100.70
        assert l2.asks[2].price == 100.80

    def test_apply_snapshot_replaces(self, populated_book: OrderBook) -> None:
        """A second snapshot should replace, not append."""
        populated_book.apply_snapshot(bids=[PriceLevel(200.0, 1000)], asks=[PriceLevel(201.0, 1000)])
        l2 = populated_book.l2_snapshot()
        assert len(l2.bids) == 1
        assert l2.bids[0].price == 200.0

    # ------------------------------------------------------------------
    # L1 snapshot
    # ------------------------------------------------------------------

    def test_l1_snapshot(self, populated_book: OrderBook) -> None:
        l1 = populated_book.l1_snapshot()
        assert l1.bid_price == 100.50
        assert l1.ask_price == 100.60
        assert l1.bid_size == 5000
        assert l1.ask_size == 7000
        assert isclose(l1.spread, 0.10)
        assert isclose(l1.mid_price, 100.55)

    def test_l1_snapshot_empty_book(self) -> None:
        book = OrderBook()
        with pytest.raises(ValueError):
            book.l1_snapshot()

    def test_l1_snapshot_missing_side(self) -> None:
        book = OrderBook()
        book.apply_snapshot(bids=[PriceLevel(100.0, 1000)], asks=[])
        with pytest.raises(ValueError):
            book.l1_snapshot()

    # ------------------------------------------------------------------
    # Diff application — ask side
    # ------------------------------------------------------------------

    def test_apply_diff_update_ask(self, populated_book: OrderBook) -> None:
        """Updating an existing ask level should change its size."""
        populated_book.apply_diff("ask", 100.70, 20000)
        l2 = populated_book.l2_snapshot()
        assert l2.asks[1].size == 20000

    def test_apply_diff_insert_ask(self, populated_book: OrderBook) -> None:
        """Inserting a new ask level should maintain ascending order."""
        populated_book.apply_diff("ask", 100.65, 5000)
        l2 = populated_book.l2_snapshot()
        assert len(l2.asks) == 4
        assert l2.asks[1].price == 100.65

    def test_apply_diff_delete_ask(self, populated_book: OrderBook) -> None:
        """Setting size=0 should remove the level."""
        populated_book.apply_diff("ask", 100.70, 0)
        l2 = populated_book.l2_snapshot()
        assert len(l2.asks) == 2
        # 100.80 should now be at index 1
        assert l2.asks[1].price == 100.80

    # ------------------------------------------------------------------
    # Diff application — bid side
    # ------------------------------------------------------------------

    def test_apply_diff_update_bid(self, populated_book: OrderBook) -> None:
        populated_book.apply_diff("bid", 100.40, 25000)
        l2 = populated_book.l2_snapshot()
        assert l2.bids[1].size == 25000

    def test_apply_diff_insert_bid(self, populated_book: OrderBook) -> None:
        """Inserting a new bid level should maintain descending order."""
        populated_book.apply_diff("bid", 100.45, 8000)
        l2 = populated_book.l2_snapshot()
        assert len(l2.bids) == 4
        assert l2.bids[1].price == 100.45

    def test_apply_diff_new_best_bid(self, populated_book: OrderBook) -> None:
        """A new level above the current best bid should become the new top."""
        populated_book.apply_diff("bid", 100.55, 3000)
        l1 = populated_book.l1_snapshot()
        assert l1.bid_price == 100.55

    def test_apply_diff_delete_bid(self, populated_book: OrderBook) -> None:
        populated_book.apply_diff("bid", 100.40, 0)
        l2 = populated_book.l2_snapshot()
        assert len(l2.bids) == 2
        assert l2.bids[1].price == 100.30

    def test_apply_diff_delete_best_bid(self, populated_book: OrderBook) -> None:
        """Deleting the best bid should cascade to the next level."""
        populated_book.apply_diff("bid", 100.50, 0)
        l1 = populated_book.l1_snapshot()
        assert l1.bid_price == 100.40

    # ------------------------------------------------------------------
    # Depth summary
    # ------------------------------------------------------------------

    def test_depth_summary(self, populated_book: OrderBook) -> None:
        depth = populated_book.depth_summary()
        assert depth.bid_volume == 30000  # 5000 + 10000 + 15000
        assert depth.ask_volume == 24000  # 7000 + 8000 + 9000
        assert depth.num_bid_levels == 3
        assert depth.num_ask_levels == 3
        # Imbalance = (30000 - 24000) / (30000 + 24000) = 6000 / 54000 = 0.1111
        assert isclose(depth.imbalance, 0.1111, abs_tol=0.001)

    def test_depth_summary_empty_book(self) -> None:
        book = OrderBook()
        depth = book.depth_summary()
        assert depth.bid_volume == 0
        assert depth.ask_volume == 0
        assert depth.imbalance == 0.0

    def test_depth_summary_max_levels(self, populated_book: OrderBook) -> None:
        depth = populated_book.depth_summary(max_levels=1)
        assert depth.num_bid_levels == 1
        assert depth.num_ask_levels == 1
        assert depth.bid_volume == 5000
        assert depth.ask_volume == 7000

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_apply_diff_nonexistent_ask(self, populated_book: OrderBook) -> None:
        """Deleting a nonexistent level should be a no-op."""
        populated_book.apply_diff("ask", 999.0, 0)  # should not raise
        l2 = populated_book.l2_snapshot()
        assert len(l2.asks) == 3

    def test_apply_diff_nonexistent_bid(self, populated_book: OrderBook) -> None:
        populated_book.apply_diff("bid", 999.0, 0)  # should not raise
        l2 = populated_book.l2_snapshot()
        assert len(l2.bids) == 3

    def test_empty_book_depth(self) -> None:
        book = OrderBook()
        depth = book.depth_summary()
        assert depth.bid_volume == 0.0
        assert depth.ask_volume == 0.0

    def test_l2_snapshot_max_levels(self, populated_book: OrderBook) -> None:
        l2 = populated_book.l2_snapshot(max_levels=2)
        assert len(l2.bids) == 2
        assert len(l2.asks) == 2


class TestOrderBookPurePython:
    """
    Tests running exclusively through the pure-Python OrderBook path.
    These verify the Python fallback works when C++ is disabled.
    """

    @pytest.fixture
    def book(self) -> OrderBook:
        return OrderBook(use_cpp=False)

    def test_apply_diff_and_l1(self, book: OrderBook) -> None:
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        l1 = book.l1_snapshot()
        assert l1.bid_price == 100.50
        assert l1.ask_price == 101.00

    def test_apply_diff_update_bid(self, book: OrderBook) -> None:
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        book.apply_diff("bid", 100.50, 6000)
        l1 = book.l1_snapshot()
        assert l1.bid_size == 6000

    def test_apply_diff_delete(self, book: OrderBook) -> None:
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("ask", 101.00, 3000)
        book.apply_diff("bid", 100.50, 0)
        with pytest.raises(ValueError):
            book.l1_snapshot()

    def test_apply_diff_insert_mid(self, book: OrderBook) -> None:
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("bid", 100.30, 3000)
        book.apply_diff("bid", 100.40, 4000)
        l2 = book.l2_snapshot()
        assert len(l2.bids) == 3
        assert l2.bids[1].price == 100.40

    def test_apply_diff_insert_ask_mid(self, book: OrderBook) -> None:
        book.apply_diff("ask", 101.00, 3000)
        book.apply_diff("ask", 101.20, 4000)
        book.apply_diff("ask", 101.10, 5000)
        l2 = book.l2_snapshot()
        assert len(l2.asks) == 3
        assert l2.asks[1].price == 101.10

    def test_apply_snapshot(self, book: OrderBook) -> None:
        book.apply_snapshot(
            bids=[PriceLevel(100.50, 5000), PriceLevel(100.25, 3000)],
            asks=[PriceLevel(101.00, 4000)],
        )
        l1 = book.l1_snapshot()
        assert l1.bid_price == 100.50
        assert l1.ask_price == 101.00

    def test_l2_snapshot_max_levels(self, book: OrderBook) -> None:
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("bid", 100.25, 3000)
        book.apply_diff("ask", 101.00, 4000)
        book.apply_diff("ask", 101.50, 2000)
        l2 = book.l2_snapshot(max_levels=1)
        assert len(l2.bids) == 1
        assert len(l2.asks) == 1

    def test_depth_summary(self, book: OrderBook) -> None:
        book.apply_diff("bid", 100.50, 5000)
        book.apply_diff("bid", 100.25, 3000)
        book.apply_diff("ask", 101.00, 4000)
        depth = book.depth_summary()
        assert depth.bid_volume == 8000
        assert depth.ask_volume == 4000

    def test_auto_detect_cpp(self) -> None:
        """When hft_cpp is available, OrderBook() auto-selects C++."""
        try:
            import hft_engine.hft_cpp
            book = OrderBook()  # no use_cpp arg
            assert book._use_cpp is True
        except ImportError:
            book = OrderBook()
            assert book._use_cpp is False
