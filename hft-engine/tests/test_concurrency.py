"""
Concurrency contract for the lock-free path.

The C++ stress binary runs under ThreadSanitizer in CI; these tests assert the
functional contract from Python: ordering, no loss, full detection, and that a
reader never observes a torn snapshot while a writer is running.
"""

import threading
import time

import pytest

from hft_engine.book.order_book import OrderBook  # noqa: F401  (ensures package import)

try:
    import hft_engine.hft_cpp as hft_cpp
    _HAS_CPP = hasattr(hft_cpp, "ConcurrentOrderBook")
except ImportError:  # extension not built in this job
    hft_cpp = None
    _HAS_CPP = False

pytestmark = pytest.mark.skipif(
    not _HAS_CPP,
    reason="hft_cpp not built (or built before the concurrency additions)",
)


class TestTickRing:
    def test_capacity_is_usable_minus_one(self):
        ring = hft_cpp.TickRing()
        assert hft_cpp.TickRing.capacity() > 0
        assert ring.size() == 0

    def test_full_then_drained_in_order(self):
        ring = hft_cpp.TickRing()
        capacity = hft_cpp.TickRing.capacity()
        for i in range(capacity):
            assert ring.try_push(i, 1.0, 2.0, 3.0) is True
        assert ring.try_push(0, 0.0, 0.0, 0.0) is False
        assert ring.size() == capacity

        seen = []
        while True:
            item = ring.try_pop()
            if item is None:
                break
            seen.append(item[0])
        assert seen == list(range(capacity))
        assert ring.try_pop() is None

    def test_no_loss_or_reordering_across_threads(self):
        ring = hft_cpp.TickRing()
        total = 20_000
        received: list[int] = []

        def producer() -> None:
            for i in range(total):
                while not ring.try_push(i, 1.0, 2.0, 3.0):
                    time.sleep(0)

        def consumer() -> None:
            while len(received) < total:
                item = ring.try_pop()
                if item is None:
                    time.sleep(0)
                    continue
                received.append(item[0])

        threads = [threading.Thread(target=producer),
                   threading.Thread(target=consumer)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=20)

        assert len(received) == total
        assert received == list(range(total))


class TestConcurrentBook:
    def test_l1_snapshot_and_update_count(self):
        book = hft_cpp.ConcurrentOrderBook()
        book.apply_l1(100.0, 100.5, 3.0, 4.0)
        snap = book.snapshot_l1()
        assert snap.bid_price == pytest.approx(100.0)
        assert snap.ask_price == pytest.approx(100.5)
        assert book.updates() == 1
        assert book.snapshots() == 1

    def test_diff_keeps_levels_sorted_and_deletes_at_zero(self):
        book = hft_cpp.ConcurrentOrderBook()
        book.apply_diff("bid", 100.0, 5.0)
        book.apply_diff("bid", 99.0, 3.0)
        book.apply_diff("bid", 101.0, 2.0)
        prices = [level.price for level in book.snapshot_l2().bids]
        assert prices == [101.0, 100.0, 99.0]

        book.apply_diff("bid", 100.0, 0.0)
        prices = [level.price for level in book.snapshot_l2().bids]
        assert prices == [101.0, 99.0]

    def test_reader_never_sees_a_torn_book(self):
        book = hft_cpp.ConcurrentOrderBook()
        updates = 20_000
        done = threading.Event()
        counts = {"valid": 0, "torn": 0, "unavailable": 0}

        def writer() -> None:
            for i in range(updates):
                price = 100.0 + 0.01 * (i % 1000)
                book.apply_l1(price - 0.5, price + 0.5, 1.0, 1.0)
            done.set()

        def reader() -> None:
            while not done.is_set():
                try:
                    snap = book.snapshot_l1()
                except RuntimeError:
                    counts["unavailable"] += 1
                    continue
                if snap.bid_price < snap.ask_price:
                    counts["valid"] += 1
                else:
                    counts["torn"] += 1

        threads = [threading.Thread(target=writer)] + [
            threading.Thread(target=reader) for _ in range(3)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert book.updates() == updates
        assert counts["torn"] == 0
        assert counts["valid"] > 0
