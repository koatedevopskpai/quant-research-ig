import time
import pytest
from hft_engine.latency.monitor import LatencyMonitor, TimingRecord


class TestLatencyMonitor:
    def test_record_creates_timing(self):
        m = LatencyMonitor()
        ns = m.now_ns()
        rec = m.record("parse", ns, ns + 1000, "EPIC")
        assert rec.stage == "parse"
        assert rec.elapsed_ns == 1000
        assert rec.epic == "EPIC"

    def test_stats_single_stage(self):
        m = LatencyMonitor()
        for _ in range(10):
            start = m.now_ns()
            time.sleep(0.001)
            m.record("book", start, m.now_ns())
        s = m.stats("book")
        assert s["count"] == 10
        assert s["mean_us"] > 0
        assert s["min_us"] <= s["max_us"]

    def test_stats_empty_stage(self):
        m = LatencyMonitor()
        s = m.stats("nonexistent")
        assert s["count"] == 0

    def test_timer_context_manager(self):
        m = LatencyMonitor()
        with m.timer("parse", "EPIC"):
            pass
        s = m.stats("parse")
        assert s["count"] == 1
        assert s["mean_us"] >= 0

    def test_all_stats(self):
        m = LatencyMonitor()
        with m.timer("parse"):
            pass
        with m.timer("book"):
            pass
        all_s = m.all_stats()
        assert "parse" in all_s
        assert "book" in all_s

    def test_total_pipeline_time(self):
        m = LatencyMonitor()
        t1 = m.now_ns()
        t2 = t1 + 2000
        t3 = t2 + 3000
        m.record("feed", t1, t2)
        m.record("book", t2, t3)
        total = m.total_pipeline_time()
        assert total >= 5.0

    def test_clear(self):
        m = LatencyMonitor()
        m.record("test", 0, 100)
        assert m.stats("test")["count"] == 1
        m.clear("test")
        assert m.stats("test")["count"] == 0

    def test_clear_all(self):
        m = LatencyMonitor()
        m.record("a", 0, 100)
        m.record("b", 0, 100)
        m.clear()
        assert m.all_stats() == {}

    def test_now_ns_returns_int(self):
        m = LatencyMonitor()
        ns = m.now_ns()
        assert isinstance(ns, int)
        assert ns > 1_000_000_000_000
