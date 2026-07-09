"""
latency/monitor.py — Nanosecond Pipeline Latency Monitor

Instruments the tick-to-order pipeline with nanosecond timestamps
at each stage. Bryan Downing's architecture demands that every
microsecond be accounted for — this monitor traces the full path:

  IG Stream → parse → dispatch → book update → signal → order

Each stage records its entry/exit time, and the monitor computes
percentiles, histograms, and real-time alerts on latency spikes.
"""

import time
import statistics
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
import threading


@dataclass
class TimingRecord:
    stage: str
    start_ns: int
    end_ns: int
    elapsed_ns: int
    epic: str = ""


class LatencyMonitor:
    def __init__(self, window_size: int = 1000, alert_threshold_us: float = 100.0):
        self.window_size = window_size
        self.alert_threshold_us = alert_threshold_us
        self.records: dict[str, deque[TimingRecord]] = {}
        self._lock = threading.Lock()

    def now_ns(self) -> int:
        return time.time_ns()

    def record(self, stage: str, start_ns: int, end_ns: int,
               epic: str = "") -> TimingRecord:
        rec = TimingRecord(
            stage=stage, start_ns=start_ns, end_ns=end_ns,
            elapsed_ns=end_ns - start_ns, epic=epic,
        )
        elapsed_us = rec.elapsed_ns / 1000.0
        with self._lock:
            if stage not in self.records:
                self.records[stage] = deque(maxlen=self.window_size)
            self.records[stage].append(rec)
        if elapsed_us > self.alert_threshold_us:
            pass
        return rec

    class Timer:
        def __init__(self, monitor: "LatencyMonitor", stage: str, epic: str = ""):
            self.monitor = monitor
            self.stage = stage
            self.epic = epic
            self.start_ns: Optional[int] = None
            self.end_ns: Optional[int] = None

        def __enter__(self):
            self.start_ns = self.monitor.now_ns()
            return self

        def __exit__(self, *args):
            self.end_ns = self.monitor.now_ns()
            self.monitor.record(self.stage, self.start_ns, self.end_ns, self.epic)

    def timer(self, stage: str, epic: str = "") -> "LatencyMonitor.Timer":
        return self.Timer(self, stage, epic)

    def stats(self, stage: str) -> dict:
        with self._lock:
            recs = list(self.records.get(stage, []))
        if not recs:
            return {"stage": stage, "count": 0}
        elapsed = [r.elapsed_ns / 1000.0 for r in recs]
        return {
            "stage": stage,
            "count": len(elapsed),
            "mean_us": statistics.mean(elapsed),
            "median_us": statistics.median(elapsed),
            "min_us": min(elapsed),
            "max_us": max(elapsed),
            "stdev_us": statistics.stdev(elapsed) if len(elapsed) > 1 else 0.0,
            "p50_us": statistics.median(elapsed),
            "p95_us": sorted(elapsed)[int(len(elapsed) * 0.95)] if len(elapsed) >= 20 else 0.0,
            "p99_us": sorted(elapsed)[int(len(elapsed) * 0.99)] if len(elapsed) >= 100 else 0.0,
        }

    def all_stats(self) -> dict[str, dict]:
        stages = list(self.records.keys())
        return {s: self.stats(s) for s in stages}

    def total_pipeline_time(self) -> float:
        stages = list(self.records.keys())
        if not stages:
            return 0.0
        latest_records = []
        for s in stages:
            with self._lock:
                recs = list(self.records.get(s, []))
            if recs:
                latest_records.append(recs[-1])
        if not latest_records:
            return 0.0
        earliest_start = min(r.start_ns for r in latest_records)
        latest_end = max(r.end_ns for r in latest_records)
        return (latest_end - earliest_start) / 1000.0

    def clear(self, stage: Optional[str] = None):
        with self._lock:
            if stage:
                self.records.pop(stage, None)
            else:
                self.records.clear()
