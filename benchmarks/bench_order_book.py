"""
bench_order_book.py — measure the C++ order book against the pure-Python one.

WHAT THIS MEASURES:
    Per-operation latency for the three calls the hot path actually makes
    (apply_diff, l1_snapshot, depth_summary) on a 100-level book, for both
    implementations, through the same public API.

WHAT THIS DOES NOT MEASURE:
    Network, ingest parsing, or anything outside the book. Numbers are for this
    machine, this interpreter and this build only — they are not a claim about
    colocated production latency. Every run prints the platform so the result
    can be compared honestly.

WHY IT EXISTS:
    The C++ layer is only worth its complexity if the pybind11 boundary costs
    less than the Python work it replaces. That is a measurable question, not a
    matter of belief, so measure it.

USAGE:
    python benchmarks/bench_order_book.py
    python benchmarks/bench_order_book.py --ops 20000 --json evidence/bench.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from pathlib import Path

# Allow running from a clone without installing the workspace root package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "hft-engine" / "src"))

from hft_engine.book.order_book import OrderBook, PriceLevel  # noqa: E402


def build_book(book: OrderBook, levels: int, mid: float) -> None:
    """Seed both sides with `levels` price levels around `mid`."""
    bids = [PriceLevel(mid - 0.5 * (i + 1), 10.0 + i, 1) for i in range(levels)]
    asks = [PriceLevel(mid + 0.5 * (i + 1), 10.0 + i, 1) for i in range(levels)]
    book.apply_snapshot(bids=bids, asks=asks)


def timeit(fn, iterations: int) -> dict:
    """Median and p99 of per-call latency, in microseconds."""
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1e6)
    samples.sort()
    return {
        "median_us": round(statistics.median(samples), 3),
        "p99_us": round(samples[int(0.99 * (len(samples) - 1))], 3),
        "samples": len(samples),
    }


def describe(use_cpp: bool) -> dict:
    book = OrderBook(use_cpp=use_cpp)
    build_book(book, levels=100, mid=1000.0)

    rng = random.Random(42)
    prices = [1000.0 - 0.5 * (i + 1) for i in range(100)]

    def one_diff() -> None:
        book.apply_diff(side="bid", price=rng.choice(prices), size=rng.random() * 20.0,
                        order_count=1)

    return {
        "implementation": "cpp" if use_cpp else "python",
        "apply_diff": timeit(one_diff, args.ops),
        "l1_snapshot": timeit(book.l1_snapshot, args.ops),
        "depth_summary": timeit(book.depth_summary, args.ops),
    }


def main() -> int:
    global args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=int, default=5000, help="iterations per operation")
    parser.add_argument("--json", type=Path, default=None, help="write results as JSON")
    args = parser.parse_args()

    cpp_available = OrderBook(use_cpp=True)._use_cpp
    results = [describe(use_cpp=False)]
    if cpp_available:
        results.append(describe(use_cpp=True))
    else:
        print("! hft_cpp not built — run hft-engine/cpp/build_cpp.sh to compare\n")

    payload = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "levels": 100,
        "iterations": args.ops,
        "results": results,
    }

    header = f"{'impl':<8}{'operation':<16}{'median us':>12}{'p99 us':>12}"
    print(header)
    print("-" * len(header))
    for row in results:
        for op in ("apply_diff", "l1_snapshot", "depth_summary"):
            stats = row[op]
            print(f"{row['implementation']:<8}{op:<16}{stats['median_us']:>12}{stats['p99_us']:>12}")

    if len(results) == 2:
        speedup = results[0]["apply_diff"]["median_us"] / results[1]["apply_diff"]["median_us"]
        print(f"\napply_diff speedup (python/cpp, includes pybind11 boundary): {speedup:.2f}x")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwritten: {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
