# Lock-free C++ path

Status: **implemented and measured** (2026-10-27). The repository now contains
lock-free structures in the hot path — with a precise definition, because
"lock-free order book" is only true for the part that is actually lock-free.

## What is lock-free, exactly

| Component | Guarantee |
|---|---|
| `SpscRing<T, Capacity>` (`spsc_ring.h`) | lock-free bounded queue: one producer, one consumer, two atomic indices, acquire/release, no mutex, no CAS |
| `ConcurrentOrderBook` snapshots (`concurrent_book.h`) | lock-free **readers**: a seqlock over atomics, readers retry, they never block the writer |
| `ConcurrentOrderBook` mutations | **single writer** by contract — not multi-writer, not claimed to be |
| L2 snapshots | consistent copy under the same seqlock; copying costs are the reader's |

This is the standard split in a market-data pipeline: the feed thread owns the
book, message handoff is lock-free, and readers take consistent snapshots
without stopping the writer. A fully concurrent multi-writer book would need CAS
on levels and memory reclamation for no benefit to a single feed.

## Why the implementation looks the way it does

**SPSC over MPSC.** There is exactly one feed producer and one book consumer. An
MPSC queue needs a CAS loop and pays for contention that does not exist here.

**Acquire/release, not seq_cst.** The producer publishes `tail_` with release
after writing the slot; the consumer acquires `tail_` before reading it. This is
the minimum ordering that makes the payload visible, and it compiles to a plain
store/load pair on x86.

**Atomics inside the seqlock, not plain doubles.** A seqlock over non-atomic
fields is undefined behaviour even when readers retry — the concurrent read is
still a data race. Every shared field is `std::atomic<uint64_t>` (doubles are
bit-cast), loaded relaxed inside the sequence protocol. ThreadSanitizer enforces
this.

**Fixed arrays, no allocation.** `MaxLevels = 64` per side, so the writer never
reallocates while a reader is copying. `static_assert`s require lock-free 64-bit
and 32-bit atomics on the target.

**GIL released on the boundary.** `apply_l1`, `apply_diff` and the snapshot
methods use `py::gil_scoped_release`, so Python threads can genuinely exercise
the concurrency rather than serialising on the interpreter.

## Measurements

Windows 11 x64, AMD Ryzen 5 8640HS, MSVC 19.51, Release. Raw output:
[`evidence/lockfree-bench.txt`](../evidence/lockfree-bench.txt).

| Measurement | Result |
|---|---|
| SPSC handoff (5,000,000 messages) | **4.6 ns / message** (23.0 ms total) |
| Book updates (5,000,000 L1 updates, 1 reader) | **60.4 ns / update** |
| Reader snapshots during that run | 4,094,610 valid, 439,566 retries |
| Stress: 1,000,000 queued messages | consumed in order, no loss |
| Stress: 500,000 updates with 3 readers | **0 torn snapshots** |

The retry counts are the honest part: readers do re-read when the writer is
mid-update. At 60 ns per write they still complete millions of snapshots per
run, but the number is reported, not hidden.

## Proof

- **Functional stress** (`hft-engine/cpp/tests/stress.cpp`): ordering/loss on the
  queue and torn-snapshot detection on the book. Exits non-zero on failure.
- **ThreadSanitizer in CI** (`tsan` job): builds the stress test with
  `-fsanitize=thread` and fails on any race report.
- **Python contract tests** (`hft-engine/tests/test_concurrency.py`): full/empty
  behaviour, cross-thread ordering, sorted diffs, deletion at size 0, and
  readers never seeing `bid >= ask` while a writer runs.

```bash
# local functional + benchmark run (Windows path shown)
cmake -S hft-engine/cpp -B build-stress -DHFT_BUILD_STRESS=ON
cmake --build build-stress --config Release --parallel
./hft-engine/cpp/build-stress/Release/hft_stress.exe
./hft-engine/cpp/build-stress/Release/hft_bench.exe
```

## Limitations

- **Single writer.** Two threads must not call `apply_l1`/`apply_diff`
  concurrently. The queue is the handoff boundary.
- **Reader starvation is possible in theory.** A writer that never pauses could
  keep the sequence odd; `kMaxRetries` bounds the reader and it returns "no
  snapshot" rather than blocking. Measured retries are reported above.
- **L2 snapshot copies are the reader's cost.** L1 is a fixed four-field copy;
  L2 allocates a vector per snapshot.
- **Linear level search** is unchanged inside the writer (K ≤ 64), so the book
  is not a multi-writer structure and does not pretend to be.
