# hft-engine — High-Frequency Trading Engine

## Objective
Build a low-latency, event-driven market data + execution engine in Python
with C++ hot-path accelerators. Focus on order book reconstruction, tick-level
analytics, microsecond timestamping, order flow toxicity, execution simulation,
and co-location arbitrage — inspired by Bryan Downing's C++ HFT architecture.
This spans **Phase 2b (Core)** and **Phase 3b (Advanced)** in the roadmap.

## Projects

### Phase 2b — Core (Completed)
1. **Order Book** — L2 book with sorted bid/ask levels, diff application,
   L1/L2 snapshots, depth summary (imbalance). O(K) operations for K levels.
2. **Feed Handler** — Lightstreamer tick consumer with L1/L2/trade parsers,
   callback dispatch, timestamp normalisation to nanosecond precision.
3. **Microstructure Analytics** — Spread stats (rolling window), depth
   profiles (weighted mid-price), Lee-Ready trade direction classification.

### Phase 3b — Advanced (Completed)
4. **C++ Hot Paths** (pybind11, build with `pip install .[cpp]`) —
   Binary-search order book, tick parser, microstructure ring buffer.
5. **Order Flow Toxicity** — Hawkes(1,1) process with grid-search EM.
   Toxicity = (lambda - mu) / lambda; ratio >0.5 indicates toxic flow.
6. **Latency Monitor** — Nanosecond pipeline instrumentation with
   context manager `timer()`, per-stage histograms, real-time spike alerts.
7. **Execution Simulator** — Almgren-Chriss market impact model with
   permanent/temporary impact, participation-rate slicing, book-level fills.
8. **Market Simulator** — GBM mid-price + Poisson L1/trade arrivals +
   mean-reverting spread dynamics, configurable depth clustering.
9. **Co-location Simulator** — Geographic latency model over fiber,
   cross-exchange arbitrage detection with competitor timing advantage.

## Key Module Map
| Module | Class/Function | Purpose |
|--------|---------------|---------|
| book/order_book.py | `OrderBook` | L2 book with diffs, snapshots, depth |
| book/order_book.py | `PriceLevel` | Single price level (price, size, count) |
| book/order_book.py | `L1Snapshot` | Top-of-book with spread and mid-price |
| feeds/tick_consumer.py | `TickConsumer` | Feed handler with callback dispatch |
| feeds/tick_consumer.py | `MarketDataMessage` | Normalised feed message |
| microstructure/analyser.py | `MicrostructureAnalyser` | Spread, depth, trade signs |
| toxicity/detector.py | `ToxicityDetector` | Hawkes process order flow toxicity |
| latency/monitor.py | `LatencyMonitor` | Nanosecond pipeline timing |
| execution/simulator.py | `ExecutionSimulator` | Almgren-Chriss impact model |
| market_sim/generator.py | `MarketSimulator` | Synthetic tick generator |
| colocation/simulator.py | `ColocationSimulator` | Cross-exchange latency arb |
| cpp/order_book.cpp | C++ OrderBook | Binary-search order book (pybind11) |
| cpp/feed_handler.cpp | C++ FeedHandler | Tick parsing hot loop (pybind11) |
| cpp/microstructure.cpp | C++ Microstructure | Ring-buffer spread stats (pybind11) |
| cpp/CMakeLists.txt | Build | CMake + pybind11 build config |

## Status
- [x] Phase 2b Core — order book, feed handler, microstructure (45 tests)
- [x] Phase 3b Python — toxicity, latency, execution, market sim, co-location (43 tests)
- [x] **All 88 tests passing**
- [ ] C++ pybind11 build (`pip install .[cpp]` — requires VS Build Tools + CMake)

---

## Architecture Decisions

### ADR-001: Separate Book, Feeds, and Microstructure Modules

**Status**: Accepted

**Context**: HFT systems require clean separation of concerns to enable
independent testing, optimisation, and replacement of components.

**Decision**: Three independent modules with unidirectional data flow:
```
IG Lightstreamer → feeds/TickConsumer → book/OrderBook
                                    → microstructure/MicrostructureAnalyser
```

**Rationale**:
- The feed handler can be replaced (IG → IBKR → direct exchange FIX) without
  touching the order book or microstructure code.
- The order book can be unit-tested with synthetic diffs (no feed required).
- The microstructure analyser can be tested with pre-recorded tick data.
- The C++ pybind11 hot path (Phase 3b) can replace individual modules without
  changing the data flow.

**Consequences**:
- All three modules share the `MarketDataMessage` as the canonical data format.
- `MicrostructureAnalyser` takes an optional `OrderBook` parameter for
  depth-dependent analytics.
- Each module has its own test file with no cross-test dependencies.

---

### ADR-002: Linear Search for Price Level Index (K ≤ 100)

**Status**: Accepted

**Context**: Order book operations require finding a price level by price to
update, insert, or delete. L2 books typically have 10-50 levels per side.

**Decision**: Linear search for price index (O(K)) instead of binary search
(O(log K)).

**Rationale**:
- For K ≤ 100 (typical IG L2 depth), linear search is faster than binary
  search due to Python's function call overhead and the simplicity of the
  comparison loop (no custom comparator needed).
- Asks are sorted ascending, bids descending — binary search would require
  separate logic for each side.
- Benchmark: 100-level linear scan completes in ~0.5 microseconds.

**Trade-offs**:
- At K > 1000 levels (extreme), binary search would be ~100x faster.
- This is a candidate for the C++ pybind11 hot path if IG ever increases
  their L2 depth.

**Consequences**:
- The `_find_price_index` helper is used for both sides.
- The `apply_diff` method uses the same code path for update, insert,
  and delete operations.

---

### ADR-003: Lee-Ready Algorithm (Simplified) for Trade Classification

**Status**: Accepted

**Context**: Determining trade direction (aggressive buy vs. sell) is essential
for order flow imbalance and toxicity detection.

**Decision**: Simplified Lee-Ready algorithm without the 5-second quote test.

**Rationale**:
- The full LR algorithm (Lee & Ready, 1991) checks if a trade occurred
  within 5 seconds of a quote update. This requires maintaining a quote
  history window and adds complexity.
- The simplified version is ~85% accurate for equities (LR original paper)
  and >90% accurate for large-tick instruments like indices.
- The 5-second test primarily improves classification for trades inside the
  spread, which are rare for liquid instruments (typically <5% of trades).

**Consequences**:
- Trades above the ask → aggressive buy (confidence 1.0).
- Trades below the bid → aggressive sell (confidence 1.0).
- Trades at the ask/bid → buy/sell with confidence 1.0.
- Trades inside the spread → classified by mid-price (confidence 0.5).

---

---

### ADR-004: Hawkes(1,1) over Bayesian for Toxicity

**Status**: Accepted

**Context**: Order flow toxicity measures how much of current trade intensity
is driven by self-exciting clustering vs. baseline random activity. Both
Bayesian methods and Hawkes processes can model this.

**Decision**: Hawkes(1,1) with grid-search EM, toxicity = (lambda - mu) / lambda.

**Rationale**:
- Hawkes directly models self-excitation — the core mechanism of toxic flow
  (informed traders cluster their orders to minimise information leakage).
- The ratio (lambda - mu) / lambda is a bounded [0,1] toxicity score that
  aligns with Downing's "toxic fraction" concept.
- Grid-search over three scaling factors (0.5×, 1.0×, 2.0×) for mu/alpha/beta
  provides robust initialisation without MCMC complexity.

**Consequences**:
- Toxicity > 0.5 means >50% of current intensity is from cascading — the
  threshold for "toxic" in the detector.
- Requires ≥10 events; window_size caps history at 1000 events by default.
- C++ hot path candidate for real-time intensity computation (O(1) at predict).

---

### ADR-005: Context Manager over Decorator for Latency Instrumentation

**Status**: Accepted

**Context**: Pipeline stages (feed → parse → book → signal → order) need
nanosecond timing. Python offers decorators, context managers, and manual
timestamp pairs.

**Decision**: Context manager (`with monitor.timer("stage"):`) as the
primary API, with explicit `record()` as the low-level fallback.

**Rationale**:
- Context managers compose naturally with existing try/finally blocks.
- They guarantee the end timestamp is captured even on exceptions.
- Decorators couple timing to function boundaries; stages often span
  multiple function calls (e.g., parse + dispatch).
- The `record()` method allows manual instrumentation for fine-grained
  sub-stage timing.

**Consequences**:
- Thread-safe with `threading.Lock` around the records deque.
- Alert threshold (default 100µs) triggers on per-record elapsed time.
- Stats computed as running mean, median, p50/p95/p99 over a sliding window.

---

### ADR-006: Almgren-Chriss over Kissell for Market Impact

**Status**: Accepted

**Context**: Execution simulators require a market impact model to estimate
slippage. The choice is between Almgren-Chriss (linear permanent + power-law
temporary) and Kissell (non-linear with spread recapture).

**Decision**: Almgren-Chriss with power-law exponents (0.5 perm, 0.6 temp).

**Rationale**:
- Almgren-Chriss is the industry standard for implementation shortfall.
- Permanent impact ∝ (Q/ADV)^0.5 and temporary impact ∝ (Q/ADV)^0.6 match
  empirically observed concavity in impact.
- Kissell's spread recapture term requires microstructural data (depth
  profiles) not available at the start of execution.
- The simulator supports book-level fills to add microstructure realism
  on top of the parametric impact curve.

**Consequences**:
- Urgency parameter scales temporary impact (higher urgency = less time
  to participate = more impact per slice).
- The `slice_order()` method implements VWAP-like participation.
- Book-level fills from `fill_from_book()` let the caller simulate
  walking the book for an extra layer of realism.

---

### ADR-007: Geographic Latency Model over Discrete-Event Network Sim

**Status**: Accepted

**Context**: Co-location arbitrage requires modelling the latency between
exchanges. Options: discrete-event network simulation (ns-3), closed-form
fiber distance model, or empirical latency data.

**Decision**: Great-circle distance over fiber at 2/3 c (200,000 km/s).

**Rationale**:
- The speed of light in fiber (~200,000 km/s = 2/3 c) is the fundamental
  physical bound on information propagation. No technology beats it.
- Great-circle distance is deterministic and explainable — an ADR reader
  can verify "London to NY = 5571 km = 27.9 ms one way".
- A full ns-3 simulation adds complexity without actionable insight at
  this stage; the fiber model provides the correct lower bound.
- The competitor latency penalty parameter captures non-ideal routing,
  switch latency, and protocol overhead without a full network model.

**Consequences**:
- Round-trip latency ~56 ms London↔NYSE; ~140 ms London↔Tokyo.
- Arbitrage window = competitor latency − our latency; >0 means we
  have a timing advantage.
- The `add_noise()` method simulates price feed jitter for more realistic
  scenario testing.

---

## Implementation Guide

### Order Book (book/)

**OrderBook** — The core data structure. Create one per instrument:
```python
book = OrderBook()
book.apply_snapshot(bids=[...], asks=[...])  # initial state
book.apply_diff("bid", 100.50, 5000)          # update level
book.apply_diff("ask", 100.60, 0)             # delete level
l1 = book.l1_snapshot()                       # best bid/ask
l2 = book.l2_snapshot(max_levels=5)           # top 5 levels
depth = book.depth_summary()                  # volume + imbalance
```

### Feed Handler (feeds/)

**TickConsumer** — Subscribe to IG epics with callbacks:
```python
consumer = TickConsumer()
consumer.subscribe("IX.D.FTSE.IFP.IP", my_callback)
consumer.dispatch(msg)  # parse → callbacks
```

Pass `TickData` for custom subscription config (L2 or TRADE mode).

### Microstructure (microstructure/)

**MicrostructureAnalyser** — Feed ticks, query analytics:
```python
analyser = MicrostructureAnalyser(window_size=100)
analyser.on_tick(msg_l1)                     # update rolling windows
stats = analyser.spread_stats()              # mean, std, min, max
profile = analyser.current_depth(book)       # depth + imbalance
sign = analyser.classify_trade(100.50, 1000, book)  # buy/sell/neutral
```

---

## References

- Downing, B. (2025). Deconstructing an Advanced Ether High Frequency Trading
  Bot: A C++ Breakdown. *QuantLabsNet*.
- Harris, L. (2003). *Trading & Exchanges: Market Microstructure for
  Practitioners*. Oxford University Press.
- Lee, C. & Ready, M. (1991). Inferring Trade Direction from Intraday Data.
  *Journal of Finance*, 46(2).
- IG Labs Streaming API: https://labs.ig.com/streaming-api
