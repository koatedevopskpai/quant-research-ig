# hft-engine — High-Frequency Trading Engine

## Objective
Build a low-latency, event-driven market data + execution engine in Python.
Focus on order book reconstruction, tick-level analytics, microsecond
timestamping, and FIX-like messaging — inspired by Bryan Downing's C++ HFT
work. This is the **Phase 2b** deliverable in the STRATEGY.md roadmap.

## Projects (Completed — Phase 2b Core)
1. **Order Book** — L2 book with sorted bid/ask levels, diff application,
   L1/L2 snapshots, depth summary (imbalance). O(K) operations for K levels.
2. **Feed Handler** — Lightstreamer tick consumer with L1/L2/trade parsers,
   callback dispatch, timestamp normalisation to nanosecond precision.
3. **Microstructure Analytics** — Spread stats (rolling window), depth
   profiles (weighted mid-price), Lee-Ready trade direction classification.

## Key Module Map
| Module | Class/Function | Purpose |
|--------|---------------|---------|
| book/order_book.py | `OrderBook` | L2 book with diffs, snapshots, depth |
| book/order_book.py | `PriceLevel` | Single price level (price, size, count) |
| book/order_book.py | `L1Snapshot` | Top-of-book with spread and mid-price |
| feeds/tick_consumer.py | `TickConsumer` | Feed handler with callback dispatch |
| feeds/tick_consumer.py | `MarketDataMessage` | Normalised feed message |
| microstructure/analyser.py | `MicrostructureAnalyser` | Spread, depth, trade signs |

## Status
- [x] Project skeleton (pyproject.toml, package structure)
- [x] L2 Order Book — sorted bid/ask levels, diff apply, snapshots (22 tests)
- [x] Feed Handler — L1/L2/trade parsing, callback dispatch (12 tests)
- [x] Microstructure Analytics — spread stats, depth profile, Lee-Ready (11 tests)
- [x] **All 45 tests passing**

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
