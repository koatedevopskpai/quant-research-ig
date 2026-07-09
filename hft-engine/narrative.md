# hft-engine — High-Frequency Trading Engine

## Objective
Build a low-latency, event-driven market data + execution engine in Python.
Focus on order book reconstruction, tick-level analytics, microsecond
timestamping, and FIX-like messaging — inspired by Bryan Downing's C++ HFT work.

## Projects (Planned)
1. **Order Book Reconstructor** — L1/L2 book from Lightstreamer ticks
2. **Microstructure Analytics** — Trade signs, spread, depth profiles
3. **Latency Monitor** — Nanosecond-level pipeline instrumentation
4. **Execution Simulator** — Market impact + slippage model
5. **Co-location Sim** — Network topology, latency arbitrage

## Key Files
| File | Purpose |
|------|---------|
| src/hft_engine/book/ | Order book data structures |
| src/hft_engine/feeds/ | Market data feed handlers (Lightstreamer) |
| src/hft_engine/execution/ | Order management and execution |

## Status
- [ ] Project skeleton created
