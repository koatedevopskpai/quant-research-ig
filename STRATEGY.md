# Implementation Strategy — Phased Roadmap

## Overarching Strategy

This project bridges two complementary approaches to quantitative trading:

- **Bryan Downing's influence** → Low-latency market data handling, order book
  reconstruction, microstructure analytics, and disciplined execution
  architecture. His C++ HFT concepts are implemented in Python with C++
  hot-path extensions via pybind11 where latency is critical.
- **Quant Beckman's influence** → Rigorous hypothesis-driven research,
  statistical falsification of strategies, systematic backtesting validation
  (PSR, DSR, CPCV, WFA), and the Research→Know→Do→Get cycle that defines the
  agentic-finance module.

The phases follow a strict dependency chain: data infrastructure must exist
before models, models before validation, and validated strategies before
autonomous agents.

---

## Phase 0: Project Housekeeping
**Goal**: Establish shared conventions and tooling across all 5 sub-projects.

Files to create:
- Root `.gitignore` (Python, venv, `__pycache__`, `.env`, C++ build artifacts)
- Root `.env.template` (shared IG credentials template)
- Root `pyproject.toml` (workspace coordination)

---

## Phase 1: Complete market-data DB Layer
**Goal**: Persistent storage for all market data.

**Alignment**: Quant Beckman's data ingestion layer — the foundation of the
end-to-end solution pipeline.

Deliverables:
- `src/ig_market_data/models.py` — SQLAlchemy ORM (OHLCV, ticks, quotes)
- Alembic migrations for PostgreSQL schema
- `src/ig_market_data/ingestion.py` — scheduled/on-demand pipeline

---

## Phase 2a: Expand quant-methods
**Goal**: Classical quant finance models on real IG data.

**Alignment**: Quant Beckman's foundational models feed into the strategy
evaluation pipeline.

Deliverables:
- Binomial Tree (CRR lattice, American options)
- Monte Carlo Engine (path-dependent options)
- GARCH(1,1) volatility model
- Portfolio Optimisation (mean-variance, Black-Litterman, risk parity)
- Unit tests for each

---

## Phase 2b: hft-engine Core
**Goal**: Real-time order book and microstructure analytics.

**Alignment**: Bryan Downing's HFT architecture — market data structures and
the feed handler hot path.

Deliverables:

### Python Layer
- `src/hft_engine/book/order_book.py` — L1/L2 book orchestration, diff application
- `src/hft_engine/feeds/` — Lightstreamer tick consumer (extends IGStreamer)
- `src/hft_engine/microstructure/` — spread analysis, depth profiles, trade signs

### C++ Layer (pybind11)
- `cpp/order_book.cpp` — Lock-free L1/L2 book, nanosecond timestamps
- `cpp/feed_handler.cpp` — Tick parsing hot loop, direct memory pool
- `cpp/microstructure.cpp` — Volume imbalance, trade clustering
- `CMakeLists.txt` + `setup.py` for builds

---

## Phase 3a: ml-for-finance
**Goal**: ML models for regime detection and return forecasting.

**Alignment**: Quant Beckman's ML pipeline — models subjected to his
statistical validation framework.

Deliverables:
- HMM Regime Detection (low/med/high volatility — Downing's concept,
  scikit-learn implementation)
- XGBoost/LightGBM Return Forecasting
- LSTM Price Predictor (PyTorch)
- Sentiment-Augmented Model (NLP + price features)
- All models gated through PSR/DSR/CPCV validation

---

## Phase 3b: hft-engine Advanced
**Goal**: Full Bryan Downing HFT architecture integrated.

**Alignment**: Bryan Downing's order flow toxicity, execution simulation, and
latency monitoring.

Deliverables:
- Order Flow Toxicity Detector (Bayesian + Hawkes process — C++ hot path)
- Latency Monitor (nanosecond pipeline instrumentation — C++)
- Execution Simulator (market impact + slippage — Python)
- Market Simulator (synthetic tick generator — Python)
- Co-location Simulator (network topology, latency arbitrage — Python)

---

## Phase 4: Quant Beckman's Validation Framework
**Goal**: Centralised statistical validation shared across all modules.

**Alignment**: Quant Beckman's entire "Know" stage — the falsification engine.

Deliverables (in `quant-methods/src/quant_methods/validation/`):
- Vectorized `Backtester` class (pandas/numpy)
- Monkey Test (Monte Carlo permutation vs random signals)
- Probabilistic Sharpe Ratio (PSR)
- Deflated Sharpe Ratio (DSR)
- Block Bootstrap Resampling
- Combinatorial Purged Cross-Validation (CPCV)
- Walk-Forward Analysis (WFA)

---

## Phase 5: agentic-finance (Capstone)
**Goal**: Autonomous trading agents combining both mentors' work.

**Alignment**:
- **Quant Beckman**: The Research→Know→Do→Get loop as agent architecture
- **Bryan Downing**: The strategy conductor pattern (regime-dependent entry,
  toxicity veto, multi-exit logic)

Deliverables:
- **Research Agent** — Fetches data, runs models, generates LLM reports
  (Beckman's Research phase)
- **Execution Agent** — Reads signals, sizes positions, submits orders
  (Downing's conductor + Beckman's Know stage)
- **Risk Manager** — VaR, drawdown limits, kill switch (Downing's risk
  framework)
- **Portfolio Agent** — Capital allocation across strategies (Beckman's
  layered architecture)
- **Audit Trail** — Full decision logging with LLM explanations

---

## Dependency Graph

```
Phase 0 (Housekeeping)
    └─> Phase 1 (market-data DB)
           ├─> Phase 2a (quant-methods)
           │      └─> Phase 4 (Validation Framework)
           │             └─> Phase 5 (agentic-finance)
           └─> Phase 2b (hft-engine C++ core)
                  └─> Phase 3b (hft-engine advanced)
                         └─> Phase 3a (ml-for-finance)
                                └─> Phase 5 (agentic-finance)
```
