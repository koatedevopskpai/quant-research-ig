# IG Market Data Project — Systematic Quant Trading Platform

[![Python](https://img.shields.io/badge/python-3.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-in%20repair-orange)]()

A systematic trading platform built on [IG Group's API](https://labs.ig.com):
streaming market data, classical derivative pricing, a C++/pybind11 order book
and microstructure layer, statistically gated ML signals, and a multi-agent
decision loop with an immutable audit trail.

This repository is **mid-repair**. An independent review of revision `dacb89d`
reproduced a set of correctness, validation and control defects. The repairs are
being delivered in eight phases, each on its own branch with its own evidence
and its own exit criteria. This README states the plan and, for each phase, what
is actually demonstrated versus merely intended.

---

## Repair programme

| Phase | Scope | Status | Evidence |
|---|---|---|---|
| **1. Reproducible baseline** | supported interpreter, locked dependencies, credential-free install and collection, CI | **Complete** | [`docs/phase-1-baseline.md`](docs/phase-1-baseline.md) |
| **2. Correct statistical validation** | PSR/DSR units and moments, look-ahead removal, model fitting inside purged CPCV / walk-forward folds | **Complete on this branch** ([PR #2](https://github.com/koatedevopskpai/quant-research-ig/pull/2)) | [`docs/phase-2-validation.md`](docs/phase-2-validation.md) |
| **3. Market-data handling** | Lightstreamer callback and field mapping, L1 vs L2 semantics, per-instrument state isolation, opening prices, timestamps | **In progress** — provider layer, replay and adapter fixes done; runner routing and ingestion outstanding | [`docs/phase-3-market-data.md`](docs/phase-3-market-data.md) |
| **4. Research model contracts** | typed model input/output adapters, HMM/XGBoost registration, explicit regime → direction/confidence policy | Planned | — |
| **5. Paper execution** | order IDs, lifecycle states, simulated fills, position ledger, reductions/exits/reversals/partials, stop and target handling | Planned | — |
| **6. Risk controls** | P&L and exposure from the ledger, daily reset and capital updates, idempotent halt/resume, enforcement on every path | Planned | — |
| **7. Security and replay** | loopback-by-default API, authentication for shared deployments, restricted bus publication, deterministic end-to-end replay | Planned | — |
| **8. Published research evidence** | one fixed experiment with costs, benchmark, chronological holdout, promotion gates, reproducible from the documented revision | Planned | — |

### Exit criteria

A phase is only marked complete when its stated test holds — not when its code
lands.

1. **Baseline** — a clean checkout installs and runs the documented tests with
   no credentials.
2. **Validation** — independent reference calculations agree, and no future
   observation can change an earlier result.
3. **Market data** — recorded updates produce accurate, isolated per-instrument
   state.
4. **Models** — real models generate valid signals through the research agent.
5. **Execution** — holdings and cash reconcile with fills and costs.
6. **Risk** — breaches stop new risk, emergency halt works unconditionally, and
   repeated events are safe.
7. **Security and replay** — one deterministic replay passes the accounting,
   isolation and risk invariants.
8. **Evidence** — another person can reproduce the results from the documented
   revision.

Phases are sequential because each one depends on the last: replay in phase 7
needs the isolated data of 3, the ledger of 5 and the controls of 6, and phase 8
cannot publish anything it cannot reproduce.

---

## Verified state

Reproduced locally on 2026-10-26: Windows 11, CPython 3.12.15, `uv sync --locked`,
MSVC 19.51 for the C++ job. No `.env`, IG session, database or LLM key.

| Command | Result |
|---|---|
| `uv run --locked python scripts/check_baseline.py` | all five distributions import |
| `uv run --locked python scripts/check_validation.py` | published DSR reference `0.9003968344` |
| `uv run --locked pytest` | **504 passed**, 24 skipped (C++ extension unbuilt) |
| `uv run --locked pytest` after `build_cpp` | **528 passed**, 0 skipped |

| Module | Tests |
|---|---|
| `market-data` | 54 |
| `quant-methods` | 128 |
| `hft-engine` | 154 (24 need the compiled extension) |
| `ml-for-finance` | 25 |
| `agentic-finance` | 167 |

Live read-only provider checks (see [`docs/phase-3-market-data.md`](docs/phase-3-market-data.md)):
Binance `BTCUSDT` delivered 5,453 L2 events and 676 trades in 15 s; IG delivered
63 ticks in 30 s with 401–548 ms provider timestamps before the account was
suspended.

Exact commands, failure modes and outstanding defects:
[`evidence/R-03.md`](evidence/R-03.md). Remote CI has not been observed yet; the
first runs are on this branch.

Statistical repairs were checked against an external reference, not just against
themselves: the tests reproduce the published Bailey & López de Prado (2014)
worked example (benchmark `0.1132`, DSR `0.9004`, `0.9505` for 46 trials), and a
zero-skill calibration over 1000 random walks dropped from **47.5%** false
positives under the old formula to **5.9%**.

---

## Quick start

Requires `uv` 0.13.5 and CPython 3.12.x. `uv` installs the interpreter if needed;
no environment activation or `PYTHONPATH` is required.

```bash
git clone https://github.com/koatedevopskpai/quant-research-ig.git
cd quant-research-ig
python -m pip install uv==0.13.5

uv sync --locked                              # 108 pinned packages + 5 local members
uv run --locked python scripts/check_baseline.py
uv run --locked pytest                        # credential-free
```

Optional C++ accelerator (needs a C++17 compiler, `pybind11`, `cmake`):

```bash
uv pip install pybind11 cmake
./hft-engine/cpp/build_cpp.sh                 # Windows: .\hft-engine\cpp\build_cpp.ps1
uv run --no-sync --locked pytest hft-engine/tests/ -q -rs
```

Provider credentials are needed only for live integrations. Copy
`.env.template` to `.env`, fill in demo credentials from
[labs.ig.com](https://labs.ig.com), and set `QUANT_LOAD_DOTENV=1` if you need
`.env` discovery outside pytest. Every test runs offline.

IG is not required for development or verification:

```bash
# Real L2 depth + trades, no account or key
python scripts/provider_smoke.py binance --symbol BTCUSDT --seconds 20

# Historical FX bid/ask ticks, no account
python scripts/provider_smoke.py dukascopy --instrument EURUSD --date 2026-10-20 --hour 10
```

```python
# Deterministic playback of a captured session (what CI uses)
from ig_market_data.providers import ReplayProvider

provider = ReplayProvider.from_jsonl("fixtures/session.jsonl")
provider.connect()
provider.subscribe("CS.D.EURUSD.CFD.IP", "l1", print)
provider.replay()          # speed=10 to pace by provider time
```

---

## Project structure

```
quant-research-ig/
├── market-data/          IG client, provider layer (IG / Dukascopy / Binance / replay), PostgreSQL
├── quant-methods/        Black-Scholes, trees, Monte Carlo, GARCH, portfolio opt., validation
├── hft-engine/           C++ order book (pybind11) + microstructure + execution simulation
├── ml-for-finance/       HMM regimes, XGBoost forecasting, LSTM, sentiment models
├── agentic-finance/      Research, risk, portfolio and execution agents + audit trail
├── benchmarks/           Order book microbenchmark (Python vs C++, same API)
├── docs/                 ADR index + phase notes
├── evidence/             Reproducible output behind the claims in this file
├── scripts/              Baseline and validation verification scripts
├── README.md             This file
└── STRATEGY.md           Full phased implementation roadmap
```

---

## Methodology & influences

### Bryan Downing — HFT and market microstructure

*Founder of QuantLabsNet.com*

Downing's C++ HFT architecture is the blueprint for `hft-engine`. Implemented
here:

- **Hidden Markov Model regime detection** (`ml-for-finance/regime`) — low /
  medium / high volatility states fitted with `hmmlearn`.
- **Hawkes(1,1) order-flow toxicity** (`hft-engine/toxicity`) — self-exciting
  intensity estimated by maximum likelihood and used as a veto on new exposure.
  [`ADR-004`](hft-engine/narrative.md#adr-004-hawkes11-over-bayesian-for-toxicity)
  records why this is Hawkes rather than Downing's Bayesian formulation.
- **C++ hot-path structures via pybind11** — sorted-vector L1/L2 book, string
  field tick parsers, microstructure aggregates. Single-threaded, called from
  Python; [`ADR-002`](hft-engine/narrative.md#adr-002-linear-search-for-price-level-index-k--100)
  covers the data-structure choice.
- **Stop-loss / take-profit sizing** in the execution agent.

**Not implemented, and not claimed:** reinforcement-learning adaptation of the
HMM transition matrix, and toxicity-spike or time-based exits (toxicity vetoes
entries, it does not force exits).

### Quant Beckman — quantitative research and falsification

*Author of "Quantitative Research Foundations"*

Beckman's **Research → Know → Do → Get** cycle and falsification-first stance
drive `quant-methods/validation`:

- **Probabilistic Sharpe Ratio** and **Deflated Sharpe Ratio** — per-period
  inference with non-normal moments and multiple-testing correction.
- **Monkey Test** — permutation of directional exposure against a random-sign
  null.
- **Circular block bootstrap** — preserves serial dependence.
- **Combinatorial Purged Cross-Validation** — a fresh model is fitted inside
  each purged fold; label intervals overlapping a test window are removed.
- **Walk-Forward Analysis** — chronological refits with an embargo gap.

These are tools, and they gate signals before an execution path. They have not
been applied to a live P&L record, because there is no live P&L record, and no
strategy has yet been promoted through them end to end (phase 8).

---

## Architecture highlights

| Decision | Choice | Rationale / status |
|---|---|---|
| **Data storage** | PostgreSQL + BRIN indexes | SQL JOINs across instruments; timestamp range indexing. No throughput comparison established here. |
| **API auth** | IG API v3 OAuth 2.0 Bearer tokens | v3 returns `oauthToken.access_token` (1800 s TTL, refreshed at 1400 s). Lightstreamer still requires the legacy v2 CST/XST pair; `stream.py` documents this. |
| **Real-time streaming** | Provider layer over Lightstreamer (IG), Binance WebSocket, Dukascopy files, and JSONL replay | Adapters and field mapping are fixed and test-covered; the IG transport is unverified while the account is suspended. IG has no depth feed — L2 comes from Binance. See [`docs/phase-3-market-data.md`](docs/phase-3-market-data.md). |
| **Quant research** | Python-first (pandas, numpy, scipy) | Rapid iteration; broad financial computing ecosystem. |
| **Hot-path performance** | C++ via pybind11 | Sorted-vector L2 book + tick parsers, single-threaded, called from Python. Measured on a 100-level book: `apply_diff` 0.9 µs vs 2.9 µs in Python (3.2×); `l1_snapshot` is **slower** in C++ (2.7 vs 1.4 µs) because the boundary dominates. See [`benchmarks/`](benchmarks/bench_order_book.py). |
| **Strategy validation** | PSR, DSR, purged model CV, walk-forward refits | Tools implemented and reference-tested; automatic promotion gates are phase 8. |
| **Agent orchestration** | Plain-Python agent classes over an in-process `MessageBus` | Research → Risk → Portfolio → Execution, every transition written to an `AuditTrail`. `openai` (`gpt-4o-mini`) is optional in the research agent. **No LangGraph dependency.** |

---

## Current status

| Module | Phase | State |
|---|---|---|
| `market-data` | 1–3 | IG v3 OAuth client, PostgreSQL schema with BRIN, Lightstreamer code. Streaming callback/field mapping and L1-as-L2 handling remain defective (phase 3). No live run. |
| `quant-methods` | 1–2 | Pricing, GARCH, portfolio optimisation, and the reference-tested validation suite. |
| `hft-engine` | 1–2 | Python book/feeds/microstructure plus a compiled C++ accelerator: order book, tick parsers, toxicity detector, execution/colocation/latency/market-sim. |
| `ml-for-finance` | 2 | HMM regimes, XGBoost with chronological partitions, LSTM, sentiment models. Temporal leakage removed. |
| `agentic-finance` | 1 | Four agents, MessageBus, AuditTrail, FastAPI REST + WebSocket API, Streamlit and terminal dashboards. Execution accounting and risk wiring are not reconciled (phases 5–6). |

---

## Known limitations

The defects below were reproduced in the 2026-10 review and are **not** fixed by
phases 1–2. Each is assigned to the phase that will fix it.

- **Streaming adapters** *(phase 3, code fixed / live unverified)* — `onItemUpdate`
  and `getFields()` are implemented and the tick-casing bug is fixed, but IG
  delivery is unverified while the account is suspended. An IG L1 `MARKET:` item
  delivered nothing even with the corrected callback and no subscription error;
  cause unknown. See `docs/phase-3-market-data.md`.
- **L1 vs L2** *(phase 3)* — successive top-of-book quotes are applied as
  incremental depth updates, so stale levels persist, and the live L1
  subscription supplies none of the sizes the book update assumes.
- **Instrument isolation** *(phase 3)* — the runner keeps one book, analyser and
  price cache for all EPICs instead of routing on `msg.epic`.
- **Ingestion** *(phase 3)* — opening prices are filled from closing prices, and
  timezone-naive timestamps are not normalised to UTC.
- **Model contracts** *(phase 4)* — no models are registered by the runner, the
  HMM adapter calls `predict()` without `X`, the XGBoost adapter reads a
  non-existent `forecast` attribute, and regime labels are names where numeric
  states are assumed. Confidence therefore collapses to zero.
- **Execution accounting** *(phase 5)* — holdings update on order creation, not
  on fill; orders carry no ID or lifecycle; position limits computed from
  absolute holdings block risk-reducing exits.
- **Risk controls** *(phase 6)* — `risk.kill` is republished by its own handler,
  so halts recurse; `resume()` publishes the kill topic again; P&L is derived
  from price level rather than mark-to-market holdings.
- **API surface** *(phase 7)* — binds `0.0.0.0`, mutation endpoints are
  unauthenticated, CORS accepts all origins. Run on loopback only.
- **Claims not implemented** — no LangGraph, no reinforcement learning, no
  toxicity-spike or time-based exits, and the C++ layer is a sorted vector, not
  lock-free, without GIL release.

---

## Evidence and records

| Document | Purpose |
|---|---|
| [`docs/phase-1-baseline.md`](docs/phase-1-baseline.md) | Interpreter, lockfile, credential-free testing, CI |
| [`docs/phase-2-validation.md`](docs/phase-2-validation.md) | Sharpe inference conventions, purged fold APIs, temporal isolation |
| [`docs/phase-3-market-data.md`](docs/phase-3-market-data.md) | Provider boundary, replay, adapter fixes, live verification results |
| [`docs/README.md`](docs/README.md) | Index of architecture decision records (they live in each module's `narrative.md`) |
| [`evidence/R-03.md`](evidence/R-03.md) | Independent verification of phases 1–2, defects found, C++ build |
| [`evidence/bench-order-book.json`](evidence/bench-order-book.json) | Raw benchmark output |
| [`evidence/W5-07.md`](evidence/W5-07.md) | Historical claim sweep on the earlier revision |

---

## Implementation roadmap

[`STRATEGY.md`](STRATEGY.md) holds the original eight-phase build roadmap for the
platform itself. The repair programme above is the current gating sequence; the
roadmap describes what the platform is intended to become once the defects are
closed.

## License

MIT — see [LICENSE](LICENSE) for details.
