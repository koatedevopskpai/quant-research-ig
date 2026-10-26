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
delivered in eight phases, each on its own branch with its own evidence and its
own exit criteria. This README states the plan and, for each phase, what is
actually demonstrated versus merely intended.

**This branch contains phase 1 only.** It proves the workspace installs, collects
and tests without credentials. It does not correct any statistical, streaming or
trading behaviour — that is phase 2 onward (see the table).

---

## Repair programme

| Phase | Scope | Status | Evidence |
|---|---|---|---|
| **1. Reproducible baseline** | supported interpreter, locked dependencies, credential-free install and collection, CI | **Complete on this branch** | [`docs/phase-1-baseline.md`](docs/phase-1-baseline.md) |
| **2. Correct statistical validation** | PSR/DSR units and moments, look-ahead removal, model fitting inside purged CPCV / walk-forward folds | In review — [PR #2](https://github.com/koatedevopskpai/quant-research-ig/pull/2), **not in this branch** | [`docs/phase-2-validation.md`](https://github.com/koatedevopskpai/quant-research-ig/blob/phase-2-validation/docs/phase-2-validation.md) |
| **3. Market-data handling** | Lightstreamer callback and field mapping, L1 vs L2 semantics, per-instrument state isolation, opening prices, timestamps | Planned | — |
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

## Verified state (phase 1)

Reproduced locally on 2026-10-26: Windows 11, CPython 3.12.15,
`uv sync --locked`, 108 pinned packages, no `.env`, IG session, database or LLM
key.

| Command | Result |
|---|---|
| `uv run --locked python scripts/check_baseline.py` | all five distributions import |
| `uv run --locked pytest --collect-only -q` | 467 tests across all five modules |
| `uv run --locked pytest` | **443 passed**, 24 skipped |

| Module | Tests |
|---|---|
| `market-data` | 26 |
| `quant-methods` | 106 |
| `hft-engine` | 150 |
| `ml-for-finance` | 18 |
| `agentic-finance` | 167 |

The 24 skips are the optional C++ accelerator tests: this phase does not build
the extension. A later change makes it buildable and adds a CI job that fails if
those tests skip (see PR #2).

**Not verified on this branch:** statistical inference, leakage behaviour, model
fold fitting, streaming adapters, execution accounting and risk controls. Those
are phases 2–6; a green run here is infrastructure evidence, not evidence that
the strategies or integrations are correct.

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

Provider credentials are needed only for live integrations. Copy
`.env.template` to `.env`, fill in demo credentials from
[labs.ig.com](https://labs.ig.com), and set `QUANT_LOAD_DOTENV=1` if you need
`.env` discovery outside pytest. Every test runs offline; the root `conftest.py`
disables `.env` loading and blocks external `httpx` transports.

---

## Project structure

```
quant-research-ig/
├── market-data/          IG API client + Lightstreamer streaming + PostgreSQL storage
├── quant-methods/        Black-Scholes, trees, Monte Carlo, GARCH, portfolio opt.
├── hft-engine/           Order book reconstruction, microstructure analytics, execution sim.
├── ml-for-finance/       HMM regimes, XGBoost forecasting, LSTM, sentiment models
├── agentic-finance/      Research, risk, portfolio and execution agents + audit trail
├── docs/                 ADR index + phase notes
├── evidence/             Reproducible output behind the claims in this file
├── scripts/              Baseline verification script
├── README.md             This file
└── STRATEGY.md           Full phased implementation roadmap
```

---

## Methodology & influences

### Bryan Downing — HFT and market microstructure

*Founder of QuantLabsNet.com*

Downing's C++ HFT architecture is the blueprint for `hft-engine`. The module
contains an HMM regime detector (`ml-for-finance/regime`), a Hawkes(1,1)
order-flow toxicity estimator used as a veto on new exposure
([`ADR-004`](hft-engine/narrative.md#adr-004-hawkes11-over-bayesian-for-toxicity)
records why Hawkes rather than a Bayesian formulation), C++ book and tick-parser
extensions behind pybind11, and stop-loss / take-profit sizing in the execution
agent.

**Not implemented, and not claimed:** reinforcement-learning adaptation of the
HMM transition matrix, and toxicity-spike or time-based exits.

### Quant Beckman — quantitative research and falsification

*Author of "Quantitative Research Foundations"*

Beckman's falsification-first stance drives `quant-methods/validation`: the
Probabilistic and Deflated Sharpe Ratios, a permutation (monkey) test, circular
block bootstrap, combinatorial purged cross-validation and walk-forward
analysis.

**Caveat for this branch:** the units, moment conventions and fold-fitting
behaviour of those tools are the subject of phase 2, where they were found
incorrect and were repaired. Do not rely on validation output from this revision.

---

## Architecture highlights

| Decision | Choice | Rationale / status |
|---|---|---|
| **Data storage** | PostgreSQL + BRIN indexes | SQL JOINs across instruments; timestamp range indexing. No throughput comparison established here. |
| **API auth** | IG API v3 OAuth 2.0 Bearer tokens | v3 returns `oauthToken.access_token`; Lightstreamer still requires the legacy v2 CST/XST pair. No live run. |
| **Real-time streaming** | Lightstreamer push protocol | L1/L2 subscriptions with reconnect support — **adapter defects outstanding** (phase 3). |
| **Quant research** | Python-first (pandas, numpy, scipy) | Rapid iteration; broad financial computing ecosystem. |
| **Hot-path performance** | C++ via pybind11 | Sorted-vector L2 book + tick parsers, single-threaded, called from Python. **Not benchmarked or built on this branch.** |
| **Strategy validation** | PSR, DSR, purged CV, walk-forward | Present but incorrect at this revision — repaired in phase 2 (PR #2). |
| **Agent orchestration** | Plain-Python agent classes over an in-process `MessageBus` | Research → Risk → Portfolio → Execution with an `AuditTrail`. `openai` (`gpt-4o-mini`) is optional. **No LangGraph dependency.** |

---

## Known limitations

The defects below were reproduced in the 2026-10 review and are **not** fixed by
phase 1. Each is assigned to the phase that will fix it.

- **Statistical validation** *(phase 2)* — PSR used an annualised Sharpe in a
  per-period variance term and excess kurtosis in a raw-kurtosis formula; DSR
  inherited the error; CPCV computed a training mask and never fitted a model;
  WFA discarded the fitted model; LSTM scaling and volatility-target sizing used
  future observations.
- **Streaming adapters** *(phase 3)* — the bridge implements `on_item_update`
  while the SDK calls `onItemUpdate`; it enumerates attributes of `getValue`
  rather than the update's fields; the tick subscription requests `LTP`/`LTV`
  while the mapping expects `Ltp`/`Ltv`.
- **L1 vs L2** *(phase 3)* — top-of-book quotes are applied as incremental depth
  updates, so stale levels persist; the live L1 subscription supplies no sizes.
- **Instrument isolation** *(phase 3)* — one book, analyser and price cache for
  all EPICs in the runner.
- **Ingestion** *(phase 3)* — opening prices filled from closing prices;
  timezone-naive timestamps not normalised to UTC.
- **Model contracts** *(phase 4)* — no models registered by the runner; HMM
  adapter calls `predict()` without `X`; XGBoost adapter reads a non-existent
  `forecast` attribute; regime labels are names where numeric states are assumed.
- **Execution accounting** *(phase 5)* — holdings update on order creation, not
  fill; no order IDs or lifecycle; position limits block risk-reducing exits.
- **Risk controls** *(phase 6)* — `risk.kill` republished by its own handler, so
  halts recurse; `resume()` republishes the kill topic; P&L from price level
  rather than mark-to-market holdings.
- **API surface** *(phase 7)* — binds `0.0.0.0`, unauthenticated mutation
  endpoints, permissive CORS. Run on loopback only.
- **Claims not implemented** — no LangGraph, no reinforcement learning, no
  toxicity-spike or time-based exits, and the C++ layer is a sorted vector, not
  lock-free, without GIL release.

---

## Evidence and records

| Document | Purpose |
|---|---|
| [`docs/phase-1-baseline.md`](docs/phase-1-baseline.md) | Interpreter, lockfile, credential-free testing, CI |
| [`docs/README.md`](docs/README.md) | Index of architecture decision records (they live in each module's `narrative.md`) |
| [`evidence/W5-07.md`](evidence/W5-07.md) | Historical claim sweep on the earlier revision |
| [PR #2](https://github.com/koatedevopskpai/quant-research-ig/pull/2) | Phase 2 validation repair, independent verification, C++ build |

---

## Implementation roadmap

[`STRATEGY.md`](STRATEGY.md) holds the original eight-phase build roadmap for the
platform itself. The repair programme above is the current gating sequence.

## License

MIT — see [LICENSE](LICENSE) for details.
