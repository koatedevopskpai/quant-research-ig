# Architecture Decision Records

The ADRs for this project live beside the code they describe, in each module's
`narrative.md`, rather than in this folder. This file is the index.

Each record states the context, the decision, and — importantly — the
trade-off that was accepted, so a reviewer can challenge the reasoning rather
than reverse-engineer it from the code.

## market-data

Source: [`market-data/narrative.md`](../market-data/narrative.md)

| ADR | Decision |
|---|---|
| [ADR-001](../market-data/narrative.md#adr-001-postgresql-for-time-series-storage) | PostgreSQL for time-series storage (BRIN on ordered timestamps) |
| [ADR-002](../market-data/narrative.md#adr-002-ig-api-v3-with-oauth-20-bearer-tokens) | IG API v3 with OAuth 2.0 Bearer tokens |
| [ADR-003](../market-data/narrative.md#adr-003-class-based-igclient-wrapping-httpx) | Class-based `IGClient` wrapping `httpx` |
| [ADR-004](../market-data/narrative.md#adr-004-lightstreamer-push-protocol-for-real-time-data) | Lightstreamer push protocol for real-time data |
| [ADR-005](../market-data/narrative.md#adr-005-ohlcv-storage-schema-design) | OHLCV storage schema design (one table per resolution) |
| [ADR-006](../market-data/narrative.md#adr-006-config-auto-discovery-via-directory-tree-walk) | Config auto-discovery via directory tree walk |

## quant-methods

Source: [`quant-methods/narrative.md`](../quant-methods/narrative.md)

| ADR | Decision |
|---|---|
| ADR-001 | Closed-form BSM + numerical methods for options |
| ADR-002 | GARCH(1,1) instead of higher-order GARCH |
| ADR-003 | Pure-Python matrix operations (no NumPy dependency) |
| ADR-004 | Closed-form min-variance and max-Sharpe without a QP solver |
| ADR-005 | PSR/DSR over simple Sharpe ratio for strategy selection |
| ADR-006 | Permutation test (monkey test) over a parametric null |
| ADR-007 | Circular block bootstrap over IID bootstrap |
| ADR-008 | Combinatorial purged CV over a standard time-series split |

## hft-engine

Source: [`hft-engine/narrative.md`](../hft-engine/narrative.md)

| ADR | Decision |
|---|---|
| ADR-001 | Separate book, feeds, and microstructure modules |
| ADR-002 | Linear search for price level index (K = 100) |
| ADR-003 | Lee-Ready (simplified) for trade classification |
| ADR-004 | Hawkes(1,1) over a Bayesian estimator for toxicity |
| ADR-005 | Context manager over decorator for latency instrumentation |
| ADR-006 | Almgren-Chriss over Kissell for market impact |
| ADR-007 | Geographic latency model over a discrete-event network sim |

## ml-for-finance

Source: [`ml-for-finance/narrative.md`](../ml-for-finance/narrative.md)

| ADR | Decision |
|---|---|
| ADR-7 | HMM over GMM for regime detection |
| ADR-8 | XGBoost over LightGBM for return forecasting |
| ADR-9 | Single-step LSTM over seq2seq |
| ADR-10 | Ensemble blending over stacking for sentiment |

## agentic-finance

Source: [`agentic-finance/narrative.md`](../agentic-finance/narrative.md)

| ADR | Decision |
|---|---|
| ADR-1 | Message bus over direct calls |
| ADR-2 | Optional LLM integration |
| ADR-3 | Three-pillar risk (VaR, drawdown, daily loss) |
| ADR-4 | Plugin model registry |
| ADR-5 | Toxicity veto in execution |

## Claim and evidence records

| Document | Purpose |
|---|---|
| [`phase-1-baseline.md`](phase-1-baseline.md) | Python 3.12 / uv baseline, CI and credential-free testing |
| [`phase-2-validation.md`](phase-2-validation.md) | Sharpe inference, purged fold APIs, temporal isolation |
| [`phase-3-market-data.md`](phase-3-market-data.md) | Provider boundary, replay, adapter fixes, live verification |
| [`phase-4-models.md`](phase-4-models.md) | Model contract, adapters, signal policy, demo API models |
| [`phase-5-execution.md`](phase-5-execution.md) | Order lifecycle, fills, ledger, reconciliation, protective exits |
| [`phase-6-risk.md`](phase-6-risk.md) | Ledger-driven risk, daily reset, idempotent halts, enforcement |
| [`phase-7-security-replay.md`](phase-7-security-replay.md) | API security posture, bus restriction, deterministic replay invariants |
| [`phase-8-experiment.md`](phase-8-experiment.md) | Fixed experiment, holdout result, promotion gates |
| [`persistence.md`](persistence.md) | Durable SQLite state and restart behaviour |
| [`W5-07.md`](W5-07.md) | Claim-sweep: what was removed from the README and why |
| [`../evidence/W5-07.md`](../evidence/W5-07.md) | Raw command output behind that sweep |
| [`../evidence/R-03.md`](../evidence/R-03.md) | Independent verification of phases 1–2, defects found, C++ build |
