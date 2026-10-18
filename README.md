# IG Market Data Project — Systematic Quant Trading Platform

[![Python](https://img.shields.io/badge/python-3.14-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-alpha-yellow)]()

Historical saved coverage badge assets remain under `.github/badges/`; their values
are not a fresh coverage measurement. See the claim-sweep evidence for attempted
test commands and their exact results.

A reference collection of IG market-data integration code, quantitative research,
execution simulation, ML models and agent components. The current full-suite
attempt fails during collection; no passing total or coverage is asserted by this
wording sweep. See [verification and blockers](evidence/W5-07.md). No live market,
provider or trading-execution run was performed for this review.

Designed as a portfolio project demonstrating the quantitative researcher skill
set: **data engineering, derivative pricing, statistical validation,
market microstructure, machine learning, and agentic systems** — all grounded
in the methodologies of two mentors whose work defines modern quant finance.

---

## Project Structure

```
quant-research-ig/
├── market-data/          IG API client + Lightstreamer streaming + PostgreSQL storage
├── quant-methods/        Black-Scholes, binomial trees, Monte Carlo, GARCH, portfolio opt.
├── hft-engine/           Order book reconstruction, microstructure analytics, execution sim.
├── ml-for-finance/       HMM regimes, XGBoost forecasting, LSTM, sentiment-augmented models
├── agentic-finance/      Autonomous trading agents with LLM reasoning + audit trails
├── docs/                 Architecture Decision Records per module
├── README.md             This file
└── STRATEGY.md           Full phased implementation roadmap
```

## Methodology & Influences

### Bryan Downing — HFT & Market Microstructure

*Founder of QuantLabsNet.com*

Bryan Downing's C++ HFT architecture provides the blueprint for the
`hft-engine` module. His Ether HFT bot design combines a **Hidden Markov
Model** for regime detection (low/med/high volatility), a **Bayesian Order
Flow Toxicity Detector** using Hawkes processes to sniff out informed trading,
**Reinforcement Learning** to adapt the HMM transition matrix from profitable
vs. losing trades, and a multi-layered exit strategy (take-profit, stop-loss,
toxicity spike exit, time-based). These concepts are implemented in Python
with C++ hot-path extensions via pybind11 where microsecond latency is
critical.

**Key references:**
- Downing, B. "Deconstructing an Advanced Ether High Frequency Trading Bot:
  A C++ Breakdown" — *QuantLabsNet.com* (2025)
- Downing, B. "Unlocking AI High Frequency Trading C++ and the Future of
  Quantitative Finance" — *QuantLabsNet.com* (2025)
- [LinkedIn](https://www.linkedin.com/in/bryan-downing-165a1523)

### Quant Beckman — Quantitative Research & Statistical Validation

*Author of "Quantitative Research Foundations"*

Quant Beckman's **Research → Know → Do → Get** cycle provides the rigorous
statistical framework for the `quant-methods`, `ml-for-finance`, and
`agentic-finance` modules. His falsification-first approach ensures every
strategy survives a gauntlet of tests before deployment: the **Monkey Test**
(Monte Carlo permutation vs. random signals), **Probabilistic Sharpe Ratio**
(accounting for non-normal return distributions), **Deflated Sharpe Ratio**
(correcting for multiple testing bias), **Combinatorial Purged
Cross-Validation** (preventing time-series leakage), and **Walk-Forward
Analysis** (testing for parameter drift across regimes).

**Key references:**
- Beckman, Q. "Quantitative Research Foundations" (2025)
- Beckman, Q. "Trading the Breaking" Substack — [quantbeckman.com](https://quantbeckman.com)
- [LinkedIn](https://www.linkedin.com/in/quantbeckman)

## Architecture Highlights

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Data storage** | PostgreSQL + BRIN indexes | SQL JOINs across instruments and indexing for timestamp queries; no throughput comparison established here |
| **API auth** | IG API v3 OAuth 2.0 Bearer tokens | IG deprecated v2; simpler flow than CST/XST |
| **Real-time streaming** | Lightstreamer push protocol | Push updates and reconnect support; no latency comparison reproduced here |
| **Quant research** | Python-first (pandas, numpy, scipy) | Rapid iteration; vast financial computing ecosystem |
| **Hot-path performance** | C++ via pybind11 | Order book + feed handler; lock-free structures in C++, orchestration in Python |
| **Strategy validation** | Beckman's falsification gauntlet | PSR, DSR, CPCV, WFA — every strategy statistically gated |
| **Agent orchestration** | LangGraph + OpenAI GPT-4o-mini | State-machine agents with auditable reasoning |

## Current Status

| Module | Phase | Status | Tests |
|--------|-------|--------|-------|
| `market-data` | Phase 1 | IG API authentication and L1/L2 streaming integration code; no live run inspected in this sweep | — |
| `quant-methods` | Phase 2a | Black-Scholes, Greeks, Monte Carlo, GARCH, portfolio optimisation | — |
| `hft-engine` | Phase 2b | Order book (C++ pybind11), microstructure analyser, toxicity detector, execution sim, colocation sim, latency monitor, market sim feed | Not reproduced in this sweep |
| `ml-for-finance` | Phase 3a | HMM regimes, XGBoost forecasting, LSTM, sentiment models | — |
| `agentic-finance` | Phase 5 | Research, execution, risk, portfolio agents; LLM research reports; MessageBus; AuditTrail; FastAPI REST + WebSocket API; Streamlit dashboard; terminal dashboard (rich) | Not reproduced in this sweep |

> The old static test/coverage headline is removed. Re-run the documented suites
> for the selected revision and platform; test counts and coverage are separate measures.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/koatedevopskpai/quant-research-ig.git
cd quant-research-ig

# 2. Configure credentials
cp .env.template .env
# Edit .env with your IG API credentials (obtain from labs.ig.com)

# 3. Install sub-modules
pip install -e agentic-finance/
pip install -e hft-engine/

# 4. Run all tests
pytest agentic-finance/tests/ hft-engine/tests/

# 5. Regenerate coverage badges
python .github/badges/generate_badges.py
```

## Implementation Roadmap

See [`STRATEGY.md`](STRATEGY.md) for the full phased roadmap across all
8 phases, from project housekeeping through to autonomous agent deployment.

## License

MIT — see [LICENSE](LICENSE) for details.
