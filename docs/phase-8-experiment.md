# Phase 8 — published research evidence

Status: **complete** (2026-10-26). One fixed experiment, one chronological
holdout, costs, a benchmark, and explicit promotion gates. **The strategy is not
promoted.**

## What the experiment is

`agentic_finance/experiment.py`

| Element | Choice |
|---|---|
| Data | deterministic daily prices, 6 years, two volatility regimes, seed `20261026` |
| Signal | momentum: position = sign of the trailing `lookback` return (causal) |
| Selection | lookback ∈ {5, 10, 20, 40} chosen on train+validation only (60% / 20%) |
| Holdout | final 20%, touched once after the parameter is frozen |
| Costs | 2 bps per unit turnover |
| Benchmark | buy-and-hold on the same instrument and holdout |
| Gates | Sharpe > 0; beats benchmark; PSR ≥ 0.95; DSR ≥ 0.95; monkey test p < 0.05; \|max drawdown\| ≤ 25% |

DSR receives the actual Sharpe ratios of the four lookback trials, so the
correction uses measured cross-trial dispersion rather than an invented variance.

## Result

Dataset digest `3d2f20a8f4222492` (seed `20261026`); selected lookback **5**.

| Holdout | Strategy | Buy & hold |
|---|---:|---:|
| Total return | 4.94% | 18.46% |
| Sharpe (annualised) | **0.29** | **0.70** |
| Observations | 302 | 302 |

| Gate | Value | Threshold | Pass |
|---|---:|---:|:--:|
| Positive Sharpe | 0.286 | > 0 | ✅ |
| Beats benchmark | 0.286 vs 0.704 | > benchmark | ❌ |
| PSR | 0.623 | ≥ 0.95 | ❌ |
| DSR (4 trials) | 0.515 | ≥ 0.95 | ❌ |
| Monkey test | p = 0.581 | < 0.05 | ❌ |
| Drawdown | within limit | ≤ 25% | ✅ |

**Promoted: false.** The momentum rule underperforms buy-and-hold on this dataset
and fails every statistical gate. That is the intended outcome: the gate is a
filter, not a formality, and a synthetic dataset with positive drift is a hard
benchmark for a long/flat timing rule.

## Reproduce

```bash
uv run --locked python -m agentic_finance.experiment --json evidence/experiment.json
```

The report records the git revision, seed, dataset digest, split dates,
configuration, environment versions and the full gate detail. Two runs with the
same seed produce identical numbers; changing the seed changes the digest.

## What this does and does not establish

**Does:** the method is reproducible from a documented revision, the holdout is
chronological and untouched during selection, costs are charged, a benchmark is
present, and promotion requires passing independent falsification tests.

**Does not:** demonstrate a tradeable edge. The dataset is synthetic and fixed by
seed. A single holdout is one sample and can pass by luck. There is no slippage
model beyond a flat bps cost, no borrow/financing, and one instrument. The live
signal path is also still in-sample (see `docs/phase-4-models.md`).

## Persistence

Every run can record its summary and equity curve in SQLite; see
[`persistence.md`](persistence.md). The experiment writes a JSON report to
`evidence/` regardless, so the result is reviewable without a database.

## Remaining work (beyond the eight phases)

- A recorded-IG replay with the real models, once the account is restored.
- CPCV / walk-forward over the strategy rather than a single holdout.
- A slippage and queue-position model, and a multi-instrument universe.
- Currency, borrow and financing costs.
