# Phase 2 — Statistical validation and temporal isolation

This change builds on the Phase 1 Python 3.12.15 workspace. It corrects Sharpe
inference, removes warm-up/scaler leakage, and replaces return-subset summaries
with model-fitting CPCV and walk-forward evaluations.

## Sharpe inference conventions

`ProbabilisticSharpeRatio.compute` accepts equally spaced simple returns, subtracts
annual arithmetic risk-free return divided by periods per year, and reports
annualised Sharpe/threshold values. Its test statistic uses per-observation Sharpe
and an observation count at the same frequency. The default benchmark is zero
excess Sharpe, not the risk-free return itself. Sample volatility uses `ddof=1`;
empirical central moments use `bias=True`. `kurtosis` now means ordinary kurtosis
(Gaussian = 3). Constant, nonfinite, or malformed return series raise `ValueError`
instead of being assigned fabricated confidence. Invalid sampling variance is
rejected rather than clamped to a tiny positive value.

The formulas and fixed numerical reference come from Bailey and Lopez de Prado,
[The Deflated Sharpe Ratio (2014)](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf),
Eq. 2 and the numerical example on pages 9–10. The native-frequency benchmark is
approximately 0.1132 and its DSR approximately 0.9004. Tests reproduce these and
the example's 46-trial result, 0.9505.

DSR requires dispersion across the independent trials searched, not a sampling
variance manufactured from the winning strategy alone. Supply either all trial
Sharpes or an explicitly measured variance and trial count:

```python
from quant_methods.validation.psr import ProbabilisticSharpeRatio
from quant_methods.validation.dsr import DeflatedSharpeRatio

psr = ProbabilisticSharpeRatio().compute(strategy_returns, threshold_sharpe=0.0)
dsr = DeflatedSharpeRatio().compute(
    strategy_returns,
    trial_sharpes=annualised_sharpes_from_all_independent_trials,
)
# Alternatively: num_trials=100, trial_sharpe_variance=measured_annual_variance
```

Public DSR Sharpe values are annualised; its variance input is annualised Sharpe
squared. The single-trial case equals PSR without evaluating an infinite normal
quantile. Both variance inputs at once, invalid counts, and undisclosed multi-trial
dispersion are rejected. The empirical variance uses `ddof=1` and the null mean is
zero. Trial dependence/effective trial count must be assessed by the researcher;
this implementation does not estimate it. PSR likewise does not correct serial
dependence, variable-frequency sampling, or repeated reuse of a final holdout.

## Model validation API migration

The old `cv.run(strategy_returns)` and WFA `train_fn`/scalar `test_fn` APIs are
replaced deliberately. They could not verify that the reported returns came from
a model fitted on the retained training data. Both evaluators now use:

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from quant_methods.validation.cpcv import CombinatorialPurgedCV
from quant_methods.validation.wfa import WalkForwardAnalysis

def model_factory():
    return make_pipeline(StandardScaler(), Ridge())  # a fresh model per fold

def returns_fn(realised_log_returns, predictions):
    return np.sign(predictions) * np.expm1(realised_log_returns)

cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, embargo=5)
result = cv.run(x, y, model_factory, returns_fn, label_end=label_end)

wfa = WalkForwardAnalysis(train_size=250, test_size=50, gap=5, mode="rolling")
walk = wfa.run(x, y, model_factory, returns_fn, label_end=label_end)
```

Here `x` is a numeric feature matrix and `y` a one-dimensional target array with
matching observation rows. `label_end[i]` is the inclusive integer row position
where observation i's outcome becomes known. Ends must lie between i and the
final row. Exclude incomplete labels before evaluation; map event ends to the
aligned row coordinate system. If event ends extend beyond the final input row,
they can be clipped there for purging because there are no later training starts
in that evaluation universe. Omitting `label_end` means instantaneous labels;
forward-looking labels require their actual horizons.

Each fold instantiates the factory, fits on training features/labels, predicts
only held-out features, and passes predictions plus realised outcomes to
`returns_fn`. That callback must return one finite strategy return per test row;
it is responsible for units, signal timing, and costs. The example omits costs.
All fitted preprocessing belongs in the factory's pipeline. Features computed
before splitting must already be causal. The evaluator cannot certify a callback
that deliberately uses future information to choose positions.

CPCV removes every training label interval overlapping any test label interval.
Embargo removes starts immediately after each test outcome interval. Size the
embargo for feature memory/dependence as appropriate. A deterministic combination
cap avoids materialising the full combinatorial search. Each result exposes
training/test indices, held-out predictions, returns, and scores for auditing.
CPCV may train on later observations, as combinatorial CV does; WFA is the
chronological evaluation. CPCV reports fold scores, not reconstructed combinatorial
backtest paths. Overlapping folds are not independent significance trials.
`is_significant` is retained as a compatibility field and is always False;
`lower_quantile` and `positive_fold_fraction` are descriptive. The old
`significance_level` argument selects the descriptive quantile only.

WFA fits each rolling or anchored window, purges labels not yet known at the test
start, applies its training gap, and concatenates actual policy returns from
nonoverlapping test windows. It never mutates configured train size. Optional
`min_train_size` controls the initial nominal training window before label purging;
rolling windows then grow to the configured cap. `is_robust` is a descriptive
positive-window heuristic, not an inferential significance result.

## Causal sizing and ML preprocessing

- Volatility-target positions stay flat until 20 past price-return observations
  are available. They never fill warm-up volatility from the full sample.
- LSTM fitting determines the chronological split first. Mean/std are fitted only
  through the last training target; validation sequences can use historical
  training context, but validation targets never affect the scaler. Prediction
  reuses the fitted scaler. Predictions over training history remain in-sample.
- XGBoost uses separate chronological training, early-stopping validation, and
  outer holdout sets. Boundaries are fixed on raw observation positions, so
  dropping future feature rows cannot move a historical cutoff. Labels reaching
  the next partition are purged. `target_horizon=h` now consistently predicts
  cumulative log return `log(price[t+h]/price[t])`; for h>1 this deliberately
  changes the previous single-return-at-offset target. The configured early
  stopping actually applies and never uses the outer test set.

## Verification and scope

Reference tests cover the published numerical case, manually calculated moments,
calendar-unit invariance, a seeded zero-skill false-positive smoke check, exact
purging/embargo indices, fold-local scaling, fitted-policy returns, repeatable
anchored runs, and invariance to future edits. XGBoost tests verify that editing
only its outer holdout leaves the fitted booster unchanged. LSTM tests verify
that validation edits leave training tensors and fitted scaling unchanged.

```sh
uv sync --locked
uv run --locked python scripts/check_validation.py
uv run --locked pytest quant-methods/tests ml-for-finance/tests -q
uv run --locked pytest --junitxml=test-results/junit.xml
```

Live data integration, execution accounting, runtime signal adapters, risk halts,
and strategy-promotion gates remain assigned to later phases. Sentiment-ensemble
weight calibration and timestamp alignment are not repaired here; validating an
ensemble requires evaluating its full fitted pipeline on a separate holdout.
These components provide validation tools; they do not establish a profitable strategy.

## Local verification — 26 October 2026

Local branch: `phase-2-statistical-validation`, built on the Phase 1 scaffold.
The Python baseline and dependency lock are unchanged.

- Full suite: **472 passed, 24 skipped, 145 warnings in 52.13 seconds**.
  All 496 tests collected. Skips remain the optional unbuilt C++ accelerator;
  warnings are existing pandas/hmmlearn deprecations and are not suppressed.
- The initial focused implementation run passed 75 tests; the final full suite
  also includes the subsequent frequency-conversion and zero-skill smoke checks.
- The deterministic verification script reports modest-sample PSR
  `0.6245365683`, published-case DSR `0.9003968344`, 15 CPCV model fits and
  three chronological WFA model fits.
- Ruff passed for the five new/reworked validation modules, new/reworked
  validation tests, temporal ML tests, and the evidence script. This is not a
  whole-repository lint claim. `git diff --check` passed.
- No provider credentials or live market/trading services were used. The first
  full attempt stalled in Windows asyncio's local socket-pair setup under the
  execution sandbox's network restriction. After granting networking for that
  local event-loop requirement, the complete run passed; provider HTTP remains
  mocked by the Phase 1 fixtures.
- Remote GitHub Actions have not run. Windows local evidence is not a Linux CI result.

## Applying the delivered changes

`phase-2-solution.zip` contains complete source with both phases, without Git
metadata, credentials or a virtual environment. For an existing checkout, choose
the patch matching its starting point:

```sh
# After applying the Phase 1 scaffold:
git apply --check phase-2-incremental.patch
git apply phase-2-incremental.patch

# OR from original reviewed revision dacb89d (includes both phases):
git apply --check phase-2-cumulative.patch
git apply phase-2-cumulative.patch
```

Use one path, then run `uv sync --locked` and the verification commands above.
Both delivered patches are checked against their respective clean baselines.
The evidence JSON and JUnit XML accompany the source archive.
