# Phase 4 — research model contracts

Status: **complete** (2026-10-26). The research path now operates on a typed
contract, the real models are registered and exercised end to end, and the
direction/confidence policy is defined in one place.

## Defects fixed

All five were reproduced in the 2026-10 review and are pinned by tests.

| Defect | Effect | Fix |
|---|---|---|
| `_run_regime` called `.pooled_mean()` on its argument | `AttributeError` inside a `try/except` → `{"error": ...}`, never a signal | `RegimeAdapter` converts prices to the HMM's expected array itself |
| `_run_forecast` read `result.forecast` | The result exposes `forecasts` (plural), so every successful fit yielded `[]` → `prediction_mean = 0.0` | `ForecastAdapter` reads `forecasts` and raises if none are finite |
| Regime labels used as direction (`regime_labels[-1] == 1`) | Labels are strings ("Low Vol"), so the comparison was always false → every signal was `-1` | Direction comes only from a return forecast; a regime returns direction `0.0` |
| Confidence taken from `prediction_mean` | Regime results have no such field → confidence `0.0` | Confidence is `|expected_return| / (2 × recent volatility)`, clipped |
| No models registered by the runner | `research()` always returned "model not registered" | `setup()` registers `ForecastAdapter` and `RegimeAdapter` |

## The contract

`agentic_finance/model_contract.py`

```python
ModelOutput(
    model: str,
    direction: float,          # -1.0 / 0.0 / +1.0
    confidence: float,         # 0.0 .. 1.0
    expected_return: float | None,
    regime: int | None,        # numeric state, never a label
    regime_label: str | None,
    diagnostics: dict,
)
```

Validation is at construction: finite values, `direction ∈ [-1, 1]`,
`confidence ∈ [0, 1]`, and **a non-zero direction must carry an
`expected_return`**. That last rule makes it impossible to reintroduce
"direction from a regime label" without an exception.

`ModelAdapter` is the only accepted registration: `name`, `fit(prices)`,
`predict(prices) -> ModelOutput`. `register_model` rejects anything else with a
`TypeError` instead of accepting a raw model and failing later.

## The signal policy

Defined once, in `ResearchAgent.research_signal`:

1. Fit and predict the **forecast** adapter. No forecast → no signal (an error is
   published, and no order can be created from regime context alone).
2. `direction = sign(expected_return)`.
3. `confidence = min(1, |expected_return| / (2 × 20-day realised volatility))` —
   a two-sigma expected move is full conviction.
4. Fit and predict the **regime** adapter for context. If the current state is
   the highest-volatility regime, halve confidence.
5. The runner refuses to trade below `MIN_SIGNAL_CONFIDENCE = 0.10`, logging the
   decision instead of sending a near-zero-size order.

The rationale for separating the two: a volatility regime says how much risk is
appropriate, never which way to trade.

## API demo models

`/research` previously returned `prediction_mean` from stub models that looked
like real output. The stubs are now adapters that emit
`diagnostics["demo"] = True`, fixed direction/confidence, and a `(demo)` regime
label. The endpoint's response uses the same contract as production, so a client
cannot distinguish demo data by shape — only by the explicit marker.

## Verification

- **554 tests pass** (was 546). New: 18 research-agent contract tests, 8 adapter
  tests, 2 API contract assertions.
- `test_xgboost_and_hmm_produce_a_signal` fits the **real** XGBoost forecaster
  and the **real** HMM on 400 synthetic prices and asserts a valid signal:
  direction in {-1, 0, 1}, confidence in [0, 1], a numeric regime with a label,
  a finite expected return, and non-zero forecast/regime diagnostics.

## Known limitation (belongs to phase 8)

The adapters refit on the recent window each research interval and use the last
prediction from that same fit, which is **in-sample**. The contract and the
policy are correct, but no walk-forward promotion gate stands between this
signal and an order yet. Phase 8 is where a strategy must pass the validation
gauntlet on out-of-sample data before it is promoted.
