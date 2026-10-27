# Adaptive regime matrix and extended exits

Status: **implemented and tested** (2026-10-27). These were the last two claims in
the original write-up: RL adaptation of the HMM transition matrix, and
toxicity-spike / time-based exits.

## Reward-modulated transition adaptation

`ml-for-finance/src/ml_for_finance/regime/adaptive_hmm.py`

```text
T[i, j] <- normalise( T[i, j] * exp(eta * tanh(pnl / scale)) )   for the observed (i -> j)
then      T <- (1 - decay) * T + decay * T_baseline
```

- `eta` (`learning_rate`, default 0.05) bounds how strongly one outcome moves the
  matrix; `tanh` saturates so a single huge P&L cannot dominate.
- `decay` (default 0.01) pulls the matrix back toward the fitted baseline, so a
  run of one-sided outcomes cannot drift without bound.
- Each update is capped at `exp(±max_log_adjustment)`; the row is renormalised
  and clipped. Every row remains a probability distribution.
- The adapted `transmat_` is the matrix the estimator uses for Viterbi and
  posterior probabilities, so the change is real rather than a display value.

**Wiring.** `ExecutionAgent` publishes `position.closed` whenever a fill reduces
or reverses a position, carrying `closed_quantity`, `realized_pnl`, the exit
`reason` and the `order_id`. `ProductionRunner` tracks the regime at entry and at
close, then calls `ResearchAgent.record_outcome(epic, prev, next, pnl)`, which
calls `RegimeAdapter.learn()` and records an `rl_update` audit decision.

**Honest scope.** This is a bandit-style reward modulation of a transition
model — the mechanism the design describes. It is not a policy-gradient agent,
has no value function or exploration, and has not been calibrated on live
outcomes. The docstrings and the README say so.

## Extended exits

`agentic-finance/src/agentic_finance/execution_agent.py`

Exit priority on every price update:

| Order | Reason | Trigger |
|---|---|---|
| 1 | `stop_loss` | price beyond the protective level |
| 2 | `take_profit` | price beyond the target |
| 3 | `toxicity_spike` | detector toxicity ≥ `toxicity_exit_threshold` (default 0.8) |
| 4 | `time_exit` | position held ≥ `max_holding_seconds` |

- Exits are **flattening** orders sized to the current position, not to the
  signal target, so none can leave a residual.
- The holding clock starts when a position is opened or reversed and is cleared
  when it goes flat.
- An unfitted toxicity detector raises internally and is ignored, so a missing
  model cannot block exits.
- The production runner enables a 60-minute time exit; the deterministic replay
  leaves it disabled and has no detector, so its invariants are unchanged.

## Verification

- **9 adaptation tests** (`ml-for-finance/tests/test_adaptive_hmm.py`): stochastic
  rows, profit raises / loss lowers the traversed cell, other rows untouched,
  invalid inputs rejected, per-update bound, decay back to baseline, `predict()`
  still works.
- **13 exit/feedback tests** (`agentic-finance/tests/test_exits_and_adaptation.py`):
  toxicity spike closes, below threshold does not, broken detector ignored, time
  exit and its boundary, reversal restarts the clock, disable switch,
  `position.closed` payload and realised P&L, reduction is not a full close,
  real-model learning, agent audit, and runner wiring.
- Full suite: **662 passed** with the extension built.

## Limitations

- **Reward scale is a policy choice** (`reward_scale=0.01` means 1% P&L saturates
  the update). It is not derived from data.
- **No live calibration.** Thresholds are defaults; a live deployment needs a
  study of false toxicity spikes and turnover cost.
- **Single-step adaptation.** The update sees entry and exit regimes, not the
  full path; credit assignment across intermediate regimes is not modelled.
- **Time exits are wall-clock**, not trading-session aware; a 60-minute limit
  spans overnight or weekend gaps if the process keeps running.
