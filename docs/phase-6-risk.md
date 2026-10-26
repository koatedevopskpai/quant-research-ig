# Phase 6 — risk controls

Status: **complete** (2026-10-26). P&L and exposure come from the execution
ledger, the daily loss limit resets on a UTC boundary, halting is idempotent, and
both the runner and the API enforce it.

## Defects fixed

| Defect | Effect | Fix |
|---|---|---|
| P&L booked as `(mid − 100) × 10` on every refresh | not P&L; a constant price added 100 units with no position | risk values the ledger at current marks |
| Exposure never wired | limits could not see the book | monetary gross exposure and `exposure_pct` from the ledger |
| `risk.kill` published by its own handler | one halt re-entered the handler and republished — nine publications from one event in the review | commands and state notifications are separated; halt publishes `risk.halted` once |
| `resume()` republished the kill topic | resuming immediately re-halted execution | `risk.resumed` notification, idempotent transition |
| Manual kill only checked for an existing breach | the "emergency stop" did nothing on a healthy book | `/risk/kill` halts unconditionally |
| API state never subscribed execution to halts | API-triggered risk events left execution active | `AppState` subscribes `risk.halted`/`risk.resumed` to execution |
| Daily loss compared against current capital | limit drifted with P&L | measured against equity at the UTC day start |

## Halt semantics

```
active ──halt(reason)──> halted
halted ──resume()──────> active
```

Both transitions return `True` only when they change state. Repeated halt events
produce one audit entry, one `risk.halted` message, and no recursion.

Topics:

| Topic | Direction | Purpose |
|---|---|---|
| `risk.alert` | notification | limits breached (metrics attached) |
| `risk.halted` | notification | risk has stopped trading |
| `risk.resumed` | notification | risk has cleared the halt |
| `execution.halted` / `execution.resumed` | notification | execution state changed |
| `risk.kill` | **retired** | was a command published as a notification |

`resume()` deliberately does **not** override an active breach: if drawdown is
still beyond the limit, `can_trade()` remains false even after the operator
clears the halt.

## Daily reset

- Boundary: UTC date change (`_reset_daily_if_needed`, with an injectable clock
  for tests).
- At the boundary the day's start equity is set to current equity and
  `daily_pnl` resets to zero; losses remain reflected in drawdown, which is
  measured from the all-time peak.
- `record_pnl()` is ignored when a ledger is attached, so ad-hoc P&L cannot
  double count.

## Enforcement

| Path | Check |
|---|---|
| Runner (`_on_research_completed`) | halted, execution inactive, or `can_trade()` false → research discarded before a signal is built |
| Runner (`_check_risk`, every minute) | `kill_if_needed()` halts once on breach |
| API (`POST /signal`) | same three checks → HTTP 409 with a reason |
| API (`POST /risk/kill`) | unconditional halt; execution is halted by subscription |
| API (`POST /risk/resume`) | idempotent resume, subject to the breach check above |

## Verification

- **591 tests pass** (was 578). New: 13 risk/ledger tests.
- Ledger-driven equity and exposure equal `ledger.equity()`/`gross_exposure()`
  and reconcile.
- Daily-loss breach measured against the day start, and cleared by an injected
  UTC boundary.
- Halt/resume publish exactly one notification each; a breach halts once;
  repeated events are no-ops; emergency halt needs no breach.
- Runner and API both refuse to create risk while halted (409 from the API).

## Known limitations (phase 7+)

- **No persistence.** Risk and ledger state are in memory; a restart loses the
  day's P&L and positions. Deterministic replay is phase 7.
- **VaR uses a manual return history.** `record_pnl` is the only source of
  returns; with a ledger attached there is no equity time series yet, so the VaR
  figures remain zero until phase 7 records marks over time. Drawdown, daily loss
  and exposure are fully live.
- **Exposure is reported, not limited.** There is no maximum gross-exposure
  threshold; the position cap is enforced in sizing (phase 5).
- **Reset boundary is UTC only.** A configurable exchange session boundary is not
  implemented.
