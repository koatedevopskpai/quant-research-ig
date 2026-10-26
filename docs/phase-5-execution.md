# Phase 5 — paper execution ledger

Status: **complete** (2026-10-26). Orders have identities and lifecycle states,
fills are simulated and booked to a position ledger, and holdings and cash
reconcile with fills and costs.

## Defects fixed

All four were reproduced in the 2026-10 review.

| Defect | Effect | Fix |
|---|---|---|
| Capacity sized from `abs(holdings)` | a full position could never be reduced; exits were rejected as "no capacity" | size against a **signed target position**: `target − current` |
| Holdings updated when the order was created | the book showed positions before any fill, and never moved cash | positions and cash change only in `PositionLedger.apply_fill` |
| No order IDs, status stuck at `"pending"` | orders were unidentifiable and no state machine existed | `uuid` order IDs, `new → partially_filled → filled`, `rejected`, `cancelled` |
| Stop-loss and take-profit attached but never consumed | protective levels were decoration | `on_price()` checks them and submits a flattening order with `reason="stop_loss"` / `"take_profit"` |

## Accounting model (`execution_ledger.py`)

- **Fill**: execution price = reference price moved adversely by slippage;
  commission charged on notional. Both in basis points, overridable per fill.
- **Position**: signed quantity, average cost, realised P&L. Reductions book
  realised P&L against average cost; a reversal closes the old position and
  opens the remainder at the fill price.
- **Cash**: decreases by notional + commission on a buy, increases by
  notional − commission on a sell.
- **Invariant**, checked by `reconcile()`:

  ```
  equity = initial_cash + realised + unrealised − commissions
  ```

  Unrealised uses supplied marks and falls back to average cost when a mark is
  missing (contributing zero rather than a guess).

## Order lifecycle (`execution_agent.py`)

```
new ──fill──> partially_filled ──fill──> filled
  └──reject (toxicity)──> (no order)
```

- Market orders fill inside `process_signal`; `fill_ratio < 1` fills a fraction
  and leaves the order `partially_filled`, with remaining size filled on later
  `on_price()` calls.
- Every fill publishes `order.filled` with `order_id`, `fill_id`, quantity,
  price and commission; `order.created` carries the protective levels.
- A minimum order notional (default 1.0) suppresses micro-orders caused by
  commission nudging equity between signals.

## Sizing

```
target_quantity = direction × confidence × (equity × max_position_pct / price)
order_quantity  = target_quantity − current_quantity
```

`direction × confidence` is a signed conviction in `[-1, 1]`. Because the order
is the **difference** from the current position, reductions, exits and reversals
require no extra risk capacity — the defect that blocked exits is structurally
gone rather than patched.

## Protective exits

The levels of the order that established (or last increased) the position are
stored per EPIC. On each price update:

- long: exit when `price ≤ stop_loss` or `price ≥ take_profit`
- short: exit when `price ≥ stop_loss` or `price ≤ take_profit`

The exit order is sized to flatten the position exactly, not to the signal
target, so a protective exit cannot leave a residual.

## Verification

- **578 tests pass** (was 554). New: 12 ledger tests and 12 lifecycle tests.
- Ledger tests use hand-checkable arithmetic (buy/reduce/reverse, slippage
  direction, short unrealised sign) and a deliberately corrupted book to prove
  `reconcile()` can fail.
- Lifecycle tests cover: identity and immediate fills, partial then complete,
  reduction at full capacity, reversal, stop-loss and take-profit for both
  sides, no exit between levels, and reconciliation after mixed activity.

## Known limitations (phase 6+)

- **Risk still does not consume the ledger.** `RiskManager` keeps its own P&L
  and position count; wiring exposure and daily loss to `PositionLedger` is
  phase 6.
- **In-memory only.** A restart loses fills and positions. Deterministic replay
  in phase 7 is the place to persist and rebuild.
- **Market orders only.** Limit fields exist but there is no matching engine, and
  there is no cancel/replace endpoint yet.
- **Deterministic fill model.** Slippage and `fill_ratio` are modelled, not a
  queue-position simulation; the execution simulator can supply the slippage
  estimate, but fill probability by level is out of scope.
