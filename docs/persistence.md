# State persistence

Status: **complete** (2026-10-26). Orders, fills, the audit trail and risk state
survive a process restart in a single SQLite file.

## Why

Phases 5-7 kept a reconciling ledger, a risk halt flag and an audit trail in
memory. A restart lost the day: positions, realised P&L, day-start equity and the
halt state all reset, which makes "paper trading" a demo rather than an
accounting record.

## Design

`agentic_finance/persistence.py`

| Table | Contents |
|---|---|
| `orders` | identity, side, size, lifecycle status, filled size, average fill, protective levels, reason |
| `fills` | fill id, order id, epic, side, quantity, price, reference price, commission, timestamp |
| `audit` | every recorded decision, as JSON payload (lossless for restore) |
| `risk_state` | single row: halt flag, day-start equity, peak, current capital, daily P&L, reset date |
| `runs` | replay/experiment summaries by id |
| `equity_curve` | marks per run, for reporting and (future) VaR |

**Fills are the source of truth.** On restore, fills are replayed through the
same booking code used live, so cash, positions, average cost and realised P&L are
rebuilt exactly rather than trusted from a separate snapshot. Orders are loaded
and their fills reattached by `order_id`; the audit trail is reconstructed from
its JSON payloads; the halt flag and day-start equity come from `risk_state`.

## Usage

```bash
# Production runner (on by default; set QUANT_PERSIST=0 to disable)
python -m agentic_finance.run_production

# API (on by default; path from QUANT_STATE_DB)
QUANT_STATE_DB=state/quant.db python -m agentic_finance.api.server

# Replay with persistence
python -m agentic_finance.replay --steps 160 --state-db state/replay.db
```

| Setting | Default | Meaning |
|---|---|---|
| `QUANT_STATE_DB` | `state/quant.db` | SQLite path; `state/` is git-ignored |
| `QUANT_PERSIST` | `1` | `0` disables persistence in the runner |

Library constructors default to **no** store, so unit tests and one-off agents
stay isolated; the runner entry point and the API opt in. Tests point
`QUANT_STATE_DB` at a temporary directory (root `conftest.py`), so no state file
is ever written into the repository.

## Verification

Seven tests in `agentic-finance/tests/test_persistence.py`:

- fills restore cash, positions, average cost and realised P&L exactly;
- restore is idempotent;
- orders, fill attachments and the audit trail survive a reopen;
- the halt flag, day-start equity and peak survive, and `can_trade()` remains
  false after a restore of a halted state;
- a resume is persisted;
- a replay stores its run summary, equity curve and row counts consistently.

## Known limitations

- **Single writer.** SQLite serialises writers; this is correct for one process,
  not for several runners sharing a file. A server deployment needs one writer or
  a real database.
- **No encryption or access control.** Protect the file like the credentials it
  outlives.
- **Marks are not persisted** (prices are transient). Equity is rebuilt from
  fills against fresh marks; the stored `equity_curve` is per run, not a global
  time series, so VaR still has no continuous series.
- **No schema migrations.** The schema is created if absent; changing it later
  needs a migration step.
