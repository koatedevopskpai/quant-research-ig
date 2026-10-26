# Phase 7 — security posture and deterministic replay

Status: **complete** (2026-10-26). The API is loopback-by-default with an optional
key, bus publication is debug-only and allow-listed, demo models are labelled, and
a deterministic replay drives the full pipeline while asserting accounting,
isolation and risk invariants.

## API security

| Defect | Fix |
|---|---|
| Entry point bound `0.0.0.0` | default `127.0.0.1`; override via `QUANT_API_HOST` |
| Mutation endpoints unauthenticated | `QUANT_API_KEY` enables an `X-API-Key` check on every endpoint except `/health` |
| CORS `allow_origins=["*"]` | loopback origins only by default; set `QUANT_CORS_ORIGINS` for a shared deployment |
| Any client could publish any bus topic | `/bus/publish` is 403 unless `QUANT_DEBUG=1` **and** the topic matches `QUANT_BUS_ALLOWED_TOPICS` (default: the `sim.` namespace) |
| Demo models indistinguishable from real ones | `/health` exposes `demo_models`; `/research` returns `demo: true`; model output carries `diagnostics["demo"]` |
| WebSocket stream unauthenticated | `X-API-Key` checked before accept when a key is configured |

The `/bus/publish` restriction is the important one: that endpoint could inject
`risk.halted` or `research.completed` and drive the system. It is now only a
simulated-data tap, and only in a debug session.

## Deterministic replay

`agentic_finance/replay.py`

```
synthetic events -> OrderBook (per EPIC) -> momentum/regime adapters
                 -> research_signal -> ExecutionAgent fills -> PositionLedger
                 -> RiskManager (marks, daily loss, halt) -> AuditTrail
```

Everything that could vary is fixed: the fixture is generated arithmetically, the
adapters are momentum and volatility-tercile (no training, no randomness), the
clock is not consulted, and research cadence is a config value. Order IDs are
UUIDs and differ between runs; no number does.

Asserted at the end of every run:

| Invariant | Meaning |
|---|---|
| `accounting_balanced` | `equity = initial + realised + unrealised − commissions` |
| `fills_have_orders` | every fill carries an `order_id` that exists |
| `no_new_entries_after_halt` | no `reason="signal"` order is created after a halt |
| `position_within_cap` | largest position ≤ `max_position_pct × equity` (5% tolerance) |
| `instruments_isolated` | each EPIC's mark is its own last event; no shared book object |
| `audit_recorded` | research, order and risk decisions all reach the trail |
| `equity_matches_reconcile` | `execution.equity()` equals the reconcile figure |

### Results

| Run | Result |
|---|---|
| Baseline, 160 steps | final equity **100,360.97**; 60 fills; all seven invariants pass (`evidence/replay-baseline.json`) |
| Adverse shock at step 60 (−40%, stop widened, 0.5% daily limit) | **halted**: `daily_loss=-874.72 > limit=0.50%`; **0 entries after halt**; book balanced (`evidence/replay-halt.json`) |

The halt scenario also demonstrates the intended asymmetry: protective and
reducing orders are still allowed after a halt, while new risk is refused.

## Verification

- **613 tests pass** (was 591). New: 8 replay tests and 13 API-security tests, plus
  the bus test rewritten to the new contract.
- Determinism is asserted directly: two runs of the same fixture produce
  identical equity curves, fill counts and final equity.
- Security tests cover: loopback default and override, key required/wrong/missing
  and `/health` exemption, CORS defaults, debug gating, allowlist enforcement, and
  demo labelling in both `/health` and `/research`.

## Known limitations (phase 8)

- **No persistence** — resolved after this phase: orders, fills, audit and risk
  state now survive a restart in SQLite. See [`persistence.md`](persistence.md).
- **WebSocket auth uses a header**, which browsers cannot set on a WebSocket
  handshake. A shared browser deployment needs a query-token or session flow.
- **No rate limiting or TLS.** Terminate TLS and rate-limit at a reverse proxy.
- **Replay is a plumbing proof.** The fixture is synthetic and the adapters are
  deterministic by design; it demonstrates invariants, not a strategy result. A
  replay over recorded IG data and the real models is phase 8, and is blocked on
  the IG account.
- **VaR has no stored equity series.** Marks are recomputed per event, not
  persisted, so VaR remains zero.
