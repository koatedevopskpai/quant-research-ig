# Market Data Architecture — Decision Records

This document catalogs all Architecture Decision Records (ADRs) for the
market-data module. Each ADR captures a significant design decision, the
context that motivated it, the alternatives considered, and the trade-offs
accepted.

---

## ADR Index

| ID | Title | Status |
|----|-------|--------|
| ADR-001 | PostgreSQL for Time-Series Storage | Accepted |
| ADR-002 | IG API v3 with OAuth 2.0 Bearer Tokens | Accepted |
| ADR-003 | Class-based IGClient Wrapping httpx | Accepted |
| ADR-004 | Lightstreamer Push Protocol for Real-Time Data | Accepted |
| ADR-005 | OHLCV Storage Schema Design | Proposed |
| ADR-006 | Config Auto-Discovery via Directory Tree Walk | Accepted |

---

## ADR-001: PostgreSQL for Time-Series Storage

**Status**: Accepted

**Context**: We need a database for OHLCV candles, tick-by-tick data, and
quote snapshots. The primary query pattern is time-range scans across one or
more instruments (e.g., "SELECT close FROM prices WHERE epic='US500' AND ts
BETWEEN '2025-01-01' AND '2025-06-01'").

**Options considered:**
1. **PostgreSQL + BRIN indexes** — General-purpose RDBMS with Block Range
   INdexes optimised for monotonically increasing values (timestamps).
2. **TimescaleDB** — PostgreSQL extension purpose-built for time-series.
   Adds hypertables, continuous aggregates, and automatic compression.
3. **InfluxDB** — Dedicated time-series database. Excellent write throughput.
   Limited SQL support and no JOINs across metadata tables.
4. **QuestDB** — Columnar time-series DB with PostgreSQL wire protocol. Very
   fast ingestion but smaller ecosystem and community.

**Decision**: PostgreSQL with BRIN indexes.

**Rationale**:
- BRIN indexes on ordered timestamps give approximately 95% of the query
  performance of a purpose-built time-series DB for our access pattern.
- PostgreSQL enables SQL JOINs across instruments, categories, and metadata
  tables — critical for portfolio-level queries that InfluxDB cannot do.
- The Python quant ecosystem (pandas, SQLAlchemy, psycopg2) has deepest
  PostgreSQL integration.
- Single PostgreSQL instance serves reference data and time-series,
  simplifying backup and administration.

**Trade-offs:**
- Write throughput ceiling (~50K rows/sec on modest hardware vs. 100K+ for
  QuestDB). Mitigation: batch writes in ingestion.
- No automatic downsampling (TimescaleDB continuous aggregates). Mitigation:
  materialised views refreshed on schedule.

**Consequences**:
- Schema migrations via Alembic.
- Long-term: if tick data exceeds 1B rows, evaluate TimescaleDB (same
  underlying PostgreSQL, same connection pool).

---

## ADR-002: IG API v3 with OAuth 2.0 Bearer Tokens

**Status**: Accepted

**Context**: IG deprecated v2 API authentication (CST + X-SECURITY-TOKEN) in
2024. New applications must use `POST /session` (v3) returning OAuth 2.0
Bearer tokens.

**Options considered:**
1. **IG API v3** — Standard Bearer token in Authorization header. 1800s
   expiry with refresh_token support.
2. **IG API v2 (legacy)** — CST/XST headers. Still functional but deprecated.
3. **FIX connection** — Direct market access. Requires additional licensing.

**Decision**: Use `POST /session` (v3) for Bearer token.

**Rationale**:
- V3 is IG's recommended and future-proofed auth method.
- Bearer tokens are a standard OAuth 2.0 pattern — well-documented and
  supported by all HTTP libraries.
- Auto-refresh at 1500s provides 5-minute safety margin before 1800s expiry.

**Trade-offs:**
- Lightstreamer does NOT support v3 OAuth — requires v2 CST/XST format.
- IG does not yet support true OAuth 2.0 refresh_token grants; refresh
  requires full re-authentication.

**Consequences**:
- IGClient manages token lifecycle internally.
- IGStreamer performs dual auth (v2 for LS, v3 for REST) transparently.

---

## ADR-003: Class-based IGClient Wrapping httpx

**Status**: Accepted

**Context**: Need a reusable, testable client for all IG REST API endpoints.
Must handle authentication, token refresh, error handling, and consistent
header management.

**Options considered:**
1. **Class-based** — Single IGClient class, single httpx.Client session.
2. **Functional** — Standalone functions creating sessions per call.
3. **Auto-generated** — OpenAPI generator from IG's (incomplete) spec.

**Decision**: Single IGClient class with methods per endpoint category.

**Rationale**:
- Connection pooling via single httpx.Client session saves ~50ms/request
  in DNS + TLS overhead.
- Token refresh centralised — consumers never handle tokens directly.
- Mockable in tests via `unittest.mock.patch`.

**Trade-offs:**
- Adding endpoints requires modifying the class.
- Not thread-safe (no current requirement).

**Consequences**:
- All IG REST endpoints through one import (`from ig_market_data.client import IGClient`).
- Error handling and rate-limit detection centralised.

---

## ADR-004: Lightstreamer Push Protocol for Real-Time Data

**Status**: Accepted

**Context**: IG provides two real-time data mechanisms: Lightstreamer (push)
and REST polling (pull at intervals).

**Options considered:**
1. **Lightstreamer** — Push-based, 10-50ms latency. Requires
   `lightstreamer-client-lib` (Java bridge via JNI).
2. **REST polling at 1s intervals** — Simple HTTP. ~86K requests/day per
   instrument. Up to 1 second latency.
3. **WebSocket** — IG does not expose a WebSocket endpoint for market data.

**Decision**: Lightstreamer push-based streaming.

**Rationale**:
- 10-50ms latency vs. 1s+ polling — essential for order book reconstruction.
- 10 instruments at 1s polling = 864K requests/day — risks rate limiting.
- Lightstreamer includes built-in reconnection and gap-filling.

**Trade-offs:**
- Java dependency via JNI bridge.
- IG-specific protocol (not reusable with other brokers).
- Dual auth (v2 for LS, v3 for REST).

**Consequences**:
- IGStreamer abstracts all LS complexity behind a callback API.
- Long-term: consider pure-Python LS implementation.

---

## ADR-005: OHLCV Storage Schema Design

**Status**: Proposed

**Context**: Historical price data must be stored in PostgreSQL with fast
time-range queries and clean resolution separation.

**Decision**: Separate tables per resolution (`ohlcv_daily`, `ohlcv_hourly`,
`ohlcv_minute`, `ticks`, `quotes`) with compound primary key `(epic, ts)`
and BRIN index on `ts`.

**Rationale**:
- Separate tables keep index sizes small — BRIN on a daily-only table is
  much smaller than a combined table.
- Query optimisation: daily scans skip hourly rows entirely.
- Cleaner Alembic migrations per-table.

**Trade-offs:**
- Application code must know which table to query.
- Cross-resolution queries require UNION or separate queries.

**Consequences**:
- `fetch_prices()` maps IG resolution to table internally.
- Scheduled ingestion writes to correct table based on resolution.

---

## ADR-006: Config Auto-Discovery via Directory Tree Walk

**Status**: Accepted

**Context**: Users invoke scripts from different directories. `.env` location
should not depend on current working directory.

**Decision**: Walk up the directory tree from the config module until a
`.env` file is found.

**Rationale**:
- `Path(__file__).resolve().parents` provides absolute path independent of
  working directory.
- Eliminates "file not found" errors from incorrect working directory.
- Compatible with `python -m` invocation from any location.

**Trade-offs:**
- O(depth) startup cost (negligible).
- Missing `.env` raises a clear `KeyError` with variable name.

**Consequences**:
- `.env` at repository root. All modules find it.
- CI/CD uses environment variables directly.
