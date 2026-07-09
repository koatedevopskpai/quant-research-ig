# market-data — IG Market Data Ingestion Engine

## Objective

Build the data foundation for all downstream quant research: an IG API client
that authenticates via OAuth 2.0, discovers available instruments, and fetches
OHLCV historical prices into a local PostgreSQL database.

This is the **Phase 1** deliverable in the STRATEGY.md roadmap. Every
downstream module — from Black-Scholes pricers to LSTM forecasters — depends
on this layer for data. A 1-microsecond timestamp error in tick data can
produce a 5% error in high-frequency volatility estimates, so correctness and
traceability are paramount.

---

## Domain Background

### What Is Market Data?

Market data refers to real-time and historical prices, quotes, and trade
information from financial exchanges. This project uses IG Group's API, which
aggregates data from major global exchanges (NYSE, NASDAQ, LSE, Eurex, etc.)
and offers it through two channels:

1. **REST API (v3)** — Request/response for historical OHLCV candles, market
   details, account information. Used for backtesting, research, and batch
   processing.
2. **Lightstreamer** — A proprietary push protocol where the server streams
   real-time updates to subscribed clients. Used for live trading, order book
   reconstruction, and tick-level analytics.

### Why IG Group?

IG Group is one of the largest CFD and spread-betting providers globally, with
particularly strong coverage of UK, European, and Asia-Pacific markets. Their
API v3 (introduced in 2024) modernised authentication from the legacy v2
CST/X-SECURITY-TOKEN pattern to standard OAuth 2.0 Bearer tokens.

**Alternatives considered:**
- **Interactive Brokers** — More comprehensive but requires TWS/Gateway
  installation and has a steeper API learning curve.
- **Alpaca** — US-equities only; no forex, indices, or commodities.
- **Yahoo Finance** — Free but unreliable for production; no real-time data.

---

## Current Status

| Component | Status | Details |
|-----------|--------|---------|
| IG API credentials | Verified | HTTP 200 on POST /session (v3) |
| Category discovery | Working | 8 categories: indices, FX, commodities, crypto, etc. |
| Price history | Working | Daily OHLCV for US 500 confirmed |
| IGClient class | **Implemented** | OAuth 2.0, auto-refresh, 5 API methods |
| IGStreamer | **Implemented** | L1, ticks, candles, account P&L subscriptions |
| PostgreSQL schema | Pending | Phase 1b — models.py + Alembic migrations |
| Ingestion pipeline | Pending | Phase 1b — scheduled/on-demand data collection |

---

## Architecture Decisions

### ADR-001: PostgreSQL for Time-Series Storage

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
3. **InfluxDB** — Dedicated time-series database. Excellent write throughput
   but limited SQL support and no JOINs across metadata tables.
4. **QuestDB** — Columnar time-series DB with PostgreSQL wire protocol. Very
   fast ingestion but smaller ecosystem and community.

**Decision**: PostgreSQL with BRIN indexes.

**Rationale**:
- BRIN indexes on ordered timestamps give approximately 95% of the query
  performance of a purpose-built time-series DB for our access pattern.
- PostgreSQL enables SQL JOINs across instruments, categories, and metadata
  tables — critical for portfolio-level queries that InfluxDB cannot perform.
- Team familiarity: the Python quant ecosystem (pandas, SQLAlchemy, psycopg2)
  has deepest PostgreSQL integration. No new infrastructure to learn.
- Same PostgreSQL instance serves reference data (instrument metadata,
  categories) and time-series data, simplifying backup and administration.

**Trade-offs:**
- Write throughput ceiling (approximately 50K rows/sec on modest hardware vs.
  100K+ for QuestDB). Mitigation: batch writes in the ingestion pipeline.
- No automatic downsampling (TimescaleDB's continuous aggregates). Mitigation:
  materialised views refreshed on a schedule.

**Consequences:**
- Schema migrations managed via Alembic.
- Long-term migration path: if tick data exceeds 1 billion rows, evaluate
  TimescaleDB (same underlying PostgreSQL, same connection pool).

---

### ADR-002: IG API v3 with OAuth 2.0 Bearer Tokens

**Status**: Accepted

**Context**: IG deprecated the v2 API authentication (CST header +
X-SECURITY-TOKEN header) in 2024. New applications must use the v3
`POST /session` endpoint that returns an OAuth 2.0 Bearer token.

**Options considered:**
1. **IG API v3 (recommended by IG)** — Standard Bearer token in the
   `Authorization` header. Token expires after 1800 seconds; refresh via
   `refresh_token`.
2. **IG API v2 (legacy)** — CST and X-SECURITY-TOKEN headers returned from
   `POST /session` (v2). Still functional but marked for deprecation.
3. **FIX connection** — Direct market access via the FIX protocol. Requires
   additional licensing and infrastructure. Typically used by institutions
   only.

**Decision**: Use `POST /session` (v3) for Bearer token, passed as
`Authorization: Bearer <token>` header.

**Rationale**:
- V3 is IG's recommended and future-proofed authentication method.
- Bearer tokens are a standard OAuth 2.0 pattern — well-documented and
  supported by all HTTP client libraries.
- The token auto-refresh pattern (refresh at 1500s, expire at 1800s) is
  simple to implement and provides 5 minutes of safety margin.

**Trade-offs:**
- Lightstreamer does NOT support v3 OAuth tokens — it requires the legacy v2
  CST/XST format for its password field. This means the streaming client must
  perform TWO authentication calls (one v2 for LS, one v3 for REST).
- Token refresh requires a full re-authentication call (not a true OAuth 2.0
  refresh token flow). IG's API does not yet support `refresh_token` grants.

**Consequences:**
- The `IGClient` class manages token lifecycle internally. Consumers never
  handle tokens directly.
- The `IGStreamer` class handles dual auth transparently.

---

### ADR-003: Class-based IGClient Wrapping httpx

**Status**: Accepted

**Context**: We need a reusable, testable client for all IG REST API
endpoints. The client must handle authentication, token refresh, error
handling, and consistent header management.

**Options considered:**
1. **Class-based client** — Single `IGClient` class with methods per endpoint
   category (`list_categories()`, `fetch_prices()`, etc.). Wraps a single
   `httpx.Client` session.
2. **Functional approach** — Standalone functions that create a new HTTP
   session per call. Simpler but duplicates connection overhead and auth logic.
3. **Auto-generated client** — OpenAPI generator from IG's (incomplete) spec.
   Fragile; IG's spec has known gaps and inaccuracies.

**Decision**: A single `IGClient` class with methods per endpoint category.

**Rationale**:
- Connection pooling via a single `httpx.Client` session reduces DNS
  resolution and TLS handshake overhead by approximately 50ms per request.
- Token refresh logic is centralised — consumers never need to check token
  expiry or call `_authenticate()` manually.
- Mocking in tests is straightforward: `unittest.mock.patch` the class or
  inject a pre-configured `httpx.Client` with a mock transport.

**Trade-offs:**
- The class must be thread-safe if used from multiple threads (currently not
  required; all consumption is single-threaded).
- Adding endpoints requires modifying the class — but this is proportional to
  actual usage.

**Consequences:**
- All IG REST endpoints are accessible through a single import.
- Error handling and rate-limit detection are centralised in one location.

---

### ADR-004: Lightstreamer Push Protocol for Real-Time Data

**Status**: Accepted

**Context**: IG provides two real-time data mechanisms: (1) Lightstreamer, a
proprietary push protocol where the server pushes updates to subscribed
clients, and (2) REST polling, where the client repeatedly calls
`GET /prices/{epic}` at intervals.

**Options considered:**
1. **Lightstreamer** — Push-based protocol. Updates arrive within 10-50ms of
   the exchange event. Requires the `lightstreamer-client-lib` Python library.
2. **REST polling at 1-second intervals** — Simple HTTP GET loop. No
   additional library. Updates delayed by up to 1 second + network latency.
   Generates approximately 86K HTTP requests per day per instrument.
3. **WebSocket** — IG does not expose a WebSocket endpoint for market data.

**Decision**: Lightstreamer push-based streaming.

**Rationale**:
- Push latency is 10-50ms vs. 1 second+ for polling — essential for order
  book reconstruction and microstructure analytics (Phase 2b).
- Polling 10 instruments at 1-second intervals generates 864,000 HTTP
  requests per day — this creates unnecessary load on IG's API and risks
  triggering rate limits.
- Lightstreamer includes built-in automatic reconnection and gap-filling
  using a sequence-number-based recovery mechanism.

**Trade-offs:**
- Additional library dependency (`lightstreamer-client-lib`). This library
  is maintained by Lightstreamer and wraps their Java client via a JNI bridge
  (which requires a Java runtime). This is a significant operational concern.
- IG-specific protocol — cannot be reused with other brokers.
- Dual auth requirement (v2 for LS, v3 for REST) adds complexity.

**Consequences:**
- The `IGStreamer` class abstracts all Lightstreamer complexity. Consumers
  provide API credentials and a callback; the class handles auth, connection,
  subscription, and reconnection internally.
- Long-term: consider moving to a pure-Python Lightstreamer implementation
  to remove the Java dependency.

---

### ADR-005: OHLCV Storage Schema Design

**Status**: Proposed

**Context**: Historical price data must be stored in PostgreSQL with a schema
that supports fast time-range queries, easy joins across instruments, and
clear separation of data resolutions (daily, hourly, minute-level, tick).

**Decision**: Separate tables per resolution level, with a BRIN index on the
timestamp column and a compound primary key of (epic, ts).

Schema outline:
```sql
CREATE TABLE ohlcv_daily (
    epic       TEXT    NOT NULL,
    ts         TIMESTAMPTZ NOT NULL,
    open_bid   NUMERIC(12,5),
    high_bid   NUMERIC(12,5),
    low_bid    NUMERIC(12,5),
    close_bid  NUMERIC(12,5),
    open_ask   NUMERIC(12,5),
    high_ask   NUMERIC(12,5),
    low_ask    NUMERIC(12,5),
    close_ask  NUMERIC(12,5),
    volume     BIGINT,
    PRIMARY KEY (epic, ts)
);
CREATE INDEX idx_ohlcv_daily_ts ON ohlcv_daily USING BRIN (ts);
```

Separate tables: `ohlcv_hourly`, `ohlcv_minute`, `ticks`, `quotes`.

**Rationale**:
- Separate tables keep index sizes small — a BRIN index on a daily-only table
  is orders of magnitude smaller than a combined table with all resolutions.
- Query optimisation: daily scans skip hourly rows entirely.
- Alembic migrations are cleaner per-table.

**Trade-offs:**
- Application code must know which table to query based on resolution.
- Cross-resolution queries (e.g., daily close combined with hourly volatility)
  require UNION or separate queries in application code.

**Consequences:**
- The `fetch_prices()` method maps IG's resolution parameter to the correct
  table internally.
- Scheduled ingestion writes to the appropriate table based on the requested
  resolution.

---

### ADR-006: Config Auto-Discovery via Directory Tree Walk

**Status**: Accepted

**Context**: Users may invoke scripts from any directory (project root,
module root, tests subdirectory). The `.env` file location should not depend
on the current working directory.

**Decision**: Walk up the directory tree from the config module's location
until a `.env` file is found.

**Rationale**:
- `Path(__file__).resolve().parents` provides an absolute path independent of
  the working directory.
- This eliminates the "file not found" errors that plague projects requiring a
  specific working directory or `export PYTHONPATH`.
- Compatible with `python -m ig_market_data.verify_ig` from any location.

**Trade-offs:**
- Walking parent directories is an O(depth) operation — negligible at
  startup but context-dependent.
- If no `.env` is found, `os.environ["IG_API_KEY"]` raises a clear `KeyError`
  with the missing variable name.

**Consequences:**
- Users place `.env` at the repository root. All modules find it.
- CI/CD systems set environment variables directly (`.env` is gitignored).

---

## Implementation Guide

### IGClient (client.py)

The `IGClient` class wraps all IG REST API interactions. See the code for
detailed inline comments explaining each method.

**Entry points for new developers:**
- Start with `verify_ig.py` to confirm credentials work.
- Then explore `list_categories()` and `search_markets()` to discover
  instrument identifiers (epics).
- Use `fetch_prices()` to download historical data for backtesting.

### IGStreamer (stream.py)

The `IGStreamer` class provides real-time market data via Lightstreamer. It
is the foundation for the hft-engine module (Phase 2b).

**Subscription modes:**
- **MERGE** — Server sends only changed fields (L1 data, candles). Client
  merges with previous state.
- **DISTINCT** — Every update is a complete new row (tick data). Client
  receives every event.

**Callbacks** receive a flat dictionary of field names to string values.
Consumers parse numeric fields as needed.

### Config (config.py)

The `IGSettings` class provides validated configuration from environment
variables. It auto-loads `.env` from the nearest parent directory.

**Testing mock strategy:**
```python
from unittest.mock import patch
with patch("ig_market_data.config.IGSettings") as mock:
    mock.api_key = "test_key"
    # ... test with mocked config
```

## References

- IG Labs API Documentation: https://labs.ig.com
- Lightstreamer Client Library: https://github.com/Lightstreamer/Lightstreamer-lib-client-python
- Downing, B. "Deconstructing an Advanced Ether High Frequency Trading Bot:
  A C++ Breakdown" — QuantLabsNet.com (2025)
- PostgreSQL BRIN Indexes: https://www.postgresql.org/docs/current/brin-intro.html
