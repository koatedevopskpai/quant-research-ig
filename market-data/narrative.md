# market-data — IG Market Data Ingestion Engine

## Objective
Build the data foundation for all downstream quant research: an IG API client
that authenticates via OAuth 2.0, discovers available instruments, and fetches
OHLCV historical prices into a local PostgreSQL database.

## Current Status
- [x] IG API credentials verified (HTTP 200 on POST /session)
- [x] Market categories discovered (8 categories: indices, FX, crypto, etc.)
- [x] Price history retrieval working (5 daily points for US 500)
- [ ] IGClient class implementation
- [ ] PostgreSQL schema + Alembic migrations
- [ ] Ingestion pipeline (scheduled/on-demand)

## ADRs
- ADR-001: PostgreSQL for time-series storage
- ADR-002: IG API v3 with OAuth 2.0 Bearer tokens
- ADR-003: Class-based IGClient wrapping httpx

## Key Files
| File | Purpose |
|------|---------|
| src/ig_market_data/verify_ig.py | Credential verification script |
| src/ig_market_data/client.py | IG API client (planned) |
| src/ig_market_data/models.py | SQLAlchemy ORM models (planned) |
| src/ig_market_data/ingestion.py | Data ingestion pipeline (planned) |
