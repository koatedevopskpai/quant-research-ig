# Market Data Architecture

## Design Decisions

### ADR-001: Use PostgreSQL for Time-Series Storage
**Status**: Proposed
**Context**: Need a database for OHLCV, tick, and quote data with time-series queries.
**Decision**: PostgreSQL with BRIN indexes on timestamp columns.
**Consequences**: Handles concurrent reads/writes well, enables SQL analytics.

### ADR-002: IG API v3 with OAuth 2.0 Bearer Tokens
**Status**: Accepted
**Context**: IG deprecated v2 CST/X-SECURITY-TOKEN auth.
**Decision**: Use POST /session (v3) for Bearer token, pass as Authorization header.
**Consequences**: Simpler auth flow; tokens expire after 1800s (refresh via refresh_token).

### ADR-003: Class-based IGClient Wrapping httpx
**Status**: Proposed
**Context**: Need reusable, testable client for all IG REST endpoints.
**Decision**: Single IGClient class with methods per endpoint category.
**Consequences**: Easy to mock in tests; centralises retry, rate-limit, and token refresh logic.
