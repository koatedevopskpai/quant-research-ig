"""
ig_market_data — IG API client, PostgreSQL schema, and data ingestion engine.

This package is the foundational data layer for all quant research projects.
It wraps the IG REST API (v3, OAuth 2.0) and persists OHLCV + tick/quote
data into a local PostgreSQL instance.

Exports:
    IGClient         — Authenticated HTTP client for IG API
    MarketDataEngine — Orchestrates ingestion from IG → PostgreSQL
    SchemaManager    — Alembic-based migration management

Typical usage:
    from ig_market_data import IGClient
    client = IGClient()
    prices = client.fetch_prices("IX.D.SPTRD.IFD.IP", resolution="DAY", max=100)
"""
