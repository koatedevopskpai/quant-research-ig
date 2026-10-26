"""
models.py — SQLAlchemy ORM models for market data storage.

WHY THIS EXISTS:
    The IG REST API returns raw JSON price data. To run quantitative research
    (backtesting, signal analysis, ML training), we need this data in a
    queryable, indexed, persistent format. SQLAlchemy models provide the
    bridge between IG's API and our PostgreSQL database.

SCHEMA DESIGN (ADR-005):
    We use separate tables per resolution level rather than a single
    monolithic table. This keeps index sizes small and query performance
    high — a BRIN index on a daily-only table is orders of magnitude
    smaller than a combined table with all resolutions.

    Resolution hierarchy:
        ticks (raw) -> 1min -> 1hour -> 1day
        Each level aggregates from the level below it.

    WHY NOT a single "prices" table with a resolution column:
        A single table would mix tick-level rows (hundreds per second per
        instrument) with daily rows (one per day). The BRIN index on
        timestamp would span both, reducing scan efficiency. Daily research
        queries would scan through millions of tick rows to find the one
        daily row they need.

        Trade-off: application code must know which table to query based
        on resolution. This is handled by the ingestion pipeline.

COLUMN DESIGN:
    - Bid/ask prices are stored separately (not just mid). Many quant
      strategies depend on spread dynamics, which require both sides.
    - NUMERIC(12,5) for prices gives 5 decimal places at the precision
      needed for forex (pips) and indices (fractional points).
    - TIMESTAMPTZ (TIMESTAMP WITH TIME ZONE) ensures timezone-naive
      timestamps from IG are stored consistently. All queries use UTC.

INDEX STRATEGY:
    - PRIMARY KEY (epic, ts): Ensures uniqueness per instrument per period.
    - BRIN index on ts: Block Range INdex is ideal for time-ordered data.
      It divides the table into block ranges and stores the min/max
      timestamp per range, enabling fast range scans without a full B-tree.
    - Optionally, a B-tree index on (epic, ts) for equality lookups on
      specific instrument-time combinations.

METHODOLOGY NOTE (Quant Beckman):
    Clean, well-structured data storage is the foundation of Beckman's
    "Research" stage. Without trustworthy data, every downstream analysis
    — from PSR calculations to ML model training — is suspect. These
    models enforce data integrity at the database level (type constraints,
    uniqueness, referential integrity) so that researchers can trust
    their queries.
"""

from datetime import datetime

from sqlalchemy import (
    Column,
    BigInteger,
    DateTime,
    Integer,
    Numeric,
    String,
    create_engine,
    Index,
)
from sqlalchemy.orm import declarative_base

# WHY using __file__ based engine URL:
#   The engine is created here for schema introspection by Alembic.
#   Runtime connections use the pool from the ingestion pipeline.
#   config.pg_settings.dsn provides the connection string.

# WHY declarative_base():
#   SQLAlchemy 2.0 supports both declarative and imperative mapping.
#   Declarative is more readable for quant researchers who may not be
#   database experts — the schema is visible directly in the class
#   definition.
Base = declarative_base()


class OHLCVDaily(Base):
    """
    Daily OHLCV candles — one row per instrument per trading day.

    This is the most commonly queried resolution for quant research:
    - Backtesting strategies over multi-year periods
    - Computing daily returns and volatility
    - Training ML models on daily features
    - Portfolio optimisation and rebalancing

    Data volume: approximately 252 rows per instrument per year (trading
    days). For 100 instruments over 5 years: ~126,000 rows.
    """

    __tablename__ = "ohlcv_daily"

    # WHY epic is TEXT not VARCHAR:
    #   IG epics have variable length (e.g., "IX.D.SPTRD.IFD.IP" vs. "CS.D.EURUSD.CFD.IP").
    #   TEXT in PostgreSQL is performance-equivalent to VARCHAR with no length
    #   limit, avoiding future issues if IG changes their epic format.
    epic = Column(String, primary_key=True)

    # WHY TIMESTAMPTZ:
    #   IG timestamps do not always include timezone information.
    #   TIMESTAMPTZ coerces all values to UTC on insertion, ensuring
    #   consistent timezone handling across all queries.
    ts = Column(DateTime(timezone=True), primary_key=True)

    # WHY separate bid/ask columns:
    #   Many quant strategies depend on the bid-ask spread. Storing bid
    #   and ask separately enables spread analysis without reconstructing
    #   from mid prices (which loses spread information).
    open_bid = Column(Numeric(12, 5))
    high_bid = Column(Numeric(12, 5))
    low_bid = Column(Numeric(12, 5))
    close_bid = Column(Numeric(12, 5))
    open_ask = Column(Numeric(12, 5))
    high_ask = Column(Numeric(12, 5))
    low_ask = Column(Numeric(12, 5))
    close_ask = Column(Numeric(12, 5))

    # WHY volume is BIGINT not INTEGER:
    #   High-volume instruments (e.g., EUR/USD, S&P 500) can exceed
    #   2^31 - 1 in daily volume. BIGINT (8 bytes) provides adequate range.
    volume = Column(BigInteger)

    # WHY BRIN index on timestamp:
    #   BRIN (Block Range INdex) is specifically designed for monotonically
    #   increasing columns like timestamps. It scans block ranges instead
    #   of individual tuples, providing excellent performance for range
    #   queries ("WHERE ts BETWEEN '2025-01-01' AND '2025-06-01'") at a
    #   fraction of the storage cost of a B-tree index.
    __table_args__ = (
        Index("idx_ohlcv_daily_ts", ts, postgresql_using="brin"),
    )


class OHLCVHourly(Base):
    """
    Hourly OHLCV candles.

    Used for intraday strategy research, volatility modelling, and
    higher-frequency signal generation.

    Data volume: approximately 4,200 rows per instrument per year
    (24 hours * 252 days * ~70% market open time).
    """

    __tablename__ = "ohlcv_hourly"

    epic = Column(String, primary_key=True)
    ts = Column(DateTime(timezone=True), primary_key=True)
    open_bid = Column(Numeric(12, 5))
    high_bid = Column(Numeric(12, 5))
    low_bid = Column(Numeric(12, 5))
    close_bid = Column(Numeric(12, 5))
    open_ask = Column(Numeric(12, 5))
    high_ask = Column(Numeric(12, 5))
    low_ask = Column(Numeric(12, 5))
    close_ask = Column(Numeric(12, 5))
    volume = Column(BigInteger)

    __table_args__ = (Index("idx_ohlcv_hourly_ts", ts, postgresql_using="brin"),)


class Tick(Base):
    """
    Tick-by-tick trade data — every individual trade as it happens.

    WHY ticks exist as a separate table:
        Ticks are the most granular data available from IG. They are the
        atomic unit of market microstructure analysis (Bryan Downing's
        HFT architecture).

    USED BY (Phase 2b+):
        - Order book reconstruction: building L1/L2 books from tick sequences
        - Trade sign classification: was the trade buyer-initiated or
          seller-initiated? (requires comparing trade price to prevailing
          bid/ask at the time of the trade)
        - Order Flow Toxicity Detector: Hawkes process on trade arrival
          times to detect informed trading
        - Microstructure analytics: spread, depth, trade clustering

    Data volume: ~5-50 rows per second per instrument (varies by liquidity).
    For a single actively-traded instrument: potentially millions per day.

    WHY UTM (Unix Time Milliseconds):
        IG sends UTM as a millisecond timestamp. Storing it as BigInteger
        preserves precision for latency analysis and order book
        reconstruction. PostgreSQL's TIMESTAMPTZ has microsecond precision,
        but storing the raw UTM enables direct millisecond-difference
        calculations without timestamp conversion overhead.
    """

    __tablename__ = "ticks"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    epic = Column(String, nullable=False)
    utm = Column(BigInteger, nullable=False)  # Unix time in milliseconds
    bid = Column(Numeric(12, 5))
    ask = Column(Numeric(12, 5))
    last_traded_price = Column(Numeric(12, 5))
    last_traded_volume = Column(Numeric(12, 5))

    # WHY the compound index on (epic, utm):
    #   Tick queries always filter by instrument (epic) and often by time
    #   range (utm). The compound index supports both patterns efficiently.
    __table_args__ = (
        Index("idx_ticks_epic_utm", "epic", "utm"),
    )


class Quote(Base):
    """
    Level 1 quote snapshots — bid/ask at a point in time.

    WHY separate from ticks:
        Not every price update generates a trade. Quote data captures the
        state of the order book (best bid/offer) at a point in time,
        regardless of whether a trade occurred. This is essential for:
        - Spread analysis: bid-ask spread over time
        - Market depth estimation: quote sizes indicate liquidity
        - Volume-weighted mid price (VWMP): using quote sizes as weights

    DIFFERENCE FROM tick data:
        - Ticks: a trade happened (last_traded_price is set)
        - Quotes: the market state changed (bid/ask updated, no trade)

    This distinction is critical for Bryan Downing's Order Flow Toxicity
    Detector, which analyses the relationship between quote updates and
    trade arrivals to detect informed trading.
    """

    __tablename__ = "quotes"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    epic = Column(String, nullable=False)
    ts = Column(DateTime(timezone=True), nullable=False)
    bid = Column(Numeric(12, 5))
    ask = Column(Numeric(12, 5))
    bid_size = Column(Integer)
    ask_size = Column(Integer)

    __table_args__ = (Index("idx_quotes_epic_ts", "epic", "ts"),)


# =============================================================================
# Schema creation convenience
# =============================================================================


def create_all_tables(dsn: str) -> None:
    """
    Create all tables in the database.

    WHY this is separate from Alembic:
        Alembic manages schema migrations in production (altering existing
        tables, adding columns, etc.). create_all_tables() is a development
        convenience that creates the full schema from scratch. In production,
        use: alembic upgrade head

    Args:
        dsn: PostgreSQL connection string (from config.pg_settings.dsn).
    """
    engine = create_engine(dsn)
    Base.metadata.create_all(engine)
    engine.dispose()
