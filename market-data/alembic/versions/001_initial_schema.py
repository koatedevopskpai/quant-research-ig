"""
Initial schema: create OHLCV, tick, and quote tables.

WHY THIS MIGRATION:
    This is the first Alembic migration. It creates all tables needed for
    market data storage. Subsequent migrations will add columns, indexes,
    or tables as the schema evolves.

DOWNGRADE NOTE:
    Dropping these tables in a downgrade destroys all market data.
    In production, downgrades should only be performed after verifying
    a full backup exists.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # ohlcv_daily — Daily OHLCV candles
    # ------------------------------------------------------------------
    # WHY BRIN index on timestamp:
    #   BRIN indexes are ideal for append-only time-series data. They are
    #   significantly smaller than B-tree indexes and provide excellent
    #   scan performance for time-range queries.
    #
    # WHY compound primary key (epic, ts):
    #   Ensures exactly one row per instrument per day. This simplifies
    #   upsert logic in the ingestion pipeline.
    op.create_table(
        "ohlcv_daily",
        sa.Column("epic", sa.String(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open_bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("high_bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("low_bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("close_bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("open_ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("high_ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("low_ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("close_ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("epic", "ts"),
    )
    op.create_index(
        "idx_ohlcv_daily_ts", "ohlcv_daily", ["ts"], postgresql_using="brin"
    )

    # ------------------------------------------------------------------
    # ohlcv_hourly — Hourly OHLCV candles
    # ------------------------------------------------------------------
    op.create_table(
        "ohlcv_hourly",
        sa.Column("epic", sa.String(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("open_bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("high_bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("low_bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("close_bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("open_ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("high_ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("low_ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("close_ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("volume", sa.BigInteger(), nullable=True),
        sa.PrimaryKeyConstraint("epic", "ts"),
    )
    op.create_index(
        "idx_ohlcv_hourly_ts", "ohlcv_hourly", ["ts"], postgresql_using="brin"
    )

    # ------------------------------------------------------------------
    # ticks — Tick-by-tick trade data
    # ------------------------------------------------------------------
    # WHY autoincrement primary key:
    #   Ticks can arrive at the same millisecond for the same instrument.
    #   A compound primary key of (epic, utm) could have collisions.
    #   An autoincrement id ensures uniqueness regardless of arrival timing.
    #
    # WHY compound index on (epic, utm):
    #   Most tick queries filter by instrument and time range. This index
    #   supports both WHERE clauses efficiently.
    op.create_table(
        "ticks",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("epic", sa.String(), nullable=False),
        sa.Column("utm", sa.BigInteger(), nullable=False),
        sa.Column("bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("last_traded_price", sa.Numeric(12, 5), nullable=True),
        sa.Column("last_traded_volume", sa.Numeric(12, 5), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_ticks_epic_utm", "ticks", ["epic", "utm"])

    # ------------------------------------------------------------------
    # quotes — Level 1 quote snapshots
    # ------------------------------------------------------------------
    op.create_table(
        "quotes",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("epic", sa.String(), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("bid", sa.Numeric(12, 5), nullable=True),
        sa.Column("ask", sa.Numeric(12, 5), nullable=True),
        sa.Column("bid_size", sa.Integer(), nullable=True),
        sa.Column("ask_size", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("idx_quotes_epic_ts", "quotes", ["epic", "ts"])


def downgrade() -> None:
    op.drop_table("quotes")
    op.drop_table("ticks")
    op.drop_table("ohlcv_hourly")
    op.drop_table("ohlcv_daily")
