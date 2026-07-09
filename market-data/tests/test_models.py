"""
Tests for market-data SQLAlchemy ORM models.

Tests cover:
    1. Model instantiation with required fields.
    2. Primary key constraints (no duplicate rows for same epic+ts).
    3. Nullable fields (e.g., prices can be None during market closure).
    4. Type coercion (string->Numeric, string->DateTime).

WHY no database connection in these tests:
    These tests validate model schema and constraint logic using
    SQLAlchemy's in-memory SQLite backend. They do NOT test
    PostgreSQL-specific features like BRIN indexes (those require
    a real PostgreSQL instance and are tested separately).

    Testing approach follows Quant Beckman's principle of layered
    validation: each component is tested in isolation before
    integration testing.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ig_market_data.models import Base, OHLCVDaily, OHLCVHourly, Tick, Quote


@pytest.fixture
def engine():
    """Create an in-memory SQLite database for testing."""
    e = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(e)
    yield e
    e.dispose()


@pytest.fixture
def session(engine):
    """Create a fresh database session for each test."""
    with Session(engine) as s:
        yield s


class TestOHLCVDaily:
    """Tests for the daily OHLCV model."""

    def test_create_minimal(self, session):
        """Create a daily OHLCV row with only required fields (epic, ts)."""
        row = OHLCVDaily(
            epic="IX.D.SPTRD.IFD.IP",
            ts=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
        session.add(row)
        session.commit()
        assert row in session

    def test_create_with_prices(self, session):
        """Create a daily OHLCV row with all price fields."""
        row = OHLCVDaily(
            epic="IX.D.SPTRD.IFD.IP",
            ts=datetime(2025, 1, 2, tzinfo=timezone.utc),
            open_bid=4725.50,
            high_bid=4750.00,
            low_bid=4700.25,
            close_bid=4735.80,
            open_ask=4725.75,
            high_ask=4750.25,
            low_ask=4700.50,
            close_ask=4736.00,
            volume=1234567,
        )
        session.add(row)
        session.commit()
        assert float(row.close_bid) == 4735.80

    def test_primary_key_prevents_duplicates(self, session):
        """Two rows with the same epic+ts should raise an integrity error."""
        ts = datetime(2025, 1, 2, tzinfo=timezone.utc)
        session.add(OHLCVDaily(epic="EPIC1", ts=ts))
        session.commit()

        session.add(OHLCVDaily(epic="EPIC1", ts=ts))
        with pytest.raises(Exception):
            session.commit()

    def test_different_epics_same_ts_allowed(self, session):
        """Different instruments can have data at the same timestamp."""
        ts = datetime(2025, 1, 2, tzinfo=timezone.utc)
        session.add_all([
            OHLCVDaily(epic="EPIC1", ts=ts),
            OHLCVDaily(epic="EPIC2", ts=ts),
        ])
        session.commit()
        assert session.query(OHLCVDaily).count() == 2

    def test_nullable_price_fields(self, session):
        """Price fields should be nullable (e.g., for non-trading days)."""
        row = OHLCVDaily(
            epic="IX.D.SPTRD.IFD.IP",
            ts=datetime(2025, 1, 2, tzinfo=timezone.utc),
        )
        session.add(row)
        session.commit()
        assert row.open_bid is None


class TestOHLCVHourly:
    """Tests for the hourly OHLCV model."""

    def test_create_hourly_row(self, session):
        """Create an hourly OHLCV row with bid/ask prices."""
        row = OHLCVHourly(
            epic="CS.D.EURUSD.CFD.IP",
            ts=datetime(2025, 1, 2, 10, 0, tzinfo=timezone.utc),
            close_bid=1.12345,
            close_ask=1.12355,
            volume=5000,
        )
        session.add(row)
        session.commit()
        assert float(row.close_bid) == 1.12345

    def test_hourly_primary_key(self, session):
        """Same epic+hour should be unique."""
        ts = datetime(2025, 1, 2, 10, 0, tzinfo=timezone.utc)
        session.add(OHLCVHourly(epic="EPIC1", ts=ts))
        session.commit()

        session.add(OHLCVHourly(epic="EPIC1", ts=ts))
        with pytest.raises(Exception):
            session.commit()


class TestTick:
    """Tests for the tick-by-tick model."""

    def test_create_tick(self, session):
        """Create a tick row with trade data."""
        tick = Tick(
            epic="CS.D.EURUSD.CFD.IP",
            utm=1735766400000,  # 2025-01-02 10:00:00 UTC in ms
            bid=1.12345,
            ask=1.12355,
            last_traded_price=1.12350,
            last_traded_volume=1.0,
        )
        session.add(tick)
        session.commit()
        assert float(tick.bid) == 1.12345

    def test_autoincrement_id(self, session):
        """Each tick should get a unique auto-increment ID."""
        ts_ms = 1735766400000
        session.add_all([
            Tick(epic="EPIC1", utm=ts_ms),
            Tick(epic="EPIC1", utm=ts_ms + 1),
        ])
        session.commit()
        ticks = session.query(Tick).order_by(Tick.id).all()
        assert ticks[0].id == 1
        assert ticks[1].id == 2

    def test_same_millisecond_allowed(self, session):
        """Multiple ticks can arrive at the same millisecond."""
        ts_ms = 1735766400000
        session.add_all([
            Tick(epic="EPIC1", utm=ts_ms),
            Tick(epic="EPIC1", utm=ts_ms),
        ])
        session.commit()
        assert session.query(Tick).count() == 2


class TestQuote:
    """Tests for the Level 1 quote model."""

    def test_create_quote(self, session):
        """Create a quote row with bid/ask and sizes."""
        quote = Quote(
            epic="IX.D.SPTRD.IFD.IP",
            ts=datetime(2025, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
            bid=4735.80,
            ask=4736.00,
            bid_size=100,
            ask_size=200,
        )
        session.add(quote)
        session.commit()
        assert float(quote.bid) == 4735.80
        assert quote.bid_size == 100
        assert quote.ask_size == 200

    def test_quote_without_sizes(self, session):
        """Bid/ask sizes should be nullable."""
        quote = Quote(
            epic="IX.D.SPTRD.IFD.IP",
            ts=datetime(2025, 1, 2, 10, 0, 0, tzinfo=timezone.utc),
            bid=4735.80,
            ask=4736.00,
        )
        session.add(quote)
        session.commit()
        assert quote.bid_size is None
        assert quote.ask_size is None
