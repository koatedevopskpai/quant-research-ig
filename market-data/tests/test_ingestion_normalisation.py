"""
Ingestion normalisation: opening prices and UTC timestamps.

Both were concrete defects from the 2026-10 review:
  * open_bid/open_ask were read from closePrice, silently destroying the open;
  * naive timestamps were stored as-is, so TIMESTAMPTZ interpreted them in the
    session timezone and the same row meant different instants per machine.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ig_market_data.ingestion import _parse_snapshot_time, ingest_prices


class TestSnapshotTime:
    def test_naive_timestamp_is_treated_as_utc(self):
        ts = _parse_snapshot_time({"snapshotTime": "2025-01-02T16:00:00"})
        assert ts is not None
        assert ts.tzinfo is not None
        assert ts == datetime(2025, 1, 2, 16, 0, tzinfo=timezone.utc)

    def test_utc_field_is_preferred(self):
        ts = _parse_snapshot_time({
            "snapshotTimeUTC": "2025-01-02T16:00:00Z",
            "snapshotTime": "2025-01-02T15:00:00",
        })
        assert ts == datetime(2025, 1, 2, 16, 0, tzinfo=timezone.utc)

    def test_explicit_offset_is_converted_to_utc(self):
        ts = _parse_snapshot_time({"snapshotTimeUTC": "2025-01-02T17:00:00+01:00"})
        assert ts == datetime(2025, 1, 2, 16, 0, tzinfo=timezone.utc)

    def test_slash_format_is_accepted(self):
        ts = _parse_snapshot_time({"snapshotTime": "2025/01/02 16:00:00"})
        assert ts == datetime(2025, 1, 2, 16, 0, tzinfo=timezone.utc)

    def test_missing_returns_none(self):
        assert _parse_snapshot_time({}) is None

    def test_unparseable_returns_none(self):
        assert _parse_snapshot_time({"snapshotTime": "not-a-time"}) is None


class TestOpenPriceIngestion:
    def _run(self, payload):
        captured = {}

        class FakeSession:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def merge(self, row):
                captured["row"] = row

            def commit(self):
                pass

        client = MagicMock()
        client.fetch_prices.return_value = payload
        with patch("ig_market_data.ingestion.create_engine"), \
             patch("ig_market_data.ingestion.Session", return_value=FakeSession()):
            rows = ingest_prices("IX.D.SPTRD.IFD.IP", resolution="DAY",
                                 max_points=1, client=client)
        return rows, captured.get("row")

    def test_open_comes_from_open_not_close(self):
        rows, row = self._run([{
            "snapshotTime": "2025-01-02T16:00:00",
            "openPrice": {"bid": "4700.0", "ask": "4701.0"},
            "closePrice": {"bid": "4735.8", "ask": "4736.8"},
            "highPrice": {"bid": "4740.0", "ask": "4741.0"},
            "lowPrice": {"bid": "4690.0", "ask": "4691.0"},
        }])
        assert rows == 1
        assert float(row.open_bid) == pytest.approx(4700.0)
        assert float(row.open_ask) == pytest.approx(4701.0)
        assert float(row.close_bid) == pytest.approx(4735.8)

    def test_missing_open_price_is_none_not_close(self):
        rows, row = self._run([{
            "snapshotTime": "2025-01-02T16:00:00",
            "closePrice": {"bid": "4735.8", "ask": "4736.8"},
        }])
        assert rows == 1
        assert row.open_bid is None
        assert row.open_ask is None

    def test_stored_timestamp_is_utc_aware(self):
        _, row = self._run([{
            "snapshotTime": "2025-01-02T16:00:00",
            "closePrice": {"bid": "1", "ask": "1"},
        }])
        assert row.ts.tzinfo is not None
        assert row.ts.utcoffset().total_seconds() == 0
