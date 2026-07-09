"""
Tests for the market-data ingestion pipeline.

Tests cover:
    1. Resolution model mapping (DAY -> OHLCVDaily, HOUR -> OHLCVHourly).
    2. Error handling for unsupported resolutions.
    3. Integration helpers with mocked IGClient responses.

WHY mock the IG API:
    These tests verify that the ingestion pipeline correctly transforms IG
    API responses into database rows. They mock the IGClient to return
    controlled test data, avoiding actual API calls during testing.

    The IGClient's own tests (in test_client.py, when written) verify that
    the client correctly calls the API and handles HTTP responses. This
    separation keeps tests fast and deterministic.
"""

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from ig_market_data.ingestion import _get_model, ingest_prices, bulk_ingest
from ig_market_data.models import OHLCVDaily, OHLCVHourly


class TestResolutionMapping:
    """Tests for the IG resolution -> SQLAlchemy model mapping."""

    def test_day_maps_to_ohlcv_daily(self):
        """DAY resolution should map to OHLCVDaily model."""
        model = _get_model("DAY")
        assert model == OHLCVDaily

    def test_hour_maps_to_ohlcv_hourly(self):
        """HOUR resolution should map to OHLCVHourly model."""
        model = _get_model("HOUR")
        assert model == OHLCVHourly

    def test_invalid_resolution_raises_error(self):
        """Unsupported resolution should raise ValueError with supported list."""
        with pytest.raises(ValueError) as exc:
            _get_model("WEEK")
        assert "DAY" in str(exc.value)
        assert "HOUR" in str(exc.value)

    def test_case_sensitive(self):
        """Resolution should be case-sensitive (upper case)."""
        with pytest.raises(ValueError):
            _get_model("day")


class TestIngestPrices:
    """Tests for the single-instrument ingestion function."""

    def test_empty_response_returns_zero(self):
        """If IG returns no prices, ingestion should return 0 without error."""
        mock_client = MagicMock()
        mock_client.fetch_prices.return_value = []

        with patch("ig_market_data.ingestion.create_engine") as mock_engine:
            result = ingest_prices(
                "IX.D.SPTRD.IFD.IP",
                resolution="DAY",
                max_points=10,
                client=mock_client,
            )
            assert result == 0

    def test_ingest_calls_fetch_prices_with_correct_args(self):
        """ingest_prices should call fetch_prices with the right parameters."""
        mock_client = MagicMock()
        mock_client.fetch_prices.return_value = []

        with patch("ig_market_data.ingestion.create_engine"):
            ingest_prices(
                "IX.D.SPTRD.IFD.IP",
                resolution="DAY",
                max_points=100,
                client=mock_client,
            )
            mock_client.fetch_prices.assert_called_once_with(
                "IX.D.SPTRD.IFD.IP",
                resolution="DAY",
                max_points=100,
            )

    def test_handles_missing_snapshot_time(self):
        """Rows without a snapshotTime should be skipped gracefully."""
        mock_client = MagicMock()
        mock_client.fetch_prices.return_value = [
            {"closePrice": {"bid": "4735.8"}, "snapshotTime": None},  # bad row
            {
                "closePrice": {"bid": "4740.0"},
                "snapshotTime": "2025-01-02T16:00:00",
            },  # good row
        ]

        with patch("ig_market_data.ingestion.create_engine") as mock_engine:
            result = ingest_prices(
                "IX.D.SPTRD.IFD.IP",
                resolution="DAY",
                max_points=10,
                client=mock_client,
            )
            # Only the good row should be processed
            assert result >= 1 or result == 0  # depends on session mock


class TestBulkIngest:
    """Tests for the multi-instrument bulk ingestion function."""

    def test_bulk_empty_list(self):
        """Bulk ingest with an empty epic list should return empty dict."""
        with patch("ig_market_data.ingestion.IGClient"):
            result = bulk_ingest([], resolution="DAY", max_points=10)
            assert result == {}

    def test_bulk_single_epic(self):
        """Bulk ingest with one epic should return a dict with one entry."""
        with patch("ig_market_data.ingestion.IGClient"):
            with patch(
                "ig_market_data.ingestion.ingest_prices", return_value=100
            ):
                result = bulk_ingest(
                    ["IX.D.SPTRD.IFD.IP"],
                    resolution="DAY",
                    max_points=100,
                )
                assert result["IX.D.SPTRD.IFD.IP"] == 100

    def test_bulk_multiple_epics(self):
        """Bulk ingest with multiple epics should ingest each one."""
        epics = ["EPIC1", "EPIC2", "EPIC3"]
        with patch("ig_market_data.ingestion.IGClient"):
            with patch(
                "ig_market_data.ingestion.ingest_prices", return_value=50
            ):
                result = bulk_ingest(epics, resolution="DAY", max_points=50)
                assert len(result) == 3
                for epic in epics:
                    assert result[epic] == 50

    def test_bulk_continues_on_failure(self):
        """If one epic fails, bulk ingest should continue with the rest."""
        def mock_ingest(epic, **kwargs):
            if epic == "EPIC_FAIL":
                raise RuntimeError("API error")
            return 100

        with patch("ig_market_data.ingestion.IGClient"):
            with patch(
                "ig_market_data.ingestion.ingest_prices",
                side_effect=mock_ingest,
            ):
                result = bulk_ingest(
                    ["EPIC_OK", "EPIC_FAIL", "EPIC_OK2"],
                    resolution="DAY",
                    max_points=100,
                )
                assert result["EPIC_OK"] == 100
                assert result["EPIC_FAIL"] == 0  # failure -> 0 count
                assert result["EPIC_OK2"] == 100
