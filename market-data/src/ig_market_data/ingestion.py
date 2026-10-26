"""
ingestion.py — Scheduled data ingestion from IG REST API to PostgreSQL.

WHY THIS EXISTS:
    Quantitative research requires historical data. The IG REST API provides
    OHLCV candles at various resolutions, but calling the API for every
    research session is slow and rate-limited. The ingestion pipeline:

    1. Fetches price data from IG for specified instruments.
    2. Stores it in PostgreSQL via SQLAlchemy models.
    3. Handles upserts (insert or update) to avoid duplicate rows.
    4. Can be run on-demand (single instrument) or scheduled (full refresh).

ARCHITECTURE (Quant Beckman's Data Ingestion Layer):
    This pipeline is the first stage of Beckman's end-to-end solution
    pipeline: Data Ingestion -> Signal Extraction -> Signal Calculation ->
    Sizing Logic -> Risk Overlays -> Execution.

    Without clean, reliable data ingestion, the entire downstream pipeline
    is compromised. This module enforces data integrity at every step.

USAGE:
    # Ingest a single instrument (on-demand)
    python -c "
        from ig_market_data.ingestion import ingest_prices
        ingest_prices('IX.D.SPTRD.IFD.IP', resolution='DAY', max_points=500)
    "

    # Ingest multiple instruments (bulk)
    python -c "
        from ig_market_data.ingestion import bulk_ingest
        bulk_ingest(['IX.D.SPTRD.IFD.IP', 'CS.D.EURUSD.CFD.IP'])
    "

SCHEDULING (cron / task scheduler):
    # Daily refresh at 18:00 UTC (after market close)
    0 18 * * 1-5 cd /path/to/project && python -c "
        from ig_market_data.ingestion import bulk_ingest
        from ig_market_data.client import IGClient
        client = IGClient()
        categories = client.list_categories()
        # ... select instruments ...
        bulk_ingest(epics)
    "

DESIGN DECISIONS:
    1. Batch upserts: SQLAlchemy's merge() inserts new rows and updates
       existing ones in a single transaction. This is idempotent — running
       the same ingestion twice produces no duplicates.

    2. Configurable max_points: The IG API limits historical data depth
       per resolution (e.g., ~2000 days for daily data). The caller
       specifies how much history to fetch.

    3. Progress logging: Each instrument is logged individually, making it
       easy to identify which instruments failed in a bulk run.

METHODOLOGY NOTE (Bryan Downing):
    While Downing focuses on real-time tick data for HFT, the historical
    data pipeline is equally important for backtesting strategies before
    deploying them live. Every HFT strategy must be validated on historical
    data first — this is the "Know" stage of Beckman's cycle.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ig_market_data.client import IGClient
from ig_market_data.config import pg_settings
from ig_market_data.models import OHLCVDaily, OHLCVHourly


_TS_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
    "%Y/%m/%d %H:%M:%S",
)


def _parse_snapshot_time(entry: dict) -> Optional[datetime]:
    """
    Return a timezone-aware UTC timestamp for an IG price entry.

    WHY THIS EXISTS:
        IG returns `snapshotTimeUTC` (ISO with an offset) and `snapshotTime`
        (ISO with NO timezone). `datetime.fromisoformat` on the latter produces
        a naive datetime, and storing a naive value in TIMESTAMPTZ makes
        PostgreSQL interpret it in the session timezone — so the same row means
        different instants on different machines. Timestamps are now normalised
        explicitly, preferring the provider's UTC field.
    """
    raw = entry.get("snapshotTimeUTC") or entry.get("snapshotTime")
    if not raw:
        return None

    text = str(raw).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        parsed = None
        for fmt in _TS_FORMATS:
            try:
                parsed = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if parsed is None:
            return None

    if parsed.tzinfo is None:
        # IG's snapshotTime carries no zone; the feed is UTC by convention.
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolution mapping
# ---------------------------------------------------------------------------

# WHY mapping IG resolutions to model classes:
#   IG uses string identifiers for resolutions ("DAY", "HOUR").
#   Each resolution maps to a specific SQLAlchemy model class.
#   This dictionary centralises the mapping and makes it easy to add
#   new resolutions.
_RESOLUTION_MODEL = {
    "DAY": OHLCVDaily,
    "HOUR": OHLCVHourly,
}


def _get_model(resolution: str):
    """
    Get the SQLAlchemy model class for a given IG resolution string.

    Raises ValueError for unsupported resolutions.

    WHY explicit mapping vs. dynamic resolution:
        Dynamic resolution (e.g., class name = f"OHLCV{resolution}")
        would work but violates the principle of explicit over implicit.
        A dictionary makes supported resolutions visible at a glance.
    """
    model = _RESOLUTION_MODEL.get(resolution)
    if model is None:
        supported = list(_RESOLUTION_MODEL.keys())
        raise ValueError(
            f"Unsupported resolution: '{resolution}'. "
            f"Supported: {supported}"
        )
    return model


# =============================================================================
# Ingestion functions
# =============================================================================


def ingest_prices(
    epic: str,
    resolution: str = "DAY",
    max_points: int = 500,
    client: Optional[IGClient] = None,
) -> int:
    """
    Fetch price history for an instrument and store in PostgreSQL.

    Args:
        epic: The instrument epic (e.g., "IX.D.SPTRD.IFD.IP").
        resolution: Candle resolution ("DAY" or "HOUR").
        max_points: Maximum candles to fetch.
        client: Optional IGClient instance. If None, creates a new one.
            Reusing a client avoids re-authentication overhead.

    Returns:
        Number of rows ingested.

    FLOW:
        1. Create IGClient if not provided.
        2. Fetch price data from IG REST API.
        3. Transform IG JSON response to SQLAlchemy model instances.
        4. Upsert into PostgreSQL (merge on primary key).
        5. Commit and return row count.

    WHY upsert vs. insert:
        Running the ingestion twice for the same day should update existing
        rows (e.g., if the original candle was incomplete). SQLAlchemy's
        merge() handles this automatically — it inserts new rows and
        updates existing ones in one operation.

    EXAMPLE:
        ingest_prices("IX.D.SPTRD.IFD.IP", resolution="DAY", max_points=100)
    """
    own_client = False
    if client is None:
        client = IGClient()
        own_client = True

    model_class = _get_model(resolution)

    # WHY creating a new engine per ingestion call:
    #   SQLAlchemy engines are designed to be long-lived, but creating one
    #   per call is acceptable for on-demand ingestion. The scheduled
    #   pipeline (bulk_ingest) creates a single engine for all instruments.
    engine = create_engine(pg_settings.dsn)

    try:
        # WHY the IG API parameter is called "max" not "max_points":
        #   IG's API uses "max" as the query parameter name for the maximum
        #   number of data points. The client.py method takes max_points
        #   for clarity and maps it internally.
        prices = client.fetch_prices(
            epic, resolution=resolution, max_points=max_points
        )

        if not prices:
            logger.warning("No prices returned for %s (%s)", epic, resolution)
            return 0

        rows = 0
        with Session(engine) as session:
            for p in prices:
                ts = _parse_snapshot_time(p)
                if ts is None:
                    logger.warning(
                        "Skipping row with missing/unparseable timestamp: %s",
                        p.get("snapshotTimeUTC") or p.get("snapshotTime"),
                    )
                    continue

                open_ = p.get("openPrice", {})
                close = p.get("closePrice", {})
                high = p.get("highPrice", {})
                low = p.get("lowPrice", {})

                # WHY merge() vs. add():
                #   merge() checks for an existing row with the same primary
                #   key (epic, ts). If found, it updates the columns. If
                #   not found, it inserts a new row. This makes the
                #   ingestion idempotent.
                row = model_class(
                    epic=epic,
                    ts=ts,
                    close_bid=close.get("bid"),
                    close_ask=close.get("ask"),
                    high_bid=high.get("bid"),
                    high_ask=high.get("ask"),
                    low_bid=low.get("bid"),
                    low_ask=low.get("ask"),
                    open_bid=open_.get("bid"),
                    open_ask=open_.get("ask"),
                    volume=p.get("lastTradedVolume"),
                )
                session.merge(row)
                rows += 1

            session.commit()

        logger.info(
            "Ingested %d rows for %s (%s)", rows, epic, resolution
        )
        return rows

    finally:
        engine.dispose()
        if own_client:
            # WHY deleting the client we created:
            #   If we created a fresh IGClient, clean it up to release the
            #   HTTP connection pool. If the caller provided one, leave it
            #   for the caller to manage.
            del client


def bulk_ingest(
    epics: list[str],
    resolution: str = "DAY",
    max_points: int = 500,
) -> dict[str, int]:
    """
    Ingest price data for multiple instruments.

    Args:
        epics: List of instrument epics to ingest.
        resolution: Candle resolution.
        max_points: Maximum candles per instrument.

    Returns:
        Dictionary mapping epic -> rows ingested for that epic.

    WHY bulk_ingest exists:
        A quant researcher may want data for 50 instruments at once.
        Calling ingest_prices() 50 times creates 50 separate database
        connections. bulk_ingest() creates a single IGClient and reuses it
        for all instruments, reducing auth overhead by ~50ms per instrument.

    WHY not multi-threaded:
        The IG API has rate limits. Sequential requests are safer and
        simpler. If speed becomes an issue, add a small delay between
        requests (e.g., time.sleep(0.2) to stay under 5 req/s).

    EXAMPLE:
        epics = ["IX.D.SPTRD.IFD.IP", "CS.D.EURUSD.CFD.IP"]
        results = bulk_ingest(epics, resolution="DAY", max_points=1000)
        print(results)
        # {'IX.D.SPTRD.IFD.IP': 1000, 'CS.D.EURUSD.CFD.IP': 1000}
    """
    client = IGClient()
    results: dict[str, int] = {}

    for epic in epics:
        try:
            count = ingest_prices(
                epic,
                resolution=resolution,
                max_points=max_points,
                client=client,
            )
            results[epic] = count
        except Exception as e:
            logger.error("Failed to ingest %s: %s", epic, e)
            results[epic] = 0

    return results
