"""
providers/dukascopy.py — free historical FX tick data (bid/ask), no account.

WHY DUKASCOPY
    It is the closest free analogue to IG's FX/CFD ticks: real bid/ask with
    millisecond timestamps and volume, years deep, no signup. That makes it the
    right source for building realistic fixtures and for backtests while the IG
    account is unavailable. It is **historical only** — no streaming, so it
    cannot test reconnect behaviour.

FORMAT (public knowledge, implemented here)
    URL:  https://datafeed.dukascopy.com/datafeed/{INSTRUMENT}/{YYYY}/{MM}/{DD}/{HH}h_ticks.bi5
          MM is ZERO-INDEXED (00 = January).
    Body: LZMA "alone" stream. Decompressed records are 20 bytes, big-endian:
          uint32 ms_offset_within_hour, uint32 ask, uint32 bid,
          uint32 ask_volume, uint32 bid_volume.
    Price: integer / 10**digits, where digits depends on the instrument
           (EURUSD 5, USDJPY 3, ...). Pass `digits` if the instrument is unusual.

LICENSING
    The data belongs to Dukascopy. This module downloads it at runtime; no vendor
    data is committed to the repository. Check their terms before redistributing.

EMITTED EVENTS
    One `l1` "replace" event per record with bid/ask and their sizes. Each record
    is a complete quote for that instant, which is exactly IG's L1 semantics.
"""

from __future__ import annotations

import lzma
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator

import httpx

from ig_market_data.providers.base import Capabilities, MarketEvent
from ig_market_data.providers.replay import ReplayProvider

_BASE = "https://datafeed.dukascopy.com/datafeed"
_RECORD = struct.Struct(">IIIII")
_RECORD_SIZE = _RECORD.size  # 20

# Decimal places per instrument, for the integer -> price conversion.
_DEFAULT_DIGITS: dict[str, int] = {
    "EURUSD": 5, "GBPUSD": 5, "AUDUSD": 5, "NZDUSD": 5,
    "USDCHF": 5, "USDCAD": 5, "EURGBP": 5, "EURCHF": 5,
    "USDJPY": 3, "EURJPY": 3, "GBPJPY": 3, "AUDJPY": 3,
    "XAUUSD": 2, "XAGUSD": 3,
}


def _to_ns(hour_start: datetime) -> int:
    return int(hour_start.timestamp()) * 1_000_000_000


class DukascopyTicks:
    """Historical bid/ask ticks from Dukascopy's public datafeed."""

    name = "dukascopy"
    capabilities = Capabilities(
        l1=True, l2=False, trades=False, sizes=True, historical=True, live=False
    )

    def __init__(
        self,
        cache_dir: str | Path | None = None,
        digits: int | None = None,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.digits = digits
        self._client = client or httpx.Client(timeout=timeout, follow_redirects=True)
        self._owns_client = client is None

    # -- fetching ----------------------------------------------------------
    def hour_url(self, instrument: str, hour_start: datetime) -> str:
        month_zero_indexed = hour_start.month - 1
        return (
            f"{_BASE}/{instrument.upper()}/{hour_start.year:04d}/"
            f"{month_zero_indexed:02d}/{hour_start.day:02d}/"
            f"{hour_start.hour:02d}h_ticks.bi5"
        )

    def _cache_path(self, instrument: str, hour_start: datetime) -> Path | None:
        if self.cache_dir is None:
            return None
        return (
            self.cache_dir / instrument.upper()
            / f"{hour_start:%Y%m%d%H}.bi5"
        )

    def fetch_hour_bytes(self, instrument: str, hour_start: datetime) -> bytes:
        cached = self._cache_path(instrument, hour_start)
        if cached and cached.is_file():
            return cached.read_bytes()
        response = self._client.get(self.hour_url(instrument, hour_start))
        if response.status_code == 404:
            return b""
        response.raise_for_status()
        payload = response.content
        if cached and payload:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_bytes(payload)
        return payload

    # -- decoding ----------------------------------------------------------
    def decode_hour(
        self, instrument: str, hour_start: datetime, payload: bytes
    ) -> list[MarketEvent]:
        if not payload:
            return []
        digits = self.digits if self.digits is not None else _DEFAULT_DIGITS.get(
            instrument.upper(), 5
        )
        scale = float(10**digits)
        base_ns = _to_ns(hour_start)
        events: list[MarketEvent] = []

        raw = lzma.decompress(payload, format=lzma.FORMAT_ALONE)
        if len(raw) % _RECORD_SIZE:
            raise ValueError(
                f"malformed bi5: {len(raw)} bytes is not a multiple of {_RECORD_SIZE}"
            )
        for offset in range(0, len(raw), _RECORD_SIZE):
            ms, ask_i, bid_i, ask_vol, bid_vol = _RECORD.unpack_from(raw, offset)
            ts_ns = base_ns + ms * 1_000_000
            events.append(
                MarketEvent(
                    epic=instrument.upper(),
                    kind="l1",
                    provider_ts_ns=ts_ns,
                    receive_ts_ns=ts_ns,   # historical: no separate receive time
                    bid=bid_i / scale,
                    ask=ask_i / scale,
                    size=float(ask_vol),
                    book_action="replace",
                    raw={"ask_volume": ask_vol, "bid_volume": bid_vol, "ms": ms},
                )
            )
        return events

    def fetch_hour(self, instrument: str, hour_start: datetime) -> list[MarketEvent]:
        if hour_start.tzinfo is None:
            hour_start = hour_start.replace(tzinfo=timezone.utc)
        payload = self.fetch_hour_bytes(instrument, hour_start)
        return self.decode_hour(instrument, hour_start, payload)

    def fetch_range(
        self, instrument: str, start: datetime, end: datetime
    ) -> list[MarketEvent]:
        """Fetch whole hours in [start, end). Times are UTC."""
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        hour = start.replace(minute=0, second=0, microsecond=0)
        events: list[MarketEvent] = []
        while hour < end:
            events.extend(self.fetch_hour(instrument, hour))
            hour += timedelta(hours=1)
        return events

    # -- integration -------------------------------------------------------
    def to_replay(self, events: Iterable[MarketEvent]) -> ReplayProvider:
        """Wrap downloaded ticks as a deterministic provider for tests/demos."""
        return ReplayProvider(events)

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
