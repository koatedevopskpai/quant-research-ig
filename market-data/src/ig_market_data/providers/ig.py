"""
providers/ig.py — IG adapter on the provider contract.

STATUS: transport unverified as of 2026-10-26.
    A single authorised demo run confirmed two defects and proved the fixes
    below work (63 ticks in 30 s, 401-548 ms provider timestamps, delivery after
    reconnect), after which the account was suspended. Until it is restored, IG
    delivery is unverified — see `docs/phase-3-market-data.md`.

WHAT THIS WRAPS
    `ig_market_data.stream.IGStreamer` for transport/token handling. The
    subscription bridge there now implements the SDK's `onItemUpdate` and reads
    `getFields()`, so this adapter only maps IG's string fields onto
    `MarketEvent`.

CAPABILITY HONESTY
    IG's API offers L1 top-of-book and trade ticks. It does **not** expose an L2
    depth feed and does not include bid/ask sizes on L1. Claiming otherwise was
    one of the original review findings; `capabilities` says so explicitly.
"""

from __future__ import annotations

import time
from typing import Callable

from ig_market_data.providers.base import Capabilities, EventKind, MarketEvent
from ig_market_data.stream import IGStreamer


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        from math import isfinite

        number = float(text)
        return number if isfinite(number) else None
    except (TypeError, ValueError):
        return None


class IGProvider:
    name = "ig"
    capabilities = Capabilities(
        l1=True, l2=False, trades=True, sizes=False, historical=True, live=True
    )

    def __init__(self, streamer: IGStreamer | None = None) -> None:
        self._streamer = streamer or IGStreamer()
        self._connected = False

    # -- provider protocol -------------------------------------------------
    def connect(self) -> None:
        self._streamer.connect()
        self._connected = True

    def disconnect(self) -> None:
        self._streamer.disconnect()
        self._connected = False

    def subscribe(
        self,
        epic: str,
        kind: EventKind,
        callback: Callable[[MarketEvent], None],
    ) -> None:
        if kind == "l2":
            raise ValueError(
                "IG's API provides no L2 depth feed; subscribe to 'l1' or 'trade'"
            )
        if not self._connected:
            raise RuntimeError("connect() before subscribe()")

        if kind == "l1":
            def on_l1(item_name: str, fields: dict[str, str]) -> None:
                receive_ns = time.time_ns()
                # IG L1 carries UPDATE_TIME as HH:MM:SS with no date, so it
                # cannot be turned into an epoch timestamp without guessing.
                # Record that we used local receive time instead of inventing one.
                provider_ns = receive_ns
                raw = dict(fields)
                raw["provider_ts_source"] = "receive"
                callback(
                    MarketEvent(
                        epic=epic,
                        kind="l1",
                        provider_ts_ns=provider_ns,
                        receive_ts_ns=receive_ns,
                        bid=_as_float(fields.get("BID")),
                        ask=_as_float(fields.get("OFR") or fields.get("OFFER")),
                        book_action="replace",  # L1 is a snapshot, not a diff
                        raw=raw,
                    )
                )

            self._streamer.subscribe_l1(epic, on_l1)
            return

        def on_tick(item_name: str, fields: dict[str, str]) -> None:
            receive_ns = time.time_ns()
            provider_ns = receive_ns
            utm = fields.get("UTM")
            if utm:
                # IG sends UTM as Unix milliseconds. Parse as an integer: going
                # through float64 loses tens of nanoseconds at current epoch
                # magnitudes (1761486000123 * 1e6 came out 64 ns high).
                text = str(utm).strip()
                try:
                    if text.isdigit():
                        provider_ns = int(text) * 1_000_000
                    else:
                        provider_ns = int(round(float(text) * 1_000_000))
                except (TypeError, ValueError):
                    provider_ns = receive_ns
            callback(
                MarketEvent(
                    epic=epic,
                    kind="trade",
                    provider_ts_ns=provider_ns,
                    receive_ts_ns=receive_ns,
                    last=_as_float(fields.get("LTP")),
                    size=_as_float(fields.get("LTV")),
                    raw=dict(fields),
                )
            )

        self._streamer.subscribe_ticks(epic, on_tick)
