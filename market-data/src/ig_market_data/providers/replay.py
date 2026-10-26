"""
providers/replay.py — deterministic, credential-free playback of market events.

WHY THIS EXISTS
    The I in CI must not be an account. This provider turns a JSONL capture into
    the same `MarketEvent` stream a live provider produces, with two clocks kept
    distinct:

      * provider_ts_ns  — when the venue said it happened (ordering, latency)
      * receive_ts_ns   — when the original capture received it

    Replay can therefore be fast (ordering and causality only) or paced by
    provider time (to exercise timing/backpressure). Tests use fast mode; demos
    and reconnect simulations can use paced mode.

FORMAT
    One JSON object per line:

    {"epic": "CS.D.EURUSD.CFD.IP", "kind": "l1", "provider_ts_ns": 1761...,
     "receive_ts_ns": 1761..., "bid": 1.11937, "ask": 1.11943,
     "book_action": "replace", "raw": {"BID": "1.11937", "OFR": "1.11943"}}
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ig_market_data.providers.base import (
    Capabilities,
    EventKind,
    MarketEvent,
    replay_key,
)


class ReplayProvider:
    name = "replay"
    capabilities = Capabilities(
        l1=True, l2=True, trades=True, sizes=True, historical=True, live=False
    )

    def __init__(self, events: Iterable[MarketEvent]) -> None:
        self._events: list[MarketEvent] = sorted(events, key=replay_key)
        self._subs: list[tuple[str, EventKind, Callable[[MarketEvent], None]]] = []
        self._connected = False
        self.dispatched = 0

    # -- construction ------------------------------------------------------
    @classmethod
    def from_jsonl(cls, path: str | Path) -> "ReplayProvider":
        path = Path(path)
        events: list[MarketEvent] = []
        with path.open(encoding="utf-8") as handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    payload = json.loads(line)
                    payload.setdefault("raw", {})
                    events.append(MarketEvent(**payload))
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}:{lineno}: {exc}") from exc
        return cls(events)

    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for event in self._events:
                handle.write(json.dumps(asdict(event), separators=(",", ":")) + "\n")
        return path

    # -- provider protocol -------------------------------------------------
    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def subscribe(
        self,
        epic: str,
        kind: EventKind,
        callback: Callable[[MarketEvent], None],
    ) -> None:
        self._subs.append((epic, kind, callback))

    # -- playback ----------------------------------------------------------
    def events(self) -> Iterator[MarketEvent]:
        return iter(self._events)

    def replay(self, speed: float = 0.0) -> int:
        """
        Dispatch every event to matching subscribers.

        Args:
            speed: 0 replays as fast as possible (tests/CI). Any positive value
                paces by provider time, divided by `speed` (e.g. 10.0 = 10x).
        """
        if not self._connected:
            raise RuntimeError("connect() before replay(); replay is a provider, not a script")
        last_ns: int | None = None
        for event in self._events:
            for epic, kind, callback in self._subs:
                if event.epic == epic and event.kind == kind:
                    callback(event)
                    self.dispatched += 1
            if speed > 0 and last_ns is not None:
                delta = event.provider_ts_ns - last_ns
                if delta > 0:
                    time.sleep(delta / 1e9 / speed)
            last_ns = event.provider_ts_ns
        return self.dispatched
