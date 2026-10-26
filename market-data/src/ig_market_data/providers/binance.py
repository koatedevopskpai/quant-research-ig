"""
providers/binance.py — real L2 depth and trades from Binance's public streams.

WHY BINANCE
    It needs no account, no API key and cannot be suspended, and it is the only
    source here that provides genuine **L2 depth diffs** (IG has no depth feed at
    all). That makes it the right place to exercise order-book maintenance,
    per-instrument isolation, backpressure and toxicity detection on real
    clustering.

WHAT IT IS NOT
    It is not IG. Instruments are crypto, the market never closes, there is no
    market-state concept, and symbol/side semantics differ. Use it to verify the
    plumbing, never as evidence about IG behaviour.

CORRECTNESS RULE (Binance's documented procedure)
    1. Fetch the REST snapshot and remember `lastUpdateId`.
    2. Buffer depth events until one satisfies U <= lastUpdateId+1 <= u.
    3. Drop every event with u <= lastUpdateId.
    4. Afterwards apply each event in order; quantity 0 removes a level.
    Deviating from this is how books silently drift.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

from ig_market_data.providers.base import Capabilities, EventKind, MarketEvent

_SNAPSHOT_URL = "https://api.binance.com/api/v3/depth"
_WS_URL = "wss://stream.binance.com:9443/stream"


@dataclass
class _Book:
    symbol: str
    last_update_id: int = 0
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    synced: bool = False

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None


class BinanceProvider:
    name = "binance"
    capabilities = Capabilities(
        l1=False, l2=True, trades=True, sizes=True, historical=False, live=True
    )

    def __init__(
        self,
        snapshot_url: str = _SNAPSHOT_URL,
        ws_url: str = _WS_URL,
        depth_limit: int = 1000,
        client: httpx.Client | None = None,
    ) -> None:
        self._snapshot_url = snapshot_url
        self._ws_url = ws_url
        self._depth_limit = depth_limit
        self._client = client or httpx.Client(timeout=20.0)
        self._owns_client = client is None

        self._books: dict[str, _Book] = {}
        self._subs: list[tuple[str, EventKind, Callable[[MarketEvent], None]]] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._counts: dict[str, int] = {"l2": 0, "trade": 0, "l1": 0}
        self._last_resync: dict[str, float] = {}

    # -- provider protocol -------------------------------------------------
    def connect(self) -> None:
        """Mark ready; the socket starts on the first subscribe()."""

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None
        self._started = False

    def subscribe(
        self,
        epic: str,
        kind: EventKind,
        callback: Callable[[MarketEvent], None],
    ) -> None:
        if kind == "l1":
            raise ValueError(
                "Binance publishes L2 depth here; use kind='l2' or 'trade'"
            )
        symbol = epic.upper()
        if self._started and symbol not in self._books:
            raise RuntimeError(
                "start a new provider to add symbols after streaming has begun"
            )
        with self._lock:
            self._subs.append((symbol, kind, callback))
            self._books.setdefault(symbol, _Book(symbol=symbol))
        if not self._started:
            self._start()

    # -- snapshots and book maintenance ------------------------------------
    def snapshot(self, symbol: str) -> dict[str, Any]:
        response = self._client.get(
            self._snapshot_url, params={"symbol": symbol, "limit": self._depth_limit}
        )
        response.raise_for_status()
        payload = response.json()
        book = self._books.setdefault(symbol, _Book(symbol=symbol))
        with self._lock:
            book.bids = {float(p): float(q) for p, q in payload["bids"]}
            book.asks = {float(p): float(q) for p, q in payload["asks"]}
            book.last_update_id = int(payload["lastUpdateId"])
            book.synced = False
        return payload

    def book_state(self, symbol: str) -> tuple[dict[float, float], dict[float, float]]:
        book = self._books[symbol]
        with self._lock:
            return dict(book.bids), dict(book.asks)

    def best(self, symbol: str) -> tuple[float | None, float | None]:
        book = self._books[symbol]
        with self._lock:
            return book.best_bid, book.best_ask

    def _apply_depth(self, data: dict[str, Any]) -> list[tuple[str, float, float]]:
        symbol = str(data["s"]).upper()
        with self._lock:
            book = self._books.setdefault(symbol, _Book(symbol=symbol))
            first, final = int(data["U"]), int(data["u"])
            if final <= book.last_update_id:
                return []                      # stale event, drop
            gap = first > book.last_update_id + 1
            book.synced = not gap

        if gap:
            # WHY THIS EXISTS:
            #   Binance requires the snapshot to be NEWER than the first event
            #   you apply. Taking the snapshot before opening the socket makes
            #   the first streamed event always appear as a gap; dropping it and
            #   waiting leaves the book permanently unsynchronised (observed:
            #   1231 trades, 0 depth events in 20 s). Refreshing the snapshot
            #   closes the gap, and the next event applies cleanly.
            self._maybe_resync(symbol)
            return []

        changed: list[tuple[str, float, float]] = []
        with self._lock:
            book = self._books.setdefault(symbol, _Book(symbol=symbol))
            for side, levels in (("bid", data.get("b", [])), ("ask", data.get("a", []))):
                target = book.bids if side == "bid" else book.asks
                for price_s, qty_s in levels:
                    price, qty = float(price_s), float(qty_s)
                    if qty == 0:
                        target.pop(price, None)
                    else:
                        target[price] = qty
                    changed.append((side, price, qty))
            book.last_update_id = final
        return changed

    def _maybe_resync(self, symbol: str, min_interval: float = 2.0) -> None:
        """Refresh the REST snapshot, at most every `min_interval` seconds."""
        now = time.time()
        if now - self._last_resync.get(symbol, 0.0) < min_interval:
            return
        self._last_resync[symbol] = now
        try:
            self.snapshot(symbol)
        except Exception as exc:  # noqa: BLE001
            print(f"[binance] resync failed for {symbol}: {type(exc).__name__}: {exc}")

    def _handle(self, message: dict[str, Any]) -> None:
        data = message.get("data", message)
        event_type = data.get("e")
        receive_ns = time.time_ns()

        if event_type == "depthUpdate":
            symbol = str(data["s"]).upper()
            event_ns = int(data["E"]) * 1_000_000
            for side, price, qty in self._apply_depth(data):
                self._emit(
                    MarketEvent(
                        epic=symbol,
                        kind="l2",
                        provider_ts_ns=event_ns,
                        receive_ts_ns=receive_ns,
                        level_side=side,          # type: ignore[arg-type]
                        level_price=price,
                        level_size=qty,
                        book_action="diff",
                        raw=data,
                    )
                )
        elif event_type == "trade":
            self._emit(
                MarketEvent(
                    epic=str(data["s"]).upper(),
                    kind="trade",
                    provider_ts_ns=int(data["T"]) * 1_000_000,
                    receive_ts_ns=receive_ns,
                    last=float(data["p"]),
                    size=float(data["q"]),
                    raw=data,
                )
            )

    def _emit(self, event: MarketEvent) -> None:
        self._counts[event.kind] += 1
        for symbol, kind, callback in list(self._subs):
            if event.epic == symbol and event.kind == kind:
                callback(event)

    # -- streaming ---------------------------------------------------------
    def _stream_url(self) -> str:
        symbols = sorted({symbol.lower() for symbol, _, _ in self._subs})
        kinds = {kind for _, kind, _ in self._subs}
        streams: list[str] = []
        if "l2" in kinds:
            streams += [f"{s}@depth@100ms" for s in symbols]
        if "trade" in kinds:
            streams += [f"{s}@trade" for s in symbols]
        return f"{self._ws_url}?streams={'/'.join(streams)}"

    def _start(self) -> None:
        symbols = sorted({symbol for symbol, kind, _ in self._subs if kind == "l2"})
        for symbol in symbols:
            self.snapshot(symbol)

        self._stop.clear()
        self._started = True
        self._thread = threading.Thread(target=self._run, name="binance-ws", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from websockets.sync.client import connect

        url = self._stream_url()
        try:
            with connect(url, ping_interval=20, close_timeout=5, max_size=2**23) as ws:
                while not self._stop.is_set():
                    try:
                        frame = ws.recv(timeout=1.0)
                    except TimeoutError:
                        continue
                    self._handle(json.loads(frame))
        except Exception as exc:  # noqa: BLE001
            # Surface rather than swallow: a silent dead stream is worse than a
            # loud failure in a data pipeline.
            print(f"[binance] stream stopped: {type(exc).__name__}: {exc}")

    def run_for(self, seconds: float) -> dict[str, int]:
        """Blocking helper for smoke tests. Returns per-kind event counts."""
        deadline = time.time() + seconds
        while time.time() < deadline:
            time.sleep(0.5)
        return dict(self._counts)

    def close(self) -> None:
        self.disconnect()
        if self._owns_client:
            self._client.close()
