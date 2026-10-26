"""
providers/base.py — provider-neutral market-data contract.

WHY THIS EXISTS
    Market data arrived through one concrete class (`IGStreamer`), so every test,
    every claim and every downstream component was coupled to a broker account.
    When that account was suspended, the whole data path became untestable. A
    provider boundary fixes that: the pipeline consumes `MarketEvent`s and does
    not know whether they came from IG, a recorded file, Dukascopy or Binance.

WHAT IS DELIBERATELY IN THE EVENT
    - `provider_ts_ns`: when the *venue* says the change happened (IG `UTM`,
      Binance `T`/`E`, Dukascopy record offset). Never synthesised from local
      time unless the source genuinely has no timestamp, in which case
      `raw["provider_ts_source"]` records that.
    - `receive_ts_ns`: when *we* received it. Keeping both is what makes latency
      measurement and clock-skew diagnosis possible.
    - `book_action`: "replace" for top-of-book snapshots (IG L1) versus "diff"
      for incremental depth updates (Binance L2). Mixing these two is the defect
      that made stale quote levels persist.
    - `raw`: the provider's own fields, untouched, so mapping bugs are
      diagnosable rather than lost at the boundary.

CAPABILITIES ARE EXPLICIT
    IG has no depth feed and no sizes on L1. Binance has real L2. Dukascopy is
    historical bid/ask only. A pipeline that assumes depth everywhere cannot be
    honest about any of them, so each provider declares what it can do.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Mapping, Protocol, runtime_checkable

EventKind = Literal["l1", "l2", "trade"]
BookAction = Literal["replace", "diff"]
Side = Literal["bid", "ask"]


@dataclass(frozen=True)
class Capabilities:
    """What a provider can actually deliver. Never inferred from its name."""

    l1: bool = True
    l2: bool = False
    trades: bool = True
    sizes: bool = False          # bid/ask sizes present on L1 quotes
    historical: bool = False
    live: bool = True


@dataclass(frozen=True)
class MarketEvent:
    epic: str
    kind: EventKind
    provider_ts_ns: int
    receive_ts_ns: int
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    size: float | None = None
    level_side: Side | None = None
    level_price: float | None = None
    level_size: float | None = None
    book_action: BookAction | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.epic:
            raise ValueError("epic must be a non-empty string")
        if self.kind not in ("l1", "l2", "trade"):
            raise ValueError(f"unknown kind: {self.kind!r}")
        if self.provider_ts_ns <= 0 or self.receive_ts_ns <= 0:
            raise ValueError("timestamps must be positive nanoseconds since epoch")

        for name in ("bid", "ask", "last", "size", "level_price", "level_size"):
            value = getattr(self, name)
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")

        if self.kind in ("l1", "l2"):
            if self.book_action is None:
                raise ValueError(f"{self.kind} events require book_action")
            if self.bid is None and self.ask is None and self.level_price is None:
                raise ValueError(f"{self.kind} events carry prices")
        elif self.book_action is not None:
            raise ValueError("trade events must not set book_action")

        if self.kind == "trade" and self.last is None:
            raise ValueError("trade events require last price")
        if self.kind == "l2":
            if self.level_side is None or self.level_price is None:
                raise ValueError("l2 events require level_side and level_price")
            if self.level_size is None:
                raise ValueError("l2 events require level_size (0 means remove)")


@runtime_checkable
class MarketDataProvider(Protocol):
    """Minimal contract every source implements."""

    name: str
    capabilities: Capabilities

    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def subscribe(
        self,
        epic: str,
        kind: EventKind,
        callback: Callable[[MarketEvent], None],
    ) -> None: ...


def replay_key(event: MarketEvent) -> tuple[int, str, str]:
    """Deterministic ordering: provider time, then epic, then kind."""
    return (event.provider_ts_ns, event.epic, event.kind)
