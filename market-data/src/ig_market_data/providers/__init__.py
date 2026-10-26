"""
providers/__init__.py — provider-neutral market data.

Import surface:
    ReplayProvider   deterministic playback of JSONL captures (CI default)
    IGProvider       IG API (L1 + trades only; live transport currently unverified)
    DukascopyTicks   historical FX bid/ask ticks (no account)
    BinanceProvider  real L2 depth + trades (public, no key)
"""

from ig_market_data.providers.base import (
    BookAction,
    Capabilities,
    EventKind,
    MarketDataProvider,
    MarketEvent,
)
from ig_market_data.providers.binance import BinanceProvider
from ig_market_data.providers.dukascopy import DukascopyTicks
from ig_market_data.providers.ig import IGProvider
from ig_market_data.providers.replay import ReplayProvider

__all__ = [
    "BookAction",
    "Capabilities",
    "EventKind",
    "MarketDataProvider",
    "MarketEvent",
    "BinanceProvider",
    "DukascopyTicks",
    "IGProvider",
    "ReplayProvider",
]
