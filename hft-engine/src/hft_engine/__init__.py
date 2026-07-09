"""
hft_engine/__init__.py — High-frequency trading engine package.
"""

from hft_engine.book import OrderBook, PriceLevel, L1Snapshot, L2Snapshot, BookDepth
from hft_engine.feeds import TickConsumer, TickData, MarketDataMessage
from hft_engine.microstructure import MicrostructureAnalyser, SpreadMetrics, DepthProfile, TradeSignResult

__all__ = [
    "OrderBook", "PriceLevel", "L1Snapshot", "L2Snapshot", "BookDepth",
    "TickConsumer", "TickData", "MarketDataMessage",
    "MicrostructureAnalyser", "SpreadMetrics", "DepthProfile", "TradeSignResult",
]
