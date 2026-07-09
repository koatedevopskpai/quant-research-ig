"""
hft_engine/__init__.py — High-frequency trading engine package.
"""

from hft_engine.book import OrderBook, PriceLevel, L1Snapshot, L2Snapshot, BookDepth
from hft_engine.feeds import TickConsumer, TickData, MarketDataMessage
from hft_engine.microstructure import MicrostructureAnalyser, SpreadMetrics, DepthProfile, TradeSignResult
from hft_engine.toxicity import ToxicityDetector, ToxicityResult
from hft_engine.latency import LatencyMonitor, TimingRecord
from hft_engine.execution import ExecutionSimulator, ImpactResult, OrderFill
from hft_engine.market_sim import MarketSimulator, SyntheticTick
from hft_engine.colocation import ColocationSimulator, ExchangeNode, LatencyArb

# Optional C++ accelerator module (build with `pip install .[cpp]`)
try:
    from hft_engine import hft_cpp  # pybind11 native module
    _HAS_CPP = True
except ImportError:
    hft_cpp = None  # type: ignore[assignment]
    _HAS_CPP = False

__all__ = [
    "OrderBook", "PriceLevel", "L1Snapshot", "L2Snapshot", "BookDepth",
    "TickConsumer", "TickData", "MarketDataMessage",
    "MicrostructureAnalyser", "SpreadMetrics", "DepthProfile", "TradeSignResult",
    "ToxicityDetector", "ToxicityResult",
    "LatencyMonitor", "TimingRecord",
    "ExecutionSimulator", "ImpactResult", "OrderFill",
    "MarketSimulator", "SyntheticTick",
    "ColocationSimulator", "ExchangeNode", "LatencyArb",
]
