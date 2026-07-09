"""
colocation/simulator.py — Co-location Simulator

Models the network topology between trading venues and simulates
latency arbitrage opportunities. Inspired by Bryan Downing's
co-location architecture for multi-venue HFT.

The simulator:
  - Places exchange nodes and your server at configurable geographic
    locations (latency = distance / (c * fiber_factor))
  - Generates synthetic arbitrage signals based on cross-exchange
    price discrepancies and latency deltas
  - Computes whether your co-location advantage is sufficient to
    capture the arb before competitors
"""

import numpy as np
import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExchangeNode:
    name: str
    lat: float
    lon: float
    latency_us: float = 0.0

    def distance_to(self, other: "ExchangeNode") -> float:
        R = 6371.0
        dlat = math.radians(other.lat - self.lat)
        dlon = math.radians(other.lon - self.lon)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(self.lat)) *
             math.cos(math.radians(other.lat)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c


@dataclass
class LatencyArb:
    buy_exchange: str
    sell_exchange: str
    buy_price: float
    sell_price: float
    spread_bps: float
    our_latency_us: float
    competitor_latency_us: float
    window_us: float
    is_capturable: bool


class ColocationSimulator:
    """Simulate co-location across multiple exchange nodes."""

    SPEED_OF_LIGHT_FIBER = 200_000.0

    def __init__(self, my_node: Optional[ExchangeNode] = None,
                 competitor_latency_penalty_us: float = 50.0):
        self.my_node = my_node or ExchangeNode("MyServer", 51.5, -0.12, 0.0)
        self.competitor_latency_penalty_us = competitor_latency_penalty_us
        self.exchanges: dict[str, ExchangeNode] = {}

    def add_exchange(self, name: str, lat: float, lon: float) -> ExchangeNode:
        node = ExchangeNode(name=name, lat=lat, lon=lon)
        node.latency_us = self._latency_between(self.my_node, node)
        self.exchanges[name] = node
        return node

    @classmethod
    def _latency_between(cls, a: ExchangeNode, b: ExchangeNode) -> float:
        distance_km = a.distance_to(b)
        latency_s = distance_km / cls.SPEED_OF_LIGHT_FIBER
        return latency_s * 1_000_000

    def round_trip_latency(self, exchange_name: str) -> float:
        if exchange_name not in self.exchanges:
            raise ValueError(f"Unknown exchange: {exchange_name}")
        one_way = self.exchanges[exchange_name].latency_us
        return one_way * 2

    def competitor_round_trip(self, exchange_name: str) -> float:
        base = self.round_trip_latency(exchange_name)
        return base + self.competitor_latency_penalty_us

    def detect_arbitrage(self, prices: dict[str, float],
                         min_spread_bps: float = 0.5) -> list[LatencyArb]:
        arb_opps: list[LatencyArb] = []
        names = list(prices.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                bid_price = min(prices[a], prices[b])
                ask_price = max(prices[a], prices[b])
                if bid_price <= 0:
                    continue
                spread_bps = (ask_price - bid_price) / bid_price * 10000
                if spread_bps < min_spread_bps:
                    continue
                is_buy_a = prices[a] < prices[b]
                buy_ex = a if is_buy_a else b
                sell_ex = b if is_buy_a else a
                our_latency = (self.exchanges[buy_ex].latency_us +
                               self.exchanges[sell_ex].latency_us)
                comp_latency = our_latency + self.competitor_latency_penalty_us
                window_us = comp_latency - our_latency
                arb_opps.append(LatencyArb(
                    buy_exchange=buy_ex, sell_exchange=sell_ex,
                    buy_price=prices[buy_ex], sell_price=prices[sell_ex],
                    spread_bps=spread_bps,
                    our_latency_us=our_latency,
                    competitor_latency_us=comp_latency,
                    window_us=window_us,
                    is_capturable=spread_bps > 0.5,
                ))
        return sorted(arb_opps, key=lambda x: x.spread_bps, reverse=True)

    def add_noise(self, prices: dict[str, float],
                  noise_bps: float = 0.1,
                  seed: Optional[int] = None) -> dict[str, float]:
        rng = np.random.default_rng(seed)
        noisy = {}
        for name, price in prices.items():
            noise = price * rng.normal(0, noise_bps / 10000)
            noisy[name] = max(0.01, price + noise)
        return noisy
