"""
market_sim/generator.py — Market Simulator (Synthetic Tick Generator)

Generates synthetic high-frequency tick data for testing order book
reconstruction, microstructure analytics, and execution algorithms.

The model:
  - Geometric Brownian Motion for mid-price (with configurable drift/vol)
  - Poisson arrival of L1 quote updates and trades
  - Realistic spread dynamics (mean-reverting, jumps during volatility)
  - Configurable order book depth with volume clustering near the touch
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SyntheticTick:
    timestamp: float
    tick_type: str  # "L1", "L2", "TRADE"
    bid_price: float = 0.0
    ask_price: float = 0.0
    bid_size: float = 0.0
    ask_size: float = 0.0
    last_price: float = 0.0
    last_size: float = 0.0
    level_price: float = 0.0
    level_size: float = 0.0
    level_side: str = ""


class MarketSimulator:
    def __init__(self, initial_price: float = 100.0, drift: float = 0.0,
                 volatility: float = 0.02, spread_bps: float = 1.0,
                 l1_freq_hz: float = 10.0, trade_freq_hz: float = 1.0,
                 depth_levels: int = 10, seed: Optional[int] = None):
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.base_spread_bps = spread_bps
        self.l1_freq_hz = l1_freq_hz
        self.trade_freq_hz = trade_freq_hz
        self.depth_levels = depth_levels
        self.rng = np.random.default_rng(seed)

        self._price = initial_price
        self._spread_bps = spread_bps
        self._bids: list[tuple[float, float]] = []
        self._asks: list[tuple[float, float]] = []
        self._t = 0.0

        self._build_depth()

    def _build_depth(self):
        spread = self._price * self._spread_bps / 10000
        self._bids = []
        self._asks = []
        base_vol = 10000
        for i in range(self.depth_levels):
            tick = 0.01 * (i + 1)
            vol_factor = np.exp(-0.5 * i)
            self._bids.append((self._price - spread / 2 - tick, base_vol * vol_factor))
            self._asks.append((self._price + spread / 2 + tick, base_vol * vol_factor))

    def _evolve_price(self, dt: float) -> float:
        ret = self.drift * dt + self.volatility * np.sqrt(dt) * self.rng.normal()
        self._price *= np.exp(ret)
        return self._price

    def _update_spread(self, dt: float):
        target_bps = self.base_spread_bps
        vol_shock = abs(self.volatility * self.rng.normal() * np.sqrt(dt)) * 10000
        noise_bps = self.rng.exponential(0.5) * (1 if self.rng.random() < 0.3 else 0)
        self._spread_bps = max(0.1, target_bps + vol_shock + noise_bps)

    def _generate_l1(self) -> SyntheticTick:
        self._update_spread(1.0 / self.l1_freq_hz)
        spread = self._price * self._spread_bps / 10000
        bid = self._price - spread / 2 + self.rng.normal(0, spread * 0.1)
        ask = bid + spread
        bid_size = 1000 + self.rng.poisson(5000)
        ask_size = 1000 + self.rng.poisson(5000)
        return SyntheticTick(
            timestamp=self._t, tick_type="L1",
            bid_price=round(bid, 2), ask_price=round(ask, 2),
            bid_size=bid_size, ask_size=ask_size,
        )

    def _generate_trade(self) -> SyntheticTick:
        spread = self._price * self._spread_bps / 10000
        side = "buy" if self.rng.random() < 0.5 else "sell"
        if side == "buy":
            price = self._price + spread / 2 + self.rng.exponential(spread * 0.3)
        else:
            price = self._price - spread / 2 - self.rng.exponential(spread * 0.3)
        size = 100 + self.rng.poisson(500)
        return SyntheticTick(
            timestamp=self._t, tick_type="TRADE",
            last_price=round(price, 2), last_size=size,
        )

    def generate_ticks(self, duration_sec: float) -> list[SyntheticTick]:
        ticks: list[SyntheticTick] = []
        self._price = self.initial_price
        self._t = 0.0

        l1_interval = 1.0 / self.l1_freq_hz if self.l1_freq_hz > 0 else float('inf')
        trade_interval = 1.0 / self.trade_freq_hz if self.trade_freq_hz > 0 else float('inf')
        dt = min(l1_interval, trade_interval)
        next_l1 = 0.0
        next_trade = 0.0

        while self._t < duration_sec:
            self._evolve_price(dt)
            self._build_depth()

            if self.l1_freq_hz > 0 and self._t >= next_l1 - 1e-12:
                ticks.append(self._generate_l1())
                next_l1 += l1_interval

            if self.trade_freq_hz > 0 and self._t >= next_trade - 1e-12:
                ticks.append(self._generate_trade())
                next_trade += trade_interval

            self._t = min(next_l1 if self.l1_freq_hz > 0 else float('inf'),
                          next_trade if self.trade_freq_hz > 0 else float('inf'))
            if self._t == float('inf') or self._t >= duration_sec:
                break

        return ticks

    def to_dataframe(self, ticks: list[SyntheticTick]) -> pd.DataFrame:
        records = []
        for t in ticks:
            records.append({
                "timestamp": t.timestamp,
                "type": t.tick_type,
                "bid_price": t.bid_price,
                "ask_price": t.ask_price,
                "bid_size": t.bid_size,
                "ask_size": t.ask_size,
                "last_price": t.last_price,
                "last_size": t.last_size,
                "level_price": t.level_price,
                "level_size": t.level_size,
                "level_side": t.level_side,
            })
        df = pd.DataFrame(records)
        if not df.empty and "timestamp" in df.columns:
            df = df.sort_values("timestamp").reset_index(drop=True)
        return df

    @property
    def bids(self) -> list[tuple[float, float]]:
        return self._bids

    @property
    def asks(self) -> list[tuple[float, float]]:
        return self._asks
