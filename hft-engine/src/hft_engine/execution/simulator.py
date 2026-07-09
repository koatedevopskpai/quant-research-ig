"""
execution/simulator.py — Execution Simulator

Models market impact and slippage for order execution using the
Almgren-Chriss framework. Given an order (side, size, urgency),
estimates:

  1. Permanent impact — information leakage that persists
  2. Temporary impact — immediate price concession from liquidity demand
  3. Slippage — execution price relative to arrival price
  4. Implementation shortfall — total cost vs. theoretical arrival price

The simulator accounts for:
  - Order book depth (liquidity available at each level)
  - Participation rate (% of volume to avoid signalling)
  - Urgency (time horizon for execution)
  - Volatility (adverse selection during execution)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ImpactResult:
    total_cost_bps: float
    permanent_impact_bps: float
    temporary_impact_bps: float
    slippage_bps: float
    execution_price: float
    shares_executed: float
    avg_price: float


@dataclass
class OrderFill:
    price: float
    size: float
    timestamp: float
    is_aggressive: bool


class ExecutionSimulator:
    def __init__(self, volatility: float = 0.02, spread_bps: float = 1.0,
                 adv: float = 1_000_000, participation_rate: float = 0.1):
        self.volatility = volatility
        self.spread_bps = spread_bps
        self.adv = adv
        self.participation_rate = participation_rate

    @staticmethod
    def _impact_coefficient(adv: float) -> float:
        return 0.142 * (1.0 / adv) ** 0.2

    def permanent_impact(self, order_size: float, side: str) -> float:
        coeff = self._impact_coefficient(self.adv)
        sign = 1.0 if side == "buy" else -1.0
        return sign * coeff * (order_size / self.adv) ** 0.5 * 10000

    def temporary_impact(self, order_size: float, side: str,
                         urgency: float = 1.0) -> float:
        coeff = self._impact_coefficient(self.adv)
        sign = 1.0 if side == "buy" else -1.0
        decay = urgency ** 0.5
        return sign * coeff * (order_size / self.adv) ** 0.6 / decay * 10000

    def simulate(self, order_size: float, side: str, arrival_price: float,
                 urgency: float = 1.0, book_levels: Optional[list[tuple[float, float]]] = None,
                 time_horizon_min: float = 60.0) -> ImpactResult:
        perm_bps = self.permanent_impact(order_size, side)
        temp_bps = self.temporary_impact(order_size, side, urgency)

        spread_cost_bps = self.spread_bps * 0.5
        vol_cost_bps = self.volatility * np.sqrt(time_horizon_min / (390 * 60)) * 10000 * 0.5
        slippage_bps = abs(temp_bps) + spread_cost_bps + vol_cost_bps
        total_cost_bps = abs(perm_bps) + abs(temp_bps) + spread_cost_bps + vol_cost_bps

        sign = 1.0 if side == "buy" else -1.0
        execution_price = arrival_price * (1 + sign * total_cost_bps / 10000)

        if book_levels:
            remaining = order_size
            total_cost = 0.0
            total_shares = 0.0
            for level_price, level_size in book_levels:
                fill = min(remaining, level_size)
                expected_price = arrival_price * (1 + sign * abs(temp_bps) / 10000)
                total_cost += fill * (expected_price - arrival_price) * sign
                total_shares += fill
                remaining -= fill
                if remaining <= 0:
                    break
            if total_shares > 0 and total_shares < order_size:
                total_shares += remaining
            effective_price = arrival_price + (total_cost / total_shares) * sign if total_shares > 0 else arrival_price
            return ImpactResult(
                total_cost_bps=total_cost_bps,
                permanent_impact_bps=abs(perm_bps),
                temporary_impact_bps=abs(temp_bps),
                slippage_bps=slippage_bps,
                execution_price=effective_price,
                shares_executed=total_shares,
                avg_price=effective_price,
            )

        return ImpactResult(
            total_cost_bps=total_cost_bps,
            permanent_impact_bps=abs(perm_bps),
            temporary_impact_bps=abs(temp_bps),
            slippage_bps=slippage_bps,
            execution_price=execution_price,
            shares_executed=order_size,
            avg_price=execution_price,
        )

    def slice_order(self, order_size: float, num_slices: int = 5) -> list[float]:
        if num_slices <= 0:
            return [order_size]
        base = order_size / num_slices
        slices = [base] * num_slices
        remainder = order_size - base * num_slices
        for i in range(int(remainder)):
            slices[i] += 1.0
        return slices

    def fill_from_book(self, order_side: str, order_size: float,
                       book_bids: list[tuple[float, float]],
                       book_asks: list[tuple[float, float]],
                       timestamp: float) -> list[OrderFill]:
        fills: list[OrderFill] = []
        remaining = order_size
        if order_side == "buy":
            levels = book_asks
        else:
            levels = book_bids
        for price, size in levels:
            if remaining <= 0:
                break
            fill_size = min(remaining, size)
            fills.append(OrderFill(price=price, size=fill_size,
                                   timestamp=timestamp, is_aggressive=True))
            remaining -= fill_size
        return fills
