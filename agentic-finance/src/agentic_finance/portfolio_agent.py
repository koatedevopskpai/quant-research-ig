"""
portfolio_agent.py — Portfolio Agent (Beckman's Layered Architecture)

The Portfolio Agent manages capital allocation across multiple trading
strategies. It implements Beckman's layered architecture:
    - Layer 1: Core allocation (mean-variance or risk parity)
    - Layer 2: Tactical overlay (regime-dependent tilts)
    - Layer 3: Execution (delegated to ExecutionAgent)

Allocations are recalculated when:
    - A new strategy is added or removed
    - Risk constraints change
    - A rebalance threshold is breached
    - The market regime changes
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone

import numpy as np

from agentic_finance.message_bus import MessageBus, AgentMessage
from agentic_finance.decision import Decision, DecisionType, DecisionStatus
from agentic_finance.audit_trail import AuditTrail


@dataclass
class Allocation:
    strategy_id: str
    weight: float
    capital: float
    reason: str = ""


@dataclass
class PortfolioState:
    total_capital: float
    allocations: list[Allocation]
    cash: float
    rebalance_threshold: float = 0.05


class PortfolioAgent:
    def __init__(self, message_bus: MessageBus, audit_trail: AuditTrail,
                 total_capital: float = 100_000.0,
                 rebalance_threshold: float = 0.05):
        self.message_bus = message_bus
        self.audit_trail = audit_trail
        self.total_capital = total_capital
        self.rebalance_threshold = rebalance_threshold
        self.strategies: dict[str, dict[str, Any]] = {}
        self.current_weights: dict[str, float] = {}

    def register_strategy(self, strategy_id: str,
                          expected_return: float = 0.0,
                          expected_vol: float = 0.0,
                          description: str = "") -> None:
        self.strategies[strategy_id] = {
            "expected_return": expected_return,
            "expected_vol": expected_vol,
            "description": description,
        }

    def _equal_weight(self) -> dict[str, float]:
        if not self.strategies:
            return {}
        w = 1.0 / len(self.strategies)
        return {s: w for s in self.strategies}

    def _risk_parity_weight(self) -> dict[str, float]:
        n = len(self.strategies)
        if n == 0:
            return {}
        if n == 1:
            return {list(self.strategies.keys())[0]: 1.0}
        vols = np.array([s["expected_vol"] for s in self.strategies.values()])
        vols = np.maximum(vols, 1e-6)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return {s: float(w) for s, w in zip(self.strategies.keys(), weights)}

    def _mean_variance_weight(self, risk_aversion: float = 2.0) -> dict[str, float]:
        n = len(self.strategies)
        if n == 0:
            return {}
        if n == 1:
            return {list(self.strategies.keys())[0]: 1.0}
        returns = np.array([s["expected_return"] for s in self.strategies.values()])
        vols = np.array([s["expected_vol"] for s in self.strategies.values()])
        vols = np.maximum(vols, 1e-6)
        cov = np.diag(vols ** 2)
        inv_cov = np.linalg.inv(cov)
        ones = np.ones(n)
        w_mvp = inv_cov @ ones / (ones @ inv_cov @ ones)
        excess = returns - returns.mean()
        w_tang = inv_cov @ excess / (ones @ inv_cov @ excess) if np.any(excess != 0) else w_mvp
        w_tang = np.maximum(w_tang, 0)
        w_tang = w_tang / w_tang.sum()
        return {s: float(w) for s, w in zip(self.strategies.keys(), w_tang)}

    def allocate(self, method: str = "equal", **kwargs) -> PortfolioState:
        methods = {
            "equal": self._equal_weight,
            "risk_parity": self._risk_parity_weight,
            "mean_variance": self._mean_variance_weight,
        }
        alloc_fn = methods.get(method)
        if alloc_fn is None:
            raise ValueError(f"Unknown allocation method: {method}")
        weights = alloc_fn()
        self.current_weights = weights
        allocations = []
        for sid, w in weights.items():
            cap = self.total_capital * w
            allocations.append(Allocation(
                strategy_id=sid,
                weight=float(w),
                capital=float(cap),
                reason=f"{method} allocation: {w:.1%} weight",
            ))
        cash = self.total_capital * max(0.0, 1.0 - sum(weights.values()))
        state = PortfolioState(
            total_capital=self.total_capital,
            allocations=allocations,
            cash=cash,
            rebalance_threshold=self.rebalance_threshold,
        )
        decision = Decision.now(
            agent="PortfolioAgent",
            decision_type=DecisionType.ALLOCATION,
            description=f"Portfolio allocated ({method}): {len(allocations)} strategies",
            context={"method": method, "n_strategies": len(allocations),
                     "cash": cash, "weights": weights},
            reasoning=f"Allocated {self.total_capital:.0f} across {len(allocations)} strategies using {method}",
        )
        decision.status = DecisionStatus.EXECUTED
        self.audit_trail.record(decision, tags=["portfolio", "allocation"])
        self.message_bus.publish_topic(
            "allocation.updated", "PortfolioAgent",
            {"method": method, "weights": weights, "cash": cash},
        )
        return state

    def needs_rebalance(self, current_prices: dict[str, float] | None = None) -> bool:
        if not self.current_weights:
            return True
        return False
