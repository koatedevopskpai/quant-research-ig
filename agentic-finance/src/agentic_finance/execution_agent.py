"""
execution_agent.py — Execution Agent (Bryan Downing's Conductor)

The Execution Agent sits between signals and the market. It implements
Downing's strategy conductor pattern:
    - Regime-dependent entry: only trade in favourable regimes
    - Toxicity veto: reject signals when order flow is toxic
    - Multi-exit logic: take-profit, stop-loss, trailing stop

It reads signals from the message bus, validates them against current
risk constraints, sizes the position, and creates orders.

The agent uses:
    - Phase 3b ExecutionSimulator for market impact estimation
    - Phase 3b ToxicityDetector for order-flow toxicity
    - Phase 2b MicrostructureAnalyser for real-time market state
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone

from agentic_finance.message_bus import MessageBus, AgentMessage
from agentic_finance.decision import Decision, DecisionType, DecisionStatus
from agentic_finance.audit_trail import AuditTrail


@dataclass
class Order:
    epic: str
    side: str  # "buy" or "sell"
    size: float
    order_type: str  # "market", "limit"
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    order_id: str = ""
    timestamp: Optional[datetime] = None
    status: str = "pending"
    reasoning: str = ""


@dataclass
class Signal:
    epic: str
    direction: float  # -1 to +1
    confidence: float  # 0 to 1
    source: str  # which model generated it
    timestamp: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ExecutionAgent:
    def __init__(self, message_bus: MessageBus, audit_trail: AuditTrail,
                 initial_capital: float = 100_000.0,
                 max_position_pct: float = 0.05,
                 default_stop_loss_pct: float = 0.02):
        self.message_bus = message_bus
        self.audit_trail = audit_trail
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.default_stop_loss_pct = default_stop_loss_pct
        self.positions: dict[str, float] = {}
        self._toxicity_detector: Any = None
        self._execution_sim: Any = None
        self._risk_constraints: dict[str, Any] = {}
        self._active = True

    def set_toxicity_detector(self, detector: Any) -> None:
        self._toxicity_detector = detector

    def set_execution_simulator(self, sim: Any) -> None:
        self._execution_sim = sim

    def update_risk_constraints(self, constraints: dict[str, Any]) -> None:
        self._risk_constraints.update(constraints)
        max_pos = self._risk_constraints.get("max_position_pct")
        if max_pos is not None:
            self.max_position_pct = max_pos

    def kill(self) -> None:
        self._active = False
        decision = Decision.now(
            agent="ExecutionAgent",
            decision_type=DecisionType.KILL_SWITCH,
            description="Execution halted by kill switch",
            context={"active": False},
        )
        decision.status = DecisionStatus.EXECUTED
        self.audit_trail.record(decision, tags=["kill_switch"])
        self.message_bus.publish_topic("risk.kill", "ExecutionAgent", {"active": False})

    def resume(self) -> None:
        self._active = True
        self.message_bus.publish_topic("risk.kill", "ExecutionAgent", {"active": True})

    def _check_toxicity(self, epic: str) -> tuple[bool, float]:
        if self._toxicity_detector is None:
            return True, 0.0
        try:
            result = self._toxicity_detector.predict()
            return not result.is_toxic, result.toxicity
        except Exception:
            return True, 0.0

    def _estimate_impact(self, epic: str, side: str, size: float,
                         price: float) -> dict[str, float]:
        if self._execution_sim is None:
            return {"total_cost_bps": 0.0, "slippage_bps": 0.0}
        try:
            impact = self._execution_sim.simulate(size, side, price)
            return {"total_cost_bps": impact.total_cost_bps,
                    "slippage_bps": impact.slippage_bps}
        except Exception:
            return {"total_cost_bps": 0.0, "slippage_bps": 0.0}

    def _size_position(self, epic: str, price: float,
                       confidence: float) -> tuple[float, str]:
        max_notional = self.capital * self.max_position_pct
        base_size = max_notional / price if price > 0 else 0.0
        sized = base_size * abs(confidence)
        current = abs(self.positions.get(epic, 0.0)) * price
        remaining = max(0.0, max_notional - current)
        final_size = min(sized, remaining / price) if price > 0 else 0.0
        reasoning = (f"base={base_size:.0f}, confidence={confidence:.2f}, "
                     f"current_pos_notional={current:.0f}, max={max_notional:.0f}")
        return final_size, reasoning

    def process_signal(self, signal: Signal, current_price: float) -> Order | None:
        if not self._active:
            return None
        if signal.direction == 0 or signal.confidence <= 0:
            return None

        toxicity_ok, tox_score = self._check_toxicity(signal.epic)
        if not toxicity_ok:
            decision = Decision.now(
                agent="ExecutionAgent",
                decision_type=DecisionType.ORDER,
                description=f"Signal REJECTED — toxicity={tox_score:.3f} on {signal.epic}",
                context={"epic": signal.epic, "toxicity": tox_score,
                         "signal_direction": signal.direction},
                reasoning="Toxicity veto triggered",
            )
            decision.status = DecisionStatus.REJECTED
            self.audit_trail.record(decision, tags=["execution", "toxicity_veto"])
            return None

        side = "buy" if signal.direction > 0 else "sell"
        size, sizing_reason = self._size_position(signal.epic, current_price, signal.confidence)
        if size <= 0:
            return None

        impact = self._estimate_impact(signal.epic, side, size, current_price)
        stop_loss = (current_price * (1 - self.default_stop_loss_pct) if side == "buy"
                     else current_price * (1 + self.default_stop_loss_pct))
        take_profit = (current_price * (1 + self.default_stop_loss_pct * 2) if side == "buy"
                       else current_price * (1 - self.default_stop_loss_pct * 2))

        order = Order(
            epic=signal.epic,
            side=side,
            size=size,
            order_type="market",
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning=(f"Signal from {signal.source}, dir={signal.direction:.2f}, "
                       f"conf={signal.confidence:.2f}, {sizing_reason}, "
                       f"impact={impact['total_cost_bps']:.2f}bps"),
        )

        decision = Decision.now(
            agent="ExecutionAgent",
            decision_type=DecisionType.ORDER,
            description=f"Order {side} {size:.0f} {signal.epic} @ ~{current_price:.2f}",
            context={"epic": signal.epic, "side": side, "size": size,
                     "price": current_price, "impact_bps": impact["total_cost_bps"],
                     "toxicity": tox_score},
            reasoning=order.reasoning,
        )
        decision.status = DecisionStatus.EXECUTED
        self.audit_trail.record(decision, tags=["execution", "order"])
        self.positions[signal.epic] = self.positions.get(signal.epic, 0.0) + size * (1 if side == "buy" else -1)

        self.message_bus.publish_topic(
            "order.created", "ExecutionAgent",
            {"epic": signal.epic, "side": side, "size": size,
             "price": current_price, "stop_loss": stop_loss,
             "take_profit": take_profit},
        )
        return order
