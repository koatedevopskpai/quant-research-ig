"""
execution_agent.py — Execution Agent (Bryan Downing's Conductor)

The Execution Agent sits between signals and the market. It implements
Downing's strategy conductor pattern:
    - Regime-dependent entry: only trade in favourable regimes
    - Toxicity veto: reject signals when order flow is toxic
    - Multi-exit logic: take-profit, stop-loss, trailing stop

WHAT CHANGED IN PHASE 5
    Orders now have identities and lifecycle states, fills are simulated and
    booked to a `PositionLedger`, and protective exits are actually consumed.
    The previous version sized against `abs(holdings)`, so a full position could
    never be reduced, and it mutated holdings when the *order* was created rather
    than when it filled.

SIZING IS AGAINST A SIGNED TARGET
    target_quantity = direction x confidence x (equity x max_position_pct / price)
    order_quantity  = target_quantity - current_quantity

    That single change makes reductions, exits and reversals fall out naturally:
    a sell that shrinks a long needs no extra "new risk" allowance.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from agentic_finance.message_bus import MessageBus, AgentMessage
from agentic_finance.decision import Decision, DecisionType, DecisionStatus
from agentic_finance.audit_trail import AuditTrail
from agentic_finance.execution_ledger import Fill, PositionLedger


@dataclass
class Order:
    epic: str
    side: str  # "buy" or "sell"
    size: float
    order_type: str  # "market", "limit"
    limit_price: float | None = None
    stop_loss: float | None = None
    take_profit: float | None = None
    order_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: Optional[datetime] = None
    status: str = "new"  # new | partially_filled | filled | rejected | cancelled
    reasoning: str = ""
    filled_size: float = 0.0
    avg_fill_price: float | None = None
    fills: list[Fill] = field(default_factory=list)
    reason: str = ""  # why the order exists: signal, stop_loss, take_profit

    @property
    def remaining(self) -> float:
        return max(0.0, self.size - self.filled_size)


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
                 default_stop_loss_pct: float = 0.02,
                 commission_bps: float = 1.0,
                 slippage_bps: float = 0.0,
                 fill_ratio: float = 1.0,
                 min_order_notional: float = 1.0,
                 store: Any = None,
                 toxicity_exit_threshold: float = 0.8,
                 max_holding_seconds: float | None = None,
                 clock: Any = None):
        self.message_bus = message_bus
        self.audit_trail = audit_trail
        self.capital = initial_capital
        self.max_position_pct = max_position_pct
        self.default_stop_loss_pct = default_stop_loss_pct
        self.fill_ratio = fill_ratio
        self.min_order_notional = min_order_notional
        self.toxicity_exit_threshold = float(toxicity_exit_threshold)
        self.max_holding_seconds = max_holding_seconds
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._entry_time: dict[str, datetime] = {}
        if not 0 < fill_ratio <= 1:
            raise ValueError("fill_ratio must be in (0, 1]")

        self.ledger = PositionLedger(
            initial_capital, commission_bps=commission_bps,
            slippage_bps=slippage_bps, store=store
        )
        self._store = store
        self.orders: list[Order] = []
        self._marks: dict[str, float] = {}
        self._protective: dict[str, tuple[float | None, float | None]] = {}
        self._toxicity_detector: Any = None
        self._execution_sim: Any = None
        self._risk_constraints: dict[str, Any] = {}
        self._active = True

    # ------------------------------------------------------------------
    # State access
    # ------------------------------------------------------------------

    @property
    def positions(self) -> dict[str, float]:
        return {
            epic: position.quantity
            for epic, position in self.ledger.open_positions().items()
        }

    @property
    def cash(self) -> float:
        return self.ledger.cash

    def equity(self, prices: dict[str, float] | None = None) -> float:
        marks = dict(self._marks)
        if prices:
            marks.update(prices)
        return self.ledger.equity(marks)

    def reconcile(self, prices: dict[str, float] | None = None) -> dict[str, Any]:
        marks = dict(self._marks)
        if prices:
            marks.update(prices)
        return self.ledger.reconcile(marks)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def set_toxicity_detector(self, detector: Any) -> None:
        self._toxicity_detector = detector

    def set_execution_simulator(self, sim: Any) -> None:
        self._execution_sim = sim

    def update_risk_constraints(self, constraints: dict[str, Any]) -> None:
        self._risk_constraints.update(constraints)
        max_pos = self._risk_constraints.get("max_position_pct")
        if max_pos is not None:
            self.max_position_pct = max_pos

    @property
    def is_active(self) -> bool:
        return self._active

    def halt(self, reason: str = "kill switch") -> bool:
        """
        Stop accepting signals. Idempotent: returns True only if it changed state.

        WHY NOT A COMMAND TOPIC:
            The old `kill()` published `risk.kill`, the same topic its caller was
            subscribed to, so one halt re-entered the handler and republished —
            nine publications from one event in the review's reproduction. Halt
            now publishes `execution.halted`, a notification that nothing
            re-issues.
        """
        if not self._active:
            return False
        self._active = False
        decision = Decision.now(
            agent="ExecutionAgent",
            decision_type=DecisionType.KILL_SWITCH,
            description=f"Execution halted: {reason}",
            context={"active": False, "reason": reason},
        )
        decision.status = DecisionStatus.EXECUTED
        self.audit_trail.record(decision, tags=["kill_switch"])
        self.message_bus.publish_topic(
            "execution.halted", "ExecutionAgent", {"reason": reason}
        )
        return True

    def resume(self) -> bool:
        """Resume trading. Idempotent: returns True only if it changed state."""
        if self._active:
            return False
        self._active = True
        self.message_bus.publish_topic("execution.resumed", "ExecutionAgent", {})
        return True

    def kill(self) -> bool:
        """Backwards-compatible name for `halt`."""
        return self.halt("kill switch")

    # ------------------------------------------------------------------
    # Restore
    # ------------------------------------------------------------------

    def restore(self) -> int:
        """
        Rebuild orders, fills and positions from the store.

        Fills are the source of truth: replaying them reproduces cash and
        positions exactly, so a restart resumes the same book rather than a
        fresh one. Returns the number of fills restored.
        """
        if self._store is None:
            return 0
        restored = self.ledger.restore()
        if self.orders:
            return restored

        fills_by_order: dict[str, list[Fill]] = {}
        for fill in self.ledger.fills:
            fills_by_order.setdefault(fill.order_id, []).append(fill)

        for row in self._store.load_orders():
            timestamp = row.get("created_at")
            order = Order(
                epic=row["epic"],
                side=row["side"],
                size=float(row["size"]),
                order_type=row["order_type"],
                stop_loss=row.get("stop_loss"),
                take_profit=row.get("take_profit"),
                order_id=row["order_id"],
                timestamp=(datetime.fromisoformat(timestamp) if timestamp else None),
                status=row["status"],
                reasoning=row.get("reasoning") or "",
                filled_size=float(row.get("filled_size") or 0.0),
                avg_fill_price=row.get("avg_fill_price"),
                reason=row.get("reason") or "",
            )
            order.fills = fills_by_order.get(order.order_id, [])
            self.orders.append(order)
        return restored

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _check_toxicity(self, epic: str) -> tuple[bool, float]:
        if self._toxicity_detector is None:
            return True, 0.0
        try:
            result = self._toxicity_detector.predict()
            return not result.is_toxic, result.toxicity
        except Exception:
            return True, 0.0

    def _estimate_slippage_bps(self, side: str, size: float, price: float) -> tuple[float, str]:
        if self._execution_sim is None:
            return 0.0, ""
        try:
            impact = self._execution_sim.simulate(size, side, price)
            return float(impact.slippage_bps), f"impact={impact.total_cost_bps:.2f}bps"
        except Exception:
            return 0.0, ""

    def _target_order(self, epic: str, direction: float, confidence: float,
                      price: float) -> tuple[str, float, float, float]:
        equity = self.ledger.equity(self._marks)
        max_notional = equity * self.max_position_pct
        target_qty = direction * abs(confidence) * (max_notional / price) if price > 0 else 0.0
        current_qty = self.ledger.positions.get(epic)
        current_qty = current_qty.quantity if current_qty else 0.0
        delta = target_qty - current_qty
        side = "buy" if delta > 0 else "sell"
        return side, abs(delta), target_qty, current_qty

    def process_signal(self, signal: Signal, current_price: float) -> Order | None:
        if not self._active:
            return None
        if signal.direction == 0 or signal.confidence <= 0:
            return None
        if current_price <= 0:
            return None

        self._marks[signal.epic] = current_price

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

        side, size, target_qty, current_qty = self._target_order(
            signal.epic, signal.direction, signal.confidence, current_price
        )
        if size <= 0:
            return None
        # Commission changes equity by a hair, which otherwise produces
        # micro-orders that only pay costs. Below a currency minimum, do nothing.
        if size * current_price < self.min_order_notional:
            return None

        slip_bps, impact_note = self._estimate_slippage_bps(side, size, current_price)
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
            timestamp=datetime.now(timezone.utc),
            reasoning=(f"Signal from {signal.source}, dir={signal.direction:.2f}, "
                       f"conf={signal.confidence:.2f}, target_qty={target_qty:.4f}, "
                       f"current_qty={current_qty:.4f}, {impact_note}").strip(),
            reason="signal",
        )
        return self._submit(order, current_price, extra_slippage_bps=slip_bps)

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    def _submit(self, order: Order, price: float, *, reason: str = "",
                extra_slippage_bps: float = 0.0) -> Order:
        order.timestamp = order.timestamp or datetime.now(timezone.utc)
        order.reason = reason or order.reason
        self.orders.append(order)
        if self._store is not None:
            self._store.save_order(order)

        decision = Decision.now(
            agent="ExecutionAgent",
            decision_type=DecisionType.ORDER,
            description=f"Order {order.side} {order.size:.4f} {order.epic} @ ~{price:.2f}",
            context={"order_id": order.order_id, "epic": order.epic,
                     "side": order.side, "size": order.size, "price": price,
                     "target": order.reason},
            reasoning=order.reasoning,
        )
        decision.status = DecisionStatus.EXECUTED
        self.audit_trail.record(decision, tags=["execution", "order"])

        self.message_bus.publish_topic(
            "order.created", "ExecutionAgent",
            {"order_id": order.order_id, "epic": order.epic, "side": order.side,
             "size": order.size, "price": price,
             "stop_loss": order.stop_loss, "take_profit": order.take_profit,
             "reason": order.reason},
        )

        self._fill(order, price, extra_slippage_bps=extra_slippage_bps)
        return order

    def _fill(self, order: Order, price: float, *,
              extra_slippage_bps: float = 0.0) -> Fill | None:
        remaining = order.remaining
        if remaining <= 0:
            return None

        # A fill_ratio below 1 simulates partial liquidity: each price event can
        # consume another fraction of the original order size.
        chunk = remaining if self.fill_ratio >= 1 else min(
            remaining, order.size * self.fill_ratio
        )
        if chunk <= 0:
            return None

        prior = self.ledger.positions.get(order.epic)
        prior_qty = prior.quantity if prior else 0.0
        prior_avg = prior.avg_price if prior else 0.0

        fill = self.ledger.apply_fill(
            order.epic, order.side, chunk, price,
            order_id=order.order_id,
            slippage_bps=self.ledger.slippage_bps + extra_slippage_bps,
        )
        order.filled_size += fill.quantity
        notional_before = (order.avg_fill_price or 0.0) * (order.filled_size - fill.quantity)
        order.avg_fill_price = (notional_before + fill.price * fill.quantity) / order.filled_size
        order.fills.append(fill)
        order.status = "filled" if order.remaining <= 1e-9 else "partially_filled"
        if self._store is not None:
            self._store.save_order(order)

        self._marks[order.epic] = price
        position = self.ledger.positions.get(order.epic)
        now = self._clock()
        if position and not position.is_flat:
            # A new (or reversed) position starts its holding clock here.
            if abs(prior_qty) < 1e-9 or (prior_qty > 0) != (position.quantity > 0):
                self._entry_time[order.epic] = now
            # Remember the protective levels for the side actually held.
            held_side = "buy" if position.quantity > 0 else "sell"
            if order.side == held_side and order.stop_loss is not None:
                self._protective[order.epic] = (order.stop_loss, order.take_profit)
        else:
            self._entry_time.pop(order.epic, None)

        # If this fill reduced or reversed an existing position, report the
        # realised part so the research loop can adapt the regime transition
        # matrix from the outcome.
        closing = 0.0
        if abs(prior_qty) > 1e-9 and ((prior_qty > 0) != (order.side == "buy")):
            closing = min(abs(prior_qty), fill.quantity)
        if closing > 0:
            realised = closing * (
                (fill.price - prior_avg) if prior_qty > 0 else (prior_avg - fill.price)
            )
            self.message_bus.publish_topic(
                "position.closed", "ExecutionAgent",
                {"epic": order.epic, "closed_quantity": float(closing),
                 "realized_pnl": float(realised), "reason": order.reason,
                 "order_id": order.order_id},
            )

        self.message_bus.publish_topic(
            "order.filled", "ExecutionAgent",
            {"order_id": order.order_id, "fill_id": fill.fill_id,
             "epic": order.epic, "side": order.side, "quantity": fill.quantity,
             "price": fill.price, "commission": fill.commission,
             "filled_size": order.filled_size, "status": order.status},
        )
        return fill

    # ------------------------------------------------------------------
    # Price updates: completion of partials, then protective exits
    # ------------------------------------------------------------------

    def on_price(self, epic: str, price: float) -> list[Order]:
        if price <= 0:
            return []
        self._marks[epic] = price
        triggered: list[Order] = []

        for order in [o for o in self.orders
                      if o.epic == epic and o.status == "partially_filled"]:
            self._fill(order, price)
            triggered.append(order)

        position = self.ledger.positions.get(epic)
        if position is None or position.is_flat:
            return triggered

        reason = self._exit_reason(epic, position, price)
        if reason is None:
            return triggered

        closing = Order(
            epic=epic,
            side="sell" if position.quantity > 0 else "buy",
            size=abs(position.quantity),
            order_type="market",
            timestamp=datetime.now(timezone.utc),
            reasoning=f"{reason} exit at {price:.4f}",
            reason=reason,
        )
        self._submit(closing, price, reason=reason)
        self._protective.pop(epic, None)
        self._entry_time.pop(epic, None)
        triggered.append(closing)
        return triggered

    def _exit_reason(self, epic: str, position: Any, price: float) -> str | None:
        """
        First applicable exit, in priority order.

        Protective levels take precedence over toxicity, which takes precedence
        over elapsed time: a stop is a hard risk limit, a toxicity spike is a
        market-state judgement, and a time exit is housekeeping.
        """
        long = position.quantity > 0
        stop, take = self._protective.get(epic, (None, None))

        if stop is not None and (
            (long and price <= stop) or (not long and price >= stop)
        ):
            return "stop_loss"
        if take is not None and (
            (long and price >= take) or (not long and price <= take)
        ):
            return "take_profit"

        toxicity = self._current_toxicity()
        if toxicity is not None and toxicity >= self.toxicity_exit_threshold:
            return "toxicity_spike"

        if self.max_holding_seconds is not None:
            opened = self._entry_time.get(epic)
            if opened is not None:
                held = (self._clock() - opened).total_seconds()
                if held >= self.max_holding_seconds:
                    return "time_exit"
        return None

    def _current_toxicity(self) -> float | None:
        if self._toxicity_detector is None:
            return None
        try:
            return float(self._toxicity_detector.predict().toxicity)
        except Exception:  # noqa: BLE001 - detector may be unfitted
            return None
