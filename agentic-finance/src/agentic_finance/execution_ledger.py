"""
execution_ledger.py — paper fills, positions, cash and reconciliation.

WHY THIS EXISTS
    The old execution path changed a `positions` dict when an *order was created*
    and never touched cash. That made three questions unanswerable: what did the
    position actually cost, what is realised versus unrealised P&L, and does the
    book balance. Without those, "paper trading" is a counter, not accounting.

THE INVARIANT
    equity = initial_cash + realised + unrealised - commissions

    `reconcile()` checks it on demand. Every fill moves cash by exactly the
    notional plus commission; every reduction books realised P&L against the
    position's average cost; a reversal closes the old position and opens the
    remainder at the fill price.

COSTS
    Slippage is applied to the execution price (adverse in both directions) and
    commission is charged per fill, so a strategy's P&L includes what it cost to
    trade. Both are in basis points and can be overridden per fill.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Side = Literal["buy", "sell"]

_EPS = 1e-9


@dataclass(frozen=True)
class Fill:
    fill_id: str
    order_id: str
    epic: str
    side: Side
    quantity: float
    price: float             # execution price, including slippage
    reference_price: float   # quoted price the fill was priced from
    commission: float
    timestamp: datetime


@dataclass
class Position:
    epic: str
    quantity: float = 0.0        # signed: positive long, negative short
    avg_price: float = 0.0
    realized_pnl: float = 0.0

    @property
    def is_flat(self) -> bool:
        return abs(self.quantity) < _EPS


class PositionLedger:
    def __init__(self, initial_cash: float, commission_bps: float = 1.0,
                 slippage_bps: float = 0.0) -> None:
        if initial_cash <= 0:
            raise ValueError("initial_cash must be positive")
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("costs cannot be negative")
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.commission_bps = float(commission_bps)
        self.slippage_bps = float(slippage_bps)
        self.positions: dict[str, Position] = {}
        self.fills: list[Fill] = []
        self.total_commission = 0.0
        self.realized_pnl = 0.0

    # ------------------------------------------------------------------
    # Fills
    # ------------------------------------------------------------------

    def apply_fill(self, epic: str, side: Side, quantity: float,
                   reference_price: float, *, order_id: str = "",
                   commission_bps: float | None = None,
                   slippage_bps: float | None = None) -> Fill:
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        if quantity <= 0:
            raise ValueError(f"quantity must be positive, got {quantity}")
        if reference_price <= 0:
            raise ValueError(f"price must be positive, got {reference_price}")

        slip = (self.slippage_bps if slippage_bps is None else slippage_bps) / 10_000
        fill_price = (reference_price * (1 + slip) if side == "buy"
                      else reference_price * (1 - slip))
        notional = quantity * fill_price
        commission = notional * (self.commission_bps if commission_bps is None
                                 else commission_bps) / 10_000

        position = self.positions.setdefault(epic, Position(epic=epic))
        signed = quantity if side == "buy" else -quantity
        old_qty = position.quantity
        new_qty = old_qty + signed

        if abs(old_qty) < _EPS or (old_qty > 0) == (signed > 0):
            # Opening or increasing: weighted average cost.
            total = abs(old_qty) + quantity
            position.avg_price = (
                (abs(old_qty) * position.avg_price + quantity * fill_price) / total
            )
        else:
            # Reducing or reversing: book realised P&L on the closed part.
            closing = min(abs(old_qty), quantity)
            if old_qty > 0:
                realised = closing * (fill_price - position.avg_price)
            else:
                realised = closing * (position.avg_price - fill_price)
            position.realized_pnl += realised
            self.realized_pnl += realised

            if abs(new_qty) < _EPS:
                position.avg_price = 0.0
                new_qty = 0.0
            elif (new_qty > 0) != (old_qty > 0):
                # Reversal: the remainder is a new position at this price.
                position.avg_price = fill_price

        position.quantity = new_qty

        self.cash += (-notional if side == "buy" else notional) - commission
        self.total_commission += commission

        fill = Fill(
            fill_id=uuid.uuid4().hex[:12],
            order_id=order_id,
            epic=epic,
            side=side,
            quantity=float(quantity),
            price=float(fill_price),
            reference_price=float(reference_price),
            commission=float(commission),
            timestamp=datetime.now(timezone.utc),
        )
        self.fills.append(fill)
        return fill

    # ------------------------------------------------------------------
    # Valuation
    # ------------------------------------------------------------------

    def open_positions(self) -> dict[str, Position]:
        return {epic: p for epic, p in self.positions.items() if not p.is_flat}

    def unrealized_pnl(self, marks: dict[str, float]) -> float:
        total = 0.0
        for epic, position in self.open_positions().items():
            mark = marks.get(epic)
            if mark is None:
                # No mark: fall back to average cost, contributing zero.
                continue
            total += position.quantity * (mark - position.avg_price)
        return total

    def market_value(self, marks: dict[str, float]) -> float:
        return sum(
            position.quantity * marks.get(epic, position.avg_price)
            for epic, position in self.open_positions().items()
        )

    def equity(self, marks: dict[str, float]) -> float:
        return self.cash + self.market_value(marks)

    def gross_exposure(self, marks: dict[str, float]) -> float:
        return sum(
            abs(position.quantity) * marks.get(epic, position.avg_price)
            for epic, position in self.open_positions().items()
        )

    def reconcile(self, marks: dict[str, float]) -> dict[str, Any]:
        """Check equity = initial + realised + unrealised - commissions."""
        unrealised = self.unrealized_pnl(marks)
        equity = self.equity(marks)
        expected = self.initial_cash + self.realized_pnl + unrealised - self.total_commission
        difference = equity - expected
        tolerance = 1e-6 * max(1.0, abs(equity))
        return {
            "equity": equity,
            "expected": expected,
            "difference": difference,
            "balanced": abs(difference) <= tolerance,
            "cash": self.cash,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealised,
            "commissions": self.total_commission,
            "fills": len(self.fills),
        }
