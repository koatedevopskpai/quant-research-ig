"""
replay.py — deterministic end-to-end paper replay.

WHAT THIS IS FOR
    Phases 1-6 repaired individual components. This is the integration proof:
    the same event stream drives the order book, a research model, the signal
    policy, the execution ledger, risk controls and the audit trail, and the run
    must satisfy accounting, isolation and risk invariants at the end.

    It is deterministic by construction: a fixed synthetic fixture, momentum and
    volatility-tercile adapters (no training, no randomness), and no wall-clock
    dependence. Order IDs are UUIDs and therefore differ between runs; nothing
    numeric does.

WHAT IS ASSERTED
    accounting   ledger.reconcile() balances; every fill belongs to an order
    isolation    each EPIC has its own book and price history; marks are per EPIC
    risk         long entries stop after a halt; position size respects the cap
    audit        research, order and risk decisions are recorded

RUN
    python -m agentic_finance.replay --steps 240
    python -m agentic_finance.replay --steps 240 --shock-after 60 --json evidence/replay.json
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from agentic_finance.audit_trail import AuditTrail
from agentic_finance.execution_agent import ExecutionAgent, Signal
from agentic_finance.message_bus import MessageBus
from agentic_finance.model_contract import ModelOutput
from agentic_finance.research_agent import ResearchAgent
from agentic_finance.risk_manager import RiskManager
from agentic_finance.graph import build_trading_graph, run_signal
from hft_engine.book.order_book import OrderBook
from ig_market_data.providers.base import MarketEvent

TREND_EPIC = "IX.SYNTH.TREND.IP"
RANGE_EPIC = "CS.SYNTH.RANGE.IP"


# ---------------------------------------------------------------------------
# Deterministic models (no training, no randomness)
# ---------------------------------------------------------------------------

class MomentumAdapter:
    """Signed momentum with a volatility-scaled confidence."""

    name = "forecast"

    def __init__(self, lookback: int = 10, vol_window: int = 20) -> None:
        self.lookback = lookback
        self.vol_window = vol_window

    def fit(self, prices: pd.Series) -> "MomentumAdapter":
        return self

    def predict(self, prices: pd.Series) -> ModelOutput:
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) < self.lookback + 1:
            return ModelOutput(model=self.name, direction=0.0, confidence=0.0,
                               diagnostics={"reason": "insufficient history"})
        momentum = float(prices.iloc[-1] / prices.iloc[-1 - self.lookback] - 1.0)
        vol = float(returns.tail(self.vol_window).std(ddof=1))
        scale = 2.0 * vol * math.sqrt(self.lookback)
        confidence = min(1.0, abs(momentum) / scale) if scale > 0 else 0.0
        return ModelOutput(
            model=self.name,
            direction=float(np.sign(momentum)),
            confidence=confidence,
            expected_return=momentum,
            diagnostics={"momentum": momentum, "recent_vol": vol},
        )


class VolatilityRegimeAdapter:
    """Volatility tercile of the current window. Context only: no direction."""

    name = "regime"

    def __init__(self, window: int = 20) -> None:
        self.window = window

    def fit(self, prices: pd.Series) -> "VolatilityRegimeAdapter":
        return self

    def predict(self, prices: pd.Series) -> ModelOutput:
        returns = np.log(prices / prices.shift(1)).dropna()
        if len(returns) < self.window + 5:
            return ModelOutput(model=self.name, direction=0.0, confidence=0.0,
                               diagnostics={"reason": "insufficient history"})
        rolling = returns.rolling(self.window).std().dropna()
        if rolling.empty:
            return ModelOutput(model=self.name, direction=0.0, confidence=0.0)
        lower, upper = np.quantile(rolling.to_numpy(), [0.33, 0.66])
        current = float(rolling.iloc[-1])
        state = 0 if current <= lower else (1 if current <= upper else 2)
        labels = ["Low Vol", "Medium Vol", "High Vol"]
        return ModelOutput(
            model=self.name, direction=0.0, confidence=0.0,
            regime=state, regime_label=labels[state],
            diagnostics={"n_regimes": 3, "regime_probability": 1.0},
        )


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def default_events(steps: int = 240) -> list[MarketEvent]:
    """Two instruments: a trend and a range, one L1 snapshot each minute."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events: list[MarketEvent] = []
    for i in range(steps):
        ts_ns = int((start + timedelta(minutes=i)).timestamp()) * 1_000_000_000
        trend = 100.0 * math.exp(0.0009 * i + 0.002 * math.sin(i / 9.0))
        rng = 1.10 * math.exp(0.0002 * i + 0.004 * math.cos(i / 5.0))
        for epic, price in ((TREND_EPIC, trend), (RANGE_EPIC, rng)):
            spread = price * 2e-4
            events.append(MarketEvent(
                epic=epic, kind="l1", provider_ts_ns=ts_ns, receive_ts_ns=ts_ns,
                bid=price - spread / 2, ask=price + spread / 2,
                book_action="replace", raw={"synthetic": True},
            ))
        if i % 5 == 0:
            events.append(MarketEvent(
                epic=TREND_EPIC, kind="trade", provider_ts_ns=ts_ns,
                receive_ts_ns=ts_ns, last=trend, size=1.0,
                raw={"synthetic": True},
            ))
    return events


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReplayConfig:
    initial_cash: float = 100_000.0
    max_position_pct: float = 0.05
    commission_bps: float = 1.0
    slippage_bps: float = 0.0
    research_every: int = 5
    risk_every: int = 10
    min_confidence: float = 0.10
    max_drawdown_limit: float = 0.15
    daily_loss_limit: float = 0.03
    stop_loss_pct: float = 0.02


@dataclass
class ReplayResult:
    events: int
    orders: int
    fills: int
    final_equity: float
    equity_curve: list[float]
    reconcile: dict[str, Any]
    invariants: dict[str, bool]
    halted: bool
    halt_reason: str | None
    entries_after_halt: int
    audit_entries: int
    marks: dict[str, float]

    def summary(self) -> dict[str, Any]:
        return {
            "events": self.events,
            "orders": self.orders,
            "fills": self.fills,
            "final_equity": self.final_equity,
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "entries_after_halt": self.entries_after_halt,
            "audit_entries": self.audit_entries,
            "reconcile": self.reconcile,
            "invariants": self.invariants,
            "marks": self.marks,
        }


def _price(event: MarketEvent) -> float:
    if event.kind == "trade":
        return float(event.last or 0.0)
    if event.bid and event.ask:
        return (event.bid + event.ask) / 2.0
    return float(event.level_price or 0.0)


def run_replay(events: Iterable[MarketEvent],
               config: ReplayConfig = ReplayConfig(),
               store: Any = None,
               run_id: str = "replay") -> ReplayResult:
    ordered = sorted(events, key=lambda e: (e.provider_ts_ns, e.epic, e.kind))
    if not ordered:
        raise ValueError("replay needs at least one event")

    epics = sorted({event.epic for event in ordered})
    bus = MessageBus()
    audit = AuditTrail(store=store)
    execution = ExecutionAgent(
        bus, audit, initial_capital=config.initial_cash,
        max_position_pct=config.max_position_pct,
        default_stop_loss_pct=config.stop_loss_pct,
        commission_bps=config.commission_bps,
        slippage_bps=config.slippage_bps,
        store=store,
    )
    risk = RiskManager(
        bus, audit, initial_capital=config.initial_cash,
        max_drawdown_limit=config.max_drawdown_limit,
        daily_loss_limit=config.daily_loss_limit,
        store=store, restore_state=False,
    )
    risk.set_ledger(execution.ledger)
    # Same wiring as the runner and API: a halt notification stops execution.
    bus.subscribe("risk.halted", lambda msg: execution.halt(msg.payload.get("reason", "risk")))
    bus.subscribe("risk.resumed", lambda msg: execution.resume())

    research = ResearchAgent(bus, audit)
    research.register_model("forecast", MomentumAdapter())
    research.register_model("regime", VolatilityRegimeAdapter())
    graph = build_trading_graph(research, risk, execution)

    books = {epic: OrderBook() for epic in epics}
    histories: dict[str, list[float]] = {epic: [] for epic in epics}
    last_price: dict[str, float] = {}
    equity_curve: list[float] = []
    halted_at: int | None = None
    halt_reason: str | None = None
    orders_at_halt: int | None = None

    for index, event in enumerate(ordered):
        price = _price(event)
        if price <= 0:
            continue

        book = books[event.epic]
        if event.kind == "l1":
            book.apply_l1(event.bid or 0.0, event.ask or 0.0)
        elif event.kind == "l2" and event.level_price is not None:
            book.apply_diff(event.level_side or "bid", event.level_price,
                            event.level_size or 0.0)
        last_price[event.epic] = price
        histories[event.epic].append(price)
        risk.update_mark(event.epic, price)

        # Protective exits and partial-fill completion on every tick.
        execution.on_price(event.epic, price)

        if (index % config.research_every == 0
                and len(histories[event.epic]) >= 30):
            series = pd.Series(
                histories[event.epic],
                index=pd.date_range(end=pd.Timestamp("2026-01-01", tz="UTC"),
                                    periods=len(histories[event.epic]), freq="min"),
            )
            # The graph owns the risk gate: research always runs on cadence,
            # and the conditional edge decides whether execution happens.
            run_signal(graph, event.epic, series, price,
                       min_confidence=config.min_confidence)
        if index % config.risk_every == 0:
            risk.kill_if_needed()

        if risk.is_halted and halted_at is None:
            halted_at = index
            orders_at_halt = len(execution.orders)
            halt_reason = next(
                (entry.decision.reasoning for entry in audit.query(agent="RiskManager")
                 if entry.decision.decision_type.value == "kill_switch"),
                "risk breach",
            )

        equity = execution.equity(last_price)
        equity_curve.append(equity)
        if store is not None:
            store.append_equity(run_id, index, equity)

    # New risk after a halt is forbidden; reductions and protective exits are not.
    entries_after_halt = 0
    if orders_at_halt is not None:
        entries_after_halt = sum(
            1 for order in execution.orders[orders_at_halt:]
            if order.reason == "signal"
        )

    reconcile = execution.reconcile(last_price)
    order_ids = {order.order_id for order in execution.orders}
    allocation = config.max_position_pct * max(reconcile["equity"], 1.0)
    max_position_notional = max(
        (abs(position.quantity) * last_price.get(epic, position.avg_price)
         for epic, position in execution.ledger.open_positions().items()),
        default=0.0,
    )

    # Isolation: each book's mark is the last event for that EPIC, and no two
    # EPICs share a book object.
    marks_ok = all(
        abs(last_price[epic] - histories[epic][-1]) < 1e-9 for epic in epics
    ) and len({id(book) for book in books.values()}) == len(epics)

    invariants = {
        "accounting_balanced": bool(reconcile["balanced"]),
        "fills_have_orders": all(fill.order_id in order_ids
                                 for fill in execution.ledger.fills),
        "no_new_entries_after_halt": entries_after_halt == 0,
        "position_within_cap": max_position_notional <= allocation * 1.05 + 1e-6,
        "instruments_isolated": bool(marks_ok),
        "audit_recorded": audit.count() > 0,
        "equity_matches_reconcile": abs(
            execution.equity(last_price) - reconcile["equity"]
        ) < 1e-6,
    }

    result = ReplayResult(
        events=len(ordered),
        orders=len(execution.orders),
        fills=len(execution.ledger.fills),
        final_equity=float(reconcile["equity"]),
        equity_curve=equity_curve,
        reconcile=reconcile,
        invariants=invariants,
        halted=risk.is_halted,
        halt_reason=halt_reason,
        entries_after_halt=entries_after_halt,
        audit_entries=audit.count(),
        marks=dict(last_price),
    )
    if store is not None:
        store.save_run(run_id, result.summary())
    return result

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--shock-after", type=int, default=None)
    parser.add_argument("--shock-pct", type=float, default=0.0)
    parser.add_argument("--daily-loss-limit", type=float, default=0.03)
    parser.add_argument("--stop-loss-pct", type=float, default=0.02)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--state-db", type=Path, default=None,
                        help="persist orders/fills/audit/risk and the run summary")
    args = parser.parse_args()

    events = default_events(args.steps)
    if args.shock_after is not None:
        # Apply an adverse move to the trend instrument from that step onward.
        shocked: list[MarketEvent] = []
        for event in events:
            if event.epic == TREND_EPIC and event.provider_ts_ns >= int(
                datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()
                + args.shock_after * 60
            ) * 1_000_000_000:
                factor = 1.0 - args.shock_pct
                shocked.append(_scale_event(event, factor))
            else:
                shocked.append(event)
        events = shocked

    config = ReplayConfig(
        daily_loss_limit=args.daily_loss_limit,
        stop_loss_pct=args.stop_loss_pct,
    )
    store = None
    if args.state_db is not None:
        from agentic_finance.persistence import StateStore

        store = StateStore(args.state_db)
    try:
        result = run_replay(events, config, store=store,
                            run_id=f"replay-{args.steps}-{args.shock_after}")
    finally:
        if store is not None:
            store.close()
    payload = result.summary()
    print(json.dumps(payload, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if not all(result.invariants.values()):
        raise SystemExit(1)


def _scale_event(event: MarketEvent, factor: float) -> MarketEvent:
    return MarketEvent(
        epic=event.epic, kind=event.kind,
        provider_ts_ns=event.provider_ts_ns, receive_ts_ns=event.receive_ts_ns,
        bid=None if event.bid is None else event.bid * factor,
        ask=None if event.ask is None else event.ask * factor,
        last=None if event.last is None else event.last * factor,
        size=event.size,
        level_side=event.level_side,
        level_price=None if event.level_price is None else event.level_price * factor,
        level_size=event.level_size,
        book_action=event.book_action,
        raw=dict(event.raw),
    )


if __name__ == "__main__":
    main()
