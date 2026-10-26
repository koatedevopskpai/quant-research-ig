"""
Persistence: a restart must resume the same book, not an empty one.

Fills are the source of truth. If replaying them rebuilds cash, positions,
realised P&L, orders, the halt flag and the audit trail to the same values, the
paper account survives a process boundary.
"""

import pytest

from agentic_finance.audit_trail import AuditTrail
from agentic_finance.execution_agent import ExecutionAgent, Signal
from agentic_finance.execution_ledger import PositionLedger
from agentic_finance.message_bus import MessageBus
from agentic_finance.persistence import StateStore
from agentic_finance.replay import default_events, run_replay
from agentic_finance.risk_manager import RiskManager


@pytest.fixture
def store(tmp_path):
    state = StateStore(tmp_path / "state.db")
    yield state
    state.close()


class TestLedgerRoundTrip:
    def test_fills_rebuild_cash_positions_and_realised_pnl(self, store):
        ledger = PositionLedger(100_000.0, commission_bps=1.0, store=store)
        ledger.apply_fill("A", "buy", 10, 100.0)
        ledger.apply_fill("A", "sell", 4, 110.0)

        restored = PositionLedger(100_000.0, commission_bps=1.0, store=store)
        assert restored.restore() == 2

        assert restored.cash == pytest.approx(ledger.cash)
        assert restored.realized_pnl == pytest.approx(ledger.realized_pnl)
        assert restored.total_commission == pytest.approx(ledger.total_commission)
        assert restored.positions["A"].quantity == pytest.approx(6.0)
        assert restored.positions["A"].avg_price == pytest.approx(100.0)
        assert restored.reconcile({"A": 110.0})["balanced"] is True

    def test_restore_is_idempotent(self, store):
        ledger = PositionLedger(100_000.0, store=store)
        ledger.apply_fill("A", "buy", 5, 100.0)
        restored = PositionLedger(100_000.0, store=store)
        restored.restore()
        restored.restore()
        assert len(restored.fills) == 1


class TestExecutionAndAuditRoundTrip:
    def test_orders_and_fills_survive(self, store):
        bus, audit = MessageBus(), AuditTrail(store=store)
        agent = ExecutionAgent(bus, audit, initial_capital=100_000.0, store=store)
        agent.process_signal(Signal("A", 1.0, 0.8, "test"), 100.0)

        bus2, audit2 = MessageBus(), AuditTrail(store=store)
        restored = ExecutionAgent(bus2, audit2, initial_capital=100_000.0, store=store)
        assert restored.restore() == 1

        assert len(restored.orders) == 1
        assert restored.orders[0].order_id == agent.orders[0].order_id
        assert restored.orders[0].status == "filled"
        assert len(restored.orders[0].fills) == 1
        assert restored.positions["A"] == pytest.approx(agent.positions["A"])
        assert restored.cash == pytest.approx(agent.cash)

    def test_audit_trail_survives(self, store):
        bus = MessageBus()
        audit = AuditTrail(store=store)
        agent = ExecutionAgent(bus, audit, store=store)
        agent.process_signal(Signal("A", 1.0, 0.5, "test"), 100.0)

        restored = AuditTrail(store=store)
        restored.restore()
        assert restored.count() == audit.count()
        assert restored.query(agent="ExecutionAgent")


class TestRiskRoundTrip:
    def test_halt_flag_and_day_start_survive(self, store):
        bus, audit = MessageBus(), AuditTrail(store=store)
        ledger = PositionLedger(100_000.0, store=store)
        risk = RiskManager(bus, audit, store=store, restore_state=False)
        risk.set_ledger(ledger)
        risk.halt("test halt")

        restored = RiskManager(bus, audit, store=store)
        assert restored.is_halted is True
        assert restored._daily_start_capital == pytest.approx(100_000.0)
        assert restored.can_trade() is False

    def test_resume_is_persisted(self, store):
        bus, audit = MessageBus(), AuditTrail(store=store)
        risk = RiskManager(bus, audit, store=store, restore_state=False)
        risk.halt("halt")
        risk.resume()

        restored = RiskManager(bus, audit, store=store)
        assert restored.is_halted is False


class TestReplayPersistence:
    def test_run_and_equity_curve_are_stored(self, store):
        result = run_replay(default_events(40), store=store, run_id="t1")

        payload = store.load_run("t1")
        assert payload is not None
        assert payload["final_equity"] == pytest.approx(result.final_equity)
        assert payload["fills"] == result.fills

        curve = store.load_equity_curve("t1")
        assert len(curve) == len(result.equity_curve)
        assert curve[-1] == pytest.approx(result.equity_curve[-1])

        counts = store.counts()
        assert counts["fills"] == result.fills
        assert counts["orders"] == result.orders
        assert counts["audit"] == result.audit_entries
