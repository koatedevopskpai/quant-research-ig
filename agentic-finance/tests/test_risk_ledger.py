"""
Risk controls driven by the execution ledger, and halt semantics.

The review's failures were behavioural: P&L booked from a price level, exposure
never wired, a halt that recursively republished its own event, and a manual
"kill" that only fired if a breach already existed. These tests assert the
behaviours, not the presence of fields.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from agentic_finance.audit_trail import AuditTrail
from agentic_finance.execution_ledger import PositionLedger
from agentic_finance.message_bus import MessageBus
from agentic_finance.risk_manager import RiskManager
from agentic_finance.run_production import ProductionRunner


@pytest.fixture
def ledger():
    return PositionLedger(100_000.0, commission_bps=0.0, slippage_bps=0.0)


@pytest.fixture
def rm(message_bus, audit_trail, ledger):
    manager = RiskManager(message_bus, audit_trail, initial_capital=100_000.0,
                          max_drawdown_limit=0.15, daily_loss_limit=0.03)
    manager.set_ledger(ledger)
    return manager


class TestLedgerMetrics:
    def test_equity_exposure_and_daily_pnl_come_from_the_ledger(self, rm, ledger):
        ledger.apply_fill("EPIC", "buy", 100, 100.0)   # 10% of equity
        rm.update_mark("EPIC", 100.0)

        metrics = rm.compute_metrics()
        assert metrics.equity == pytest.approx(100_000.0)
        assert metrics.daily_pnl == pytest.approx(0.0)
        assert metrics.total_exposure == pytest.approx(10_000.0)
        assert metrics.exposure_pct == pytest.approx(0.10)
        assert metrics.n_positions == 1

        rm.update_mark("EPIC", 110.0)
        metrics = rm.compute_metrics()
        assert metrics.equity == pytest.approx(101_000.0)
        assert metrics.daily_pnl == pytest.approx(1_000.0)
        assert metrics.total_exposure == pytest.approx(11_000.0)
        assert metrics.daily_pnl == pytest.approx(metrics.equity - 100_000.0)

    def test_risk_equity_equals_ledger_equity_and_reconciles(self, rm, ledger):
        ledger.apply_fill("EPIC", "buy", 50, 100.0)
        rm.update_mark("EPIC", 105.0)
        assert rm.compute_metrics().equity == pytest.approx(ledger.equity({"EPIC": 105.0}))
        assert ledger.reconcile({"EPIC": 105.0})["balanced"] is True

    def test_record_pnl_is_ignored_when_a_ledger_is_attached(self, rm):
        rm.record_pnl(-50_000.0)  # must not double count with the ledger
        assert rm.compute_metrics().daily_pnl == pytest.approx(0.0)


class TestDailyLoss:
    def test_breach_measured_against_day_start(self, rm, ledger):
        ledger.apply_fill("EPIC", "buy", 1000, 100.0)
        rm.update_mark("EPIC", 96.0)   # -4,000 = -4% of 100,000 > 3% limit
        metrics = rm.compute_metrics()
        assert metrics.is_breached is True
        assert any("daily_loss" in b for b in metrics.breaches)

    def test_resets_on_a_new_utc_day(self, message_bus, audit_trail, ledger):
        now = [datetime(2026, 10, 26, 23, 0, tzinfo=timezone.utc)]
        manager = RiskManager(message_bus, audit_trail, initial_capital=100_000.0,
                              daily_loss_limit=0.03, clock=lambda: now[0])
        manager.set_ledger(ledger)
        ledger.apply_fill("EPIC", "buy", 1000, 100.0)
        manager.update_mark("EPIC", 96.0)
        assert manager.compute_metrics().daily_pnl == pytest.approx(-4_000.0)

        now[0] = now[0] + timedelta(days=1)
        metrics = manager.compute_metrics()
        assert metrics.daily_pnl == pytest.approx(0.0)
        assert metrics.equity == pytest.approx(96_000.0)
        assert metrics.is_breached is False   # the loss is now "yesterday's"


class TestHaltSemantics:
    def test_halt_and_resume_are_idempotent(self, rm, message_bus):
        halts, resumes = [], []
        message_bus.subscribe("risk.halted", halts.append)
        message_bus.subscribe("risk.resumed", resumes.append)

        assert rm.halt("manual") is True
        assert rm.halt("again") is False
        assert len(halts) == 1
        assert rm.is_halted is True

        assert rm.resume() is True
        assert rm.resume() is False
        assert len(resumes) == 1
        assert rm.is_halted is False

    def test_breach_halts_exactly_once(self, rm, ledger, message_bus):
        halts = []
        message_bus.subscribe("risk.halted", halts.append)
        ledger.apply_fill("EPIC", "buy", 1000, 100.0)
        rm.update_mark("EPIC", 80.0)   # -20%: drawdown breach

        assert rm.kill_if_needed() is True
        assert rm.kill_if_needed() is False   # repeated event is safe
        assert len(halts) == 1
        assert rm.is_halted is True
        assert rm.can_trade() is False

    def test_emergency_halt_needs_no_breach(self, rm, message_bus):
        """The manual halt is unconditional; the old endpoint only checked limits."""
        halts = []
        message_bus.subscribe("risk.halted", halts.append)
        assert rm.compute_metrics().is_breached is False
        assert rm.halt("manual emergency halt") is True
        assert len(halts) == 1

    def test_resume_does_not_clear_an_active_breach(self, rm, ledger):
        ledger.apply_fill("EPIC", "buy", 1000, 100.0)
        rm.update_mark("EPIC", 80.0)
        rm.kill_if_needed()
        rm.resume()
        # Still in drawdown, so new risk remains blocked even though the
        # operator cleared the halt.
        assert rm.can_trade() is False


class TestEnforcement:
    def test_runner_discards_research_while_halted(self):
        runner = ProductionRunner()
        runner.execution_agent = MagicMock()
        runner.risk_manager = MagicMock()
        runner.risk_manager.is_halted = True

        message = MagicMock()
        message.payload = {"epic": "EPIC", "output": {"direction": 1.0, "confidence": 0.8}}
        runner._on_research_completed(message)

        runner.execution_agent.process_signal.assert_not_called()

    def test_runner_discards_research_when_execution_is_halted(self):
        runner = ProductionRunner()
        runner.execution_agent = MagicMock()
        runner.execution_agent.is_active = False
        runner.risk_manager = MagicMock()
        runner.risk_manager.is_halted = False

        message = MagicMock()
        message.payload = {"epic": "EPIC", "output": {"direction": 1.0, "confidence": 0.8}}
        runner._on_research_completed(message)

        runner.execution_agent.process_signal.assert_not_called()
