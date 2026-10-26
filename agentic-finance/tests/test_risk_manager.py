import numpy as np
import pytest
from agentic_finance.risk_manager import RiskManager, RiskMetrics


class TestRiskManager:
    @pytest.fixture
    def rm(self, message_bus, audit_trail):
        return RiskManager(message_bus, audit_trail, initial_capital=100_000.0,
                           max_drawdown_limit=0.15, daily_loss_limit=0.03,
                           var_limit=0.02, var_confidence=0.95)

    def test_initial_state(self, rm):
        assert rm.current_capital == 100_000.0
        assert rm.peak_capital == 100_000.0
        assert rm.daily_pnl == 0.0

    def test_record_pnl_increases_capital(self, rm):
        rm.record_pnl(5000.0)
        assert rm.current_capital == 105_000.0
        assert rm.daily_pnl == 5000.0

    def test_record_pnl_decreases_capital(self, rm):
        rm.record_pnl(-3000.0)
        assert rm.current_capital == 97_000.0
        assert rm.daily_pnl == -3000.0

    def test_drawdown_zero_when_at_peak(self, rm):
        assert rm.current_drawdown() == 0.0

    def test_drawdown_positive_after_loss(self, rm):
        rm.record_pnl(-20_000.0)
        dd = rm.current_drawdown()
        assert 0.19 < dd < 0.21

    def test_drawdown_uses_peak_not_initial(self, rm):
        rm.record_pnl(10_000.0)
        rm.record_pnl(-15_000.0)
        dd = rm.current_drawdown()
        peak = 110_000.0
        expected = (peak - 95_000.0) / peak
        assert dd == pytest.approx(expected)

    def test_compute_metrics_no_positions(self, rm):
        metrics = rm.compute_metrics()
        assert metrics.n_positions == 0
        assert metrics.total_exposure == 0.0
        assert metrics.is_breached is False

    def test_compute_metrics_with_positions(self, rm):
        rm.update_positions({"IX.D.STCHEDS.DAILY.IP": 50, "IX.D.SOME.IP": -30})
        metrics = rm.compute_metrics()
        assert metrics.n_positions == 2
        assert metrics.total_exposure == 80.0

    def test_var_historical_not_enough_data(self, rm):
        metrics = rm.compute_metrics()
        assert metrics.var_historical == 0.0

    def test_var_historical_with_data_returns_negative(self, rm):
        for _ in range(30):
            rm.record_pnl(np.random.normal(0, 1000))
        metrics = rm.compute_metrics()
        assert metrics.var_historical < 0

    def test_var_parametric_not_enough_data(self, rm):
        metrics = rm.compute_metrics()
        assert metrics.var_parametric == 0.0

    def test_var_parametric_with_data_returns_negative(self, rm):
        for _ in range(10):
            rm.record_pnl(np.random.normal(0, 1000))
        metrics = rm.compute_metrics()
        assert metrics.var_parametric < 0

    def test_check_risk_no_breach(self, rm):
        metrics = rm.check_risk()
        assert metrics.is_breached is False
        assert rm.audit_trail.count() == 1
        entry = rm.audit_trail.query(decision_type="risk_check")[0]
        assert entry.decision.status.value == "approved"

    def test_drawdown_breach_detected(self, rm):
        rm.record_pnl(-20_000.0)
        metrics = rm.check_risk()
        assert metrics.is_breached is True
        assert len(metrics.breaches) >= 1

    def test_kill_switch_not_triggered_when_safe(self, rm):
        assert rm.kill_if_needed() is False

    def test_kill_switch_triggered_on_breach(self, rm):
        rm.record_pnl(-20_000.0)
        assert rm.kill_if_needed() is True

    def test_kill_switch_publishes_alert(self, rm):
        received = []

        def cb(msg):
            received.append(msg)

        rm.message_bus.subscribe("risk.alert", cb)
        rm.record_pnl(-20_000.0)
        rm.check_risk()
        assert len(received) == 1

    def test_kill_switch_publishes_halt_notification(self, rm):
        received = []

        def cb(msg):
            received.append(msg)

        rm.message_bus.subscribe("risk.halted", cb)
        rm.record_pnl(-20_000.0)
        rm.kill_if_needed()
        assert len(received) == 1
        assert "drawdown" in received[0].payload["reason"]

    def test_daily_loss_limit_breach(self, rm):
        rm.record_pnl(-4000.0)
        metrics = rm.compute_metrics()
        assert metrics.is_breached is True

    def test_no_false_positive_when_returns_flat(self, rm):
        for _ in range(30):
            rm.record_pnl(0.0)
        metrics = rm.check_risk()
        assert metrics.is_breached is False
