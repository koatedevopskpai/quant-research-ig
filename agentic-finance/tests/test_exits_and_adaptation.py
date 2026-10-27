"""
Toxicity-spike and time-based exits, plus the reward feedback loop.

The two remaining design claims:
  * a position is exited when order flow toxicity spikes, and when it has been
    held too long;
  * a realised close updates the HMM transition matrix based on the outcome.
"""

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from agentic_finance.audit_trail import AuditTrail
from agentic_finance.execution_agent import ExecutionAgent, Signal
from agentic_finance.message_bus import MessageBus
from agentic_finance.model_adapters import RegimeAdapter
from agentic_finance.research_agent import ResearchAgent
from agentic_finance.run_production import ProductionRunner


class Toxicity:
    def __init__(self, value: float):
        self.value = value

    def predict(self):
        return SimpleNamespace(toxicity=self.value, is_toxic=self.value >= 0.5)


class Clock:
    def __init__(self):
        self.now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    def __call__(self):
        return self.now

    def advance(self, seconds: float):
        self.now += timedelta(seconds=seconds)


@pytest.fixture
def clock():
    return Clock()


def _agent(bus, audit, clock, max_holding=60.0):
    return ExecutionAgent(
        bus, audit, initial_capital=100_000.0, max_position_pct=0.05,
        default_stop_loss_pct=0.02, commission_bps=0.0,
        toxicity_exit_threshold=0.8, max_holding_seconds=max_holding,
        clock=clock,
    )


def _entry(agent):
    order = agent.process_signal(Signal("EPIC", 1.0, 1.0, "test"), 100.0)
    assert order is not None
    return order


class TestToxicityExit:
    def test_spike_closes_the_position(self, message_bus, audit_trail, clock):
        agent = _agent(message_bus, audit_trail, clock)
        agent.set_toxicity_detector(Toxicity(0.1))
        _entry(agent)
        assert agent.positions["EPIC"] > 0

        agent.set_toxicity_detector(Toxicity(0.9))
        triggered = agent.on_price("EPIC", 101.0)

        assert triggered
        assert triggered[-1].reason == "toxicity_spike"
        assert triggered[-1].side == "sell"
        assert agent.positions == {}

    def test_below_threshold_does_not_exit(self, message_bus, audit_trail, clock):
        agent = _agent(message_bus, audit_trail, clock)
        agent.set_toxicity_detector(Toxicity(0.1))
        _entry(agent)
        agent.set_toxicity_detector(Toxicity(0.5))

        assert agent.on_price("EPIC", 100.5) == []
        assert agent.positions["EPIC"] > 0

    def test_unfitted_detector_is_ignored(self, message_bus, audit_trail, clock):
        class Broken:
            def predict(self):
                raise RuntimeError("not fitted")

        agent = _agent(message_bus, audit_trail, clock)
        agent.set_toxicity_detector(Broken())
        _entry(agent)
        assert agent.on_price("EPIC", 100.5) == []


class TestTimeExit:
    def test_exit_after_holding_period(self, message_bus, audit_trail, clock):
        agent = _agent(message_bus, audit_trail, clock)
        _entry(agent)
        clock.advance(61)

        triggered = agent.on_price("EPIC", 100.0)

        assert triggered
        assert triggered[-1].reason == "time_exit"
        assert agent.positions == {}

    def test_no_exit_before_the_period(self, message_bus, audit_trail, clock):
        agent = _agent(message_bus, audit_trail, clock)
        _entry(agent)
        clock.advance(30)
        assert agent.on_price("EPIC", 100.0) == []
        assert agent.positions["EPIC"] > 0

    def test_reversal_restarts_the_clock(self, message_bus, audit_trail, clock):
        agent = _agent(message_bus, audit_trail, clock)
        _entry(agent)
        clock.advance(30)
        agent.process_signal(Signal("EPIC", -1.0, 1.0, "test"), 100.0)
        clock.advance(40)  # 40s since the short opened, 70s since the original long
        assert agent.on_price("EPIC", 100.0) == []
        assert agent.positions["EPIC"] < 0

    def test_disabled_when_none(self, message_bus, audit_trail):
        agent = ExecutionAgent(message_bus, audit_trail, max_holding_seconds=None)
        clock = Clock()
        agent._clock = clock
        _entry(agent)
        clock.advance(10_000)
        assert agent.on_price("EPIC", 100.0) == []


class TestClosedEvent:
    def test_realised_close_is_published(self, message_bus, audit_trail, clock):
        agent = _agent(message_bus, audit_trail, clock)
        events = []
        message_bus.subscribe("position.closed", events.append)
        agent.set_toxicity_detector(Toxicity(0.1))
        _entry(agent)

        agent.set_toxicity_detector(Toxicity(0.9))
        agent.on_price("EPIC", 101.0)

        assert len(events) == 1
        payload = events[0].payload
        assert payload["epic"] == "EPIC"
        assert payload["reason"] == "toxicity_spike"
        assert payload["realized_pnl"] > 0
        assert payload["closed_quantity"] == pytest.approx(50.0)

    def test_reduction_does_not_emit_a_full_close(self, message_bus, audit_trail, clock):
        agent = _agent(message_bus, audit_trail, clock)
        events = []
        message_bus.subscribe("position.closed", events.append)
        _entry(agent)
        # Sell to half the target: a reduction, not an exit.
        agent.process_signal(Signal("EPIC", 0.5, 1.0, "test"), 100.0)
        assert len(events) == 1
        assert events[0].payload["closed_quantity"] < 50.0


class TestRewardFeedback:
    def test_regime_adapter_learns_from_a_real_model(self):
        rng = np.random.default_rng(3)
        prices = pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 300))),
            index=pd.date_range("2025-01-01", periods=300, freq="D"),
        )
        adapter = RegimeAdapter(n_regimes=2, min_observations=100)
        adapter.fit(prices)
        before = adapter.transition_matrix.copy()

        assert adapter.learn(0, 1, reward=0.05) is True

        after = adapter.transition_matrix
        assert after[0, 1] > before[0, 1]
        np.testing.assert_allclose(after.sum(axis=1), 1.0)

    def test_agent_records_the_update(self, message_bus, audit_trail):
        class FakeRegime:
            name = "regime"

            def __init__(self):
                self.calls = []

            def fit(self, prices):
                return self

            def predict(self, prices):
                raise AssertionError("not used")

            def learn(self, prev, nxt, reward):
                self.calls.append((prev, nxt, reward))
                return True

        adapter = FakeRegime()
        agent = ResearchAgent(message_bus, audit_trail)
        agent.register_model("regime", adapter)

        assert agent.record_outcome("EPIC", 0, 1, 12.5) is True
        assert adapter.calls == [(0, 1, 12.5)]
        assert any("rl_update" in entry.tags for entry in audit_trail._entries)

    def test_missing_regime_model_is_not_an_error(self, message_bus, audit_trail):
        agent = ResearchAgent(message_bus, audit_trail)
        assert agent.record_outcome("EPIC", 0, 1, 1.0) is False


class TestRunnerFeedbackWiring:
    def test_position_closed_updates_with_entry_and_current_regime(self):
        runner = ProductionRunner()
        runner.research_agent = MagicMock()
        runner.research_agent.record_outcome.return_value = True
        runner._regimes["EPIC"] = 2
        runner._entry_regimes["EPIC"] = 0

        message = MagicMock()
        message.payload = {"epic": "EPIC", "realized_pnl": -42.0}
        runner._on_position_closed(message)

        runner.research_agent.record_outcome.assert_called_once_with("EPIC", 0, 2, -42.0)
        assert "EPIC" not in runner._entry_regimes
