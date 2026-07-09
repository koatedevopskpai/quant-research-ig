import pytest
from agentic_finance.execution_agent import ExecutionAgent, Order, Signal


class MockToxicityDetector:
    def __init__(self, is_toxic=False, toxicity=0.0):
        self.is_toxic = is_toxic
        self.toxicity = toxicity

    def predict(self):
        return self


class MockImpactResult:
    total_cost_bps = 2.5
    slippage_bps = 1.0


class MockExecutionSim:
    def simulate(self, size, side, price):
        return MockImpactResult()


class TestExecutionAgent:
    @pytest.fixture
    def agent(self, message_bus, audit_trail):
        return ExecutionAgent(message_bus, audit_trail, initial_capital=100_000.0,
                              max_position_pct=0.05, default_stop_loss_pct=0.02)

    def _signal(self, direction=1.0, confidence=0.8, epic="IX.D.STCHEDS.DAILY.IP"):
        return Signal(epic=epic, direction=direction, confidence=confidence,
                      source="TestModel")

    def test_initial_state(self, agent):
        assert agent.capital == 100_000.0
        assert agent._active is True
        assert agent.positions == {}

    def test_kill_switch(self, agent):
        agent.kill()
        assert agent._active is False
        result = agent.process_signal(self._signal(), 100.0)
        assert result is None

    def test_resume_after_kill(self, agent):
        agent.kill()
        agent.resume()
        assert agent._active is True

    def test_kill_publishes_message(self, agent):
        received = []

        def cb(msg):
            received.append(msg)

        agent.message_bus.subscribe("risk.kill", cb)
        agent.kill()
        assert len(received) == 1
        assert received[0].payload == {"active": False}

    def test_kill_records_audit_trail(self, agent):
        agent.kill()
        entries = agent.audit_trail.query(agent="ExecutionAgent",
                                          decision_type="kill_switch")
        assert len(entries) == 1

    def test_process_buy_signal(self, agent):
        signal = self._signal(direction=1.0, confidence=0.8)
        order = agent.process_signal(signal, 100.0)
        assert order is not None
        assert order.side == "buy"
        assert order.size > 0
        assert order.stop_loss < 100.0
        assert order.take_profit > 100.0
        assert order.epic == "IX.D.STCHEDS.DAILY.IP"

    def test_process_sell_signal(self, agent):
        signal = self._signal(direction=-1.0, confidence=0.7)
        order = agent.process_signal(signal, 100.0)
        assert order is not None
        assert order.side == "sell"
        assert order.stop_loss > 100.0
        assert order.take_profit < 100.0

    def test_zero_direction_returns_none(self, agent):
        signal = self._signal(direction=0.0)
        assert agent.process_signal(signal, 100.0) is None

    def test_zero_confidence_returns_none(self, agent):
        signal = self._signal(confidence=0.0)
        assert agent.process_signal(signal, 100.0) is None

    def test_toxicity_veto(self, agent):
        agent.set_toxicity_detector(MockToxicityDetector(is_toxic=True, toxicity=0.9))
        signal = self._signal()
        order = agent.process_signal(signal, 100.0)
        assert order is None

    def test_toxicity_allows_clean_signal(self, agent):
        agent.set_toxicity_detector(MockToxicityDetector(is_toxic=False, toxicity=0.1))
        signal = self._signal()
        order = agent.process_signal(signal, 100.0)
        assert order is not None

    def test_execution_simulator_used(self, agent):
        agent.set_execution_simulator(MockExecutionSim())
        signal = self._signal()
        order = agent.process_signal(signal, 100.0)
        assert order is not None
        assert "impact" in order.reasoning

    def test_position_tracking(self, agent):
        signal = self._signal(direction=1.0, confidence=0.5)
        agent.process_signal(signal, 100.0)
        assert agent.positions["IX.D.STCHEDS.DAILY.IP"] > 0

    def test_sell_reverses_position(self, agent):
        buy = self._signal(direction=1.0, confidence=0.5)
        agent.process_signal(buy, 100.0)
        buy_pos = agent.positions["IX.D.STCHEDS.DAILY.IP"]
        sell = self._signal(direction=-1.0, confidence=0.5)
        agent.process_signal(sell, 100.0)
        sell_pos = agent.positions["IX.D.STCHEDS.DAILY.IP"]
        assert buy_pos > 0
        assert sell_pos < buy_pos

    def test_update_risk_constraints(self, agent):
        agent.update_risk_constraints({"max_position_pct": 0.10})
        assert agent.max_position_pct == 0.10

    def test_order_published_to_bus(self, agent):
        received = []

        def cb(msg):
            received.append(msg)

        agent.message_bus.subscribe("order.created", cb)
        signal = self._signal()
        agent.process_signal(signal, 100.0)
        assert len(received) == 1
        assert received[0].payload["side"] == "buy"

    def test_order_records_audit_trail(self, agent):
        signal = self._signal()
        agent.process_signal(signal, 100.0)
        entries = agent.audit_trail.query(agent="ExecutionAgent",
                                          decision_type="order")
        assert len(entries) == 1

    def test_multiple_signals_same_epic(self, agent):
        s1 = self._signal(direction=1.0, confidence=0.5)
        s2 = self._signal(direction=1.0, confidence=0.5)
        agent.process_signal(s1, 100.0)
        agent.process_signal(s2, 100.0)
        assert abs(agent.positions["IX.D.STCHEDS.DAILY.IP"]) > 0
