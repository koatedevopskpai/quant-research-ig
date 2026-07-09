import numpy as np
import pytest
from agentic_finance.portfolio_agent import PortfolioAgent, Allocation, PortfolioState


class TestPortfolioAgent:
    @pytest.fixture
    def agent(self, message_bus, audit_trail):
        return PortfolioAgent(message_bus, audit_trail, total_capital=100_000.0,
                              rebalance_threshold=0.05)

    def _add_strategies(self, agent, n=3):
        for i in range(n):
            agent.register_strategy(
                strategy_id=f"strat_{i}",
                expected_return=0.1 * (i + 1),
                expected_vol=0.2 * (i + 1),
                description=f"Strategy {i}",
            )

    def test_initial_state(self, agent):
        assert agent.total_capital == 100_000.0
        assert agent.strategies == {}
        assert agent.current_weights == {}

    def test_register_strategy(self, agent):
        agent.register_strategy("momentum", expected_return=0.15, expected_vol=0.2)
        assert "momentum" in agent.strategies
        assert agent.strategies["momentum"]["expected_return"] == 0.15

    def test_equal_weight_allocation(self, agent):
        self._add_strategies(agent, 3)
        state = agent.allocate(method="equal")
        assert isinstance(state, PortfolioState)
        assert len(state.allocations) == 3
        for alloc in state.allocations:
            assert alloc.weight == pytest.approx(1.0 / 3)
            assert alloc.capital == pytest.approx(100_000.0 / 3)
        assert state.cash == 0.0

    def test_equal_weight_single_strategy(self, agent):
        agent.register_strategy("only_one", expected_return=0.1, expected_vol=0.2)
        state = agent.allocate(method="equal")
        assert len(state.allocations) == 1
        assert state.allocations[0].weight == 1.0
        assert state.allocations[0].capital == 100_000.0

    def test_equal_weight_no_strategies(self, agent):
        state = agent.allocate(method="equal")
        assert len(state.allocations) == 0
        assert state.cash == 100_000.0

    def test_risk_parity_allocation(self, agent):
        self._add_strategies(agent, 3)
        state = agent.allocate(method="risk_parity")
        low_vol = [a for a in state.allocations if "strat_0" in a.strategy_id][0]
        high_vol = [a for a in state.allocations if "strat_2" in a.strategy_id][0]
        assert low_vol.weight > high_vol.weight

    def test_risk_parity_single_strategy(self, agent):
        agent.register_strategy("only_one", expected_return=0.1, expected_vol=0.2)
        state = agent.allocate(method="risk_parity")
        assert len(state.allocations) == 1
        assert state.allocations[0].weight == 1.0

    def test_risk_parity_zero_vol(self, agent):
        agent.register_strategy("zero_vol", expected_return=0.0, expected_vol=0.0)
        state = agent.allocate(method="risk_parity")
        assert state.allocations[0].weight == pytest.approx(1.0)

    def test_mean_variance_allocation(self, agent):
        self._add_strategies(agent, 2)
        state = agent.allocate(method="mean_variance")
        assert len(state.allocations) == 2
        weights = [a.weight for a in state.allocations]
        assert sum(weights) == pytest.approx(1.0)

    def test_mean_variance_single_strategy(self, agent):
        agent.register_strategy("only_one", expected_return=0.1, expected_vol=0.2)
        state = agent.allocate(method="mean_variance")
        assert state.allocations[0].weight == 1.0

    def test_unknown_method_raises(self, agent):
        with pytest.raises(ValueError, match="unknown"):
            agent.allocate(method="unknown")

    def test_allocation_updates_current_weights(self, agent):
        self._add_strategies(agent, 2)
        agent.allocate(method="equal")
        assert len(agent.current_weights) == 2

    def test_allocation_publishes_message(self, agent):
        self._add_strategies(agent, 2)
        received = []

        def cb(msg):
            received.append(msg)

        agent.message_bus.subscribe("allocation.updated", cb)
        agent.allocate(method="equal")
        assert len(received) == 1
        assert received[0].payload["method"] == "equal"
        assert len(received[0].payload["weights"]) == 2

    def test_allocation_records_audit_trail(self, agent):
        self._add_strategies(agent, 2)
        agent.allocate(method="equal")
        entries = agent.audit_trail.query(agent="PortfolioAgent",
                                          decision_type="allocation")
        assert len(entries) == 1

    def test_needs_rebalance_true_when_no_weights(self, agent):
        assert agent.needs_rebalance() is True

    def test_needs_rebalance_false_after_allocation(self, agent):
        self._add_strategies(agent, 2)
        agent.allocate(method="equal")
        assert agent.needs_rebalance() is False
