"""
LangGraph orchestration tests.

The claim is that Research → Risk → Portfolio → Execution is a LangGraph state
machine with an audit trail. These tests assert the branching behaviour, not the
existence of an import: a halt must stop before execution, a flat view must not
create an order, and the same inputs must produce the same decision.
"""

import numpy as np
import pandas as pd
import pytest

from agentic_finance.audit_trail import AuditTrail
from agentic_finance.execution_agent import ExecutionAgent
from agentic_finance.graph import build_trading_graph, run_signal
from agentic_finance.message_bus import MessageBus
from agentic_finance.model_contract import ModelOutput
from agentic_finance.research_agent import ResearchAgent
from agentic_finance.risk_manager import RiskManager


class FixedForecast:
    name = "forecast"

    def __init__(self, direction=1.0, confidence=0.8):
        self.direction = direction
        self.confidence = confidence

    def fit(self, prices):
        return self

    def predict(self, prices):
        return ModelOutput(
            model="forecast", direction=self.direction,
            confidence=self.confidence,
            expected_return=self.direction * 0.01,
        )


class FixedRegime:
    name = "regime"

    def fit(self, prices):
        return self

    def predict(self, prices):
        return ModelOutput(model="regime", direction=0.0, confidence=0.0,
                           regime=0, regime_label="Low Vol",
                           diagnostics={"n_regimes": 3})


def _prices(n=80):
    index = pd.date_range("2026-01-01", periods=n, freq="min")
    return pd.Series(np.linspace(100.0, 101.0, n), index=index)


def _components(direction=1.0, confidence=0.8):
    bus, audit = MessageBus(), AuditTrail()
    research = ResearchAgent(bus, audit)
    research.register_model("forecast", FixedForecast(direction, confidence))
    research.register_model("regime", FixedRegime())
    execution = ExecutionAgent(bus, audit, initial_capital=100_000.0)
    risk = RiskManager(bus, audit, initial_capital=100_000.0)
    risk.set_ledger(execution.ledger)
    graph = build_trading_graph(research, risk, execution)
    return graph, research, risk, execution, audit


class TestGraphBehaviour:
    def test_directional_view_creates_an_order(self):
        graph, _, _, execution, audit = _components()
        final = run_signal(graph, "EPIC", _prices(), 100.5, min_confidence=0.1)

        assert final["allowed"] is True
        assert final["reason"] == "approved"
        assert final["order"] is not None
        assert final["order"]["status"] == "filled"
        assert len(execution.orders) == 1
        # Research and execution both audited through the agents.
        assert audit.count() >= 2

    def test_halt_stops_before_execution(self):
        graph, _, risk, execution, _ = _components()
        risk.halt("test")

        final = run_signal(graph, "EPIC", _prices(), 100.5, min_confidence=0.1)

        assert final["allowed"] is False
        assert "halted" in final["reason"]
        assert final.get("order") is None
        assert execution.orders == []

    def test_flat_view_does_not_trade(self):
        graph, _, _, execution, _ = _components(direction=0.0, confidence=0.0)
        final = run_signal(graph, "EPIC", _prices(), 100.5, min_confidence=0.1)

        assert final["allowed"] is False
        assert "no directional view" in final["reason"]
        assert execution.orders == []

    def test_confidence_floor_is_enforced_in_the_graph(self):
        graph, _, _, execution, _ = _components(direction=1.0, confidence=0.05)
        final = run_signal(graph, "EPIC", _prices(), 100.5, min_confidence=0.10)

        assert final["allowed"] is False
        assert "below floor" in final["reason"]
        assert execution.orders == []

    def test_same_inputs_give_the_same_decision(self):
        first, _, _, first_exec, _ = _components()
        second, _, _, second_exec, _ = _components()

        a = run_signal(first, "EPIC", _prices(), 100.5, min_confidence=0.1)
        b = run_signal(second, "EPIC", _prices(), 100.5, min_confidence=0.1)

        assert a["allowed"] == b["allowed"]
        assert a["order"]["side"] == b["order"]["side"]
        assert a["order"]["size"] == pytest.approx(b["order"]["size"])
        assert first_exec.positions == pytest.approx(second_exec.positions)

    def test_breach_blocks_even_with_a_good_signal(self):
        graph, _, risk, execution, _ = _components()
        risk.max_drawdown_limit = -1.0  # any drawdown counts as a breach
        risk.current_capital = 10.0     # force drawdown = 99.99%

        final = run_signal(graph, "EPIC", _prices(), 100.5, min_confidence=0.1)

        assert final["allowed"] is False
        assert execution.orders == []
