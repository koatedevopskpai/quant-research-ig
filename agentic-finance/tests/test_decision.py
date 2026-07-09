import pytest
from agentic_finance.decision import Decision, DecisionType, DecisionStatus


class TestDecisionType:
    def test_has_expected_values(self):
        assert DecisionType.RESEARCH.value == "research"
        assert DecisionType.SIGNAL.value == "signal"
        assert DecisionType.ORDER.value == "order"
        assert DecisionType.RISK_CHECK.value == "risk_check"
        assert DecisionType.ALLOCATION.value == "allocation"
        assert DecisionType.KILL_SWITCH.value == "kill_switch"
        assert DecisionType.AUDIT.value == "audit"


class TestDecisionStatus:
    def test_has_expected_values(self):
        assert DecisionStatus.PENDING.value == "pending"
        assert DecisionStatus.APPROVED.value == "approved"
        assert DecisionStatus.REJECTED.value == "rejected"
        assert DecisionStatus.EXECUTED.value == "executed"
        assert DecisionStatus.FAILED.value == "failed"


class TestDecision:
    def test_now_creates_decision_with_defaults(self):
        d = Decision.now(
            agent="TestAgent",
            decision_type=DecisionType.RESEARCH,
            description="test decision",
        )
        assert d.agent == "TestAgent"
        assert d.decision_type == DecisionType.RESEARCH
        assert d.description == "test decision"
        assert d.status == DecisionStatus.PENDING
        assert len(d.decision_id) == 8
        assert d.timestamp is not None
        assert d.context == {}
        assert d.reasoning == ""
        assert d.llm_explanation == ""
        assert d.parent_id is None

    def test_now_with_kwargs(self):
        d = Decision.now(
            agent="TestAgent",
            decision_type=DecisionType.ORDER,
            description="test",
            reasoning="because",
            context={"price": 100.0},
            llm_explanation="llm text",
            parent_id="parent123",
            metadata={"version": 2},
        )
        assert d.reasoning == "because"
        assert d.context == {"price": 100.0}
        assert d.llm_explanation == "llm text"
        assert d.parent_id == "parent123"
        assert d.metadata == {"version": 2}

    def test_to_dict(self):
        d = Decision.now(
            agent="Agent",
            decision_type=DecisionType.RISK_CHECK,
            description="risk check",
            reasoning="all ok",
        )
        d.status = DecisionStatus.APPROVED
        dd = d.to_dict()
        assert dd["decision_id"] == d.decision_id
        assert dd["agent"] == "Agent"
        assert dd["decision_type"] == "risk_check"
        assert dd["status"] == "approved"
        assert dd["description"] == "risk check"
        assert dd["reasoning"] == "all ok"
        assert "timestamp" in dd

    def test_unique_ids(self):
        d1 = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="a")
        d2 = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="b")
        assert d1.decision_id != d2.decision_id
