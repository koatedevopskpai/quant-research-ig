import json
import pytest
from agentic_finance.audit_trail import AuditTrail, AuditEntry
from agentic_finance.decision import Decision, DecisionType, DecisionStatus


class TestAuditEntry:
    def test_to_dict(self):
        d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="test")
        entry = AuditEntry(decision=d, market_context={"vol": 0.2},
                           signals_used=[{"name": "regime"}], tags=["research"])
        dd = entry.to_dict()
        assert dd["decision"]["decision_id"] == d.decision_id
        assert dd["market_context"] == {"vol": 0.2}
        assert dd["signals_used"] == [{"name": "regime"}]
        assert dd["tags"] == ["research"]


class TestAuditTrail:
    def test_record_returns_entry(self, audit_trail):
        d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="test")
        entry = audit_trail.record(d, tags=["research"])
        assert isinstance(entry, AuditEntry)
        assert entry.decision.decision_id == d.decision_id

    def test_get_by_id(self, audit_trail):
        d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="test")
        audit_trail.record(d)
        entry = audit_trail.get(d.decision_id)
        assert entry is not None
        assert entry.decision.decision_id == d.decision_id

    def test_get_unknown_id_returns_none(self, audit_trail):
        assert audit_trail.get("nonexistent") is None

    def test_record_increments_count(self, audit_trail):
        for i in range(5):
            d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description=str(i))
            audit_trail.record(d)
        assert audit_trail.count() == 5

    def test_query_by_agent(self, audit_trail):
        for i in range(3):
            audit_trail.record(
                Decision.now(agent="Alpha", decision_type=DecisionType.RESEARCH, description=str(i)))
        for i in range(2):
            audit_trail.record(
                Decision.now(agent="Beta", decision_type=DecisionType.ORDER, description=str(i)))
        results = audit_trail.query(agent="Alpha")
        assert len(results) == 3
        assert all(e.decision.agent == "Alpha" for e in results)

    def test_query_by_decision_type(self, audit_trail):
        audit_trail.record(Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="r1"))
        audit_trail.record(Decision.now(agent="A", decision_type=DecisionType.ORDER, description="o1"))
        audit_trail.record(Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="r2"))
        results = audit_trail.query(decision_type="research")
        assert len(results) == 2
        assert all(e.decision.decision_type == DecisionType.RESEARCH for e in results)

    def test_query_by_status(self, audit_trail):
        d1 = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="r1")
        d1.status = DecisionStatus.EXECUTED
        d2 = Decision.now(agent="A", decision_type=DecisionType.ORDER, description="o1")
        d2.status = DecisionStatus.PENDING
        audit_trail.record(d1)
        audit_trail.record(d2)
        results = audit_trail.query(status="executed")
        assert len(results) == 1
        assert results[0].decision.decision_id == d1.decision_id

    def test_query_limit(self, audit_trail):
        for i in range(10):
            d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description=str(i))
            audit_trail.record(d)
        results = audit_trail.query(limit=3)
        assert len(results) == 3

    def test_to_json_returns_string(self, audit_trail):
        d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="test")
        audit_trail.record(d)
        json_str = audit_trail.to_json()
        data = json.loads(json_str)
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["decision"]["description"] == "test"

    def test_to_json_writes_file(self, audit_trail, tmp_path):
        d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="test")
        audit_trail.record(d)
        fp = tmp_path / "audit.json"
        audit_trail.to_json(str(fp))
        assert fp.exists()
        data = json.loads(fp.read_text())
        assert len(data) == 1

    def test_clear(self, audit_trail):
        for i in range(3):
            d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description=str(i))
            audit_trail.record(d)
        assert audit_trail.count() == 3
        audit_trail.clear()
        assert audit_trail.count() == 0
        assert audit_trail.get("anything") is None

    def test_lookup_by_id_after_clear(self, audit_trail):
        d = Decision.now(agent="A", decision_type=DecisionType.RESEARCH, description="test")
        audit_trail.record(d)
        audit_trail.clear()
        assert audit_trail.get(d.decision_id) is None
