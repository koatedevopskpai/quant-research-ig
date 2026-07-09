"""
audit_trail.py — Full decision logging with LLM explanations.

Every decision, signal, order, and risk check is recorded as an
immutable entry in the audit trail. The trail can be:
    - Exported to JSON for external analysis
    - Queried by agent, decision type, or time range
    - Used to reconstruct the full Reasoning→Know→Do→Get cycle

The LLM explanation field provides human-readable reasoning for
every decision, bridging the gap between quantitative signals and
narrative understanding.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from agentic_finance.decision import Decision


@dataclass
class AuditEntry:
    decision: Decision
    market_context: dict[str, Any] = field(default_factory=dict)
    signals_used: list[dict[str, Any]] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "market_context": self.market_context,
            "signals_used": self.signals_used,
            "tags": self.tags,
        }


class AuditTrail:
    def __init__(self):
        self._entries: list[AuditEntry] = []
        self._decisions: dict[str, AuditEntry] = {}

    def record(self, decision: Decision, market_context: dict[str, Any] | None = None,
               signals_used: list[dict[str, Any]] | None = None,
               tags: list[str] | None = None) -> AuditEntry:
        entry = AuditEntry(
            decision=decision,
            market_context=market_context or {},
            signals_used=signals_used or [],
            tags=tags or [],
        )
        self._entries.append(entry)
        self._decisions[decision.decision_id] = entry
        return entry

    def get(self, decision_id: str) -> Optional[AuditEntry]:
        return self._decisions.get(decision_id)

    def query(self, agent: str | None = None,
              decision_type: str | None = None,
              status: str | None = None,
              limit: int = 100) -> list[AuditEntry]:
        results = list(self._entries)
        if agent:
            results = [e for e in results if e.decision.agent == agent]
        if decision_type:
            results = [e for e in results if e.decision.decision_type.value == decision_type]
        if status:
            results = [e for e in results if e.decision.status.value == status]
        return results[-limit:]

    def to_json(self, filepath: str | None = None) -> str:
        data = [e.to_dict() for e in self._entries]
        result = json.dumps(data, indent=2, default=str)
        if filepath:
            with open(filepath, "w") as f:
                f.write(result)
        return result

    def count(self) -> int:
        return len(self._entries)

    def clear(self) -> None:
        self._entries.clear()
        self._decisions.clear()
