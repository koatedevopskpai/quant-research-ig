"""
decision.py — Decision record dataclasses for the agent framework.

Every decision made by any agent is recorded as a Decision object and
persisted in the AuditTrail. This gives full transparency into the
Research→Know→Do→Get cycle and supports post-hoc analysis.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class DecisionType(Enum):
    RESEARCH = "research"
    SIGNAL = "signal"
    ORDER = "order"
    RISK_CHECK = "risk_check"
    ALLOCATION = "allocation"
    KILL_SWITCH = "kill_switch"
    AUDIT = "audit"


class DecisionStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class Decision:
    decision_id: str
    timestamp: datetime
    agent: str
    decision_type: DecisionType
    status: DecisionStatus
    description: str
    context: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    llm_explanation: str = ""
    parent_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def now(cls, agent: str, decision_type: DecisionType,
            description: str, **kwargs) -> "Decision":
        import uuid
        return cls(
            decision_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now(timezone.utc),
            agent=agent,
            decision_type=decision_type,
            status=DecisionStatus.PENDING,
            description=description,
            **kwargs,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent,
            "decision_type": self.decision_type.value,
            "status": self.status.value,
            "description": self.description,
            "context": self.context,
            "reasoning": self.reasoning,
            "llm_explanation": self.llm_explanation,
            "parent_id": self.parent_id,
            "metadata": self.metadata,
        }
