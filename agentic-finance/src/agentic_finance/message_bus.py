"""
message_bus.py — In-memory event bus for agent communication.

Follows the publish-subscribe pattern. Agents subscribe to message
types they care about and the bus delivers asynchronously (in-process).
This decouples agents — the Research Agent doesn't need to know about
the Execution Agent, and vice versa.

Message types used across the system:
    "research.completed"   — Research Agent published a report
    "signal.generated"     — ML model produced a trading signal
    "order.created"        — Execution Agent created an order
    "risk.alert"           — Risk Manager issued a warning
    "risk.kill"            — Risk Manager triggered kill switch
    "allocation.updated"   — Portfolio Agent rebalanced
    "audit.record"         — Audit trail entry
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from datetime import datetime, timezone
import uuid
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    topic: str
    sender: str
    payload: dict[str, Any] = field(default_factory=dict)
    message_id: str = ""
    timestamp: Optional[datetime] = None

    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())[:8]
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message_id": self.message_id,
            "topic": self.topic,
            "sender": self.sender,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
        }


class MessageBus:
    def __init__(self):
        self._subscribers: dict[str, list[Callable[[AgentMessage], None]]] = {}

    def subscribe(self, topic: str, callback: Callable[[AgentMessage], None]) -> None:
        if topic not in self._subscribers:
            self._subscribers[topic] = []
        self._subscribers[topic].append(callback)

    def unsubscribe(self, topic: str, callback: Callable[[AgentMessage], None]) -> None:
        if topic in self._subscribers:
            self._subscribers[topic] = [
                cb for cb in self._subscribers[topic] if cb is not callback
            ]

    def publish(self, message: AgentMessage) -> None:
        callbacks = self._subscribers.get(message.topic, []) + self._subscribers.get("*", [])
        for cb in callbacks:
            try:
                cb(message)
            except Exception:
                logger.exception(f"Error in subscriber for {message.topic}")

    def publish_topic(self, topic: str, sender: str,
                      payload: dict[str, Any] | None = None) -> AgentMessage:
        msg = AgentMessage(topic=topic, sender=sender, payload=payload or {})
        self.publish(msg)
        return msg
