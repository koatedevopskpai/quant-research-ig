import pytest
from agentic_finance.message_bus import MessageBus, AgentMessage


class TestAgentMessage:
    def test_default_id_generated(self):
        msg = AgentMessage(topic="test", sender="TestAgent", payload={"key": "val"})
        assert len(msg.message_id) == 8
        assert msg.topic == "test"
        assert msg.sender == "TestAgent"
        assert msg.payload == {"key": "val"}
        assert msg.timestamp is not None

    def test_to_dict(self):
        msg = AgentMessage(topic="t", sender="s", payload={"a": 1}, message_id="abc123")
        d = msg.to_dict()
        assert d["message_id"] == "abc123"
        assert d["topic"] == "t"
        assert d["sender"] == "s"
        assert d["payload"] == {"a": 1}
        assert "timestamp" in d


class TestMessageBus:
    def test_subscribe_and_publish(self, message_bus):
        received = []

        def cb(msg):
            received.append(msg)

        message_bus.subscribe("test.topic", cb)
        msg = message_bus.publish_topic("test.topic", "Sender", {"x": 1})
        assert len(received) == 1
        assert received[0].topic == "test.topic"

    def test_unsubscribe(self, message_bus):
        received = []

        def cb(msg):
            received.append(msg)

        message_bus.subscribe("test.topic", cb)
        message_bus.unsubscribe("test.topic", cb)
        message_bus.publish_topic("test.topic", "Sender")
        assert len(received) == 0

    def test_wildcard_subscriber(self, message_bus):
        received = []

        def cb(msg):
            received.append(msg)

        message_bus.subscribe("*", cb)
        message_bus.publish_topic("any.topic", "Sender")
        message_bus.publish_topic("another.topic", "Sender")
        assert len(received) == 2

    def test_publish_to_no_subscribers_does_not_error(self, message_bus):
        message_bus.publish_topic("unsubscribed.topic", "Sender")

    def test_publish_returns_message(self, message_bus):
        msg = message_bus.publish_topic("t", "s", {"k": "v"})
        assert isinstance(msg, AgentMessage)
        assert msg.topic == "t"
        assert msg.sender == "s"

    def test_error_in_callback_does_not_crash_bus(self, message_bus):
        def bad_cb(msg):
            raise ValueError("boom")

        good_received = []

        def good_cb(msg):
            good_received.append(msg)

        message_bus.subscribe("t", bad_cb)
        message_bus.subscribe("t", good_cb)
        message_bus.publish_topic("t", "s")
        assert len(good_received) == 1

    def test_multiple_subscribers_all_notified(self, message_bus):
        received = []

        def cb1(msg):
            received.append("cb1")

        def cb2(msg):
            received.append("cb2")

        message_bus.subscribe("t", cb1)
        message_bus.subscribe("t", cb2)
        message_bus.publish_topic("t", "s")
        assert len(received) == 2

    def test_subscribers_on_different_topics(self, message_bus):
        a, b = [], []

        def ca(msg):
            a.append(msg)

        def cb(msg):
            b.append(msg)

        message_bus.subscribe("a", ca)
        message_bus.subscribe("b", cb)
        message_bus.publish_topic("a", "s")
        assert len(a) == 1
        assert len(b) == 0
