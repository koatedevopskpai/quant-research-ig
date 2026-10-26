"""
Order lifecycle: identity, fills, partials, reductions, protective exits.

The review reproduced four failures this file pins:
  * orders had no ID and stayed "pending" forever;
  * holdings changed on order creation, not on fill;
  * a full position could not be sold because capacity used |holdings|;
  * stop-loss and take-profit were attached and never consumed.
"""

import pytest

from agentic_finance.execution_agent import ExecutionAgent, Signal


@pytest.fixture
def agent(message_bus, audit_trail):
    return ExecutionAgent(message_bus, audit_trail, initial_capital=100_000.0,
                          max_position_pct=0.05, default_stop_loss_pct=0.02,
                          commission_bps=1.0, slippage_bps=0.0)


def _signal(direction=1.0, confidence=0.8, epic="EPIC"):
    return Signal(epic=epic, direction=direction, confidence=confidence, source="test")


class TestOrderIdentityAndFills:
    def test_order_has_identity_and_fills_immediately(self, agent):
        order = agent.process_signal(_signal(), 100.0)

        assert order is not None
        assert order.order_id, "orders must be identifiable"
        assert order.status == "filled"
        assert order.filled_size == pytest.approx(order.size)
        assert order.avg_fill_price is not None
        assert len(order.fills) == 1
        assert order.fills[0].order_id == order.order_id

    def test_holdings_change_on_fill_not_on_creation(self, agent):
        # A market order fills inside process_signal, so the ledger must already
        # reflect it; the point is that the ledger is the only source of truth.
        order = agent.process_signal(_signal(), 100.0)
        assert agent.positions["EPIC"] == pytest.approx(order.filled_size)

    def test_cash_and_equity_are_reported(self, agent):
        agent.process_signal(_signal(confidence=0.5), 100.0)
        assert agent.cash < 100_000.0
        report = agent.reconcile({"EPIC": 100.0})
        assert report["balanced"] is True
        assert report["fills"] == 1


class TestPartialFills:
    def test_partial_then_complete(self, message_bus, audit_trail):
        agent = ExecutionAgent(message_bus, audit_trail, initial_capital=100_000.0,
                               max_position_pct=0.05, fill_ratio=0.5,
                               commission_bps=0.0)
        order = agent.process_signal(_signal(confidence=1.0), 100.0)

        assert order.status == "partially_filled"
        assert order.filled_size == pytest.approx(order.size * 0.5)
        half = agent.positions["EPIC"]

        completed = agent.on_price("EPIC", 100.0)
        assert order.status == "filled"
        assert order.filled_size == pytest.approx(order.size)
        assert agent.positions["EPIC"] > half
        assert any(o.order_id == order.order_id for o in completed)
        assert agent.reconcile({"EPIC": 100.0})["balanced"] is True


class TestReductionsAndReversals:
    def test_full_position_can_be_reduced(self, agent):
        entry = agent.process_signal(_signal(confidence=1.0), 100.0)
        assert agent.positions["EPIC"] == pytest.approx(entry.size)

        # Same confidence, opposite direction: capacity is exhausted, yet this is
        # a risk-reducing sell and must be allowed.
        exit_order = agent.process_signal(_signal(direction=-1.0, confidence=1.0), 100.0)
        assert exit_order is not None
        assert exit_order.side == "sell"
        assert agent.positions["EPIC"] < 0  # 1.0 long -> 1.0 short
        assert agent.reconcile({"EPIC": 100.0})["balanced"] is True

    def test_weaker_signal_reduces_without_reversing(self, agent):
        agent.process_signal(_signal(confidence=1.0), 100.0)
        before = agent.positions["EPIC"]
        agent.process_signal(_signal(confidence=0.25), 100.0)
        after = agent.positions["EPIC"]
        assert 0 < after < before

    def test_zero_delta_produces_no_order(self, agent):
        first = agent.process_signal(_signal(confidence=0.5), 100.0)
        assert first is not None
        again = agent.process_signal(_signal(confidence=0.5), 100.0)
        assert again is None  # already at target


class TestProtectiveExits:
    def test_stop_loss_closes_the_position(self, agent):
        entry = agent.process_signal(_signal(confidence=1.0), 100.0)
        stop = entry.stop_loss
        assert stop is not None

        triggered = agent.on_price("EPIC", stop - 1.0)

        assert len(triggered) == 1
        assert triggered[0].reason == "stop_loss"
        assert triggered[0].side == "sell"
        assert triggered[0].status == "filled"
        assert agent.positions == {}
        assert agent.ledger.realized_pnl < 0

    def test_take_profit_closes_the_position(self, agent):
        entry = agent.process_signal(_signal(confidence=1.0), 100.0)
        take = entry.take_profit
        assert take is not None

        triggered = agent.on_price("EPIC", take + 1.0)

        assert len(triggered) == 1
        assert triggered[0].reason == "take_profit"
        assert agent.positions == {}
        assert agent.ledger.realized_pnl > 0

    def test_no_exit_between_the_levels(self, agent):
        agent.process_signal(_signal(confidence=1.0), 100.0)
        assert agent.on_price("EPIC", 100.5) == []
        assert agent.positions["EPIC"] > 0

    def test_short_stop_is_above_entry(self, agent):
        entry = agent.process_signal(_signal(direction=-1.0, confidence=1.0), 100.0)
        assert entry.stop_loss > 100.0
        assert entry.take_profit < 100.0

        triggered = agent.on_price("EPIC", entry.stop_loss + 1.0)
        assert triggered[0].reason == "stop_loss"
        assert triggered[0].side == "buy"
        assert agent.positions == {}


class TestReconciliationUnderActivity:
    def test_book_reconciles_after_mixed_activity(self, agent):
        agent.process_signal(_signal(confidence=1.0), 100.0)
        agent.on_price("EPIC", 101.0)
        agent.process_signal(_signal(direction=-0.5, confidence=0.5), 101.0)
        agent.on_price("EPIC", 102.0)

        report = agent.reconcile({"EPIC": 102.0})
        assert report["balanced"] is True, report
        assert report["equity"] == pytest.approx(
            100_000.0 + report["realized_pnl"] + report["unrealized_pnl"]
            - report["commissions"]
        )
