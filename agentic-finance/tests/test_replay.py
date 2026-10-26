"""
Deterministic end-to-end replay: the integration proof.

Every component now has unit tests. This file asserts the property that matters:
run the same event stream through data -> model -> signal -> fill -> P&L -> risk
-> audit and the book still balances, the instruments stay separate, and a
breach stops new risk.
"""

from datetime import datetime, timedelta, timezone

import pytest

from agentic_finance.replay import (
    RANGE_EPIC,
    TREND_EPIC,
    ReplayConfig,
    default_events,
    run_replay,
    _scale_event,
)

START = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _ns_after(minutes: int) -> int:
    return int((START + timedelta(minutes=minutes)).timestamp()) * 1_000_000_000


class TestInvariants:
    def test_full_replay_passes_accounting_isolation_and_risk(self):
        result = run_replay(default_events(120))

        assert all(result.invariants.values()), result.invariants
        assert result.orders > 0
        assert result.fills > 0
        assert result.audit_entries > 0
        assert result.reconcile["balanced"] is True
        assert result.final_equity == pytest.approx(result.reconcile["equity"])

    def test_every_fill_belongs_to_an_order(self):
        result = run_replay(default_events(80))
        assert result.invariants["fills_have_orders"] is True
        assert result.fills == result.reconcile["fills"]

    def test_audit_trail_covers_research_orders_and_risk(self):
        result = run_replay(default_events(60))
        # research + execution + risk all record; a replay with none is a bug.
        assert result.audit_entries >= result.orders
        assert result.audit_entries > 0

    def test_position_cap_is_respected(self):
        result = run_replay(default_events(120))
        assert result.invariants["position_within_cap"] is True


class TestDeterminism:
    def test_two_runs_produce_the_same_numbers(self):
        first = run_replay(default_events(100))
        second = run_replay(default_events(100))

        assert first.equity_curve == pytest.approx(second.equity_curve)
        assert first.fills == second.fills
        assert first.orders == second.orders
        assert first.final_equity == pytest.approx(second.final_equity)
        assert first.invariants == second.invariants

    def test_marks_are_per_instrument(self):
        events = default_events(30)
        result = run_replay(events)
        assert set(result.marks) == {TREND_EPIC, RANGE_EPIC}
        # Each mark equals the last event for that EPIC, not a shared price.
        for epic in (TREND_EPIC, RANGE_EPIC):
            last = [e for e in events if e.epic == epic][-1]
            expected = ((last.bid + last.ask) / 2 if last.bid and last.ask else last.last)
            assert result.marks[epic] == pytest.approx(expected)


class TestHalt:
    def _shocked(self, steps: int, after_minutes: int, pct: float):
        cutoff = _ns_after(after_minutes)
        return [
            _scale_event(event, 1.0 - pct)
            if event.epic == TREND_EPIC and event.provider_ts_ns >= cutoff
            else event
            for event in default_events(steps)
        ]

    def test_breach_halts_and_no_new_entries_follow(self):
        # Stop-loss is widened so the shock is absorbed as a mark-to-market loss
        # rather than being capped by a protective exit before risk can see it.
        result = run_replay(
            self._shocked(160, after_minutes=60, pct=0.40),
            ReplayConfig(daily_loss_limit=0.005, stop_loss_pct=0.50),
        )

        assert result.halted is True
        assert result.halt_reason
        assert result.entries_after_halt == 0
        assert result.invariants["no_new_entries_after_halt"] is True
        assert result.invariants["accounting_balanced"] is True

    def test_book_balances_after_a_halt(self):
        result = run_replay(
            self._shocked(160, after_minutes=60, pct=0.40),
            ReplayConfig(daily_loss_limit=0.005, stop_loss_pct=0.50),
        )

        report = result.reconcile
        assert report["balanced"] is True
        assert report["equity"] == pytest.approx(
            100_000.0 + report["realized_pnl"] + report["unrealized_pnl"]
            - report["commissions"]
        )
        assert result.final_equity > 0
