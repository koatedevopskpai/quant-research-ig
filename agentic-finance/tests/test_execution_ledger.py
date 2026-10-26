"""
Ledger tests: the accounting that "paper trading" claims to provide.

Every assertion here is arithmetic a reviewer can redo by hand, because the
previous implementation had no cash, no realised P&L and no reconciliation.
"""

import pytest

from agentic_finance.execution_ledger import PositionLedger


class TestFillsAndCash:
    def test_buy_moves_cash_and_position(self):
        ledger = PositionLedger(100_000.0, commission_bps=1.0, slippage_bps=0.0)
        fill = ledger.apply_fill("EPIC", "buy", 10, 100.0)

        assert fill.price == pytest.approx(100.0)
        assert ledger.positions["EPIC"].quantity == pytest.approx(10.0)
        assert ledger.positions["EPIC"].avg_price == pytest.approx(100.0)
        assert ledger.cash == pytest.approx(100_000.0 - 1_000.0 - 0.1)
        assert ledger.total_commission == pytest.approx(0.1)

    def test_slippage_is_adverse_in_both_directions(self):
        ledger = PositionLedger(100_000.0, commission_bps=0.0, slippage_bps=10.0)
        buy = ledger.apply_fill("EPIC", "buy", 1, 100.0)
        sell = ledger.apply_fill("EPIC", "sell", 1, 100.0)
        assert buy.price == pytest.approx(100.1)
        assert sell.price == pytest.approx(99.9)

    def test_reduction_books_realised_pnl(self):
        ledger = PositionLedger(100_000.0, commission_bps=0.0)
        ledger.apply_fill("EPIC", "buy", 10, 100.0)
        ledger.apply_fill("EPIC", "sell", 4, 110.0)

        position = ledger.positions["EPIC"]
        assert position.quantity == pytest.approx(6.0)
        assert position.avg_price == pytest.approx(100.0)
        assert ledger.realized_pnl == pytest.approx(40.0)
        assert ledger.cash == pytest.approx(100_000.0 - 1_000.0 + 440.0)

    def test_reversal_closes_then_opens_at_the_fill_price(self):
        ledger = PositionLedger(100_000.0, commission_bps=0.0)
        ledger.apply_fill("EPIC", "buy", 10, 100.0)
        ledger.apply_fill("EPIC", "sell", 15, 110.0)

        position = ledger.positions["EPIC"]
        assert position.quantity == pytest.approx(-5.0)
        assert position.avg_price == pytest.approx(110.0)
        assert ledger.realized_pnl == pytest.approx(100.0)

    def test_short_unrealised_pnl_sign(self):
        ledger = PositionLedger(100_000.0, commission_bps=0.0)
        ledger.apply_fill("EPIC", "sell", 5, 100.0)
        assert ledger.unrealized_pnl({"EPIC": 90.0}) == pytest.approx(50.0)
        assert ledger.unrealized_pnl({"EPIC": 110.0}) == pytest.approx(-50.0)

    def test_zero_quantity_is_flat_not_a_position(self):
        ledger = PositionLedger(100_000.0)
        ledger.apply_fill("EPIC", "buy", 5, 100.0)
        ledger.apply_fill("EPIC", "sell", 5, 100.0)
        assert ledger.positions["EPIC"].is_flat
        assert ledger.open_positions() == {}


class TestReconciliation:
    def test_identity_holds_after_fills_and_price_moves(self):
        ledger = PositionLedger(100_000.0, commission_bps=1.0, slippage_bps=2.0)
        ledger.apply_fill("A", "buy", 10, 100.0)
        ledger.apply_fill("B", "sell", 3, 50.0)
        ledger.apply_fill("A", "sell", 4, 110.0)

        marks = {"A": 108.0, "B": 48.0}
        report = ledger.reconcile(marks)

        assert report["balanced"] is True, report
        assert report["difference"] == pytest.approx(0.0, abs=1e-9)
        assert report["equity"] == pytest.approx(
            ledger.cash + ledger.market_value(marks)
        )
        assert report["fills"] == 3

    def test_reconciliation_catches_a_corrupted_book(self):
        ledger = PositionLedger(100_000.0, commission_bps=0.0)
        ledger.apply_fill("A", "buy", 10, 100.0)
        ledger.cash += 1.0  # inject an error
        assert ledger.reconcile({"A": 100.0})["balanced"] is False


class TestValidation:
    @pytest.mark.parametrize("kwargs", [
        dict(quantity=0, reference_price=100.0),
        dict(quantity=-1, reference_price=100.0),
        dict(quantity=1, reference_price=0.0),
    ])
    def test_invalid_fill_inputs_rejected(self, kwargs):
        ledger = PositionLedger(100_000.0)
        with pytest.raises(ValueError):
            ledger.apply_fill("EPIC", "buy", **kwargs)

    def test_invalid_side_rejected(self):
        ledger = PositionLedger(100_000.0)
        with pytest.raises(ValueError, match="side"):
            ledger.apply_fill("EPIC", "hold", 1, 100.0)
