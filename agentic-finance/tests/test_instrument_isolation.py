"""
Instrument isolation in the production runner.

The 2026-10 review fed instrument A at 100/101 and B at 1/1.1 into one runner and
got a crossed book (bid 100, ask 1.1) plus a mixed price series that research
labelled with the first EPIC. These tests pin the routing contract.
"""

import pytest

from agentic_finance.run_production import ProductionRunner
from hft_engine.feeds.tick_consumer import MarketDataMessage


def _l1(epic: str, bid: float, ask: float, last: float = 0.0) -> MarketDataMessage:
    return MarketDataMessage(
        epic=epic, message_type="L1",
        bid_price=bid, ask_price=ask, last_price=last,
        timestamp_ns=1,
    )


def _trade(epic: str, price: float) -> MarketDataMessage:
    return MarketDataMessage(
        epic=epic, message_type="TRADE", last_price=price, timestamp_ns=1,
    )


class TestInstrumentIsolation:
    def test_books_are_separate(self):
        runner = ProductionRunner()
        a, b = runner.epics[0], runner.epics[1]

        runner._on_market_data(_l1(a, 100.0, 101.0))
        runner._on_market_data(_l1(b, 1.0, 1.1))

        sa = runner.books[a].l1_snapshot()
        sb = runner.books[b].l1_snapshot()

        assert (sa.bid_price, sa.ask_price) == (100.0, 101.0)
        assert (sb.bid_price, sb.ask_price) == (1.0, 1.1)
        # The regression: the shared book became crossed (bid 100, ask 1.1).
        assert sa.bid_price < sa.ask_price
        assert sb.bid_price < sb.ask_price

    def test_price_histories_are_separate(self):
        runner = ProductionRunner()
        a, b = runner.epics[0], runner.epics[1]

        runner._on_market_data(_trade(a, 100.0))
        runner._on_market_data(_trade(a, 101.0))
        runner._on_market_data(_trade(b, 1.0))

        assert runner._price_caches[a] == [100.0, 101.0]
        assert runner._price_caches[b] == [1.0]

    def test_analysers_are_separate(self):
        runner = ProductionRunner()
        a, b = runner.epics[0], runner.epics[1]
        assert runner.analysers[a] is not runner.analysers[b]

    def test_unknown_epic_is_ignored(self):
        runner = ProductionRunner()
        runner._on_market_data(_l1("NOT_SUBSCRIBED", 1.0, 2.0))
        assert "NOT_SUBSCRIBED" not in runner.books

    def test_l1_replaces_instead_of_accumulating(self):
        runner = ProductionRunner()
        a = runner.epics[0]
        runner._on_market_data(_l1(a, 100.0, 101.0))
        runner._on_market_data(_l1(a, 90.0, 95.0))
        snap = runner.books[a].l1_snapshot()
        assert snap.bid_price == 90.0
        assert snap.ask_price == 95.0

    def test_price_history_is_capped_per_instrument(self):
        runner = ProductionRunner()
        a = runner.epics[0]
        for i in range(1100):
            runner._on_market_data(_trade(a, float(i)))
        assert len(runner._price_caches[a]) == 1000
        assert runner._price_caches[a][-1] == 1099.0
