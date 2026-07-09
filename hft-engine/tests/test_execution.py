import pytest
from hft_engine.execution.simulator import ExecutionSimulator, ImpactResult


class TestExecutionSimulator:
    def test_permanent_impact_sign(self):
        sim = ExecutionSimulator(adv=1_000_000)
        buy_impact = sim.permanent_impact(10000, "buy")
        sell_impact = sim.permanent_impact(10000, "sell")
        assert buy_impact > 0
        assert sell_impact < 0
        assert abs(buy_impact) == abs(sell_impact)

    def test_temporary_impact_sign(self):
        sim = ExecutionSimulator()
        buy_impact = sim.temporary_impact(10000, "buy")
        sell_impact = sim.temporary_impact(10000, "sell")
        assert buy_impact > 0
        assert sell_impact < 0

    def test_simulate_returns_result(self):
        sim = ExecutionSimulator()
        r = sim.simulate(10000, "buy", 100.0)
        assert isinstance(r, ImpactResult)
        assert r.total_cost_bps > 0
        assert r.execution_price > 100.0

    def test_simulate_sell_below_arrival(self):
        sim = ExecutionSimulator()
        r = sim.simulate(10000, "sell", 100.0)
        assert r.execution_price < 100.0

    def test_larger_order_higher_cost(self):
        sim = ExecutionSimulator()
        small = sim.simulate(1000, "buy", 100.0)
        large = sim.simulate(100000, "buy", 100.0)
        assert large.total_cost_bps >= small.total_cost_bps

    def test_higher_urgency_lower_impact(self):
        sim = ExecutionSimulator()
        slow = sim.simulate(10000, "buy", 100.0, urgency=0.1)
        fast = sim.simulate(10000, "buy", 100.0, urgency=2.0)
        assert fast.temporary_impact_bps <= slow.temporary_impact_bps

    def test_slice_order_sums_to_total(self):
        sim = ExecutionSimulator()
        slices = sim.slice_order(10000, 5)
        assert sum(slices) == 10000
        assert len(slices) == 5

    def test_slice_order_single(self):
        sim = ExecutionSimulator()
        slices = sim.slice_order(10000, 1)
        assert slices == [10000.0]

    def test_fill_from_book_buy(self):
        sim = ExecutionSimulator()
        book_bids = [(99.0, 1000), (98.0, 2000)]
        book_asks = [(101.0, 1000), (102.0, 2000)]
        fills = sim.fill_from_book("buy", 1500, book_bids, book_asks, 0.0)
        assert len(fills) == 2
        assert sum(f.size for f in fills) == 1500
        assert all(f.is_aggressive for f in fills)

    def test_fill_from_book_sell(self):
        sim = ExecutionSimulator()
        book_bids = [(99.0, 1000), (98.0, 2000)]
        book_asks = [(101.0, 1000)]
        fills = sim.fill_from_book("sell", 2500, book_bids, book_asks, 0.0)
        assert len(fills) == 2
        assert sum(f.size for f in fills) == 2500

    def test_simulate_with_book_levels(self):
        sim = ExecutionSimulator()
        book_asks = [(101.0, 1000), (102.0, 2000)]
        r = sim.simulate(1500, "buy", 100.0, book_levels=book_asks)
        assert isinstance(r, ImpactResult)
        assert r.total_cost_bps > 0
        assert r.shares_executed > 0

    def test_simulate_with_book_levels_sell(self):
        sim = ExecutionSimulator()
        book_bids = [(99.0, 1000), (98.0, 2000)]
        r = sim.simulate(1500, "sell", 100.0, book_levels=book_bids)
        assert isinstance(r, ImpactResult)
        assert r.execution_price < 100.0

    def test_simulate_with_book_larger_than_order(self):
        """When book has enough volume, all shares are filled."""
        sim = ExecutionSimulator()
        book_asks = [(101.0, 10000)]
        r = sim.simulate(500, "buy", 100.0, book_levels=book_asks)
        assert r.shares_executed == 500

    def test_simulate_with_book_smaller_than_order(self):
        """When book has insufficient volume, partial fill occurs."""
        sim = ExecutionSimulator()
        book_asks = [(101.0, 100)]
        r = sim.simulate(1000, "buy", 100.0, book_levels=book_asks)
        assert r.shares_executed <= 1000

    def test_slice_order_zero_slices(self):
        sim = ExecutionSimulator()
        slices = sim.slice_order(10000, 0)
        assert slices == [10000.0]

    def test_slice_order_negative_slices(self):
        sim = ExecutionSimulator()
        slices = sim.slice_order(10000, -1)
        assert slices == [10000.0]

    def test_fill_from_book_exact_volume(self):
        sim = ExecutionSimulator()
        book_asks = [(101.0, 1000), (102.0, 1000)]
        fills = sim.fill_from_book("buy", 2000, [], book_asks, 0.0)
        assert len(fills) == 2
        assert sum(f.size for f in fills) == 2000

    def test_fill_from_book_partial(self):
        sim = ExecutionSimulator()
        book_asks = [(101.0, 500)]
        fills = sim.fill_from_book("buy", 1000, [], book_asks, 0.0)
        assert len(fills) == 1
        assert fills[0].size == 500
