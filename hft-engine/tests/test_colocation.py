import pytest
from hft_engine.colocation.simulator import ColocationSimulator, ExchangeNode, LatencyArb


class TestColocationSimulator:
    def test_add_exchange(self):
        sim = ColocationSimulator()
        ny = sim.add_exchange("NYSE", 40.71, -74.01)
        assert isinstance(ny, ExchangeNode)
        assert ny.name == "NYSE"
        assert ny.latency_us > 0

    def test_distance_to_self_is_zero(self):
        node = ExchangeNode("A", 51.5, -0.12)
        assert node.distance_to(node) == 0.0

    def test_distance_london_ny(self):
        london = ExchangeNode("LSE", 51.5, -0.12)
        ny = ExchangeNode("NYSE", 40.71, -74.01)
        dist = london.distance_to(ny)
        assert 5000 < dist < 6000

    def test_detect_arbitrage(self):
        sim = ColocationSimulator()
        sim.add_exchange("NYSE", 40.71, -74.01)
        sim.add_exchange("LSE", 51.5, -0.12)
        prices = {"NYSE": 100.0, "LSE": 100.05}
        arbs = sim.detect_arbitrage(prices, min_spread_bps=0.1)
        assert len(arbs) >= 1
        for a in arbs:
            assert isinstance(a, LatencyArb)
            assert a.spread_bps > 0

    def test_no_arb_when_prices_equal(self):
        sim = ColocationSimulator()
        sim.add_exchange("A", 40.71, -74.01)
        sim.add_exchange("B", 51.5, -0.12)
        arbs = sim.detect_arbitrage({"A": 100.0, "B": 100.0}, min_spread_bps=0.1)
        assert len(arbs) == 0

    def test_round_trip_latency(self):
        sim = ColocationSimulator()
        sim.add_exchange("NYSE", 40.71, -74.01)
        rt = sim.round_trip_latency("NYSE")
        assert rt > 0
        assert rt < 100000  # ~56ms realistic London→NY round trip

    def test_competitor_round_trip(self):
        sim = ColocationSimulator(competitor_latency_penalty_us=100)
        sim.add_exchange("NYSE", 40.71, -74.01)
        our_rt = sim.round_trip_latency("NYSE")
        comp_rt = sim.competitor_round_trip("NYSE")
        assert comp_rt == our_rt + 100

    def test_add_noise(self):
        sim = ColocationSimulator()
        prices = {"A": 100.0, "B": 100.01}
        noisy = sim.add_noise(prices, noise_bps=0.5, seed=42)
        for k in prices:
            assert abs(prices[k] - noisy[k]) < prices[k] * 0.001

    def test_detect_multiple_arbitrages(self):
        sim = ColocationSimulator()
        sim.add_exchange("NYSE", 40.71, -74.01)
        sim.add_exchange("LSE", 51.5, -0.12)
        sim.add_exchange("TSE", 35.68, 139.77)
        prices = {"NYSE": 100.0, "LSE": 100.03, "TSE": 100.06}
        arbs = sim.detect_arbitrage(prices, min_spread_bps=0.1)
        assert len(arbs) >= 2
        assert arbs[0].spread_bps >= arbs[1].spread_bps

    def test_arbitrage_capturable_flag(self):
        sim = ColocationSimulator()
        sim.add_exchange("A", 40.71, -74.01)
        sim.add_exchange("B", 51.5, -0.12)
        arbs = sim.detect_arbitrage({"A": 100.0, "B": 100.05}, min_spread_bps=0.1)
        for a in arbs:
            assert isinstance(a.is_capturable, bool)
