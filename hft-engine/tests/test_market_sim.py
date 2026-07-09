import pytest
from hft_engine.market_sim.generator import MarketSimulator, SyntheticTick


class TestMarketSimulator:
    def test_generate_l1_ticks(self):
        sim = MarketSimulator(l1_freq_hz=10, trade_freq_hz=0, seed=42)
        ticks = sim.generate_ticks(1.0)
        assert len(ticks) > 0
        for t in ticks:
            assert t.tick_type == "L1"
            assert t.bid_price > 0
            assert t.ask_price > t.bid_price

    def test_generate_trade_ticks(self):
        sim = MarketSimulator(l1_freq_hz=0, trade_freq_hz=10, seed=42)
        ticks = sim.generate_ticks(1.0)
        assert len(ticks) > 0
        for t in ticks:
            assert t.tick_type == "TRADE"
            assert t.last_price > 0
            assert t.last_size > 0

    def test_generate_mixed_ticks(self):
        sim = MarketSimulator(l1_freq_hz=10, trade_freq_hz=5, seed=42)
        ticks = sim.generate_ticks(2.0)
        types = {t.tick_type for t in ticks}
        assert "L1" in types
        assert "TRADE" in types

    def test_price_evolves_over_time(self):
        sim = MarketSimulator(initial_price=100.0, volatility=0.0, seed=42)
        ticks = sim.generate_ticks(1.0)
        l1_ticks = [t for t in ticks if t.tick_type == "L1"]
        mid_prices = [(t.bid_price + t.ask_price) / 2.0 for t in l1_ticks]
        if mid_prices:
            mean_mid = sum(mid_prices) / len(mid_prices)
            assert abs(mean_mid - 100.0) < 1.0

    def test_depth_built_after_evolution(self):
        sim = MarketSimulator(seed=42)
        ticks = sim.generate_ticks(1.0)
        bids = sim.bids
        asks = sim.asks
        assert len(bids) == sim.depth_levels
        assert len(asks) == sim.depth_levels
        for i in range(len(bids) - 1):
            assert bids[i][0] > bids[i + 1][0]
            assert asks[i][0] < asks[i + 1][0]

    def test_to_dataframe(self):
        sim = MarketSimulator(seed=42)
        ticks = sim.generate_ticks(1.0)
        df = sim.to_dataframe(ticks)
        assert len(df) == len(ticks)
        assert "timestamp" in df.columns
        assert "type" in df.columns

    def test_spread_is_positive(self):
        sim = MarketSimulator(seed=42)
        ticks = sim.generate_ticks(5.0)
        l1_ticks = [t for t in ticks if t.tick_type == "L1"]
        if l1_ticks:
            for t in l1_ticks:
                assert t.ask_price > t.bid_price
