import numpy as np
import pandas as pd
import pytest
from quant_methods.validation.backtester import Backtester, BacktestResult


class TestBacktester:
    @pytest.fixture
    def sample_data(self):
        n = 252
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(ret))
        signal = pd.Series(np.where(np.random.rand(n) > 0.3, 1.0, -1.0), index=idx)
        return pd.Series(prices, index=idx), signal

    def test_run_returns_backtest_result(self, sample_data):
        prices, signal = sample_data
        bt = Backtester()
        result = bt.run(prices, signal)
        assert isinstance(result, BacktestResult)
        assert isinstance(result.sharpe_ratio, float)
        assert result.total_trades > 0

    def test_equal_weight_scheme(self, sample_data):
        prices, _ = sample_data
        equal_signal = pd.Series(1.0, index=prices.index)
        bt = Backtester()
        r = bt.run(prices, equal_signal, weight_scheme="equal")
        assert isinstance(r.sharpe_ratio, float)

    def test_total_return_positive_for_upward_trend(self):
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        prices = pd.Series(100 * np.exp(np.linspace(0, 0.1, n)), index=idx)
        signal = pd.Series(1.0, index=idx)
        bt = Backtester()
        r = bt.run(prices, signal)
        assert r.total_return > 0

    def test_max_drawdown_non_positive(self, sample_data):
        prices, signal = sample_data
        bt = Backtester()
        r = bt.run(prices, signal)
        assert r.max_drawdown <= 0

    def test_transaction_costs_reduce_returns(self, sample_data):
        prices, signal = sample_data
        bt0 = Backtester(transaction_cost_bps=0.0)
        bt1 = Backtester(transaction_cost_bps=10.0)
        r0 = bt0.run(prices, signal)
        r1 = bt1.run(prices, signal)
        assert r1.total_return <= r0.total_return + 1e-10

    def test_benchmark_metrics(self, sample_data):
        prices, signal = sample_data
        bt = Backtester()
        bench = prices * 0.99
        r = bt.run(prices, signal, benchmark=bench)
        assert r.alpha is not None
        assert r.beta is not None
        assert r.tracking_error is not None
        assert r.information_ratio is not None

    def test_vol_target_scheme(self, sample_data):
        prices, signal = sample_data
        bt = Backtester()
        r = bt.run(prices, signal, weight_scheme="vol_target")
        assert isinstance(r.sharpe_ratio, float)

    def test_drawdown_duration(self, sample_data):
        prices, signal = sample_data
        bt = Backtester()
        r = bt.run(prices, signal)
        assert r.max_drawdown_duration >= 0

    def test_win_rate_between_zero_and_one(self, sample_data):
        prices, signal = sample_data
        bt = Backtester()
        r = bt.run(prices, signal)
        assert 0.0 <= r.win_rate <= 1.0
