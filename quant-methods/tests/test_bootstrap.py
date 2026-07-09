import numpy as np
import pytest
from quant_methods.validation.bootstrap import BlockBootstrap, BootstrapResult


class TestBlockBootstrap:
    def test_confidence_interval_returns_result(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 200)
        bt = BlockBootstrap(n_bootstrap=50, random_seed=42)
        result = bt.confidence_interval(ret, lambda r: float(r.mean()))
        assert isinstance(result, BootstrapResult)
        assert result.ci_lower <= result.actual_value <= result.ci_upper

    def test_sharpe_ratio_method(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        bt = BlockBootstrap(n_bootstrap=50, random_seed=42)
        result = bt.sharpe_ratio(ret)
        assert result.metric_name == "Sharpe Ratio"
        assert isinstance(result.actual_value, float)

    def test_too_few_observations_raises(self):
        bt = BlockBootstrap()
        with pytest.raises(ValueError, match="Need >=5"):
            bt.confidence_interval(np.array([1.0, 2.0, 3.0]), lambda r: float(r.mean()))

    def test_block_size_auto_selection(self):
        bt = BlockBootstrap()
        bs = bt._optimal_block_size(1000)
        assert bs >= 2
        assert bs <= 100

    def test_circular_bootstrap_preserves_length(self):
        np.random.seed(42)
        series = np.random.normal(0, 1, 100)
        bt = BlockBootstrap(n_bootstrap=10, random_seed=42)
        resampled = bt._circular_block_bootstrap(series)
        assert len(resampled) == len(series)

    def test_distribution_shape(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 200)
        bt = BlockBootstrap(n_bootstrap=100, random_seed=42)
        result = bt.confidence_interval(ret, lambda r: float(r.mean()))
        assert result.distribution is not None
        assert len(result.distribution) == 100
