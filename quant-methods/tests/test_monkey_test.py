import numpy as np
import pytest
from quant_methods.validation.monkey_test import MonkeyTest, MonkeyTestResult


class TestMonkeyTest:
    def test_run_returns_result(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 200)
        mt = MonkeyTest(n_permutations=100, random_seed=42)
        result = mt.run(ret)
        assert isinstance(result, MonkeyTestResult)
        assert isinstance(result.actual_statistic, float)
        assert 0 <= result.p_value <= 1
        assert result.n_permutations == 100

    def test_p_value_low_for_real_signal(self):
        np.random.seed(42)
        ret = np.random.normal(0.005, 0.02, 500)
        mt = MonkeyTest(n_permutations=200, random_seed=42)
        result = mt.run(ret)
        assert result.is_significant or result.p_value > 0.01

    def test_too_few_observations_raises(self):
        mt = MonkeyTest()
        with pytest.raises(ValueError, match="Need >=10"):
            mt.run(np.array([1.0]))

    def test_custom_statistic_fn(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 200)
        mt = MonkeyTest(n_permutations=50, random_seed=42)
        result = mt.run(ret, statistic_fn=lambda r: float(np.mean(r)))
        assert isinstance(result.actual_statistic, float)

    def test_null_distribution_not_empty(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 200)
        mt = MonkeyTest(n_permutations=50, random_seed=42)
        result = mt.run(ret)
        assert result.null_distribution is not None
        assert len(result.null_distribution) == 50

    def test_returns_deterministic_with_seed(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 200)
        mt1 = MonkeyTest(n_permutations=50, random_seed=123)
        mt2 = MonkeyTest(n_permutations=50, random_seed=123)
        r1 = mt1.run(ret)
        r2 = mt2.run(ret)
        assert r1.p_value == r2.p_value
        assert np.allclose(r1.null_distribution, r2.null_distribution)
