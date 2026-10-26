import numpy as np
import pytest
from quant_methods.validation.dsr import DeflatedSharpeRatio, DSRResult


class TestDeflatedSharpeRatio:
    def test_compute_returns_dsr_result(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        dsr = DeflatedSharpeRatio()
        result = dsr.compute(ret, num_trials=1)
        assert isinstance(result, DSRResult)
        assert isinstance(result.dsr, float)
        assert 0 <= result.dsr <= 1

    def test_dsr_lower_than_psr_with_many_trials(self):
        np.random.seed(42)
        ret = np.random.normal(0.002, 0.02, 500)
        dsr = DeflatedSharpeRatio()
        r1 = dsr.compute(ret, num_trials=1)
        r100 = dsr.compute(ret, num_trials=100, trial_sharpe_variance=1.0)
        assert r100.dsr <= r1.dsr + 1e-10

    def test_expected_max_sr_increases_with_trials(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        dsr = DeflatedSharpeRatio()
        r1 = dsr.compute(ret, num_trials=1)
        r100 = dsr.compute(ret, num_trials=100, trial_sharpe_variance=1.0)
        assert r100.expected_max_sr >= r1.expected_max_sr - 1e-10

    def test_num_trials_recorded(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        dsr = DeflatedSharpeRatio()
        result = dsr.compute(ret, num_trials=50, trial_sharpe_variance=1.0)
        assert result.num_trials == 50
        assert result.n_observations == 500

    def test_too_few_observations_raises(self):
        dsr = DeflatedSharpeRatio()
        with pytest.raises(ValueError, match="Need >=3"):
            dsr.compute(np.array([1.0, 2.0]))

    def test_dsr_significance_flag(self):
        np.random.seed(42)
        ret = np.random.normal(0.005, 0.015, 1000)
        dsr = DeflatedSharpeRatio()
        r1 = dsr.compute(ret, num_trials=1)
        r1000 = dsr.compute(ret, num_trials=1000, trial_sharpe_variance=1.0)
        assert isinstance(r1.is_significant, bool)
        assert isinstance(r1000.is_significant, bool)
