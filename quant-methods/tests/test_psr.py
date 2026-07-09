import numpy as np
import pytest
from quant_methods.validation.psr import ProbabilisticSharpeRatio, PSRResult


class TestProbabilisticSharpeRatio:
    def test_compute_returns_psr_result(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 252)
        psr = ProbabilisticSharpeRatio()
        result = psr.compute(ret)
        assert isinstance(result, PSRResult)
        assert isinstance(result.psr, float)
        assert 0 <= result.psr <= 1

    def test_high_sharpe_high_psr(self):
        np.random.seed(42)
        ret = np.random.normal(0.005, 0.015, 500)
        psr = ProbabilisticSharpeRatio()
        result = psr.compute(ret)
        assert result.is_significant or result.psr > 0.5

    def test_zero_sharpe_low_psr(self):
        np.random.seed(42)
        ret = np.random.normal(0.0, 0.02, 252)
        psr = ProbabilisticSharpeRatio()
        result = psr.compute(ret)
        assert result.psr < 0.95

    def test_threshold_sharpe_lowers_psr(self):
        np.random.seed(42)
        ret = np.random.normal(0.002, 0.02, 500)
        psr = ProbabilisticSharpeRatio()
        r0 = psr.compute(ret, threshold_sharpe=0.0)
        r1 = psr.compute(ret, threshold_sharpe=1.0)
        assert r1.psr <= r0.psr + 1e-10

    def test_too_few_observations_raises(self):
        psr = ProbabilisticSharpeRatio()
        with pytest.raises(ValueError, match="Need >=3"):
            psr.compute(np.array([1.0, 2.0]))

    def test_skewness_and_kurtosis_recorded(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        psr = ProbabilisticSharpeRatio()
        result = psr.compute(ret)
        assert isinstance(result.skewness, float)
        assert isinstance(result.kurtosis, float)

    def test_z_stat_is_finite(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 252)
        psr = ProbabilisticSharpeRatio()
        result = psr.compute(ret)
        assert np.isfinite(result.z_stat)
