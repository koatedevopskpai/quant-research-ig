import numpy as np
import pytest
from quant_methods.validation.cpcv import CombinatorialPurgedCV, CPCVResult


class TestCombinatorialPurgedCV:
    def test_run_returns_cpcv_result(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
        result = cv.run(ret)
        assert isinstance(result, CPCVResult)
        assert isinstance(result.sharpe_mean, float)

    def test_invalid_n_test_groups_raises(self):
        with pytest.raises(ValueError, match="n_test_groups"):
            CombinatorialPurgedCV(n_groups=5, n_test_groups=5)

    def test_too_few_observations_raises(self):
        cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
        with pytest.raises(ValueError, match="Need >= 10"):
            cv.run(np.array([1.0, 2.0, 3.0]))

    def test_split_groups_exhaustive(self):
        cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
        groups = cv._split_groups(100)
        assert len(groups) == 5
        total = sum(e - s for s, e in groups)
        assert total == 100

    def test_embargo_purges_indices(self):
        cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2, embargo=5)
        start, end = cv._purge_indices(50, 60, 200)
        assert start == 45
        assert end == 65

    def test_embargo_clamps_to_boundaries(self):
        cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2, embargo=10)
        start, end = cv._purge_indices(0, 5, 200)
        assert start == 0
        start2, end2 = cv._purge_indices(195, 200, 200)
        assert end2 == 200

    def test_custom_metric_fn(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=2)
        result = cv.run(ret, metric_fn=lambda r: float(np.mean(r)))
        assert isinstance(result.sharpe_mean, float)

    def test_significance_flag(self):
        np.random.seed(42)
        ret = np.random.normal(0.003, 0.02, 1000)
        cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=2)
        result = cv.run(ret)
        assert isinstance(result.is_significant, bool)
