import numpy as np
import pytest
from quant_methods.validation.wfa import WalkForwardAnalysis, WFAResult


class TestWalkForwardAnalysis:
    def test_run_returns_wfa_result(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        wfa = WalkForwardAnalysis(train_size=100, test_size=20)
        result = wfa.run(ret)
        assert isinstance(result, WFAResult)
        assert isinstance(result.oos_sharpe, float)

    def test_anchored_mode(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        wfa = WalkForwardAnalysis(train_size=100, test_size=20, mode="anchored")
        result = wfa.run(ret)
        assert result.n_windows >= 1
        assert result.oos_sharpe is not None

    def test_rolling_mode(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        wfa = WalkForwardAnalysis(train_size=100, test_size=20, mode="rolling")
        result = wfa.run(ret)
        assert result.n_windows >= 1

    def test_too_few_observations_raises(self):
        wfa = WalkForwardAnalysis(train_size=100, test_size=20)
        with pytest.raises(ValueError, match="Need >= 120"):
            wfa.run(np.array([1.0, 2.0]))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="mode"):
            WalkForwardAnalysis(train_size=100, test_size=20, mode="invalid")

    def test_window_results_structure(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        wfa = WalkForwardAnalysis(train_size=100, test_size=20)
        result = wfa.run(ret)
        assert len(result.window_results) >= 1
        for w in result.window_results:
            assert "oos_sharpe" in w
            assert "predicted_sharpe" in w
            assert "train_start" in w

    def test_oos_returns_length(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        wfa = WalkForwardAnalysis(train_size=100, test_size=20)
        result = wfa.run(ret)
        assert len(result.oos_returns) > 0

    def test_robustness_flag(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        wfa = WalkForwardAnalysis(train_size=100, test_size=20)
        result = wfa.run(ret)
        assert isinstance(result.is_robust, bool)

    def test_custom_train_fn(self):
        np.random.seed(42)
        ret = np.random.normal(0.001, 0.02, 500)
        wfa = WalkForwardAnalysis(train_size=100, test_size=20)

        def train_fn(data):
            return data * 0.5

        result = wfa.run(ret, train_fn=train_fn)
        assert result.n_windows >= 1
