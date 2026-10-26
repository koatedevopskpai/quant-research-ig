import numpy as np
import pytest
from quant_methods.validation.wfa import WalkForwardAnalysis


class SignModel:
    def fit(self, x, y):
        self.direction = np.sign(y.mean())
        return self

    def predict(self, x):
        return np.full(len(x), self.direction)


def returns_fn(y, predictions):
    return y * predictions


@pytest.mark.parametrize("mode", ["rolling", "anchored"])
def test_chronological_oos_returns_come_from_fitted_policy(mode):
    x = np.arange(60)[:, None]
    y = np.linspace(-0.03, 0.01, 60)
    ends = np.minimum(np.arange(60) + 2, 59)
    wfa = WalkForwardAnalysis(20, 10, mode=mode, gap=1)
    result = wfa.run(x, y, SignModel, returns_fn, label_end=ends)
    expected = []
    for fold in result.window_results:
        train, test = fold["train_indices"], fold["test_indices"]
        assert train.max() < test.min()
        assert ends[train].max() < test.min()
        assert fold["train_end"] <= fold["test_start"] - 1
        expected.extend(y[test] * np.sign(y[train].mean()))
    np.testing.assert_allclose(result.oos_returns, expected)
    assert len(np.unique(result.oos_indices)) == len(result.oos_indices)
    assert not np.allclose(result.oos_returns, y[result.oos_indices])


def test_anchored_runs_do_not_mutate_configuration():
    wfa = WalkForwardAnalysis(20, 10, mode="anchored")
    x, y = np.arange(60), np.linspace(-0.02, 0.01, 60)
    a = wfa.run(x, y, SignModel, returns_fn)
    b = wfa.run(x, y, SignModel, returns_fn)
    assert wfa.train_size == a.train_size == b.train_size == 20
    np.testing.assert_array_equal(a.oos_indices, b.oos_indices)
    np.testing.assert_allclose(a.oos_returns, b.oos_returns)
    assert len(a.window_results[0]["train_indices"]) == 20
    assert len(a.window_results[-1]["train_indices"]) == 50


def test_future_changes_leave_earlier_oos_results_unchanged():
    x, y = np.arange(80), np.linspace(-0.02, 0.04, 80)
    altered = y.copy()
    altered[60:] *= -10
    wfa = WalkForwardAnalysis(20, 10)
    a = wfa.run(x, y, SignModel, returns_fn)
    b = wfa.run(x, altered, SignModel, returns_fn)
    np.testing.assert_array_equal(a.oos_returns[:40], b.oos_returns[:40])


def test_minimum_controls_initial_window_and_rolling_cap():
    windows = list(WalkForwardAnalysis(20, 10, min_train_size=10).split(50))
    assert len(windows[0][0]) == 10
    assert len(windows[-1][0]) == 20


@pytest.mark.parametrize(
    "kwargs", [dict(mode="invalid"), dict(test_size=0), dict(gap=-1), dict(min_train_size=30)]
)
def test_invalid_configuration(kwargs):
    arguments = dict(train_size=20, test_size=10)
    arguments.update(kwargs)
    with pytest.raises(ValueError):
        WalkForwardAnalysis(**arguments)


def test_too_few_rows_and_nonfinite_predictions_rejected():
    wfa = WalkForwardAnalysis(20, 10)
    with pytest.raises(ValueError, match="Need >= 30"):
        list(wfa.split(29))

    class InvalidModel(SignModel):
        def predict(self, x):
            return np.full(len(x), np.nan)

    with pytest.raises(ValueError, match="finite prediction"):
        wfa.run(np.arange(40), np.arange(40), InvalidModel, returns_fn)
