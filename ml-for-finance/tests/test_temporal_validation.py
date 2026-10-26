import numpy as np
import pandas as pd
import pytest
import torch
from ml_for_finance.deep_learning.lstm_predictor import LSTMPredictor
from ml_for_finance.forecasting.xgboost_forecaster import XGBoostForecaster


def prices(n=320):
    returns = np.random.default_rng(42).normal(0.001, 0.02, n)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=pd.date_range("2020-01-01", periods=n))


def test_lstm_training_tensors_and_scaler_ignore_validation_suffix():
    p = prices(200)
    split = int((len(p) - 1 - 10) * 0.8)
    cutoff = 10 + split
    altered = p.copy()
    altered.iloc[cutoff + 1 :] *= np.exp(np.linspace(0.1, 1, len(p) - cutoff - 1))
    a, b = LSTMPredictor(seq_len=10), LSTMPredictor(seq_len=10)
    x_a, ya = a._prepare_data(p, fit_end=cutoff)
    x_b, yb = b._prepare_data(altered, fit_end=cutoff)
    assert a._scaler_mean == b._scaler_mean
    assert a._scaler_std == b._scaler_std
    torch.testing.assert_close(x_a[:split], x_b[:split])
    torch.testing.assert_close(ya[:split], yb[:split])
    assert not torch.equal(ya[split:], yb[split:])


def test_lstm_fit_uses_only_training_prefix_for_scaling():
    p = prices(160)
    model = LSTMPredictor(seq_len=10, hidden_dim=4, num_layers=1, epochs=1, device="cpu")
    model.fit(p)
    end = 10 + int((len(p) - 1 - 10) * 0.8)
    train_returns = np.diff(np.log(p.values))[:end]
    assert model._scaler_mean == pytest.approx(train_returns.mean())
    assert model._scaler_std == pytest.approx(train_returns.std())


def test_xgboost_horizon_purges_labels_at_both_partition_boundaries():
    p = prices()
    model = XGBoostForecaster(model_params={"n_estimators": 3, "max_depth": 2, "n_jobs": 1})
    model.fit(p, target_horizon=5)
    assert model._train_indices.max() + 5 < model._validation_indices.min()
    assert model._validation_indices.max() + 5 < model._test_indices.min()
    expected = np.log(p.iloc[model._test_indices + 5].values / p.iloc[model._test_indices].values)
    np.testing.assert_allclose(model._y_test, expected)


def test_outer_holdout_is_not_used_for_training_or_early_stopping():
    p = prices()
    params = {"n_estimators": 10, "max_depth": 2, "n_jobs": 1, "random_state": 42}
    a = XGBoostForecaster(model_params=params)
    a.fit(p)
    changed = p.copy()
    boundary = a._test_indices[0]
    changed.iloc[boundary:] *= np.exp(np.linspace(0.1, 1, len(p) - boundary))
    b = XGBoostForecaster(model_params=params)
    b.fit(changed)
    np.testing.assert_array_equal(a._train_indices, b._train_indices)
    np.testing.assert_array_equal(a._validation_indices, b._validation_indices)
    assert a._model.get_booster().save_raw() == b._model.get_booster().save_raw()


@pytest.mark.parametrize("horizon", [0, -1, 1.5])
def test_invalid_forecast_horizon_rejected(horizon):
    with pytest.raises(ValueError, match="target_horizon"):
        XGBoostForecaster().fit(prices(), target_horizon=horizon)
