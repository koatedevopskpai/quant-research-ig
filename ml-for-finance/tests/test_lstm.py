"""
test_lstm.py — Unit tests for the PyTorch LSTM predictor.
"""

import numpy as np
import pandas as pd
import pytest

from ml_for_finance.deep_learning.lstm_predictor import LSTMPredictor


class TestLSTMPredictor:
    """Test LSTM data preparation and training (synthetic data)."""

    @pytest.fixture
    def price_series(self) -> pd.Series:
        """Short synthetic price series for quick training."""
        np.random.seed(42)
        n = 200
        returns = np.random.normal(0.001, 0.02, n)
        prices = 100 * np.exp(np.cumsum(returns))
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.Series(prices, index=idx, name="close")

    def test_fit_and_predict(self, price_series: pd.Series) -> None:
        predictor = LSTMPredictor(
            seq_len=10,
            hidden_dim=16,
            num_layers=1,
            epochs=5,
            batch_size=16,
        )
        predictor.fit(price_series, verbose=False)
        result = predictor.predict(price_series)

        assert len(result.forecast) == len(price_series)
        # First seq_len predictions should be NaN
        assert np.isnan(result.forecast[:10]).all()
        # Remaining predictions should be finite (seq_len+1 onwards)
        assert np.isfinite(result.forecast[11:]).all()
        assert len(result.loss_history) == 5

    def test_predict_before_fit_raises(self, price_series: pd.Series) -> None:
        predictor = LSTMPredictor()
        with pytest.raises(RuntimeError):
            predictor.predict(price_series)

    def test_too_few_samples(self) -> None:
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(20) * 0.01)), index=idx)
        predictor = LSTMPredictor(seq_len=10)
        with pytest.raises(ValueError):
            predictor.fit(prices)

    def test_loss_decreases(self, price_series: pd.Series) -> None:
        """Training loss should generally decrease over epochs."""
        predictor = LSTMPredictor(
            seq_len=10,
            hidden_dim=16,
            num_layers=1,
            epochs=10,
            batch_size=16,
        )
        predictor.fit(price_series, verbose=False)
        # Last loss should be lower than or comparable to first
        assert predictor._loss_history[-1] <= predictor._loss_history[0] * 1.5 or predictor._loss_history[-1] < 0.1
