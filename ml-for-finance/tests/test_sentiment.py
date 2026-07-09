"""
test_sentiment.py — Unit tests for the sentiment-augmented model.
"""

import numpy as np
import pandas as pd
import pytest

from ml_for_finance.sentiment.sentiment_model import SentimentAugmentedModel


class TestSentimentAugmentedModel:
    """Test sentiment-augmented forecasting with synthetic data."""

    @pytest.fixture
    def data(self) -> tuple[pd.Series, pd.Series]:
        """Generate synthetic price and sentiment data."""
        np.random.seed(42)
        n = 300
        returns = np.random.normal(0.0005, 0.015, n)
        prices = 100 * np.exp(np.cumsum(returns))
        idx = pd.date_range("2024-01-01", periods=n, freq="D")

        # Sentiment that has some predictive power
        sentiment = pd.Series(
            0.5 + 0.3 * np.sin(np.linspace(0, 4 * np.pi, n)) + 0.1 * np.random.randn(n),
            index=idx,
        )
        price_series = pd.Series(prices, index=idx, name="close")
        return price_series, sentiment

    def test_fit_and_predict(self, data: tuple[pd.Series, pd.Series]) -> None:
        prices, sentiment = data
        model = SentimentAugmentedModel(
            n_lags=5,
            rolling_windows=[5, 10],
            test_size=0.2,
        )
        model.fit(prices, sentiment)
        result = model.predict()

        assert len(result.forecast) > 0
        assert len(result.price_forecast) > 0
        assert len(result.sentiment_forecast) > 0
        assert 0.0 <= result.sentiment_weight <= 1.0

    def test_predict_before_fit_raises(self) -> None:
        model = SentimentAugmentedModel()
        with pytest.raises(RuntimeError):
            model.predict()

    def test_too_few_aligned_samples(self) -> None:
        idx = pd.date_range("2024-01-01", periods=20, freq="D")
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(20) * 0.01)), index=idx)
        sentiment = pd.Series(0.5 + 0.1 * np.random.randn(20), index=idx)

        model = SentimentAugmentedModel()
        with pytest.raises(ValueError):
            model.fit(prices, sentiment)
