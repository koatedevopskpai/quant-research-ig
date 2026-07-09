"""
test_forecasting.py — Unit tests for XGBoost return forecasting.
"""

import numpy as np
import pandas as pd
import pytest

from ml_for_finance.forecasting.xgboost_forecaster import XGBoostForecaster


class TestXGBoostForecaster:
    """Test XGBoost feature engineering and forecasting."""

    @pytest.fixture
    def price_series(self) -> pd.Series:
        """Generate synthetic price data with some predictability."""
        np.random.seed(42)
        n = 500
        # Random walk with slight mean reversion
        returns = np.random.normal(0.0005, 0.015, n)
        prices = 100 * np.exp(np.cumsum(returns))
        idx = pd.date_range("2024-01-01", periods=n, freq="D")
        return pd.Series(prices, index=idx, name="close")

    def test_fit_and_predict(self, price_series: pd.Series) -> None:
        forecaster = XGBoostForecaster(n_lags=5, test_size=0.2)
        forecaster.fit(price_series)
        result = forecaster.predict()

        assert len(result.forecasts) > 0
        assert result.r2 is not None
        assert len(result.feature_importance) > 0

    def test_feature_engineering_shapes(self, price_series: pd.Series) -> None:
        forecaster = XGBoostForecaster(n_lags=5, rolling_windows=[5, 10])
        forecaster.fit(price_series)

        # Check that feature names include expected patterns
        feature_names = [f[0] for f in forecaster.predict().feature_importance]
        assert any("ret_lag" in name for name in feature_names)
        assert any("ret_mean" in name for name in feature_names)
        assert any("rsi" in name for name in feature_names)

    def test_predict_before_fit_raises(self) -> None:
        forecaster = XGBoostForecaster()
        with pytest.raises(RuntimeError):
            forecaster.predict()

    def test_feature_importance_sorted(self, price_series: pd.Series) -> None:
        forecaster = XGBoostForecaster(n_lags=5, test_size=0.2)
        forecaster.fit(price_series)
        result = forecaster.predict()

        # Check descending order
        importances = [imp for _, imp in result.feature_importance]
        assert all(
            importances[i] >= importances[i + 1]
            for i in range(len(importances) - 1)
        )

    def test_too_few_samples_raises(self) -> None:
        """Very short price series should raise."""
        idx = pd.date_range("2024-01-01", periods=30, freq="D")
        prices = pd.Series(100 * np.exp(np.cumsum(np.random.randn(30) * 0.01)), index=idx)
        forecaster = XGBoostForecaster(n_lags=20, test_size=0.2)
        # After feature creation (20 lags + rolling windows), we may have
        # < 100 samples.
        with pytest.raises(ValueError):
            forecaster.fit(prices)
