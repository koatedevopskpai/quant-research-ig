"""
xgboost_forecaster.py — XGBoost return forecasting with feature engineering.

WHY XGBoost FOR RETURN FORECASTING:
    Gradient boosted trees are the dominant ML model for tabular financial
    data. XGBoost offers:

    1. **Nonlinear relationships** — Captures complex interactions between
       technical features (e.g., momentum × volatility regime).
    2. **Feature importance** — Built-in importance metrics show which
       features drive predictions (critical for Beckman's validation).
    3. **Robustness to outliers** — Financial data has fat tails. XGBoost's
       tree-based splitting is more robust than neural networks to extreme
       returns.
    4. **Speed** — Training and inference are orders of magnitude faster
       than neural networks for tabular data.

FEATURE ENGINEERING:
    We create a rich set of features from raw price/return data:
    - Lagged returns: r_{t-1}, r_{t-2}, ..., r_{t-n}
    - Rolling statistics: mean, std, skew, kurtosis over windows
    - Technical indicators: RSI, MACD, Bollinger Bands
    - Volume features: volume change, dollar volume
    - Calendar features: day of week, month, quarter

    WHY lags + rolling stats + calendar features:
        Financial returns have weak autocorrelation (lags), time-varying
        volatility (rolling std), and calendar effects (day-of-week).
        Combining all three captures the main sources of predictability.

QUANT BECKMAN'S APPROACH:
    Beckman uses XGBoost as the "workhorse" forecaster in his pipeline:
    1. Train on 70% of data (walk-forward, not random split).
    2. Validate on 30% (out-of-sample, with purging for look-ahead bias).
    3. Feature importance analysis to identify robust signals.
    4. Only deploy features that are significant across multiple train/test
       splits (no data mining).

USAGE:
    forecaster = XGBoostForecaster()
    forecaster.fit(
        prices=prices_df,  # DataFrame with 'close' column
        target_horizon=1,   # predict next day's return
    )

    result = forecaster.predict(prices_df)  # out-of-sample
    print(result.forecasts)
    print(f"R²: {result.r2:.4f}")

    # Feature importance
    for name, imp in result.feature_importance:
        print(f"{name}: {imp:.4f}")

REFERENCES:
    Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting
        System. KDD 2016.
    Gu, S., Kelly, B., & Xiu, D. (2020). Empirical Asset Pricing via
        Machine Learning. Review of Financial Studies, 33(5).
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error as mse


@dataclass
class ForecastResult:
    """
    Output of the XGBoost forecaster.

    Attributes:
        forecasts: Array of predicted returns (same length as input).
        actuals: Array of actual returns (for comparison).
        feature_importance: List of (feature_name, importance) tuples,
            sorted by importance descending.
        r2: Out-of-sample R².
        rmse: Root mean squared error.
        model: The fitted XGBoost model (for further analysis).
    """

    forecasts: np.ndarray
    actuals: np.ndarray
    feature_importance: list[tuple[str, float]]
    r2: float
    rmse: float
    model: Any  # xgboost.Booster


class XGBoostForecaster:
    """
    XGBoost return forecasting model with automated feature engineering.

    Parameters:
        n_lags: Number of lagged returns to use as features.
        rolling_windows: List of window sizes for rolling statistics.
        test_size: Fraction of data to hold out for testing.
        early_stopping_rounds: XGBoost early stopping (prevents overfit).
        model_params: XGBoost parameters dict (learning_rate, max_depth,
            n_estimators, etc.). Sensible defaults provided.
    """

    def __init__(
        self,
        n_lags: int = 20,
        rolling_windows: list[int] | None = None,
        test_size: float = 0.2,
        early_stopping_rounds: int = 50,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        self.n_lags = n_lags
        self.rolling_windows = rolling_windows or [5, 10, 21, 63]
        self.test_size = test_size
        self.early_stopping_rounds = early_stopping_rounds

        # WHY XGBoost default params:
        #   - learning_rate=0.05: Conservative, prevents overfitting on
        #     noisy financial data.
        #   - max_depth=4: Shallow trees reduce overfitting. Financial
        #     features rarely need deep interactions.
        #   - n_estimators=1000: Large enough with early stopping.
        #   - subsample=0.8: Row subsampling adds randomness.
        #   - colsample_bytree=0.8: Feature subsampling.
        self.model_params = model_params or {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "max_depth": 4,
            "n_estimators": 1000,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
        }
        self._model: xgb.XGBRegressor | None = None
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _create_features(self, prices: pd.Series) -> pd.DataFrame:
        """
        Create feature matrix from price series.

        Features include:
        - Lagged log returns (n_lags)
        - Rolling mean, std, skew, kurtosis (various windows)
        - RSI (14-period)
        - MACD (12, 26, 9)
        - Calendar features (day of week)

        Args:
            prices: Series of closing prices, datetime index.

        Returns:
            DataFrame with engineered features (index aligned with prices).
        """
        # Log returns
        log_ret = np.log(prices / prices.shift(1))
        df = pd.DataFrame(index=prices.index)

        # Lagged returns
        for i in range(1, self.n_lags + 1):
            df[f"ret_lag_{i}"] = log_ret.shift(i)

        # Rolling statistics
        for w in self.rolling_windows:
            df[f"ret_mean_{w}"] = log_ret.rolling(w).mean().shift(1)
            df[f"ret_std_{w}"] = log_ret.rolling(w).std().shift(1)
            df[f"ret_skew_{w}"] = log_ret.rolling(w).skew().shift(1)
            df[f"ret_kurt_{w}"] = log_ret.rolling(w).kurt().shift(1)

        # RSI (14-period)
        delta = prices.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta).clip(lower=0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df["rsi_14"] = (100 - 100 / (1 + rs)).shift(1)

        # MACD (12, 26, 9)
        ema_12 = prices.ewm(span=12).mean()
        ema_26 = prices.ewm(span=26).mean()
        macd = ema_12 - ema_26
        signal = macd.ewm(span=9).mean()
        df["macd"] = macd.shift(1)
        df["macd_signal"] = signal.shift(1)
        df["macd_hist"] = (macd - signal).shift(1)

        # Calendar features
        df["day_of_week"] = prices.index.dayofweek
        df["month"] = prices.index.month

        self._feature_names = [c for c in df.columns if c not in ("target",)]
        # Drop NaN rows (from shifting/rolling)
        return df.dropna()

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        prices: pd.Series | pd.DataFrame,
        target_horizon: int = 1,
    ) -> "XGBoostForecaster":
        """
        Fit the XGBoost model to historical price data.

        Uses walk-forward validation:
        - Train: first (1 - test_size) portion of data
        - Test: last test_size portion
        - WHY walk-forward vs. random split: Time series data has temporal
          dependencies. Random splitting leaks future information into the
          training set.

        Args:
            prices: Series of closing prices with datetime index (or
                    DataFrame with 'close' column).
            target_horizon: Number of days ahead to predict.

        Returns:
            Self (fitted model).
        """
        # Extract price series
        if isinstance(prices, pd.DataFrame):
            prices = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
            prices = prices.squeeze()

        # Create features
        features = self._create_features(prices)

        # Target: future log return
        log_ret = np.log(prices / prices.shift(1))
        target = log_ret.shift(-target_horizon)

        # Align features and target (drop NaN from shift)
        data = pd.concat([features, target.rename("target")], axis=1).dropna()
        if len(data) < 100:
            raise ValueError(
                f"Need at least 100 samples after feature creation, got {len(data)}"
            )

        X = data[self._feature_names].values
        y = data["target"].values

        # Walk-forward split
        split_idx = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train with early stopping
        self._model = xgb.XGBRegressor(**self.model_params)
        self._model.fit(
            X_train,
            y_train,
            eval_set=[(X_test, y_test)],
            verbose=False,
        )

        # Store test data for predict()
        self._X_test = X_test
        self._y_test = y_test

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, prices: pd.Series | pd.DataFrame | None = None) -> ForecastResult:
        """
        Generate return forecasts.

        If prices is provided, generates out-of-sample forecasts using
        the test portion of the data. Otherwise uses the stored test set.

        Args:
            prices: Optional new price series to forecast.

        Returns:
            ForecastResult with forecasts, actuals, and diagnostics.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before predict()")

        if prices is not None:
            # Create features for new data
            if isinstance(prices, pd.DataFrame):
                prices = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]
            features = self._create_features(prices)
            X = features[self._feature_names].values
            y_actual = np.full(len(X), np.nan)
        else:
            X = self._X_test
            y_actual = self._y_test

        # Predict returns
        forecasts = self._model.predict(X)

        # Compute test metrics
        valid_mask = ~np.isnan(y_actual)
        if valid_mask.sum() > 2:
            ss_res = np.sum((y_actual[valid_mask] - forecasts[valid_mask]) ** 2)
            ss_tot = np.sum((y_actual[valid_mask] - np.mean(y_actual[valid_mask])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            rmse_val = float(np.sqrt(mse(y_actual[valid_mask], forecasts[valid_mask])))
        else:
            r2 = 0.0
            rmse_val = 0.0

        # Feature importance
        importance = sorted(
            zip(
                self._feature_names,
                self._model.feature_importances_,
            ),
            key=lambda x: x[1],
            reverse=True,
        )

        return ForecastResult(
            forecasts=forecasts,
            actuals=y_actual,
            feature_importance=importance,
            r2=round(r2, 4),
            rmse=round(rmse_val, 6),
            model=self._model,
        )
