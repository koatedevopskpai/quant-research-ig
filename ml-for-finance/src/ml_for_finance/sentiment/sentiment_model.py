"""
sentiment_model.py — Sentiment-augmented return forecasting model.

WHY SENTIMENT AUGMENTATION:
    Price-based features capture only realised information. Sentiment
    data (news, social media, earnings call transcripts) captures
    EXPECTATIONS that are not yet priced in. Combining the two:

    1. **Price features** → "what has happened" (momentum, volatility)
    2. **Sentiment features** → "what might happen" (news tone, fear/greed)

    When price features say "up" and sentiment says "up", the signal is
    strong. When they diverge, the model learns which to trust more.

THE MODEL ARCHITECTURE:
    This model takes pre-computed sentiment scores and combines them with
    price-based features using a simple weighted ensemble:

    - Price sub-model: XGBoost regressor on technical features.
    - Sentiment sub-model: XGBoost regressor on sentiment features.
    - Ensemble: Weighted average of the two predictions.

    WHY ensemble vs. concatenation:
        Concatenating all features (price + sentiment) into a single model
        works but makes feature importance analysis harder — you can't tell
        whether the model is "listening to" price or sentiment signals.
        Separate sub-models with a learned ensemble weight makes the
        contribution of each source explicit.

SENTIMENT DATA SOURCE:
    This model expects pre-computed sentiment scores. For IG market data,
    we assume sentiment is provided as a "sentiment ratio" or "bullish
    percentage" (0-100 scale) — similar to IG's own sentiment data:

        IG Sentiment: {instrument} → {percentage_long, percentage_short}

    For a production system, sentiment would come from:
    - IG's sentiment endpoint (retail trader positioning).
    - News NLP pipeline (VADER, BERT financial models).
    - Options market sentiment (put/call ratio, skew).

USAGE:
    # Price data
    prices = pd.read_csv("prices.csv", index_col=0, parse_dates=True)

    # Sentiment data (e.g., IG sentiment percentage, or NLP scores)
    sentiment = pd.Series(
        [0.6, 0.55, 0.45, 0.52, ...],  # 0-1 scale
        index=prices.index,
    )

    model = SentimentAugmentedModel()
    model.fit(prices=prices, sentiment=sentiment)

    result = model.predict(prices=prices, sentiment=sentiment)
    print(f"R²: {result.r2:.4f}")
    print(f"Sentiment weight: {result.sentiment_weight:.2f}")

REFERENCES:
    Tetlock, P. (2007). Giving Content to Investor Sentiment: The Role
        of Media in the Stock Market. Journal of Finance, 62(3).
    Da, Z., Engelberg, J., & Gao, P. (2015). The Sum of All FEARS:
        Investor Sentiment and Asset Prices. Review of Financial
        Studies, 28(1).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ml_for_finance.forecasting.xgboost_forecaster import (
    XGBoostForecaster,
    ForecastResult,
)


@dataclass
class SentimentResult:
    """
    Output of the sentiment-augmented model.

    Attributes:
        forecast: Combined forecast (price + sentiment ensemble).
        price_forecast: Forecast from price-only model.
        sentiment_forecast: Forecast from sentiment-only model.
        sentiment_weight: Learned weight for sentiment component.
            weight=0: Model ignores sentiment entirely.
            weight=1: Model relies entirely on sentiment.
        r2: Out-of-sample R² for the combined forecast.
        rmse: Root mean squared error for the combined forecast.
    """

    forecast: np.ndarray
    price_forecast: np.ndarray
    sentiment_forecast: np.ndarray
    sentiment_weight: float
    r2: float
    rmse: float


class SentimentAugmentedModel:
    """
    Sentiment-augmented return forecasting model.

    Combines price-based technical features with external sentiment
    scores using a weighted ensemble of two XGBoost sub-models.

    Parameters:
        n_lags: Number of lagged returns for price features.
        rolling_windows: Window sizes for rolling statistics.
        test_size: Fraction of data for validation.
        model_params: XGBoost parameters (shared by both sub-models).
    """

    def __init__(
        self,
        n_lags: int = 10,
        rolling_windows: list[int] | None = None,
        test_size: float = 0.2,
        model_params: dict[str, Any] | None = None,
    ) -> None:
        self.n_lags = n_lags
        self.rolling_windows = rolling_windows or [5, 10, 21]
        self.test_size = test_size
        self.model_params = model_params or {
            "objective": "reg:squarederror",
            "learning_rate": 0.05,
            "max_depth": 3,
            "n_estimators": 500,
            "subsample": 0.8,
            "random_state": 42,
        }

        self._price_model: XGBoostForecaster | None = None
        self._sentiment_model: XGBoostForecaster | None = None
        self._sentiment_weight: float = 0.5
        self._feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Sentiment feature engineering
    # ------------------------------------------------------------------

    @staticmethod
    def _create_sentiment_features(
        sentiment: pd.Series,
    ) -> pd.DataFrame:
        """
        Create features from a sentiment time series.

        Features:
        - Raw sentiment value (level).
        - Lagged sentiment (1, 2, 3, 5 days).
        - Rolling mean (5, 10, 21 days) — smooths noise.
        - Rolling std (5, 10 days) — sentiment volatility.
        - Sentiment change (1-day, 5-day).

        WHY these features:
            - Level: current sentiment regime.
            - Lags: sentiment momentum.
            - Rolling stats: trend and noise in sentiment.
            - Changes: sentiment shocks (rapid shifts in market mood).
        """
        df = pd.DataFrame(index=sentiment.index)
        name = sentiment.name or "sentiment"

        df["sentiment_raw"] = sentiment.shift(1)

        for lag in [1, 2, 3, 5]:
            df[f"sentiment_lag_{lag}"] = sentiment.shift(lag)

        for w in [5, 10, 21]:
            df[f"sentiment_mean_{w}"] = sentiment.rolling(w).mean().shift(1)
            df[f"sentiment_std_{w}"] = sentiment.rolling(w).std().shift(1)

        df["sentiment_chg_1d"] = sentiment.diff(1).shift(1)
        df["sentiment_chg_5d"] = sentiment.diff(5).shift(1)

        return df.dropna()

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        prices: pd.Series | pd.DataFrame,
        sentiment: pd.Series,
    ) -> "SentimentAugmentedModel":
        """
        Fit both sub-models and learn the optimal ensemble weight.

        The ensemble weight is determined by comparing the out-of-sample
        performance of each sub-model:
            weight = (R²_price) / (R²_price + R²_sentiment)

        This ensures the more accurate sub-model gets higher weight.

        Args:
            prices: Series of closing prices.
            sentiment: Series of sentiment scores (same index as prices),
                       values in [0, 1] range.

        Returns:
            Self (fitted model).
        """
        if isinstance(prices, pd.DataFrame):
            prices = prices["close"] if "close" in prices.columns else prices.iloc[:, 0]

        # Align sentiment to price index
        sentiment = sentiment.reindex(prices.index)

        # Price sub-model
        self._price_model = XGBoostForecaster(
            n_lags=self.n_lags,
            rolling_windows=self.rolling_windows,
            test_size=self.test_size,
            model_params=self.model_params,
        )
        self._price_model.fit(prices)

        # For sentiment, we need to train a model that uses sentiment features
        # to predict returns. We directly create an XGBoost model for this.
        log_ret = np.log(prices / prices.shift(1))
        sent_features = self._create_sentiment_features(sentiment)
        self._feature_names = list(sent_features.columns)

        # Align returns with sentiment features
        target = log_ret.reindex(sent_features.index)

        import xgboost as xgb
        from sklearn.metrics import mean_squared_error as mse

        data = pd.concat([sent_features, target.rename("target")], axis=1).dropna()
        if len(data) < 50:
            raise ValueError(
                f"Need at least 50 aligned samples, got {len(data)}"
            )

        X = data[self._feature_names].values
        y = data["target"].values

        split = int(len(X) * (1 - self.test_size))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self._sentiment_xgb = xgb.XGBRegressor(**self.model_params)
        self._sentiment_xgb.fit(X_train, y_train, verbose=False)

        # Compute ensemble weight based on out-of-sample R²
        price_result = self._price_model.predict()
        sent_pred = self._sentiment_xgb.predict(X_test)
        y_test_aligned = y_test

        price_r2 = price_result.r2 if price_result.r2 > 0 else 0.0

        ss_res = np.sum((y_test_aligned - sent_pred) ** 2)
        ss_tot = np.sum((y_test_aligned - np.mean(y_test_aligned)) ** 2)
        sent_r2 = max(0.0, 1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        total = price_r2 + sent_r2
        # Weight reflects relative contribution. Default 0.5 when both are
        # equally good/bad.
        self._sentiment_weight = sent_r2 / total if total > 0 else 0.5

        # Store test data
        self._X_test = X_test
        self._y_test = y_test_aligned
        self._log_ret = log_ret
        self._sentiment = sentiment

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        prices: pd.Series | pd.DataFrame | None = None,
        sentiment: pd.Series | None = None,
    ) -> SentimentResult:
        """
        Generate combined forecasts.

        Args:
            prices: Optional new price data (uses stored if None).
            sentiment: Optional new sentiment data (uses stored if None).

        Returns:
            SentimentResult with combined and component forecasts.
        """
        if self._price_model is None:
            raise RuntimeError("Model must be fitted before predict()")

        # Price forecast
        if prices is not None:
            price_result = self._price_model.predict(prices)
        else:
            price_result = self._price_model.predict()

        # Sentiment forecast
        if sentiment is None:
            sentiment = self._sentiment
            X_test = self._X_test
            y_test = self._y_test
        else:
            sent_features = self._create_sentiment_features(sentiment)
            X_test = sent_features[self._feature_names].values
            y_test = np.full(len(X_test), np.nan)

        sent_forecast = self._sentiment_xgb.predict(X_test)

        # Combined forecast: weighted average
        alpha = self._sentiment_weight
        combined = (1 - alpha) * price_result.forecasts[:len(sent_forecast)] + alpha * sent_forecast

        # Metrics
        valid_mask = ~np.isnan(y_test)
        if valid_mask.sum() > 2:
            ss_res = np.sum((y_test[valid_mask] - combined[valid_mask]) ** 2)
            ss_tot = np.sum((y_test[valid_mask] - np.mean(y_test[valid_mask])) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            from sklearn.metrics import mean_squared_error as mse
            rmse_val = float(np.sqrt(mse(y_test[valid_mask], combined[valid_mask])))
        else:
            r2 = 0.0
            rmse_val = 0.0

        return SentimentResult(
            forecast=combined,
            price_forecast=price_result.forecasts[:len(sent_forecast)],
            sentiment_forecast=sent_forecast,
            sentiment_weight=round(alpha, 4),
            r2=round(r2, 4),
            rmse=round(rmse_val, 6),
        )
