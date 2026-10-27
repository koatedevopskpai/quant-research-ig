"""
model_adapters.py — wrappers that put the real research models on the contract.

Each adapter is responsible for exactly three things: converting prices into the
model's expected input, translating the model's result into `ModelOutput`, and
failing loudly when the model returns something unusable. The trading loop never
sees a model's native result object.

WHY THE ADAPTERS IMPORT LAZILY
    `agentic-finance` must remain importable in an environment where the ML
    packages are absent (the provider/replay path does not need torch). The
    training libraries are imported when an adapter is constructed, not when the
    module is imported.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agentic_finance.model_contract import ModelOutput


def _log_returns(prices: pd.Series) -> pd.Series:
    if not isinstance(prices, pd.Series):
        raise TypeError(f"prices must be a pandas Series, got {type(prices).__name__}")
    if len(prices) < 2:
        raise ValueError("at least two prices are required")
    if not np.isfinite(prices.to_numpy(dtype=float)).all() or (prices <= 0).any():
        raise ValueError("prices must be finite and positive")
    return np.log(prices / prices.shift(1)).dropna()


class RegimeAdapter:
    """
    HMM volatility regimes. Context only: direction is always 0.

    The numeric state and its probability are exposed because the signal policy
    uses the high-volatility state to scale confidence. The labels are carried
    for reporting, never parsed to infer a direction.
    """

    name = "regime"

    def __init__(self, n_regimes: int = 3, detector: object | None = None,
                 min_observations: int = 100) -> None:
        self._n_regimes = n_regimes
        self._detector = detector
        self._min_observations = min_observations

    def _model(self):
        if self._detector is None:
            from ml_for_finance.regime.adaptive_hmm import AdaptiveHMMRegimeDetector

            self._detector = AdaptiveHMMRegimeDetector(n_regimes=self._n_regimes)
        return self._detector

    def _features(self, prices: pd.Series) -> np.ndarray:
        returns = _log_returns(prices)
        if len(returns) < self._min_observations:
            raise ValueError(
                f"need >= {self._min_observations} returns for regime detection, "
                f"got {len(returns)}"
            )
        return returns.to_numpy(dtype=float).reshape(-1, 1)

    def fit(self, prices: pd.Series) -> "RegimeAdapter":
        self._model().fit(self._features(prices))
        return self

    def predict(self, prices: pd.Series) -> ModelOutput:
        features = self._features(prices)
        result = self._model().predict(features)
        state = int(result.regimes[-1])
        probability = float(result.probabilities[-1, state])
        labels = list(result.regime_labels)
        return ModelOutput(
            model=self.name,
            direction=0.0,          # a regime is not a direction
            confidence=0.0,         # and cannot create an order on its own
            regime=state,
            regime_label=labels[state] if state < len(labels) else f"Regime {state}",
            diagnostics={
                "regime_probability": probability,
                "n_regimes": int(result.n_regimes),
                "observations": len(features),
            },
        )

    # ------------------------------------------------------------------
    # Reward-modulated adaptation
    # ------------------------------------------------------------------

    def learn(self, prev_state: int, next_state: int, reward: float) -> bool:
        """
        Update the HMM transition matrix from a realised trade outcome.

        The detector must have been fitted (it is, after every research call).
        Returns False when the wrapped detector has no adaptation support.
        """
        model = self._model()
        if not hasattr(model, "adapt"):
            return False
        model.adapt(int(prev_state), int(next_state), float(reward))
        return True

    @property
    def transition_matrix(self):
        model = self._model()
        return getattr(model, "transition_matrix", None)


class ForecastAdapter:
    """
    XGBoost return forecast. This is the only default source of direction.

    Confidence is the expected move measured in recent volatility units:
        confidence = min(1, |expected_return| / (sigmas * recent_vol))
    with `sigmas = 2` meaning "a two-sigma expected move is full conviction".
    That is a policy choice, stated here because a confidence number without a
    unit is meaningless.
    """

    name = "forecast"

    def __init__(self, forecaster: object | None = None,
                 min_observations: int = 150, vol_window: int = 20,
                 sigmas: float = 2.0) -> None:
        self._forecaster = forecaster
        self._min_observations = min_observations
        self._vol_window = vol_window
        self._sigmas = sigmas

    def _model(self):
        if self._forecaster is None:
            from ml_for_finance.forecasting.xgboost_forecaster import XGBoostForecaster

            self._forecaster = XGBoostForecaster()
        return self._forecaster

    def fit(self, prices: pd.Series) -> "ForecastAdapter":
        _log_returns(prices)  # validates type, positivity and finiteness
        if len(prices) < self._min_observations:
            raise ValueError(
                f"need >= {self._min_observations} prices to fit the forecaster, "
                f"got {len(prices)}"
            )
        self._model().fit(prices)
        return self

    def predict(self, prices: pd.Series) -> ModelOutput:
        result = self._model().predict(prices)
        forecasts = np.asarray(getattr(result, "forecasts", []), dtype=float)
        finite = forecasts[np.isfinite(forecasts)]
        if finite.size == 0:
            raise ValueError(
                "forecaster produced no finite predictions; refusing to emit a "
                "zero signal (this is the 'forecasts' vs 'forecast' failure mode)"
            )

        expected_return = float(finite[-1])
        returns = _log_returns(prices)
        recent_vol = float(returns.tail(self._vol_window).std(ddof=1))

        if recent_vol > 0:
            confidence = min(1.0, abs(expected_return) / (self._sigmas * recent_vol))
        else:
            confidence = 0.0

        return ModelOutput(
            model=self.name,
            direction=float(np.sign(expected_return)),
            confidence=confidence,
            expected_return=expected_return,
            diagnostics={
                "recent_vol": recent_vol,
                "sigmas": self._sigmas,
                "n_forecasts": int(finite.size),
                "r2": float(getattr(result, "r2", float("nan"))),
                "rmse": float(getattr(result, "rmse", float("nan"))),
            },
        )
