"""
Adapter tests: the boundary where the review's "silent zero" bugs lived.

Each adapter is tested twice: against a fake model whose result shape we control
(so the regression is pinned exactly), and — for the real models — end to end
through the research agent, which is phase 4's "done when".
"""

import numpy as np
import pandas as pd
import pytest

from agentic_finance.audit_trail import AuditTrail
from agentic_finance.message_bus import MessageBus
from agentic_finance.model_adapters import ForecastAdapter, RegimeAdapter
from agentic_finance.research_agent import ResearchAgent


def _prices(n=400, seed=7, drift=0.0004, vol=0.01):
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift, vol, n)
    index = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index, name="close")


class _FakeForecastResult:
    def __init__(self, forecasts, r2=0.01, rmse=0.01):
        self.forecasts = np.asarray(forecasts, dtype=float)   # plural, as in the real API
        self.actuals = np.zeros_like(self.forecasts)
        self.r2 = r2
        self.rmse = rmse
        self.feature_importance = []


class _FakeForecaster:
    def __init__(self, forecasts):
        self._forecasts = forecasts
        self.fitted = False

    def fit(self, prices):
        self.fitted = True
        return self

    def predict(self, prices):
        return _FakeForecastResult(self._forecasts)


class TestForecastAdapter:
    def test_reads_plural_forecasts(self):
        """The regression: reading `forecast` produced 0.0 after a good fit."""
        adapter = ForecastAdapter(forecaster=_FakeForecaster([0.001, 0.02]))
        out = adapter.predict(_prices(50))
        assert out.expected_return == pytest.approx(0.02)
        assert out.direction == 1.0
        assert out.confidence > 0.0

    def test_confidence_is_the_move_in_vol_units(self):
        prices = _prices(50)
        adapter = ForecastAdapter(
            forecaster=_FakeForecaster([0.0, 0.01]), sigmas=2.0, vol_window=20
        )
        out = adapter.predict(prices)
        recent_vol = np.log(prices / prices.shift(1)).dropna().tail(20).std(ddof=1)
        assert out.confidence == pytest.approx(min(1.0, 0.01 / (2.0 * recent_vol)))

    def test_two_sigma_move_is_full_confidence(self):
        prices = _prices(50)
        recent_vol = np.log(prices / prices.shift(1)).dropna().tail(20).std(ddof=1)
        adapter = ForecastAdapter(
            forecaster=_FakeForecaster([2.1 * recent_vol]), sigmas=2.0
        )
        assert adapter.predict(prices).confidence == pytest.approx(1.0)

    def test_missing_forecasts_raises_instead_of_returning_zero(self):
        class WrongShapeForecaster:
            def fit(self, prices):
                return self

            def predict(self, prices):
                class Result:
                    forecast = np.array([0.01])  # singular: the old mistake
                    r2 = 0.0
                    rmse = 0.0
                return Result()

        adapter = ForecastAdapter(forecaster=WrongShapeForecaster())
        with pytest.raises(ValueError, match="forecasts"):
            adapter.predict(_prices(50))

    def test_rejects_non_series_input(self):
        adapter = ForecastAdapter(forecaster=_FakeForecaster([0.01]))
        with pytest.raises(TypeError, match="Series"):
            adapter.predict(np.ones(50))


class _FakeRegimeResult:
    def __init__(self, regimes, probabilities, labels, n_regimes=3):
        self.regimes = np.asarray(regimes)
        self.probabilities = np.asarray(probabilities)
        self.regime_labels = labels
        self.n_regimes = n_regimes
        self.transition_matrix = np.eye(n_regimes)


class _FakeDetector:
    def __init__(self, result):
        self._result = result
        self.fitted = False

    def fit(self, X):
        self.fitted = True
        return self

    def predict(self, X):
        return self._result


class TestRegimeAdapter:
    def test_reports_numeric_state_and_label(self):
        detector = _FakeDetector(_FakeRegimeResult(
            [0, 1, 2],
            [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.05, 0.15, 0.8]],
            ["Low Vol", "Medium Vol", "High Vol"],
        ))
        out = RegimeAdapter(detector=detector, min_observations=10).predict(_prices(50))
        assert out.regime == 2
        assert out.regime_label == "High Vol"
        assert out.diagnostics["regime_probability"] == pytest.approx(0.8)
        assert out.direction == 0.0 and out.confidence == 0.0

    def test_regime_is_not_a_direction(self):
        detector = _FakeDetector(_FakeRegimeResult(
            [2], [[0.0, 0.0, 1.0]], ["Low Vol", "Medium Vol", "High Vol"]
        ))
        out = RegimeAdapter(detector=detector, min_observations=10).predict(_prices(50))
        assert out.is_directional is False
        with pytest.raises(ValueError, match="expected_return"):
            type(out)(
                model="regime", direction=1.0, confidence=0.5,
                regime=2, regime_label="High Vol",
            )


class TestRealModelsThroughTheAgent:
    """Phase 4 exit criterion: real models produce a valid signal."""

    def test_xgboost_and_hmm_produce_a_signal(self):
        prices = _prices(400, seed=11, drift=0.0008)
        agent = ResearchAgent(MessageBus(), AuditTrail())
        agent.register_model("forecast", ForecastAdapter())
        agent.register_model("regime", RegimeAdapter())

        out = agent.research_signal("IX.D.SPTRD.IFD.IP", prices).model_output

        assert "error" not in out, out.get("error")
        assert out["direction"] in (-1.0, 0.0, 1.0)
        assert 0.0 <= out["confidence"] <= 1.0
        assert out["regime"] is not None
        assert out["regime_label"] in ("Low Vol", "Medium Vol", "High Vol")
        assert out["expected_return"] is not None
        assert "forecast" in out["models"] and "regime" in out["models"]
        # Both real models ran; neither silently returned zero.
        assert out["models"]["forecast"]["diagnostics"]["n_forecasts"] > 0
        assert out["models"]["regime"]["diagnostics"]["observations"] > 0
