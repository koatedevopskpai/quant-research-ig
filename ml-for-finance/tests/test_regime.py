"""
test_regime.py — Unit tests for HMM regime detection.
"""

import numpy as np
import pytest

from ml_for_finance.regime.hmm_detector import HMMRegimeDetector


class TestHMMRegimeDetector:
    """Test HMM regime detection with synthetic data."""

    def test_fit_and_predict(self) -> None:
        """Should fit and return regime predictions."""
        # Generate synthetic returns with 2 regimes
        np.random.seed(42)
        n = 300
        returns = np.concatenate([
            np.random.normal(0.001, 0.01, n // 2),  # low vol
            np.random.normal(-0.002, 0.03, n // 2),  # high vol
        ])

        detector = HMMRegimeDetector(n_regimes=2)
        detector.fit(returns)
        result = detector.predict(returns)

        assert result.regimes.shape == (n,)
        assert result.probabilities.shape == (n, 2)
        assert len(result.regime_labels) == 2
        assert result.n_regimes == 2
        assert result.transition_matrix.shape == (2, 2)

    def test_predict_before_fit_raises(self) -> None:
        detector = HMMRegimeDetector()
        with pytest.raises(RuntimeError):
            detector.predict(np.array([0.01, 0.02]))

    def test_too_few_observations(self) -> None:
        detector = HMMRegimeDetector(n_regimes=3)
        with pytest.raises(ValueError):
            detector.fit(np.array([0.01, 0.02, 0.03]))

    def test_regime_labels_sorted_by_volatility(self) -> None:
        """Regimes should be sorted by volatility (low → high)."""
        np.random.seed(42)
        n = 300
        returns = np.concatenate([
            np.random.normal(0.0, 0.01, n // 3),
            np.random.normal(0.0, 0.05, n // 3),
            np.random.normal(0.0, 0.10, n // 3),
        ])

        detector = HMMRegimeDetector(n_regimes=3)
        detector.fit(returns)
        result = detector.predict(returns)

        assert result.regime_labels[0] == "Low Vol"
        assert result.regime_labels[2] == "High Vol"

    def test_score_returns_float(self) -> None:
        np.random.seed(42)
        returns = np.random.normal(0.0, 0.02, 200)

        detector = HMMRegimeDetector(n_regimes=2)
        detector.fit(returns)
        score = detector.score(returns)

        assert isinstance(score, float)
        assert np.isfinite(score)

    def test_multivariate_features(self) -> None:
        """Should handle multiple feature dimensions."""
        np.random.seed(42)
        n = 200
        X = np.column_stack([
            np.random.normal(0.0, 0.02, n),
            np.random.normal(0.0, 0.01, n),
        ])

        detector = HMMRegimeDetector(n_regimes=2)
        detector.fit(X)
        result = detector.predict(X)

        assert result.regimes.shape == (n,)
        assert result.probabilities.shape == (n, 2)
