"""
Reward-modulated transition adaptation.

The claim is "RL adaptation of the HMM transition matrix based on profitable
versus losing trades". These tests pin the mechanism and its safety bounds, not a
profit claim.
"""

import numpy as np
import pytest

from ml_for_finance.regime.adaptive_hmm import AdaptiveHMMRegimeDetector


def _returns(n=400, seed=7):
    rng = np.random.default_rng(seed)
    state = 0
    values = []
    for _ in range(n):
        state = (1 - state) if rng.random() < 0.03 else state
        values.append(rng.normal(0.0, 0.006 if state == 0 else 0.018))
    return np.array(values).reshape(-1, 1)


@pytest.fixture
def detector():
    model = AdaptiveHMMRegimeDetector(n_regimes=2, learning_rate=0.2, decay=0.0)
    model.fit(_returns())
    return model


class TestShapeAndBounds:
    def test_transition_matrix_is_stochastic(self, detector):
        baseline = detector.baseline_transition_matrix
        assert baseline.shape == (2, 2)
        np.testing.assert_allclose(baseline.sum(axis=1), 1.0)

        matrix = detector.adapt(0, 1, reward=0.05)
        np.testing.assert_allclose(matrix.sum(axis=1), 1.0)
        assert (matrix > 0).all()

    def test_invalid_indices_and_rewards_rejected(self, detector):
        with pytest.raises(ValueError):
            detector.adapt(0, 2, 0.01)
        with pytest.raises(TypeError):
            detector.adapt(0.0, 1, 0.01)
        with pytest.raises(ValueError):
            detector.adapt(0, 1, float("nan"))

    def test_adapt_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            AdaptiveHMMRegimeDetector(n_regimes=2).adapt(0, 1, 0.01)


class TestRewardDirection:
    def test_profit_increases_the_traversed_transition(self, detector):
        baseline = detector.baseline_transition_matrix[0, 1]
        adapted = detector.adapt(0, 1, reward=0.05)[0, 1]
        assert adapted > baseline

    def test_loss_decreases_the_traversed_transition(self, detector):
        baseline = detector.baseline_transition_matrix[0, 1]
        adapted = detector.adapt(0, 1, reward=-0.05)[0, 1]
        assert adapted < baseline

    def test_other_rows_are_unchanged(self, detector):
        before = detector.transition_matrix
        after = detector.adapt(0, 1, reward=0.05)
        np.testing.assert_allclose(after[1], before[1])

    def test_single_update_is_bounded(self, detector):
        baseline = detector.baseline_transition_matrix[0, 1]
        for _ in range(50):
            detector.adapt(0, 1, reward=10.0)  # saturating reward
        adapted = detector.transition_matrix[0, 1]
        # Row normalisation and the per-update factor cap keep it below 1 and
        # above the baseline; it must not collapse the row.
        assert baseline < adapted < 1.0
        np.testing.assert_allclose(detector.transition_matrix.sum(axis=1), 1.0)


class TestDecay:
    def test_neutral_updates_drift_back_to_baseline(self):
        model = AdaptiveHMMRegimeDetector(n_regimes=2, learning_rate=0.5,
                                          decay=0.2)
        model.fit(_returns())
        baseline = model.baseline_transition_matrix
        model.adapt(0, 1, reward=0.1)
        moved = model.transition_matrix
        for _ in range(40):
            model.adapt(0, 1, reward=0.0)
        np.testing.assert_allclose(model.transition_matrix, baseline, atol=0.02)
        assert not np.allclose(moved, baseline)

    def test_predict_still_works_after_adaptation(self, detector):
        detector.adapt(0, 1, reward=0.05)
        result = detector.predict(_returns(60))
        assert result.regimes.shape[0] == 60
        assert result.transition_matrix.shape == (2, 2)
