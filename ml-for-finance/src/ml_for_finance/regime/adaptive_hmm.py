"""
adaptive_hmm.py — reward-modulated adaptation of an HMM transition matrix.

WHAT THIS IS
    Downing's design adapts the regime transition matrix from realised trade
    outcomes. This is the minimal honest version of that loop: a reward-modulated
    multiplicative update on the row that a trade actually traversed, with decay
    back to the fitted baseline.

        T[i, j] <- normalise( T[i, j] * exp(eta * tanh(pnl / scale)) )   for the
                    observed (i -> j), then blend toward the fitted baseline.

    Positive outcomes make the traversed transition slightly more likely;
    negative outcomes make it slightly less likely. Everything else is untouched.

WHAT THIS IS NOT
    Not full policy-gradient RL: there is no value function, no policy, and no
    exploration. It is a bandit-style update on a transition model, which is the
    mechanism the design describes ("adapt the transition matrix from profitable
    versus losing trades"). The docstring says so rather than borrowing the
    credibility of a larger claim.

SAFETY PROPERTIES (tested)
    * every row remains a probability distribution;
    * a single update is bounded by `exp(+/-max_log_adjustment)`;
    * repeated updates decay toward the fitted baseline (`decay`);
    * indices and rewards are validated;
    * `predict()` keeps working because the estimator reads `transmat_`.
"""

from __future__ import annotations

import numpy as np

from ml_for_finance.regime.hmm_detector import HMMRegimeDetector


class AdaptiveHMMRegimeDetector(HMMRegimeDetector):
    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 1000,
        learning_rate: float = 0.05,
        decay: float = 0.01,
        max_log_adjustment: float = 1.0,
        reward_scale: float = 0.01,
    ) -> None:
        super().__init__(n_regimes, covariance_type, n_iter)
        if not 0.0 < learning_rate < 1.0:
            raise ValueError("learning_rate must be in (0, 1)")
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be in [0, 1]")
        if max_log_adjustment <= 0:
            raise ValueError("max_log_adjustment must be positive")
        if reward_scale <= 0:
            raise ValueError("reward_scale must be positive")
        self.learning_rate = learning_rate
        self.decay = decay
        self.max_log_adjustment = max_log_adjustment
        self.reward_scale = reward_scale
        self._baseline: np.ndarray | None = None
        self.updates = 0
        self.last_reward = 0.0

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "AdaptiveHMMRegimeDetector":
        super().fit(X)
        assert self._model is not None
        self._baseline = np.asarray(self._model.transmat_, dtype=float).copy()
        self.updates = 0
        self.last_reward = 0.0
        return self

    @property
    def transition_matrix(self) -> np.ndarray | None:
        if self._model is None:
            return None
        return np.asarray(self._model.transmat_, dtype=float).copy()

    @property
    def baseline_transition_matrix(self) -> np.ndarray | None:
        return None if self._baseline is None else self._baseline.copy()

    # ------------------------------------------------------------------
    # Adaptation
    # ------------------------------------------------------------------

    def adapt(self, prev_state: int, next_state: int, reward: float) -> np.ndarray:
        """
        Apply one reward-modulated update to the (prev_state -> next_state) cell.

        Args:
            prev_state: regime at entry.
            next_state: regime at exit (may equal prev_state).
            reward: realised P&L, in the same units as `reward_scale`.
        """
        if self._model is None or self._baseline is None:
            raise RuntimeError("fit() before adapt()")
        if not isinstance(prev_state, (int, np.integer)) or not isinstance(
            next_state, (int, np.integer)
        ):
            raise TypeError("regime indices must be integers")
        n = self.n_regimes
        if not (0 <= int(prev_state) < n) or not (0 <= int(next_state) < n):
            raise ValueError(f"regime indices must be in [0, {n - 1}]")
        if not np.isfinite(reward):
            raise ValueError("reward must be finite")

        prev_state, next_state = int(prev_state), int(next_state)
        shaped = float(np.tanh(float(reward) / self.reward_scale))

        with np.errstate(over="ignore"):
            factor = float(np.exp(np.clip(
                self.learning_rate * shaped,
                -self.max_log_adjustment,
                self.max_log_adjustment,
            )))

        matrix = np.asarray(self._model.transmat_, dtype=float).copy()
        # Decay the whole matrix toward the fitted baseline before the update,
        # so a sequence of one-sided outcomes cannot drift without bound.
        matrix = (1.0 - self.decay) * matrix + self.decay * self._baseline
        matrix[prev_state, next_state] *= factor
        matrix[prev_state, :] = np.clip(matrix[prev_state, :], 1e-12, None)
        matrix[prev_state, :] /= matrix[prev_state, :].sum()

        # Defensive: no row may contain zeros or fail to normalise.
        matrix = np.clip(matrix, 1e-12, None)
        matrix /= matrix.sum(axis=1, keepdims=True)

        self._model.transmat_ = matrix
        self.updates += 1
        self.last_reward = float(reward)
        return matrix.copy()
