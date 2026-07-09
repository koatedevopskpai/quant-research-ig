"""
hmm_detector.py — Hidden Markov Model regime detection.

WHY HMM FOR REGIME DETECTION:
    Financial markets exhibit distinct regimes (bull, bear, range-bound,
    high/low volatility) that persist for periods before transitioning.
    Hidden Markov Models are the natural tool for this because:

    1. **Latent states** — Regimes are not directly observable. HMMs infer
       the most likely hidden state sequence from observed returns/features.
    2. **Transition probabilities** — HMMs learn the probability of switching
       from one regime to another (e.g., bull→bear is typically low).
    3. **Emission probabilities** — Each regime has a characteristic
       distribution of returns (e.g., bull: positive mean, low vol;
       bear: negative mean, high vol).

QUANT BECKMAN'S APPROACH:
    Beckman uses HMM regime detection as a "context gating" mechanism:
    - Trade strategies are only active in favourable regimes.
    - Position sizing adjusts based on regime (smaller in high-vol regimes).
    - Backtest results are segmented by regime to check for overfitting.

BRYAN DOWNING'S APPROACH:
    Downing uses regimes for microstructure adaptation:
    - Low-vol regime → tight spreads, smaller orders.
    - High-vol regime → wider spreads, larger queue positions.
    - Regime switches trigger position size rebalancing.

THE MODEL:
    We use a Gaussian HMM where each state's emission distribution is a
    multivariate Gaussian over observed features (returns, volatility,
    volume, etc.). The model learns:

        theta = {pi, A, mu_k, Sigma_k}

    where:
        pi: Initial state distribution
        A: State transition matrix (K x K)
        mu_k: Mean vector for state k
        Sigma_k: Covariance matrix for state k

USAGE:
    detector = HMMRegimeDetector(n_regimes=3)
    detector.fit(returns)  # DataFrame or numpy array

    # Predict regimes for new data
    result = detector.predict(new_returns)
    print(result.regimes)  # array of 0, 1, 2
    print(result.probabilities)  # probability of each regime

    # Describe what each regime means
    for i, desc in enumerate(result.regime_labels):
        print(f"Regime {i}: {desc}")

REFERENCES:
    Hamilton, J. (1989). A New Approach to the Economic Analysis of
        Nonstationary Time Series. Econometrica, 57(2).
    Ang, A. & Timmermann, A. (2012). Regime Changes and Financial Markets.
        Annual Review of Financial Economics, 4(1).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from hmmlearn import hmm


@dataclass
class RegimeResult:
    """
    Output of HMM regime detection.

    Attributes:
        regimes: Array of predicted regime labels (0, 1, ..., K-1) for
            each observation. Length = len(observations).
        probabilities: Array of shape (n_obs, n_regimes) with the posterior
            probability of each regime for each observation.
        regime_labels: Human-readable descriptions of each regime, sorted
            by volatility (ascending): "Low Vol", "Medium Vol", "High Vol".
        transition_matrix: Estimated transition probabilities between
            regimes. T[i,j] = probability of transitioning from i to j.
        n_regimes: Number of detected regimes.
    """

    regimes: np.ndarray
    probabilities: np.ndarray
    regime_labels: list[str]
    transition_matrix: np.ndarray
    n_regimes: int


class HMMRegimeDetector:
    """
    Hidden Markov Model for market regime detection.

    Infers latent market regimes from observed financial features
    (typically returns and volatility).

    Parameters:
        n_regimes: Number of hidden states (default 3 = low/med/high vol).
        covariance_type: Type of covariance for each state's emission
            distribution. "full" (default) allows different correlations
            in each regime. "diag" is faster but assumes independence.
        n_iter: Maximum EM iterations for the Baum-Welch algorithm.

    USAGE:
        detector = HMMRegimeDetector(n_regimes=3)
        detector.fit(returns)  # (n_obs, n_features) array

        result = detector.predict(returns)
        current_regime = result.regimes[-1]

    WHY 3 REGIMES (default):
        Financial theory and empirical evidence (Ang & Timmermann, 2012)
        support a 3-regime model:
        - Low volatility / bull: Positive returns, low variance.
        - Medium volatility / range: Near-zero returns, moderate variance.
        - High volatility / bear or crisis: Negative returns, high variance.

        More regimes (4-5) can capture finer distinctions but increase
        the risk of overfitting and label-switching problems.

    LIMITATIONS:
        - HMM assumes the Markov property (next state depends only on
          current state). This is a strong assumption for markets where
          long-range dependencies exist.
        - The number of regimes must be specified in advance. Model
          selection via BIC/AIC can help but is not implemented here.
        - Gaussian emissions may underestimate tail risk (multi-modal
          return distributions within a regime).
    """

    def __init__(
        self,
        n_regimes: int = 3,
        covariance_type: str = "full",
        n_iter: int = 1000,
    ) -> None:
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self._model: hmm.GaussianHMM | None = None

    # ------------------------------------------------------------------
    # Model fitting
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray) -> "HMMRegimeDetector":
        """
        Fit the HMM to observed features using the Baum-Welch algorithm.

        The Baum-Welch algorithm is a special case of Expectation-
        Maximisation (EM) for HMMs. It iteratively:
        1. (E-step) Computes the posterior probability of each hidden
           state given the observations (forward-backward algorithm).
        2. (M-step) Updates the model parameters (pi, A, mu, Sigma) to
           maximise the expected log-likelihood.

        WHY EM vs. direct optimisation:
            The HMM likelihood is non-convex with multiple local maxima.
            EM is guaranteed to converge to a local maximum. Running
            multiple initialisations (n_init parameter) helps find a
            better local optimum.

        Args:
            X: Observed features, shape (n_observations, n_features).
               Typically log returns or log returns + other features.

        Returns:
            Self (fitted model).

        Raises:
            ValueError: If X has fewer observations than regimes.
        """
        n_obs = X.shape[0] if hasattr(X, "shape") else len(X)
        if n_obs < self.n_regimes * 2:
            raise ValueError(
                f"Need at least {self.n_regimes * 2} observations, got {n_obs}"
            )

        # Convert to numpy if not already
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self._model = hmm.GaussianHMM(
            n_components=self.n_regimes,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=42,
            init_params="stmc",  # initialise all params
            params="stmc",       # learn all params
        )

        self._model.fit(X)
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> RegimeResult:
        """
        Predict the most likely regime for each observation.

        Uses the Viterbi algorithm to find the most probable state
        sequence given the observations.

        Args:
            X: Observed features, shape (n_observations, n_features).
               Must have the same number of features as the training data.

        Returns:
            RegimeResult with predictions, probabilities, and labels.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before predict()")

        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Viterbi: most likely state sequence
        regimes = self._model.predict(X)

        # Posterior probabilities (forward-backward)
        probabilities = self._model.predict_proba(X)

        # Transition matrix
        transmat = self._model.transmat_

        # Label regimes by volatility (low → medium → high)
        # We sort states by the variance of their emission distribution
        # to assign meaningful labels.
        means = self._model.means_
        covars = self._model.covars_

        # Compute mean variance across features for each state
        state_vols: list[float] = []
        for k in range(self.n_regimes):
            if self.covariance_type == "full":
                vol = np.trace(covars[k])  # sum of variances
            else:
                vol = np.sum(covars[k])  # diag or spherical
            state_vols.append(float(vol))

        # Sort states by volatility (ascending)
        sorted_indices = np.argsort(state_vols)
        vol_labels = ["Low Vol", "Medium Vol", "High Vol"]

        # Map original state indices to sorted order
        state_map = {orig: new for new, orig in enumerate(sorted_indices)}
        mapped_regimes = np.array([state_map[s] for s in regimes])

        # Reorder transition matrix to match sorted order
        reordered_transmat = transmat[np.ix_(sorted_indices, sorted_indices)]

        # Create regime labels
        labels = [vol_labels[i] if i < len(vol_labels) else f"Regime {i}" for i in range(self.n_regimes)]

        return RegimeResult(
            regimes=mapped_regimes,
            probabilities=probabilities[:, sorted_indices],
            regime_labels=labels,
            transition_matrix=reordered_transmat,
            n_regimes=self.n_regimes,
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def means_(self) -> np.ndarray | None:
        """State mean vectors (learned emission parameters)."""
        return self._model.means_ if self._model else None

    @property
    def transition_matrix_(self) -> np.ndarray | None:
        """State transition matrix A."""
        return self._model.transmat_ if self._model else None

    def score(self, X: np.ndarray) -> float:
        """
        Compute the log-likelihood of observations under the fitted model.

        Higher values indicate a better fit. Can be used for model
        selection (BIC/AIC) across different numbers of regimes.

        Args:
            X: Observed features (1D or 2D array).

        Returns:
            Log-likelihood score.
        """
        if self._model is None:
            raise RuntimeError("Model must be fitted before score()")
        if not isinstance(X, np.ndarray):
            X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._model.score(X)
