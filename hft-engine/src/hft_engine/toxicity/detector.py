"""
toxicity/detector.py — Order Flow Toxicity Detector

Implements Bryan Downing's toxicity framework using a mutual-exciting
Hawkes process. The key insight: toxic order flow is *self-exciting* —
aggressive market orders cluster in time, and the conditional intensity
captures how quickly the market is being "picked off."

The Hawkes(1,1) model:
    lambda(t) = mu + alpha * sum_{t_i < t} exp(-beta * (t - t_i))

where:
    mu    = baseline intensity (random background orders)
    alpha = self-excitement strength (how much each trade amplifies)
    beta  = decay rate (how fast excitement fades)

Toxicity is defined as:
    Toxicity = (lambda(t) - mu) / lambda(t)

i.e., the fraction of current intensity attributable to clustering.
When >0.5, more than half of expected activity is from cascading —
the hallmark of toxic flow (informed traders hitting the book).
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToxicityResult:
    intensity: float
    toxicity: float
    mu: float
    alpha: float
    beta: float
    n_events: int
    is_toxic: bool


class ToxicityDetector:
    def __init__(self, alpha_init: float = 0.5, beta_init: float = 1.0,
                 mu_init: float = 0.1, window_size: int = 1000,
                 toxicity_threshold: float = 0.5):
        self.alpha = alpha_init
        self.beta = beta_init
        self.mu = mu_init
        self.window_size = window_size
        self.toxicity_threshold = toxicity_threshold

        self.timestamps: list[float] = []
        self._fitted = False

    def _log_likelihood(self, params: tuple[float, float, float],
                        timestamps: np.ndarray, T: float) -> float:
        mu, alpha, beta = params
        if mu <= 0 or alpha < 0 or beta <= 0 or alpha >= beta:
            return -np.inf
        n = len(timestamps)
        if n == 0:
            return -np.inf
        ll = -mu * T
        ll -= (alpha / beta) * (n - np.sum(np.exp(-beta * (T - timestamps))))
        for i in range(n):
            if i == 0:
                integral = 0.0
            else:
                integral = np.sum(np.exp(-beta * (timestamps[i] - timestamps[:i])))
            ll += np.log(mu + alpha * integral)
        return ll

    def fit(self, timestamps: list[float]) -> "ToxicityDetector":
        arr = np.asarray(timestamps, dtype=float)
        if len(arr) < 10:
            raise ValueError(f"Need >=10 events, got {len(arr)}")
        arr = arr[arr > 0]
        arr.sort()
        if len(arr) > self.window_size:
            arr = arr[-self.window_size:]
        arr -= arr[0]
        T = arr[-1] if arr[-1] > 0 else 1.0
        best_ll = -np.inf
        best_params = (self.mu, self.alpha, self.beta)
        for mu_scale in [0.5, 1.0, 2.0]:
            for alpha_scale in [0.2, 0.5, 1.0]:
                for beta_scale in [0.5, 1.0, 2.0]:
                    mu = self.mu * mu_scale
                    alpha_val = self.alpha * alpha_scale
                    beta_val = self.beta * beta_scale
                    if alpha_val >= beta_val:
                        continue
                    ll = self._log_likelihood((mu, alpha_val, beta_val), arr, T)
                    if ll > best_ll and np.isfinite(ll):
                        best_ll = ll
                        best_params = (mu, alpha_val, beta_val)
        self.mu, self.alpha, self.beta = best_params
        self.timestamps = arr.tolist()
        self._T = T
        self._fitted = True
        return self

    def _assert_fitted(self):
        if not self._fitted:
            raise RuntimeError("Model must be fitted before predict()")

    def intensity(self, t: Optional[float] = None) -> float:
        self._assert_fitted()
        arr = np.asarray(self.timestamps, dtype=float)
        if t is None:
            t = arr[-1] if len(arr) > 0 else 0.0
        past = arr[arr < t]
        if len(past) == 0:
            return self.mu
        decay = np.exp(-self.beta * (t - past))
        return self.mu + self.alpha * np.sum(decay)

    def predict(self, t: Optional[float] = None) -> ToxicityResult:
        self._assert_fitted()
        lam = self.intensity(t)
        tox = (lam - self.mu) / lam if lam > 0 else 0.0
        return ToxicityResult(
            intensity=lam,
            toxicity=tox,
            mu=self.mu,
            alpha=self.alpha,
            beta=self.beta,
            n_events=len(self.timestamps),
            is_toxic=tox >= self.toxicity_threshold,
        )
