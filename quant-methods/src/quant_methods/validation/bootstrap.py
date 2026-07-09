"""
validation/bootstrap.py — Block Bootstrap Resampling

Computes confidence intervals for performance metrics by resampling
return series with blocks that preserve serial dependence.

The circular block bootstrap (Politis & Romano, 1992) wraps the series
into a circle and draws overlapping blocks of length L:
    - L = ceil(T^(1/3)) by default (optimal for stationary bootstrap)
    - Each block starts at a random uniformly sampled position
    - Blocks are concatenated until length >= original series

This preserves autocorrelation and volatility clustering within blocks
while destroying dependence across blocks, giving valid inference under
weak dependence conditions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class BootstrapResult:
    metric_name: str
    actual_value: float
    ci_lower: float
    ci_upper: float
    std_error: float
    n_bootstrap: int
    block_size: int
    distribution: Optional[np.ndarray] = None


class BlockBootstrap:
    def __init__(self, n_bootstrap: int = 1000, block_size: Optional[int] = None,
                 random_seed: Optional[int] = None):
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.rng = np.random.default_rng(random_seed)

    def _optimal_block_size(self, n: int) -> int:
        return max(2, int(np.ceil(n ** (1.0 / 3.0))))

    def _circular_block_bootstrap(self, series: np.ndarray) -> np.ndarray:
        n = len(series)
        bs = self.block_size or self._optimal_block_size(n)
        wrapped = np.concatenate([series, series])
        n_blocks = int(np.ceil(n / bs))
        starts = self.rng.integers(0, n, size=n_blocks)
        resampled = np.concatenate([wrapped[s:s + bs] for s in starts])
        return resampled[:n]

    def confidence_interval(self, returns: np.ndarray,
                            metric_fn: Callable[[np.ndarray], float],
                            metric_name: str = "metric",
                            confidence: float = 0.95) -> BootstrapResult:
        returns = np.asarray(returns, dtype=float).flatten()
        if len(returns) < 5:
            raise ValueError(f"Need >=5 observations, got {len(returns)}")
        actual = float(metric_fn(returns))
        bs = self.block_size or self._optimal_block_size(len(returns))
        bootstrapped = np.zeros(self.n_bootstrap)
        for i in range(self.n_bootstrap):
            sample = self._circular_block_bootstrap(returns)
            bootstrapped[i] = metric_fn(sample)
        alpha = 1.0 - confidence
        ci_lower = float(np.percentile(bootstrapped, alpha / 2 * 100))
        ci_upper = float(np.percentile(bootstrapped, (1 - alpha / 2) * 100))
        std_err = float(np.std(bootstrapped))
        return BootstrapResult(
            metric_name=metric_name,
            actual_value=actual,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            std_error=std_err,
            n_bootstrap=self.n_bootstrap,
            block_size=bs,
            distribution=bootstrapped,
        )

    def sharpe_ratio(self, returns: np.ndarray,
                     periods_per_year: int = 252,
                     confidence: float = 0.95) -> BootstrapResult:
        def _sr(r):
            return r.mean() / r.std() * np.sqrt(periods_per_year) if r.std() > 0 else 0.0
        return self.confidence_interval(returns, _sr, "Sharpe Ratio", confidence)
