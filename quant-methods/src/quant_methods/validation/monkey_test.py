"""
validation/monkey_test.py — Monkey Test (Monte Carlo Permutation Test)

Quant Beckman's signature falsification method. The Monkey Test asks:
"Would a random strategy with the same trade distribution perform
as well as my strategy?"

Procedure:
    1. Take the actual strategy's return series.
    2. Generate N random permutations of the returns (shuffling trade signs
       while preserving the magnitude distribution).
    3. Compute the test statistic (default: Sharpe ratio) for each permutation.
    4. Count how many permuted strategies beat the actual strategy.
    5. p-value = (count + 1) / (N + 1)

If p < 0.05, the strategy is statistically significant — unlikely to be
the result of overfitting or data mining.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable


@dataclass
class MonkeyTestResult:
    actual_statistic: float
    null_mean: float
    null_std: float
    p_value: float
    n_permutations: int
    is_significant: bool
    null_distribution: Optional[np.ndarray] = None


class MonkeyTest:
    def __init__(self, n_permutations: int = 1000, random_seed: Optional[int] = None):
        self.n_permutations = n_permutations
        self.rng = np.random.default_rng(random_seed)

    @staticmethod
    def _sharpe_ratio(returns: np.ndarray) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(252)

    def run(self, returns: np.ndarray,
            statistic_fn: Optional[Callable[[np.ndarray], float]] = None,
            significance_level: float = 0.05) -> MonkeyTestResult:
        returns = np.asarray(returns, dtype=float).flatten()
        if len(returns) < 10:
            raise ValueError(f"Need >=10 observations, got {len(returns)}")

        stat_fn = statistic_fn or self._sharpe_ratio
        actual_stat = stat_fn(returns)
        null_stats = np.zeros(self.n_permutations)
        for i in range(self.n_permutations):
            perm = returns.copy()
            self.rng.shuffle(perm)
            null_stats[i] = stat_fn(perm)
        p_value = (np.sum(null_stats >= actual_stat) + 1) / (self.n_permutations + 1)
        return MonkeyTestResult(
            actual_statistic=actual_stat,
            null_mean=float(np.mean(null_stats)),
            null_std=float(np.std(null_stats)),
            p_value=float(p_value),
            n_permutations=self.n_permutations,
            is_significant=p_value < significance_level,
            null_distribution=null_stats,
        )
