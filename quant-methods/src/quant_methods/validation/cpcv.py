"""
validation/cpcv.py — Combinatorial Purged Cross-Validation (CPCV)

López de Prado (2018). CPCV extends standard CV to time series by:
    1. Splitting the data into N groups (purged of leakage).
    2. Selecting K groups for testing, the rest for training.
    3. Purging training data within an embargo period before each test set.
    4. Repeating across all C(N, K) combinations.

This yields a distribution of out-of-sample performance metrics that
is more robust than a single train/test split. The combinatorial
approach ensures every observation appears in a test set multiple times,
reducing the variance of the performance estimate.

Key concepts:
    - Purge: remove training observations within 'embargo' of test set
    - Embargo: minimum gap between train and test to prevent leakage
    - C(N, K): number of combinations (capped to max_combo for tractability)
"""

import numpy as np
from itertools import combinations
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class CPCVResult:
    sharpe_mean: float
    sharpe_std: float
    sharpe_min: float
    sharpe_max: float
    sharpe_median: float
    n_groups: int
    n_test_groups: int
    embargo: int
    n_combinations: int
    all_sharpes: list[float]
    is_significant: bool


class CombinatorialPurgedCV:
    def __init__(self, n_groups: int = 10, n_test_groups: int = 2,
                 embargo: int = 0, max_combinations: int = 100,
                 periods_per_year: int = 252):
        if n_test_groups >= n_groups:
            raise ValueError(f"n_test_groups ({n_test_groups}) must be < n_groups ({n_groups})")
        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.embargo = embargo
        self.max_combinations = max_combinations
        self.periods_per_year = periods_per_year

    def _split_groups(self, n_obs: int) -> list[tuple[int, int]]:
        group_size = n_obs // self.n_groups
        remainder = n_obs % self.n_groups
        groups = []
        start = 0
        for g in range(self.n_groups):
            size = group_size + (1 if g < remainder else 0)
            groups.append((start, start + size))
            start += size
        return groups

    def _purge_indices(self, test_start: int, test_end: int,
                       n_obs: int) -> tuple[int, int]:
        purge_before = max(0, test_start - self.embargo)
        purge_after = min(n_obs, test_end + self.embargo)
        return (purge_before, purge_after)

    def run(self, strategy_returns: np.ndarray,
            metric_fn: Optional[Callable[[np.ndarray], float]] = None,
            significance_level: float = 0.05) -> CPCVResult:
        returns = np.asarray(strategy_returns, dtype=float).flatten()
        if len(returns) < self.n_groups * 2:
            raise ValueError(f"Need >= {self.n_groups * 2} observations")

        def default_metric(r):
            return r.mean() / r.std() * np.sqrt(self.periods_per_year) if r.std() > 0 else 0.0

        metric = metric_fn or default_metric
        groups = self._split_groups(len(returns))
        all_combos = list(combinations(range(self.n_groups), self.n_test_groups))
        if len(all_combos) > self.max_combinations:
            rng = np.random.default_rng(42)
            all_combos = rng.choice(all_combos, self.max_combinations, replace=False).tolist()

        sharpes = []
        for test_groups in all_combos:
            test_mask = np.zeros(len(returns), dtype=bool)
            train_mask = np.ones(len(returns), dtype=bool)
            for g in test_groups:
                s, e = groups[g]
                test_mask[s:e] = True
                train_mask[s:e] = False
                purge_before, purge_after = self._purge_indices(s, e, len(returns))
                if purge_before < s:
                    train_mask[purge_before:s] = False
                if purge_after > e:
                    train_mask[e:purge_after] = False
            if train_mask.sum() < 10 or test_mask.sum() < 5:
                continue
            oos_sharpe = metric(returns[test_mask])
            sharpes.append(oos_sharpe)

        sharpes_arr = np.array(sharpes) if sharpes else np.array([0.0])
        sr_mean = float(np.mean(sharpes_arr))
        sr_std = float(np.std(sharpes_arr)) if len(sharpes_arr) > 1 else 0.0
        return CPCVResult(
            sharpe_mean=sr_mean,
            sharpe_std=sr_std,
            sharpe_min=float(sharpes_arr.min()),
            sharpe_max=float(sharpes_arr.max()),
            sharpe_median=float(np.median(sharpes_arr)),
            n_groups=self.n_groups,
            n_test_groups=self.n_test_groups,
            embargo=self.embargo,
            n_combinations=len(sharpes),
            all_sharpes=sharpes,
            is_significant=sr_mean - 2 * sr_std > 0,
        )
