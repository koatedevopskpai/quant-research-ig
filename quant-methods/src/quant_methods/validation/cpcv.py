"""Purged combinatorial supervised CV with inclusive label event intervals.

Each fold fits a fresh model. Embargo excludes observations starting immediately
after each held-out event interval. Folds may train on later observations, as
combinatorial CV does; this is not a chronological live-trading simulation.
Fold scores overlap and are not independent significance observations.
"""

from dataclasses import dataclass
from itertools import combinations
from math import comb
from numbers import Integral

import numpy as np
from quant_methods.validation._folds import evaluate_fold, event_ends, sharpe, supervised_data


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
    is_significant: bool  # always False: correlated folds do not establish significance
    lower_quantile: float
    positive_fold_fraction: float
    fold_results: list[dict]


class CombinatorialPurgedCV:
    def __init__(
        self,
        n_groups=10,
        n_test_groups=2,
        embargo=0,
        max_combinations=100,
        periods_per_year=252,
        random_state=42,
    ):
        if not all(
            isinstance(v, Integral)
            for v in (n_groups, n_test_groups, embargo, max_combinations, periods_per_year)
        ):
            raise ValueError("Fold counts, embargo and periods_per_year must be integers")
        if not 1 <= n_test_groups < n_groups:
            raise ValueError("n_test_groups must be >=1 and < n_groups")
        if embargo < 0 or max_combinations < 1 or periods_per_year < 1:
            raise ValueError("Invalid embargo, max_combinations or periods_per_year")
        self.n_groups, self.n_test_groups = n_groups, n_test_groups
        self.embargo, self.max_combinations = embargo, max_combinations
        self.periods_per_year, self.random_state = periods_per_year, random_state

    def _split_groups(self, n_obs):
        groups = np.array_split(np.arange(n_obs), self.n_groups)
        return [(int(group[0]), int(group[-1]) + 1) for group in groups]

    def _combinations(self):
        total = comb(self.n_groups, self.n_test_groups)
        if total <= self.max_combinations:
            return list(combinations(range(self.n_groups), self.n_test_groups))
        rng, selected = np.random.default_rng(self.random_state), set()
        while len(selected) < self.max_combinations:
            selected.add(
                tuple(sorted(rng.choice(self.n_groups, self.n_test_groups, replace=False).tolist()))
            )
        return sorted(selected)

    def split(self, x, *, label_end=None):
        n = x if isinstance(x, Integral) else len(x)
        if n < self.n_groups * 2:
            raise ValueError(f"Need >= {self.n_groups * 2} observations")
        starts = np.arange(n)
        ends = event_ends(n, label_end)
        groups = self._split_groups(n)
        for selected in self._combinations():
            test_mask = np.zeros(n, dtype=bool)
            train_mask = np.ones(n, dtype=bool)
            for group in selected:
                start, stop = groups[group]
                test_mask[start:stop] = True
                event_stop = int(ends[start:stop].max())
                overlaps = (starts <= event_stop) & (ends >= start)
                embargoed = (starts > event_stop) & (starts <= event_stop + self.embargo)
                train_mask &= ~(overlaps | embargoed)
            train_mask &= ~test_mask
            train, test = starts[train_mask], starts[test_mask]
            if len(train) < 2:
                raise ValueError("Purging left fewer than two training observations")
            yield train, test

    def run(
        self,
        x,
        y,
        model_factory,
        returns_fn,
        *,
        label_end=None,
        metric_fn=None,
        significance_level=0.05,
    ):
        """Fit each fold; returns_fn(realised_y, predictions) supplies net returns.

        All preprocessing must live inside model_factory's fitted model/pipeline.
        significance_level selects a descriptive lower quantile, not a p-value.
        """
        if not 0 < significance_level < 1:
            raise ValueError("significance_level must lie strictly between 0 and 1")
        features, targets = supervised_data(x, y)
        metric = metric_fn or (lambda r: sharpe(r, self.periods_per_year))
        folds, scores = [], []
        for train, test in self.split(features, label_end=label_end):
            fold = evaluate_fold(features, targets, train, test, model_factory, returns_fn)
            score = float(metric(fold["oos_returns"]))
            if not np.isfinite(score):
                raise ValueError("metric_fn must return a finite score")
            fold["score"] = score
            folds.append(fold)
            scores.append(score)
        values = np.asarray(scores)
        return CPCVResult(
            float(values.mean()),
            float(values.std()),
            float(values.min()),
            float(values.max()),
            float(np.median(values)),
            self.n_groups,
            self.n_test_groups,
            self.embargo,
            len(folds),
            scores,
            False,
            float(np.quantile(values, significance_level)),
            float(np.mean(values > 0)),
            folds,
        )
