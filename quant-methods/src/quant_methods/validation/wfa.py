"""Chronological walk-forward model refits with label purging and a training gap."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from quant_methods.validation._folds import evaluate_fold, event_ends, sharpe, supervised_data


@dataclass
class WFAResult:
    oos_sharpe: float
    oos_returns: np.ndarray
    n_windows: int
    train_size: int
    test_size: int
    window_results: list[dict]
    is_robust: bool  # descriptive positive-window heuristic, not statistical significance
    oos_indices: np.ndarray


class WalkForwardAnalysis:
    def __init__(
        self,
        train_size,
        test_size,
        mode="rolling",
        periods_per_year=252,
        min_train_size=None,
        gap=0,
    ):
        if mode not in ("anchored", "rolling"):
            raise ValueError("mode must be 'anchored' or 'rolling'")
        minimum = train_size if min_train_size is None else min_train_size
        if not all(
            isinstance(v, Integral) for v in (train_size, test_size, periods_per_year, minimum, gap)
        ):
            raise ValueError("Window sizes, periods_per_year and gap must be integers")
        if not 2 <= minimum <= train_size or test_size < 2 or gap < 0 or periods_per_year < 1:
            raise ValueError("Invalid window sizes, minimum, gap or periods_per_year")
        self.train_size, self.test_size = train_size, test_size
        self.mode, self.periods_per_year = mode, periods_per_year
        self.min_train_size, self.gap = minimum, gap

    def split(self, x, *, label_end=None):
        n = x if isinstance(x, Integral) else len(x)
        required = self.min_train_size + self.gap + self.test_size
        if n < required:
            raise ValueError(f"Need >= {required} observations")
        ends = event_ends(n, label_end)
        for test_start in range(
            self.min_train_size + self.gap, n - self.test_size + 1, self.test_size
        ):
            train_end = test_start - self.gap
            train_start = 0 if self.mode == "anchored" else max(0, train_end - self.train_size)
            train = np.arange(train_start, train_end)
            train = train[ends[train] < test_start]
            if len(train) < 2:
                raise ValueError("Purging left fewer than two training observations")
            yield train, np.arange(test_start, test_start + self.test_size)

    def run(self, x, y, model_factory, returns_fn, *, label_end=None):
        features, targets = supervised_data(x, y)
        windows = []
        for train, test in self.split(features, label_end=label_end):
            fold = evaluate_fold(features, targets, train, test, model_factory, returns_fn)
            fold.update(
                window=len(windows),
                train_start=int(train[0]),
                train_end=int(train[-1]) + 1,
                test_start=int(test[0]),
                test_end=int(test[-1]) + 1,
                oos_sharpe=sharpe(fold["oos_returns"], self.periods_per_year),
            )
            windows.append(fold)
        returns = np.concatenate([w["oos_returns"] for w in windows])
        indices = np.concatenate([w["test_indices"] for w in windows])
        scores = np.array([w["oos_sharpe"] for w in windows])
        robust = bool(
            len(windows) >= 3
            and scores.mean() > 0
            and np.median(scores) > 0
            and np.mean(scores > 0) >= 0.5
        )
        return WFAResult(
            sharpe(returns, self.periods_per_year),
            returns,
            len(windows),
            self.train_size,
            self.test_size,
            windows,
            robust,
            indices,
        )
