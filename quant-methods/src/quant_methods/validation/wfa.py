"""
validation/wfa.py — Walk-Forward Analysis (WFA)

Walk-Forward Analysis evaluates a strategy's robustness by repeatedly
re-optimising on a training window and testing on the subsequent
out-of-sample (OOS) period. This simulates how the strategy would have
performed in production — where parameters are periodically updated.

Two modes:
    1. Anchored (expanding window): training grows over time
    2. Rolling (fixed window): training is a fixed-length sliding window

OOS periods are concatenated to form a single out-of-sample return
series, which is then evaluated with the standard metrics.

This is the final gate in Quant Beckman's "Know" stage before
a strategy is promoted to production.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable


@dataclass
class WFAResult:
    oos_sharpe: float
    oos_returns: np.ndarray
    n_windows: int
    train_size: int
    test_size: int
    window_results: list[dict]
    is_robust: bool


class WalkForwardAnalysis:
    def __init__(self, train_size: int, test_size: int,
                 mode: str = "rolling", periods_per_year: int = 252,
                 min_train_size: Optional[int] = None):
        if mode not in ("anchored", "rolling"):
            raise ValueError(f"mode must be 'anchored' or 'rolling', got {mode}")
        self.train_size = train_size
        self.test_size = test_size
        self.mode = mode
        self.periods_per_year = periods_per_year
        self.min_train_size = min_train_size or train_size

    def _sharpe(self, returns: np.ndarray) -> float:
        if len(returns) < 2 or returns.std() == 0:
            return 0.0
        return returns.mean() / returns.std() * np.sqrt(self.periods_per_year)

    def run(self, returns: np.ndarray,
            train_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
            test_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None) -> WFAResult:
        returns = np.asarray(returns, dtype=float).flatten()
        n = len(returns)
        if n < self.train_size + self.test_size:
            raise ValueError(f"Need >= {self.train_size + self.test_size} obs, got {n}")

        oos_list: list[float] = []
        window_results: list[dict] = []
        start = 0

        while start + self.train_size + self.test_size <= n:
            train_end = start + self.train_size
            test_end = train_end + self.test_size
            train_data = returns[start:train_end]
            test_data = returns[train_end:test_end]

            if train_fn is not None:
                _ = train_fn(train_data)

            if test_fn is not None:
                pred_sharpe = test_fn(train_data, test_data)
            else:
                pred_sharpe = self._sharpe(train_data)

            oos_sharpe = self._sharpe(test_data)
            oos_list.extend(test_data.tolist())

            window_results.append({
                "window": len(window_results),
                "train_start": start,
                "train_end": train_end,
                "test_end": test_end,
                "predicted_sharpe": float(pred_sharpe),
                "oos_sharpe": float(oos_sharpe),
            })

            if self.mode == "anchored":
                self.train_size += self.test_size
            else:
                start += self.test_size

        oos_returns = np.array(oos_list)
        final_oos_sharpe = self._sharpe(oos_returns) if len(oos_returns) > 0 else 0.0
        oos_sharpes = [w["oos_sharpe"] for w in window_results]
        robust = (len(oos_sharpes) >= 3 and
                  np.mean(oos_sharpes) > 0 and
                  np.median(oos_sharpes) > 0 and
                  sum(1 for s in oos_sharpes if s > 0) >= len(oos_sharpes) * 0.5)

        return WFAResult(
            oos_sharpe=final_oos_sharpe,
            oos_returns=oos_returns,
            n_windows=len(window_results),
            train_size=self.train_size,
            test_size=self.test_size,
            window_results=window_results,
            is_robust=robust,
        )
