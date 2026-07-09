"""
validation/dsr.py — Deflated Sharpe Ratio (DSR)

Bailey & López de Prado (2014). The DSR corrects the PSR for multiple
testing — the fact that we tried many strategies and picked the best one.

Key insight: if you test 1000 random strategies, the best one will have
a high Sharpe by chance alone. The DSR estimates the expected maximum
Sharpe under the null and deflates the PSR accordingly.

DSR(SR*) = PSR(E[max(SR)])

where E[max(SR)] is the expected maximum Sharpe ratio over N trials,
computed using the Euler-Mascheroni approximation for the maximum of
a normal distribution.

SR* = sqrt(V(SR)) * ( (1 - gamma) * Z^-1(1 - 1/N) + gamma * Z^-1(1 - 1/N * e^-1) )

This accounts for:
    - Number of trials (strategies tried)
    - Length of each trial (observations per strategy)
    - Variance of the Sharpe ratio estimate
    - Non-normality of returns (skewness, kurtosis)
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class DSRResult:
    dsr: float
    psr: float
    expected_max_sr: float
    deflated_threshold: float
    num_trials: int
    n_observations: int
    is_significant: bool


class DeflatedSharpeRatio:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def _sharpe_variance(self, n: int, skew: float, kurt: float, sr: float) -> float:
        return (1 + 0.5 * sr ** 2 - skew * sr + (kurt - 1) / 4 * sr ** 2) / (n - 1)

    def _expected_max_sharpe(self, n: int, num_trials: int, skew: float,
                              kurt: float, sr_hat: float) -> float:
        var_sr = self._sharpe_variance(n, skew, kurt, sr_hat)
        if var_sr <= 0:
            return 0.0
        se = np.sqrt(var_sr)
        emc = 0.5772156649
        z_max = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1.0 / num_trials) \
                + np.euler_gamma * stats.norm.ppf(1 - 1.0 / (num_trials * np.e))
        return se * z_max

    def compute(self, returns: np.ndarray, num_trials: int = 1,
                threshold_sharpe: Optional[float] = None) -> DSRResult:
        from quant_methods.validation.psr import ProbabilisticSharpeRatio
        returns = np.asarray(returns, dtype=float).flatten()
        if len(returns) < 3:
            raise ValueError(f"Need >=3 observations, got {len(returns)}")
        psr = ProbabilisticSharpeRatio(self.risk_free_rate, self.periods_per_year)
        psr_result = psr.compute(returns, threshold_sharpe)

        if threshold_sharpe is None:
            threshold_sharpe = self.risk_free_rate

        n = len(returns)
        excess = returns - self.risk_free_rate / self.periods_per_year
        sr_hat = (excess.mean() / excess.std() * np.sqrt(self.periods_per_year)
                  if excess.std() > 0 else 0.0)
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns, fisher=True))
        e_max_sr = self._expected_max_sharpe(n, num_trials, skew, kurt, sr_hat)
        deflated_threshold = max(threshold_sharpe, e_max_sr)

        dsr_result = psr.compute(returns, deflated_threshold)

        return DSRResult(
            dsr=dsr_result.psr,
            psr=psr_result.psr,
            expected_max_sr=e_max_sr,
            deflated_threshold=deflated_threshold,
            num_trials=num_trials,
            n_observations=n,
            is_significant=dsr_result.psr >= 0.95,
        )
