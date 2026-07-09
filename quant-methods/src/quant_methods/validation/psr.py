"""
validation/psr.py — Probabilistic Sharpe Ratio (PSR)

Bailey & López de Prado (2012). The PSR answers: "Given my track record,
what is the probability that the true Sharpe ratio exceeds a given
threshold (typically 0)?"

PSR(SR*) = Phi( (SR_hat - SR*) * sqrt(N-1) / sqrt(1 - gamma_3 * SR_hat + (gamma_4 - 1)/4 * SR_hat^2) )

where:
    SR_hat  = estimated Sharpe ratio (annualised)
    SR*     = threshold Sharpe ratio (typically 0, or the risk-free)
    N       = number of observations
    gamma_3 = skewness of returns
    gamma_4 = excess kurtosis of returns
    Phi     = standard normal CDF

The PSR penalises strategies with high skewness (right-tail risk) and
negative skewness (losses cluster), and high kurtosis (fat tails).
"""

import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Optional


@dataclass
class PSRResult:
    psr: float
    sharpe_ratio: float
    threshold_sharpe: float
    skewness: float
    kurtosis: float
    n_observations: int
    is_significant: bool
    z_stat: float


class ProbabilisticSharpeRatio:
    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def compute(self, returns: np.ndarray,
                threshold_sharpe: Optional[float] = None) -> PSRResult:
        returns = np.asarray(returns, dtype=float).flatten()
        if len(returns) < 3:
            raise ValueError(f"Need >=3 observations, got {len(returns)}")
        excess = returns - self.risk_free_rate / self.periods_per_year
        sr = excess.mean() / excess.std() * np.sqrt(self.periods_per_year) if excess.std() > 0 else 0.0
        if threshold_sharpe is None:
            threshold_sharpe = self.risk_free_rate
        skew = float(stats.skew(returns))
        kurt = float(stats.kurtosis(returns, fisher=True))
        n = len(returns)
        num = (sr - threshold_sharpe) * np.sqrt(n - 1)
        var_term = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
        denom = np.sqrt(max(var_term, 1e-10))
        z = float(num / denom)
        psr_val = float(stats.norm.cdf(z))
        return PSRResult(
            psr=psr_val,
            sharpe_ratio=sr,
            threshold_sharpe=threshold_sharpe,
            skewness=skew,
            kurtosis=kurt,
            n_observations=n,
            is_significant=psr_val >= 0.95,
            z_stat=z,
        )
