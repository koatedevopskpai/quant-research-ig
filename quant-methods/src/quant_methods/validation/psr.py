"""PSR with per-observation inference and annualised public Sharpe values.

Bailey & Lopez de Prado (2014), Eq. 2:
https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
Kurtosis is ordinary (Gaussian = 3), not excess. Returns must be finite,
nonconstant, equally spaced observations. Serial dependence is not corrected.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from scipy import stats


def probability_from_moments(sharpe, threshold, n, skewness, kurtosis):
    """Return (probability, z) from per-observation Sharpe and ordinary moments."""
    if not isinstance(n, Integral) or n < 3:
        raise ValueError("Need >=3 observations")
    if not np.isfinite([sharpe, threshold, skewness, kurtosis]).all() or kurtosis < 1:
        raise ValueError("Sharpe values and moments must be finite; kurtosis must be >=1")
    variance_term = 1 - skewness * sharpe + (kurtosis - 1) * sharpe**2 / 4
    if variance_term <= 0:
        raise ValueError("Sharpe sampling variance must be positive")
    z = float((sharpe - threshold) * np.sqrt(n - 1) / np.sqrt(variance_term))
    return float(stats.norm.cdf(z)), z


@dataclass
class PSRResult:
    psr: float
    sharpe_ratio: float  # annualised, for reporting
    threshold_sharpe: float  # annualised, for reporting
    skewness: float
    kurtosis: float  # ordinary kurtosis
    n_observations: int
    is_significant: bool
    z_stat: float


class ProbabilisticSharpeRatio:
    def __init__(self, risk_free_rate=0.0, periods_per_year=252):
        if not isinstance(periods_per_year, Integral) or periods_per_year <= 0:
            raise ValueError("periods_per_year must be a positive integer")
        if not np.isfinite(risk_free_rate):
            raise ValueError("risk_free_rate must be finite")
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year

    def compute(self, returns, threshold_sharpe=None, *, significance_level=0.05):
        """Test against an annualised Sharpe benchmark (default 0, above RF).

        RF is an annual arithmetic return, subtracted as RF/periods_per_year.
        The sample standard deviation uses ddof=1; central moments use the
        empirical population-moment convention (scipy bias=True).
        """
        values = np.asarray(returns, dtype=float)
        if values.ndim != 1 or len(values) < 3:
            raise ValueError("Need >=3 observations in a one-dimensional return series")
        if not np.isfinite(values).all():
            raise ValueError("returns must be finite")
        threshold = 0.0 if threshold_sharpe is None else float(threshold_sharpe)
        if not np.isfinite(threshold) or not 0 < significance_level < 1:
            raise ValueError("Invalid threshold or significance_level")
        excess = values - self.risk_free_rate / self.periods_per_year
        volatility = float(excess.std(ddof=1))
        if volatility <= 0:
            raise ValueError("Sharpe inference requires nonconstant returns")
        sr = float(excess.mean() / volatility)
        skew = float(stats.skew(excess, bias=True))
        kurt = float(stats.kurtosis(excess, fisher=False, bias=True))
        scale = np.sqrt(self.periods_per_year)
        probability, z = probability_from_moments(sr, threshold / scale, len(values), skew, kurt)
        return PSRResult(
            probability,
            sr * scale,
            threshold,
            skew,
            kurt,
            len(values),
            bool(probability >= 1 - significance_level),
            z,
        )
