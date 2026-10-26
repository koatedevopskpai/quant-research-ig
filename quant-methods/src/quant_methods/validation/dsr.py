"""DSR with explicitly supplied dispersion across independent trials.

Public Sharpe inputs/outputs are annualised; variance inputs are therefore in
annualised Sharpe squared. Sampling variance of the selected strategy is not
a substitute for dispersion across trials. Assumes independent trial estimates.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
from quant_methods.validation.psr import ProbabilisticSharpeRatio
from scipy import stats


def expected_maximum_sharpe(variance, num_trials):
    """Expected maximum under zero-mean null, in the supplied Sharpe units."""
    if not isinstance(num_trials, Integral) or num_trials < 1:
        raise ValueError("num_trials must be a positive integer")
    if not np.isfinite(variance) or variance < 0:
        raise ValueError("Trial Sharpe variance must be finite and nonnegative")
    if num_trials == 1 or variance == 0:
        return 0.0
    # isf avoids cancellation in 1 - 1/N for large searches.
    factor = (1 - np.euler_gamma) * stats.norm.isf(
        1 / num_trials
    ) + np.euler_gamma * stats.norm.isf(1 / (num_trials * np.e))
    return float(np.sqrt(variance) * factor)


@dataclass
class DSRResult:
    dsr: float
    psr: float
    expected_max_sr: float  # annualised
    deflated_threshold: float  # annualised
    num_trials: int
    n_observations: int
    is_significant: bool
    trial_sharpe_variance: float  # annualised Sharpe squared
    variance_source: str


class DeflatedSharpeRatio:
    def __init__(self, risk_free_rate=0.0, periods_per_year=252):
        self._psr = ProbabilisticSharpeRatio(risk_free_rate, periods_per_year)

    def compute(
        self,
        returns,
        num_trials=None,
        threshold_sharpe=None,
        *,
        trial_sharpes=None,
        trial_sharpe_variance=None,
        significance_level=0.05,
    ):
        if trial_sharpes is not None and trial_sharpe_variance is not None:
            raise ValueError("Supply trial_sharpes or trial_sharpe_variance, not both")
        if trial_sharpes is not None:
            candidates = np.asarray(trial_sharpes, dtype=float)
            if candidates.ndim != 1 or len(candidates) < 2 or not np.isfinite(candidates).all():
                raise ValueError("trial_sharpes must contain at least two finite annualised values")
            if num_trials is None:
                num_trials = len(candidates)
            if num_trials != len(candidates):
                raise ValueError("num_trials must match trial_sharpes length")
            variance = float(candidates.var(ddof=1))
            source = "empirical"
        else:
            num_trials = 1 if num_trials is None else num_trials
            if trial_sharpe_variance is None:
                if num_trials != 1:
                    raise ValueError(
                        "Multiple trials require trial_sharpes or trial_sharpe_variance"
                    )
                variance, source = 0.0, "single_trial"
            else:
                variance, source = float(trial_sharpe_variance), "provided"
        maximum = expected_maximum_sharpe(variance, num_trials)
        base = self._psr.compute(returns, threshold_sharpe, significance_level=significance_level)
        threshold = (
            base.threshold_sharpe if num_trials == 1 else max(base.threshold_sharpe, maximum)
        )
        deflated = self._psr.compute(returns, threshold, significance_level=significance_level)
        return DSRResult(
            deflated.psr,
            base.psr,
            maximum,
            threshold,
            num_trials,
            base.n_observations,
            deflated.is_significant,
            variance,
            source,
        )
