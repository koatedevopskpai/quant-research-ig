"""Independent numerical references and causality checks for Phase 2."""

import math

import numpy as np
import pandas as pd
import pytest
from quant_methods.validation.backtester import Backtester
from quant_methods.validation.dsr import DeflatedSharpeRatio, expected_maximum_sharpe
from quant_methods.validation.psr import ProbabilisticSharpeRatio, probability_from_moments


def modest_returns():
    r = np.random.default_rng(42).normal(size=252)
    return ((r - r.mean()) / r.std() + 0.02) * 0.01


def test_psr_matches_independent_central_moment_calculation():
    r = modest_returns()
    mean = sum(r) / len(r)
    deviations = [v - mean for v in r]
    variance = sum(v * v for v in deviations) / (len(r) - 1)
    m2 = sum(v * v for v in deviations) / len(r)
    skew = sum(v**3 for v in deviations) / len(r) / m2**1.5
    kurt = sum(v**4 for v in deviations) / len(r) / m2**2
    sr = mean / math.sqrt(variance)
    z = sr * math.sqrt(len(r) - 1) / math.sqrt(1 - skew * sr + (kurt - 1) * sr * sr / 4)
    reference = (1 + math.erf(z / math.sqrt(2))) / 2
    result = ProbabilisticSharpeRatio().compute(r)
    assert result.psr == pytest.approx(reference, abs=1e-12)
    assert 0.62 < result.psr < 0.63
    assert result.is_significant is False
    assert result.kurtosis == pytest.approx(kurt)


def test_frequency_changes_reporting_but_not_inference():
    r = modest_returns()
    daily = ProbabilisticSharpeRatio(periods_per_year=252).compute(r, threshold_sharpe=0.3)
    native = ProbabilisticSharpeRatio(periods_per_year=1).compute(
        r, threshold_sharpe=0.3 / math.sqrt(252)
    )
    assert daily.psr == pytest.approx(native.psr)
    assert daily.sharpe_ratio == pytest.approx(native.sharpe_ratio * math.sqrt(252))


def test_seeded_zero_skill_samples_do_not_routinely_pass_significance():
    samples = np.random.default_rng(2026).normal(0, 0.01, size=(1000, 252))
    psr = ProbabilisticSharpeRatio()
    fraction = np.mean([psr.compute(sample).is_significant for sample in samples])
    assert 0.03 < fraction < 0.08


def test_risk_free_rate_is_subtracted_once_and_is_not_a_sharpe_threshold():
    r = modest_returns()
    a = ProbabilisticSharpeRatio(0.05).compute(r)
    b = ProbabilisticSharpeRatio().compute(r - 0.05 / 252)
    assert a.threshold_sharpe == 0
    assert a.psr == pytest.approx(b.psr)


def test_published_dsr_numerical_example():
    # Bailey & Lopez de Prado 2014, pp. 9-10. Ordinary kurtosis; daily units.
    benchmark = expected_maximum_sharpe(0.002, 100)
    probability, _ = probability_from_moments(2.5 / math.sqrt(250), benchmark, 1250, -3, 10)
    assert benchmark == pytest.approx(0.1132, abs=5e-5)
    assert probability == pytest.approx(0.9004, abs=5e-5)
    benchmark_46 = expected_maximum_sharpe(0.002, 46)
    probability_46, _ = probability_from_moments(2.5 / math.sqrt(250), benchmark_46, 1250, -3, 10)
    assert probability_46 == pytest.approx(0.9505, abs=5e-5)


def test_multi_trial_dsr_requires_measured_trial_dispersion():
    with pytest.raises(ValueError, match="Multiple trials require"):
        DeflatedSharpeRatio().compute(modest_returns(), num_trials=100)
    trials = [-0.4, 0.2, 0.6, 1.0]
    result = DeflatedSharpeRatio().compute(modest_returns(), trial_sharpes=trials)
    explicit = DeflatedSharpeRatio().compute(
        modest_returns(), num_trials=4, trial_sharpe_variance=np.var(trials, ddof=1)
    )
    assert result.variance_source == "empirical"
    assert result.dsr == pytest.approx(explicit.dsr)
    assert result.dsr <= result.psr


def test_single_trial_is_exactly_psr_and_has_no_infinite_quantile():
    result = DeflatedSharpeRatio().compute(modest_returns())
    assert result.dsr == result.psr
    assert result.expected_max_sr == 0
    below_zero = DeflatedSharpeRatio().compute(modest_returns(), threshold_sharpe=-0.2)
    assert below_zero.dsr == below_zero.psr


def test_dsr_converts_trial_variance_and_benchmark_consistently():
    r = modest_returns()
    annual = DeflatedSharpeRatio(periods_per_year=252).compute(
        r, num_trials=100, trial_sharpe_variance=0.5
    )
    native = DeflatedSharpeRatio(periods_per_year=1).compute(
        r, num_trials=100, trial_sharpe_variance=0.5 / 252
    )
    assert annual.dsr == pytest.approx(native.dsr)
    assert annual.expected_max_sr == pytest.approx(native.expected_max_sr * math.sqrt(252))


@pytest.mark.parametrize(
    "returns", [np.ones(10), [0.1, np.nan, 0.2], [0.1, np.inf, 0.2], [[1, 2, 3]]]
)
def test_undefined_sharpe_inference_rejected(returns):
    with pytest.raises(ValueError):
        ProbabilisticSharpeRatio().compute(returns)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_trials=0),
        dict(num_trials=1.5),
        dict(num_trials=2, trial_sharpe_variance=-1),
        dict(num_trials=2, trial_sharpe_variance=np.nan),
        dict(num_trials=3, trial_sharpes=[0.1, 0.2]),
    ],
)
def test_invalid_dsr_inputs_rejected(kwargs):
    with pytest.raises(ValueError):
        DeflatedSharpeRatio().compute(modest_returns(), **kwargs)


def test_backtest_prefix_is_invariant_to_future_prices_and_sample_length():
    r = modest_returns()
    prices = pd.Series(100 * np.cumprod(1 + r[:100]))
    changed = prices.copy()
    changed.iloc[60:] *= np.exp(np.linspace(0, 2, 40))
    bt = Backtester()
    a = bt.run(prices, pd.Series(1.0, index=prices.index), weight_scheme="vol_target")
    b = bt.run(changed, pd.Series(1.0, index=prices.index), weight_scheme="vol_target")
    prefix = bt.run(
        prices.iloc[:60], pd.Series(1.0, index=prices.index[:60]), weight_scheme="vol_target"
    )
    np.testing.assert_allclose(a.returns.iloc[:60], b.returns.iloc[:60])
    np.testing.assert_allclose(a.returns.iloc[:60], prefix.returns)
    assert (a.returns.iloc[:20] == 0).all()
