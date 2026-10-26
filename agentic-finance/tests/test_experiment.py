"""
Experiment tests: reproducibility, causality, costs and gate arithmetic.

The experiment is only defensible if the same seed gives the same numbers, the
signal cannot see the future, costs actually reduce returns, and the promotion
flag is the conjunction of its gates.
"""

import numpy as np
import pandas as pd
import pytest

from agentic_finance.experiment import (
    ExperimentConfig,
    generate_prices,
    momentum_signal,
    run_experiment,
)


class TestReproducibility:
    def test_same_seed_gives_identical_numbers(self):
        first = run_experiment(ExperimentConfig(seed=7))
        second = run_experiment(ExperimentConfig(seed=7))

        assert first["dataset"]["digest"] == second["dataset"]["digest"]
        assert first["selection"] == second["selection"]
        assert first["holdout"]["strategy"]["sharpe_ratio"] == pytest.approx(
            second["holdout"]["strategy"]["sharpe_ratio"]
        )
        assert first["promotion"]["gates"] == second["promotion"]["gates"]
        assert first["promotion"]["promoted"] == second["promotion"]["promoted"]

    def test_different_seed_changes_the_dataset(self):
        a = run_experiment(ExperimentConfig(seed=1))
        b = run_experiment(ExperimentConfig(seed=2))
        assert a["dataset"]["digest"] != b["dataset"]["digest"]

    def test_report_is_self_describing(self):
        report = run_experiment(ExperimentConfig(seed=3))
        assert report["dataset"]["synthetic"] is True
        assert report["dataset"]["n_observations"] > 0
        assert report["splits"]["holdout_observations"] > 0
        assert report["limitations"]
        assert "revision" in report


class TestCausality:
    def test_signal_prefix_is_invariant_to_future_prices(self):
        prices = generate_prices(ExperimentConfig(seed=11, years=2))
        lookback = 10
        full = momentum_signal(prices, lookback)
        k = 150
        prefix = momentum_signal(prices.iloc[:k], lookback)
        np.testing.assert_allclose(prefix.to_numpy(), full.iloc[:k].to_numpy())

    def test_holdout_starts_after_validation(self):
        report = run_experiment(ExperimentConfig(seed=5))
        assert report["splits"]["holdout_start"] > report["splits"]["validation_end"]


class TestCosts:
    def test_costs_reduce_holdout_return(self):
        base = ExperimentConfig(seed=9, cost_bps=0.0)
        expensive = ExperimentConfig(seed=9, cost_bps=25.0)

        cheap_return = run_experiment(base)["holdout"]["strategy"]["total_return"]
        costly_return = run_experiment(expensive)["holdout"]["strategy"]["total_return"]

        assert costly_return < cheap_return


class TestGates:
    def test_promotion_is_the_conjunction_of_gates(self):
        report = run_experiment(ExperimentConfig(seed=13))
        gates = report["promotion"]["gates"]
        assert report["promotion"]["promoted"] == all(gates.values())
        assert set(gates) == {
            "positive_sharpe", "beats_benchmark", "psr", "dsr",
            "monkey_test", "drawdown_within_limit",
        }
        assert 0.0 <= report["promotion"]["detail"]["psr"] <= 1.0
        assert 0.0 <= report["promotion"]["detail"]["monkey_p_value"] <= 1.0

    def test_gate_detail_exposes_thresholds_and_trial_count(self):
        detail = run_experiment(ExperimentConfig(seed=17))["promotion"]["detail"]
        assert detail["n_trials"] == len(ExperimentConfig().lookbacks)
        assert detail["thresholds"]["psr"] == pytest.approx(0.95)
