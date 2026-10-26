"""
experiment.py — one fixed, reproducible research experiment with promotion gates.

WHAT THIS IS
    A single documented experiment, not a platform demo: a deterministic daily
    dataset, a momentum rule with a lookback selected on train+validation, one
    chronological holdout, transaction costs, a buy-and-hold benchmark, and
    phase-2 falsification tools (PSR, DSR, monkey test) acting as promotion
    gates.

WHAT IT IS NOT
    Evidence of a tradeable edge. The dataset is synthetic and fixed by seed so
    that another person can reproduce the run exactly; that makes it a
    reproducible *method*, not a market claim. The report says so, and the gate
    result is reported as-is — including "not promoted".

WHY A CHRONOLOGICAL HOLDOUT
    Selecting a lookback on data you later report is the most common way to
    manufacture a result. Selection sees train+validation only; the holdout is
    touched once, after the parameter is frozen.

RUN
    python -m agentic_finance.experiment
    python -m agentic_finance.experiment --json evidence/experiment.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_methods.validation.backtester import Backtester
from quant_methods.validation.dsr import DeflatedSharpeRatio
from quant_methods.validation.monkey_test import MonkeyTest
from quant_methods.validation.psr import ProbabilisticSharpeRatio


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 20261026
    years: int = 6
    start_price: float = 100.0
    cost_bps: float = 2.0
    lookbacks: tuple[int, ...] = (5, 10, 20, 40)
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    psr_threshold: float = 0.95
    dsr_threshold: float = 0.95
    monkey_p_threshold: float = 0.05
    max_drawdown_threshold: float = 0.25


def generate_prices(config: ExperimentConfig) -> pd.Series:
    """
    Deterministic daily prices with two volatility regimes.

    A two-state Markov chain (calm / stressed) drives volatility so that the
    holdout contains both, rather than a single lucky regime.
    """
    rng = np.random.default_rng(config.seed)
    n = int(config.years * 252)
    state = 0
    vols = (0.008, 0.020)
    means = (0.00035, -0.00010)
    returns = np.empty(n)
    for i in range(n):
        if state == 0 and rng.random() < 0.02:
            state = 1
        elif state == 1 and rng.random() < 0.05:
            state = 0
        returns[i] = rng.normal(means[state], vols[state])
    index = pd.bdate_range("2020-01-01", periods=n, name="date")
    return pd.Series(config.start_price * np.exp(np.cumsum(returns)), index=index, name="close")


def momentum_signal(prices: pd.Series, lookback: int) -> pd.Series:
    """Position at t is the sign of the trailing lookback return, known at t."""
    signal = np.sign(prices.pct_change(lookback))
    return signal.fillna(0.0).clip(-1.0, 1.0)


def _metrics(prices: pd.Series, signal: pd.Series, cost_bps: float,
             benchmark: pd.Series | None = None) -> tuple[dict[str, Any], pd.Series]:
    backtester = Backtester(transaction_cost_bps=cost_bps)
    result = backtester.run(prices, signal, benchmark=benchmark, weight_scheme="signal")
    returns = result.returns
    return (
        {
            "total_return": float(result.total_return),
            "annualised_return": float(result.annualised_return),
            "annualised_volatility": float(result.annualised_volatility),
            "sharpe_ratio": float(result.sharpe_ratio),
            "max_drawdown": float(result.max_drawdown),
            "turnover_trades": int((signal.diff().abs() > 1e-12).sum()),
            "observations": int(len(returns)),
            "cost_bps": cost_bps,
        },
        returns,
    )


def _promotion_gates(holdout_returns: pd.Series, holdout_metrics: dict[str, Any],
                     benchmark_metrics: dict[str, Any], trial_sharpes: list[float],
                     config: ExperimentConfig) -> dict[str, Any]:
    psr = ProbabilisticSharpeRatio().compute(holdout_returns.to_numpy())
    # DSR needs the dispersion across the trials actually searched (the lookback
    # grid), not a variance invented from the winner.
    dsr = DeflatedSharpeRatio().compute(
        holdout_returns.to_numpy(), trial_sharpes=trial_sharpes
    )
    monkey = MonkeyTest(n_permutations=1000, random_seed=config.seed).run(
        holdout_returns.to_numpy()
    )

    gates = {
        "positive_sharpe": holdout_metrics["sharpe_ratio"] > 0.0,
        "beats_benchmark": holdout_metrics["sharpe_ratio"] > benchmark_metrics["sharpe_ratio"],
        "psr": psr.psr >= config.psr_threshold,
        "dsr": dsr.dsr >= config.dsr_threshold,
        "monkey_test": monkey.p_value < config.monkey_p_threshold,
        "drawdown_within_limit": abs(holdout_metrics["max_drawdown"])
        <= config.max_drawdown_threshold,
    }
    detail = {
        "psr": float(psr.psr),
        "dsr": float(dsr.dsr),
        "expected_max_sharpe": float(dsr.expected_max_sr),
        "monkey_p_value": float(monkey.p_value),
        "n_trials": len(trial_sharpes),
        "thresholds": {
            "psr": config.psr_threshold,
            "dsr": config.dsr_threshold,
            "monkey_p": config.monkey_p_threshold,
            "max_drawdown": config.max_drawdown_threshold,
        },
    }
    return {"gates": gates, "detail": detail, "promoted": all(gates.values())}


def _digest(prices: pd.Series) -> str:
    payload = ",".join(f"{value:.10f}" for value in prices.to_numpy())
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _revision() -> str:
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def run_experiment(config: ExperimentConfig = ExperimentConfig()) -> dict[str, Any]:
    prices = generate_prices(config)
    n = len(prices)
    train_end = int(n * config.train_fraction)
    validation_end = int(n * (config.train_fraction + config.validation_fraction))

    # Selection sees train + validation only.
    selection_prices = prices.iloc[:validation_end]
    best: tuple[float, int] | None = None
    trial_sharpes: list[float] = []
    for lookback in config.lookbacks:
        signal = momentum_signal(selection_prices, lookback)
        metrics, _ = _metrics(selection_prices, signal, config.cost_bps)
        trial_sharpes.append(metrics["sharpe_ratio"])
        if best is None or metrics["sharpe_ratio"] > best[0]:
            best = (metrics["sharpe_ratio"], lookback)
    assert best is not None
    selected_lookback = best[1]

    # Freeze the parameter, then evaluate the holdout once.
    holdout_prices = prices.iloc[validation_end:]
    holdout_signal = momentum_signal(prices, selected_lookback).iloc[validation_end:]
    benchmark_returns = holdout_prices.pct_change().fillna(0.0)
    benchmark_metrics, _ = _metrics(
        holdout_prices, pd.Series(1.0, index=holdout_prices.index),
        config.cost_bps, benchmark=holdout_prices,
    )
    strategy_metrics, strategy_returns = _metrics(
        holdout_prices, holdout_signal, config.cost_bps, benchmark=holdout_prices
    )

    gate_report = _promotion_gates(
        strategy_returns, strategy_metrics, benchmark_metrics,
        trial_sharpes=trial_sharpes, config=config,
    )

    return {
        "experiment": "momentum-lookback-holdout",
        "revision": _revision(),
        "seed": config.seed,
        "dataset": {
            "n_observations": n,
            "digest": _digest(prices),
            "start": str(prices.index[0].date()),
            "end": str(prices.index[-1].date()),
            "synthetic": True,
        },
        "splits": {
            "train_end": str(prices.index[train_end - 1].date()),
            "validation_end": str(prices.index[validation_end - 1].date()),
            "holdout_start": str(prices.index[validation_end].date()),
            "holdout_observations": int(len(holdout_prices)),
        },
        "selection": {
            "lookback": selected_lookback,
            "selection_sharpe": best[0],
            "grid": list(config.lookbacks),
        },
        "holdout": {
            "strategy": strategy_metrics,
            "benchmark": benchmark_metrics,
        },
        "promotion": gate_report,
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "platform": platform.platform(),
        },
        "limitations": [
            "Dataset is synthetic and fixed by seed; this demonstrates a "
            "reproducible method, not a market edge.",
            "A single chronological holdout is one sample; the gate can pass by "
            "luck. Phase-2 CPCV/WFA exist for a broader assessment.",
            "No slippage model beyond a flat bps cost; no borrow/financing.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--cost-bps", type=float, default=2.0)
    args = parser.parse_args()

    report = run_experiment(ExperimentConfig(cost_bps=args.cost_bps))
    print(json.dumps(report, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["promotion"]["promoted"]:
        print("\nResult: NOT PROMOTED (see gates)", flush=True)


if __name__ == "__main__":
    main()
