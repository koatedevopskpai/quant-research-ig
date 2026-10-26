"""Print deterministic numerical and model-fold evidence for Phase 2."""

import json
import math

import numpy as np
from quant_methods.validation.cpcv import CombinatorialPurgedCV
from quant_methods.validation.dsr import expected_maximum_sharpe
from quant_methods.validation.psr import ProbabilisticSharpeRatio, probability_from_moments
from quant_methods.validation.wfa import WalkForwardAnalysis


class MeanModel:
    def fit(self, x, y):
        self.mean = float(y.mean())
        return self

    def predict(self, x):
        return np.full(len(x), self.mean)


def main():
    r = np.random.default_rng(42).normal(size=252)
    r = ((r - r.mean()) / r.std() + 0.02) * 0.01
    psr = ProbabilisticSharpeRatio().compute(r)
    benchmark = expected_maximum_sharpe(0.002, 100)
    dsr, _ = probability_from_moments(2.5 / math.sqrt(250), benchmark, 1250, -3, 10)
    assert 0.62 < psr.psr < 0.63
    assert abs(dsr - 0.9004) < 5e-5

    x = np.arange(60)[:, None]
    y = np.linspace(-0.02, 0.03, 60)
    ends = np.minimum(np.arange(60) + 2, 59)

    def returns_fn(realised, predictions):
        return np.sign(predictions) * realised

    cv = CombinatorialPurgedCV(6, 2, embargo=1).run(x, y, MeanModel, returns_fn, label_end=ends)
    walk = WalkForwardAnalysis(20, 10, gap=2).run(x, y, MeanModel, returns_fn, label_end=ends)
    assert cv.n_combinations == 15
    assert all(w["train_indices"].max() < w["test_indices"].min() for w in walk.window_results)
    print(
        json.dumps(
            {
                "modest_sample_annual_sharpe": psr.sharpe_ratio,
                "modest_sample_psr": psr.psr,
                "paper_daily_benchmark": benchmark,
                "paper_dsr": dsr,
                "cpcv_model_fits": cv.n_combinations,
                "walk_forward_model_fits": walk.n_windows,
                "walk_forward_unique_oos_rows": len(np.unique(walk.oos_indices)),
                "scope": (
                    "Synthetic verification of calculations and fold orchestration; "
                    "no strategy claim"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
