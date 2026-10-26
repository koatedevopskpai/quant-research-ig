"""
model_contract.py — the contract between a research model and the trading loop.

WHY THIS EXISTS
    The research path used to speak to models through duck-typed dicts:
    `_run_forecast` read `result.forecast` while `ForecastResult` exposes
    `forecasts`, so a successful fit produced an empty list and a prediction mean
    of zero — silently. `_run_regime` called `.pooled_mean()` on whatever it was
    given, so a NumPy array (no `.values`) or a mismatched `predict()` signature
    failed inside a try/except and became another zero. The runner then derived
    direction from `regime_labels[-1] == 1` (labels are strings, not states) and
    confidence from `prediction_mean` (which regime results do not have).

    Every one of those is a contract failure, not a model failure. This module
    makes the contract explicit and validates it at construction time.

THE SIGNAL POLICY (why a regime cannot be a direction)
    A volatility regime is context: it says how much risk is appropriate, not
    which way to trade. Only a model that predicts a return may produce a
    direction. Therefore:

      * `direction` is +1 / -1 from the sign of an expected return, or 0 when the
        model has no directional view (a regime model always returns 0);
      * `confidence` is in [0, 1] and is the model's own conviction. For a
        forecast it is |expected_return| / (k * recent volatility), clipped:
        a two-sigma expected move is full confidence;
      * a directional output MUST carry `expected_return`, so a direction can
        never be fabricated from a label;
      * downstream, the regime may only scale confidence (high-volatility
        regime halves it in `ResearchAgent.research_signal`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class ModelOutput:
    """What every model must return. No free-form dicts at the boundary."""

    model: str
    direction: float                    # -1.0 / 0.0 / +1.0
    confidence: float                   # 0.0 .. 1.0
    expected_return: float | None = None
    regime: int | None = None           # numeric state, not a label
    regime_label: str | None = None     # human-readable description
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be a non-empty name")
        if not math.isfinite(self.direction):
            raise ValueError("direction must be finite")
        if not -1.0 <= self.direction <= 1.0:
            raise ValueError(f"direction must be in [-1, 1], got {self.direction}")
        if not math.isfinite(self.confidence):
            raise ValueError("confidence must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.expected_return is not None and not math.isfinite(self.expected_return):
            raise ValueError("expected_return must be finite or None")
        if self.direction != 0.0 and self.expected_return is None:
            raise ValueError(
                "a directional output must carry expected_return; a direction "
                "cannot be inferred from a regime label"
            )

    @property
    def is_directional(self) -> bool:
        return self.direction != 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "direction": float(self.direction),
            "confidence": float(self.confidence),
            "expected_return": (
                None if self.expected_return is None else float(self.expected_return)
            ),
            "regime": self.regime,
            "regime_label": self.regime_label,
            "diagnostics": dict(self.diagnostics),
        }


@runtime_checkable
class ModelAdapter(Protocol):
    """A model wrapper the research agent can fit and query."""

    name: str

    def fit(self, prices: pd.Series) -> None: ...

    def predict(self, prices: pd.Series) -> ModelOutput: ...
