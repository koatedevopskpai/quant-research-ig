"""
research_agent.py — Research Agent (Quant Beckman's Research phase)

The Research Agent is the starting point of Beckman's Research → Know → Do → Get
cycle. It:
    1. Accepts research requests (epic, model type, parameters)
    2. Runs a registered model adapter over the supplied price history
    3. Produces either a single-model result or a combined directional signal
    4. Optionally asks an LLM for a narrative report
    5. Records a Decision and publishes `research.completed`

WHY ADAPTERS
    Models are reached through `ModelAdapter` (see `model_contract.py`), not
    through raw result objects. The previous implementation guessed at each
    model's shape: it read `result.forecast` from a result that exposes
    `forecasts`, called `predict()` without the required input, and derived
    direction from regime *labels*. All three failed silently into zero. The
    contract now validates the output, and a missing or broken model produces an
    explicit `error` rather than a tradable-looking zero.

THE SIGNAL POLICY
    `research_signal` is the only path that turns models into a direction:

      * direction comes from a return forecast (never from a regime);
      * confidence is the forecast's own conviction, halved when the regime is
        the highest-volatility state;
      * if no forecast model is registered, the result is an error — regime
        context alone cannot open a position.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone

import json
import numpy as np
from agentic_finance.message_bus import MessageBus, AgentMessage
from agentic_finance.decision import Decision, DecisionType, DecisionStatus
from agentic_finance.audit_trail import AuditTrail

SUPPORTED_MODELS = ("regime", "forecast", "lstm", "sentiment")


@dataclass
class ResearchRequest:
    epic: str
    model_type: str  # "regime", "forecast", "lstm", "sentiment"
    parameters: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""


@dataclass
class ResearchResult:
    epic: str
    model_type: str
    model_output: dict[str, Any]
    summary: str
    llm_report: str = ""
    request_id: str = ""


class ResearchAgent:
    def __init__(self, message_bus: MessageBus, audit_trail: AuditTrail,
                 llm_api_key: str | None = None):
        self.message_bus = message_bus
        self.audit_trail = audit_trail
        self.llm_api_key = llm_api_key
        self._models: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_model(self, model_type: str, adapter: Any) -> None:
        """
        Register a model adapter.

        Rejects objects that do not expose `name`, `fit` and `predict`, because
        accepting them is exactly how a silently-broken model gets wired in.
        """
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown model_type: {model_type}. Expected one of {SUPPORTED_MODELS}"
            )
        missing = [attr for attr in ("name", "fit", "predict") if not hasattr(adapter, attr)]
        if missing:
            raise TypeError(
                f"{model_type} adapter is missing {missing}; register a ModelAdapter "
                "(see model_contract.py), not a raw model object"
            )
        self._models[model_type] = adapter

    def registered_models(self) -> list[str]:
        return sorted(self._models)

    # ------------------------------------------------------------------
    # Model execution
    # ------------------------------------------------------------------

    def _predict(self, model_type: str, prices: Any) -> dict[str, Any]:
        """
        Fit the adapter on the supplied history, then predict.

        WHY FIT HERE: research runs on the latest window, so each call is a
        refit followed by a prediction. Keeping that ordering in one place means
        an adapter can never be queried unfitted; the previous code's
        `fit(...).predict(...)` chain made the same assumption but without a
        contract around the result.
        """
        adapter = self._models.get(model_type)
        if adapter is None:
            return {"error": f"{model_type} model not registered"}
        try:
            adapter.fit(prices)
            output = adapter.predict(prices)
            return output.to_dict()
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _predict_combined(self, epic: str, prices: Any) -> dict[str, Any]:
        outputs = {name: self._predict(name, prices) for name in ("forecast", "regime")}
        forecast = outputs.get("forecast", {})
        regime = outputs.get("regime", {})

        if "error" in forecast:
            return {
                "error": (
                    "a directional signal requires a forecast model: "
                    f"{forecast['error']}"
                ),
                "models": outputs,
            }

        direction = float(forecast["direction"])
        confidence = float(forecast["confidence"])

        multiplier = 1.0
        if "error" not in regime and regime.get("regime") is not None:
            n_regimes = regime.get("diagnostics", {}).get("n_regimes")
            if n_regimes and int(regime["regime"]) == int(n_regimes) - 1:
                # Highest-volatility state: same forecast, less conviction.
                multiplier = 0.5
        confidence = max(0.0, min(1.0, confidence * multiplier))

        return {
            "direction": direction,
            "confidence": confidence,
            "expected_return": forecast.get("expected_return"),
            "regime": regime.get("regime") if "error" not in regime else None,
            "regime_label": regime.get("regime_label") if "error" not in regime else None,
            "regime_confidence_multiplier": multiplier,
            "models": outputs,
        }

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def _generate_summary(self, request: ResearchRequest, output: dict[str, Any]) -> str:
        if "error" in output:
            return f"Research failed for {request.epic} ({request.model_type}): {output['error']}"
        parts = [f"Research Result — {request.epic} ({request.model_type})"]
        for k, v in output.items():
            if k != "error":
                parts.append(f"  {k}: {v}")
        return "\n".join(parts)

    def _record(self, request: ResearchRequest, output: dict[str, Any],
                summary: str, llm_report: str) -> None:
        decision = Decision.now(
            agent="ResearchAgent",
            decision_type=DecisionType.RESEARCH,
            description=f"Research {request.model_type} on {request.epic}",
            context={"epic": request.epic, "model_type": request.model_type},
            reasoning=summary,
            llm_explanation=llm_report,
        )
        decision.status = DecisionStatus.EXECUTED
        self.audit_trail.record(decision,
                                tags=["research", request.model_type, request.epic])

    def _publish(self, request: ResearchRequest, output: dict[str, Any],
                 summary: str) -> None:
        self.message_bus.publish_topic(
            "research.completed", "ResearchAgent",
            {"epic": request.epic, "model_type": request.model_type,
             "output": output, "summary": summary},
        )

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

    def research(self, request: ResearchRequest, prices: Any,
                 sentiment_data: Any = None) -> ResearchResult:
        """Run one registered model. Errors are returned, not raised."""
        if request.model_type not in SUPPORTED_MODELS:
            raise ValueError(f"Unknown model_type: {request.model_type}")

        output = self._predict(request.model_type, prices)
        summary = self._generate_summary(request, output)
        llm_report = self._llm_report(request, output) if self.llm_api_key else ""

        result = ResearchResult(
            epic=request.epic,
            model_type=request.model_type,
            model_output=output,
            summary=summary,
            llm_report=llm_report,
            request_id=request.request_id,
        )
        self._record(request, output, summary, llm_report)
        self._publish(request, output, summary)
        return result

    def research_signal(self, epic: str, prices: Any,
                        request_id: str = "", publish: bool = True) -> ResearchResult:
        """
        Combine forecast and regime into one directional signal.

        The result's `model_output` always carries `direction` and `confidence`
        (0.0 when there is no usable forecast) plus the per-model details, so the
        execution path never has to guess which model produced what.
        """
        request = ResearchRequest(epic=epic, model_type="signal",
                                  request_id=request_id)
        output = self._predict_combined(epic, prices)
        if "error" not in output and output.get("direction") == 0.0:
            # A zero direction is not an error, but it is not a signal either.
            output["note"] = "no directional view"
        summary = self._generate_summary(request, output)
        llm_report = self._llm_report(request, output) if self.llm_api_key else ""

        result = ResearchResult(
            epic=epic,
            model_type="signal",
            model_output=output,
            summary=summary,
            llm_report=llm_report,
            request_id=request_id,
        )
        self._record(request, output, summary, llm_report)
        if publish:
            # The graph owns the decision to act, so it suppresses the bus
            # message and executes explicitly. External callers keep the message.
            self._publish(request, output, summary)
        return result

    def _llm_report(self, request: ResearchRequest, output: dict[str, Any]) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.llm_api_key)
            prompt = (
                f"You are a quantitative research analyst. "
                f"Summarise the following {request.model_type} analysis for {request.epic}:\n"
                f"{json.dumps(output, indent=2, default=str)}\n"
                f"Provide a clear, actionable interpretation for a portfolio manager."
            )
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            return f"[LLM report unavailable: {e}]"
