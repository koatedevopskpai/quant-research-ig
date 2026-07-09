"""
research_agent.py — Research Agent (Quant Beckman's Research phase)

The Research Agent is the starting point of Beckman's Research→Know→Do→Get
cycle. It:
    1. Accepts research requests (epic, model type, parameters)
    2. Fetches market data for the requested epic
    3. Runs the selected ML model (regime detection, forecasting, LSTM)
    4. Calls an LLM to generate a natural-language research report
    5. Publishes the research result to the message bus

The LLM integration is optional — without an API key, the agent falls
back to a structured summary of model outputs. This makes the agent
fully testable without external dependencies.
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from datetime import datetime, timezone

import json
import numpy as np
from agentic_finance.message_bus import MessageBus, AgentMessage
from agentic_finance.decision import Decision, DecisionType, DecisionStatus
from agentic_finance.audit_trail import AuditTrail


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

    def register_model(self, model_type: str, model: Any) -> None:
        self._models[model_type] = model

    def _run_regime(self, prices: Any) -> dict[str, Any]:
        model = self._models.get("regime")
        if model is None:
            return {"error": "Regime model not registered"}
        try:
            result = model.fit(prices.values).predict()
            labels = result.regime_labels if hasattr(result, 'regime_labels') else []
            probs = result.regime_probs.tolist() if hasattr(result, 'regime_probs') else []
            return {"regime_labels": labels, "regime_probs": probs, "n_regimes": len(labels)}
        except Exception as e:
            return {"error": str(e)}

    def _run_forecast(self, prices: Any) -> dict[str, Any]:
        model = self._models.get("forecast")
        if model is None:
            return {"error": "Forecast model not registered"}
        try:
            result = model.fit(prices).predict(prices)
            pred = result.forecast.tolist() if hasattr(result, 'forecast') else []
            return {"prediction_mean": float(np.mean(pred)) if pred else 0.0,
                    "prediction_std": float(np.std(pred)) if pred else 0.0,
                    "n_predictions": len(pred)}
        except Exception as e:
            return {"error": str(e)}

    def _run_lstm(self, prices: Any) -> dict[str, Any]:
        model = self._models.get("lstm")
        if model is None:
            return {"error": "LSTM model not registered"}
        try:
            result = model.predict(prices)
            pred = result.forecast.tolist() if hasattr(result, 'forecast') else []
            return {"prediction_mean": float(np.mean(pred)) if pred else 0.0,
                    "n_predictions": len(pred)}
        except Exception as e:
            return {"error": str(e)}

    def _run_sentiment(self, prices: Any, sentiment_data: Any = None) -> dict[str, Any]:
        model = self._models.get("sentiment")
        if model is None:
            return {"error": "Sentiment model not registered"}
        try:
            result = model.predict(prices)
            forecast = result.forecast.tolist() if hasattr(result, 'forecast') else []
            return {"prediction_mean": float(np.mean(forecast)) if forecast else 0.0,
                    "n_predictions": len(forecast)}
        except Exception as e:
            return {"error": str(e)}

    def _generate_summary(self, request: ResearchRequest, output: dict[str, Any]) -> str:
        if "error" in output:
            return f"Research failed for {request.epic} ({request.model_type}): {output['error']}"
        parts = [f"Research Result — {request.epic} ({request.model_type})"]
        for k, v in output.items():
            if k != "error":
                parts.append(f"  {k}: {v}")
        return "\n".join(parts)

    def research(self, request: ResearchRequest, prices: Any,
                 sentiment_data: Any = None) -> ResearchResult:
        runners = {
            "regime": self._run_regime,
            "forecast": self._run_forecast,
            "lstm": self._run_lstm,
            "sentiment": self._run_sentiment,
        }
        runner = runners.get(request.model_type)
        if runner is None:
            raise ValueError(f"Unknown model_type: {request.model_type}")
        output = runner(prices, sentiment_data) if request.model_type == "sentiment" else runner(prices)
        summary = self._generate_summary(request, output)
        llm_report = ""
        if self.llm_api_key:
            llm_report = self._llm_report(request, output)
        result = ResearchResult(
            epic=request.epic,
            model_type=request.model_type,
            model_output=output,
            summary=summary,
            llm_report=llm_report,
            request_id=request.request_id,
        )
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
        self.message_bus.publish_topic(
            "research.completed", "ResearchAgent",
            {"epic": request.epic, "model_type": request.model_type,
             "output": output, "summary": summary},
        )
        return result

    def _llm_report(self, request: ResearchRequest, output: dict[str, Any]) -> str:
        try:
            import openai
            client = openai.OpenAI(api_key=self.llm_api_key)
            prompt = (
                f"You are a quantitative research analyst. "
                f"Summarise the following {request.model_type} analysis for {request.epic}:\n"
                f"{json.dumps(output, indent=2)}\n"
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
