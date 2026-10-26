"""
Research agent contract tests.

The old suite asserted that dicts had keys. It passed while every model path was
silently broken: `result.forecast` (absent; the field is `forecasts`), `.values`
on a ndarray, and direction derived from regime *labels*. These tests assert
behaviour: what the agent does with a working adapter, a broken one, and a
missing one.
"""

import numpy as np
import pytest

from agentic_finance.model_contract import ModelOutput
from agentic_finance.research_agent import ResearchAgent, ResearchRequest, ResearchResult


class FakeAdapter:
    def __init__(self, name, output=None, error=None):
        self.name = name
        self._output = output
        self._error = error
        self.fitted = False
        self.predictions = 0

    def fit(self, prices):
        self.fitted = True
        return self

    def predict(self, prices):
        self.predictions += 1
        if self._error:
            raise RuntimeError(self._error)
        return self._output


def forecast_adapter(expected=0.01, confidence=0.6):
    return FakeAdapter("forecast", ModelOutput(
        model="forecast", direction=float(np.sign(expected)), confidence=confidence,
        expected_return=expected,
    ))


def regime_adapter(state=0, n_regimes=3, label="Low Vol"):
    return FakeAdapter("regime", ModelOutput(
        model="regime", direction=0.0, confidence=0.0,
        regime=state, regime_label=label,
        diagnostics={"n_regimes": n_regimes, "regime_probability": 0.9},
    ))


@pytest.fixture
def agent(message_bus, audit_trail):
    return ResearchAgent(message_bus, audit_trail)


class TestRegistration:
    def test_registers_adapter(self, agent):
        adapter = forecast_adapter()
        agent.register_model("forecast", adapter)
        assert agent.registered_models() == ["forecast"]

    def test_rejects_raw_model_object(self, agent):
        class RawModel:
            def fit(self, prices):
                return self

            def predict(self, prices):
                return object()

        with pytest.raises(TypeError, match="ModelAdapter"):
            agent.register_model("forecast", RawModel())

    def test_rejects_unknown_type(self, agent):
        with pytest.raises(ValueError, match="Unknown model_type"):
            agent.register_model("magic", forecast_adapter())


class TestSingleModelResearch:
    def test_forecast_output_is_the_contract(self, agent):
        agent.register_model("forecast", forecast_adapter())
        result = agent.research(ResearchRequest("EPIC", "forecast"), [1.0] * 10)
        assert isinstance(result, ResearchResult)
        assert result.model_output["direction"] == 1.0
        assert result.model_output["confidence"] == pytest.approx(0.6)
        assert result.model_output["expected_return"] == pytest.approx(0.01)

    def test_regime_has_no_direction(self, agent):
        agent.register_model("regime", regime_adapter(state=2, label="High Vol"))
        result = agent.research(ResearchRequest("EPIC", "regime"), [1.0] * 10)
        assert result.model_output["direction"] == 0.0
        assert result.model_output["confidence"] == 0.0
        assert result.model_output["regime"] == 2
        assert result.model_output["regime_label"] == "High Vol"

    def test_missing_model_is_an_error_not_a_zero(self, agent):
        result = agent.research(ResearchRequest("EPIC", "forecast"), [1.0] * 10)
        assert "error" in result.model_output
        assert "not registered" in result.model_output["error"]

    def test_model_exception_is_surfaced(self, agent):
        agent.register_model("forecast", FakeAdapter("forecast", error="boom"))
        result = agent.research(ResearchRequest("EPIC", "forecast"), [1.0] * 10)
        assert "RuntimeError: boom" in result.model_output["error"]

    def test_unknown_model_type_raises(self, agent):
        with pytest.raises(ValueError, match="Unknown model_type"):
            agent.research(ResearchRequest("EPIC", "unknown"), [1.0] * 10)

    def test_publishes_once(self, agent, message_bus):
        agent.register_model("regime", regime_adapter())
        received = []
        message_bus.subscribe("research.completed", received.append)
        agent.research(ResearchRequest("EPIC", "regime"), [1.0] * 10)
        assert len(received) == 1
        assert received[0].payload["epic"] == "EPIC"

    def test_records_audit_decision(self, agent, audit_trail):
        agent.register_model("regime", regime_adapter())
        agent.research(ResearchRequest("EPIC", "regime"), [1.0] * 10)
        assert audit_trail.count() == 1
        assert audit_trail.query(agent="ResearchAgent")[0].decision.decision_type.value == "research"

    def test_request_id_propagates(self, agent):
        agent.register_model("regime", regime_adapter())
        result = agent.research(
            ResearchRequest("EPIC", "regime", request_id="req-001"), [1.0] * 10
        )
        assert result.request_id == "req-001"


class TestCombinedSignal:
    def test_direction_and_confidence_come_from_the_forecast(self, agent):
        agent.register_model("forecast", forecast_adapter(expected=-0.01, confidence=0.6))
        agent.register_model("regime", regime_adapter(state=0, label="Low Vol"))

        result = agent.research_signal("EPIC", [1.0] * 10)
        out = result.model_output
        assert out["direction"] == -1.0
        assert out["confidence"] == pytest.approx(0.6)
        assert out["regime_confidence_multiplier"] == 1.0
        assert out["regime_label"] == "Low Vol"

    def test_high_vol_regime_halves_confidence(self, agent):
        agent.register_model("forecast", forecast_adapter(confidence=0.6))
        agent.register_model("regime", regime_adapter(state=2, n_regimes=3, label="High Vol"))

        out = agent.research_signal("EPIC", [1.0] * 10).model_output
        assert out["confidence"] == pytest.approx(0.3)
        assert out["regime_confidence_multiplier"] == 0.5

    def test_without_forecast_there_is_no_signal(self, agent):
        agent.register_model("regime", regime_adapter(state=2, label="High Vol"))
        out = agent.research_signal("EPIC", [1.0] * 10).model_output
        assert "error" in out
        assert "forecast model" in out["error"]
        assert "direction" not in out

    def test_zero_direction_is_flagged_not_traded(self, agent):
        agent.register_model("forecast", forecast_adapter(expected=0.0, confidence=0.0))
        out = agent.research_signal("EPIC", [1.0] * 10).model_output
        assert out["direction"] == 0.0
        assert out.get("note") == "no directional view"

    def test_publishes_one_combined_message(self, agent, message_bus):
        agent.register_model("forecast", forecast_adapter())
        agent.register_model("regime", regime_adapter())
        received = []
        message_bus.subscribe("research.completed", received.append)
        agent.research_signal("EPIC", [1.0] * 10)
        assert len(received) == 1
        assert received[0].payload["model_type"] == "signal"
        assert "direction" in received[0].payload["output"]


class TestLLMFallback:
    def test_no_key_produces_no_report(self, agent):
        agent.register_model("regime", regime_adapter())
        result = agent.research(ResearchRequest("EPIC", "regime"), [1.0] * 10)
        assert result.llm_report == ""

    def test_provider_failure_is_reported_not_raised(self, agent, monkeypatch):
        agent.llm_api_key = "sk-invalid"
        agent.register_model("regime", regime_adapter())

        def rejected(*args, **kwargs):
            raise RuntimeError("provider rejected test credentials")

        monkeypatch.setattr("openai.OpenAI", rejected)
        result = agent.research(ResearchRequest("EPIC", "regime"), [1.0] * 10)
        assert "[LLM report unavailable:" in result.llm_report
