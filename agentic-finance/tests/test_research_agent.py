import numpy as np
import pytest
from agentic_finance.research_agent import ResearchAgent, ResearchRequest, ResearchResult


class MockRegimeResult:
    regime_labels = [0, 0, 1, 1, 0]
    regime_probs = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8], [0.85, 0.15]])


class MockForecastResult:
    forecast = np.array([0.01, 0.02, -0.01, 0.005, 0.015])


class MockModel:
    result_type = "regime"

    def fit(self, *args, **kwargs):
        return self

    def predict(self, *args, **kwargs):
        if self.result_type == "regime":
            return MockRegimeResult()
        return MockForecastResult()


class MockErrorModel:
    def fit(self, *args, **kwargs):
        return self

    def predict(self, *args, **kwargs):
        raise RuntimeError("model failure")


class MockPrices:
    """Mimics a pandas Series with .values attribute."""

    def __init__(self, n=100):
        self.values = np.random.randn(n).cumsum() + 100


class TestResearchAgent:
    @pytest.fixture
    def agent(self, message_bus, audit_trail):
        return ResearchAgent(message_bus, audit_trail, llm_api_key=None)

    def test_register_model(self, agent):
        model = MockModel()
        agent.register_model("regime", model)
        assert "regime" in agent._models

    def test_research_regime(self, agent):
        model = MockModel()
        model.result_type = "regime"
        agent.register_model("regime", model)
        request = ResearchRequest(epic="IX.D.STCHEDS.DAILY.IP", model_type="regime")
        result = agent.research(request, MockPrices())
        assert isinstance(result, ResearchResult)
        assert result.model_type == "regime"
        assert "regime_labels" in result.model_output
        assert result.summary != ""
        assert result.llm_report == ""

    def test_research_forecast(self, agent):
        model = MockModel()
        model.result_type = "forecast"
        agent.register_model("forecast", model)
        request = ResearchRequest(epic="IX.D.STCHEDS.DAILY.IP", model_type="forecast")
        result = agent.research(request, MockPrices())
        assert result.model_type == "forecast"
        assert "prediction_mean" in result.model_output

    def test_research_sentiment(self, agent):
        model = MockModel()
        model.result_type = "sentiment"
        agent.register_model("sentiment", model)
        request = ResearchRequest(epic="IX.D.STCHEDS.DAILY.IP", model_type="sentiment")
        result = agent.research(request, MockPrices())
        assert result.model_type == "sentiment"

    def test_research_model_not_registered(self, agent):
        request = ResearchRequest(epic="EPIC", model_type="lstm")
        result = agent.research(request, MockPrices())
        assert "error" in result.model_output

    def test_research_unknown_model_type(self, agent):
        request = ResearchRequest(epic="EPIC", model_type="unknown")
        with pytest.raises(ValueError, match="unknown"):
            agent.research(request, MockPrices())

    def test_research_publishes_message(self, agent):
        model = MockModel()
        model.result_type = "regime"
        agent.register_model("regime", model)
        received = []

        def cb(msg):
            received.append(msg)

        agent.message_bus.subscribe("research.completed", cb)
        request = ResearchRequest(epic="EPIC", model_type="regime")
        agent.research(request, MockPrices())
        assert len(received) == 1
        assert received[0].payload["epic"] == "EPIC"

    def test_research_records_audit_trail(self, agent):
        model = MockModel()
        model.result_type = "regime"
        agent.register_model("regime", model)
        request = ResearchRequest(epic="EPIC", model_type="regime")
        agent.research(request, MockPrices())
        assert agent.audit_trail.count() == 1
        entry = agent.audit_trail.query(agent="ResearchAgent")[0]
        assert entry.decision.decision_type.value == "research"

    def test_request_id_propagates(self, agent):
        model = MockModel()
        model.result_type = "regime"
        agent.register_model("regime", model)
        request = ResearchRequest(epic="EPIC", model_type="regime", request_id="req-001")
        result = agent.research(request, MockPrices())
        assert result.request_id == "req-001"

    def test_error_in_model_returns_error_output(self, agent):
        agent.register_model("regime", MockErrorModel())
        request = ResearchRequest(epic="EPIC", model_type="regime")
        result = agent.research(request, MockPrices())
        assert "error" in result.model_output
