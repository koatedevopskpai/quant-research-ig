"""
Tests for the FastAPI REST API server.

Uses TestClient for fast in-process testing without a live server.
"""

import pytest
from fastapi.testclient import TestClient

from agentic_finance.api.server import app, get_state

client = TestClient(app)


class TestApiHealth:
    """Verify the API starts and basic endpoints respond."""

    def test_get_risk(self):
        resp = client.get("/risk")
        assert resp.status_code == 200
        data = resp.json()
        assert "var_parametric" in data
        assert "current_drawdown" in data

    def test_get_portfolio(self):
        resp = client.get("/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_capital"] == 100_000.0
        assert "allocations" in data
        assert "cash" in data

    def test_get_audit(self):
        resp = client.get("/audit")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_get_positions(self):
        resp = client.get("/positions")
        assert resp.status_code == 200
        assert isinstance(resp.json(), dict)


class TestApiResearch:
    """Research endpoint tests."""

    def test_run_regime(self):
        resp = client.post("/research/IX.D.SPTRD.IFD.IP", json={
            "model_type": "regime",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["epic"] == "IX.D.SPTRD.IFD.IP"
        assert data["model_type"] == "regime"
        assert "model_output" in data
        assert "summary" in data
        # Regime context only: no direction, and clearly labelled as demo data.
        assert data["model_output"]["direction"] == 0.0
        assert data["model_output"]["diagnostics"]["demo"] is True

    def test_run_forecast(self):
        resp = client.post("/research/TEST", json={
            "model_type": "forecast",
            "prices": [100.0, 101.0, 102.0],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == "forecast"
        # The old response exposed "prediction_mean"; the contract exposes a
        # direction and a confidence, plus an explicit demo marker.
        assert data["model_output"]["direction"] in (-1.0, 0.0, 1.0)
        assert 0.0 <= data["model_output"]["confidence"] <= 1.0
        assert data["model_output"]["diagnostics"]["demo"] is True

    def test_invalid_model(self):
        resp = client.post("/research/TEST", json={
            "model_type": "invalid",
        })
        assert resp.status_code == 422


class TestApiExecution:
    """Execution/signal endpoint tests."""

    def test_post_signal(self):
        resp = client.post("/signal", json={
            "epic": "TEST",
            "direction": 0.5,
            "confidence": 0.7,
            "source": "test",
            "current_price": 100.0,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["side"] in ("buy", "sell")
        assert "order_id" in data

    def test_signal_low_confidence_rejected(self):
        resp = client.post("/signal", json={
            "epic": "TEST",
            "direction": 0.5,
            "confidence": 0.0,
            "source": "test",
            "current_price": 100.0,
        })
        assert resp.status_code == 400

    def test_signal_default_price(self):
        """current_price defaults to 100.0 if omitted."""
        resp = client.post("/signal", json={
            "epic": "TEST",
            "direction": 0.5,
            "confidence": 0.7,
        })
        assert resp.status_code == 200


class TestApiRisk:
    """Risk endpoint tests."""

    def test_check_risk(self):
        resp = client.post("/risk/check")
        assert resp.status_code == 200
        data = resp.json()
        assert "is_breached" in data
        assert "breaches" in data

    def test_kill_switch(self):
        resp = client.post("/risk/kill")
        assert resp.status_code == 200
        data = resp.json()
        assert "killed" in data
        # Reset the kill state for subsequent tests
        get_state().execution.resume()


class TestApiPortfolio:
    """Portfolio/allocate endpoint tests."""

    def test_allocate_equal(self):
        agent = get_state().portfolio
        agent.register_strategy("strat_a", expected_return=0.05, expected_vol=0.10)
        agent.register_strategy("strat_b", expected_return=0.08, expected_vol=0.15)

        resp = client.post("/allocate/equal", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["allocations"]) == 2
        assert abs(data["allocations"][0]["weight"] - 0.5) < 0.01

    def test_allocate_risk_parity(self):
        resp = client.post("/allocate/risk_parity", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["allocations"]) == 2

    def test_allocate_invalid_method(self):
        resp = client.post("/allocate/invalid", json={})
        assert resp.status_code == 400

    def test_get_portfolio_after_allocate(self):
        resp = client.get("/portfolio")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["allocations"]) > 0
        assert data["total_capital"] > 0

    def test_get_strategies(self):
        resp = client.get("/portfolio/strategies")
        assert resp.status_code == 200
        data = resp.json()
        assert "strat_a" in data


class TestApiAudit:
    """Audit trail endpoint tests."""

    def test_audit_has_entries(self):
        resp = client.get("/audit")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) > 0

    def test_audit_with_filters(self):
        resp = client.get("/audit?agent=PortfolioAgent&limit=5")
        assert resp.status_code == 200
        entries = resp.json()
        for e in entries:
            assert e["agent"] == "PortfolioAgent"

    def test_audit_get_by_id(self):
        entries = client.get("/audit").json()
        if entries:
            eid = entries[0]["decision_id"]
            resp = client.get(f"/audit/{eid}")
            assert resp.status_code == 200
            assert resp.json()["decision_id"] == eid

    def test_audit_get_missing(self):
        resp = client.get("/audit/nonexistent")
        assert resp.status_code == 404

    def test_export_audit(self):
        resp = client.get("/audit/export")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestApiBus:
    """Message bus endpoint tests."""

    def test_bus_publish(self):
        received = []

        def handler(msg):
            received.append(msg)

        get_state().bus.subscribe("test.topic", handler)
        resp = client.post("/bus/publish", json={
            "topic": "test.topic",
            "sender": "test",
            "payload": {"key": "value"},
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["topic"] == "test.topic"
        assert data["sender"] == "test"
        assert len(received) == 1
        assert received[0].payload["key"] == "value"
