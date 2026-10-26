"""
API security posture: loopback default, optional key, restricted bus, demo labels.

The review's finding: the entry point bound 0.0.0.0, mutation endpoints were
unauthenticated, CORS accepted every origin, and any client could publish to any
bus topic.
"""

import pytest
from fastapi.testclient import TestClient

from agentic_finance.api.server import (
    allowed_bus_topics,
    allowed_origins,
    api_host,
    app,
)

client = TestClient(app)


class TestBinding:
    def test_default_bind_is_loopback(self, monkeypatch):
        monkeypatch.delenv("QUANT_API_HOST", raising=False)
        assert api_host() == "127.0.0.1"

    def test_bind_can_be_overridden_explicitly(self, monkeypatch):
        monkeypatch.setenv("QUANT_API_HOST", "0.0.0.0")
        assert api_host() == "0.0.0.0"

    def test_default_cors_is_loopback_only(self, monkeypatch):
        monkeypatch.delenv("QUANT_CORS_ORIGINS", raising=False)
        origins = allowed_origins()
        assert origins and all("127.0.0.1" in o or "localhost" in o for o in origins)

    def test_cors_is_configurable(self, monkeypatch):
        monkeypatch.setenv("QUANT_CORS_ORIGINS", "https://desk.example.com")
        assert allowed_origins() == ["https://desk.example.com"]


class TestApiKey:
    def test_key_is_required_on_data_endpoints_when_set(self, monkeypatch):
        monkeypatch.setenv("QUANT_API_KEY", "s3cret")
        assert client.get("/positions").status_code == 401
        assert client.get("/positions", headers={"X-API-Key": "s3cret"}).status_code == 200

    def test_wrong_key_is_rejected(self, monkeypatch):
        monkeypatch.setenv("QUANT_API_KEY", "s3cret")
        assert client.get("/positions", headers={"X-API-Key": "nope"}).status_code == 401

    def test_health_is_always_reachable(self, monkeypatch):
        monkeypatch.setenv("QUANT_API_KEY", "s3cret")
        assert client.get("/health").status_code == 200

    def test_no_key_configured_leaves_endpoints_open(self, monkeypatch):
        monkeypatch.delenv("QUANT_API_KEY", raising=False)
        assert client.get("/positions").status_code == 200


class TestBusRestriction:
    def test_publication_is_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("QUANT_DEBUG", raising=False)
        resp = client.post("/bus/publish", json={
            "topic": "sim.tick", "sender": "test", "payload": {},
        })
        assert resp.status_code == 403

    def test_debug_allows_only_allowlisted_topics(self, monkeypatch):
        monkeypatch.setenv("QUANT_DEBUG", "1")
        assert allowed_bus_topics() == {"sim."}

        allowed = client.post("/bus/publish", json={
            "topic": "sim.tick", "sender": "test", "payload": {"p": 1},
        })
        assert allowed.status_code == 200

        blocked = client.post("/bus/publish", json={
            "topic": "risk.halted", "sender": "test", "payload": {},
        })
        assert blocked.status_code == 403

    def test_allowlist_is_configurable(self, monkeypatch):
        monkeypatch.setenv("QUANT_DEBUG", "1")
        monkeypatch.setenv("QUANT_BUS_ALLOWED_TOPICS", "sim.tick")
        assert client.post("/bus/publish", json={
            "topic": "sim.tick", "sender": "test", "payload": {},
        }).status_code == 200
        assert client.post("/bus/publish", json={
            "topic": "sim.other", "sender": "test", "payload": {},
        }).status_code == 403


class TestDemoLabelling:
    def test_health_exposes_the_demo_posture(self, monkeypatch):
        monkeypatch.delenv("QUANT_API_KEY", raising=False)
        monkeypatch.delenv("QUANT_DEBUG", raising=False)
        data = client.get("/health").json()
        assert data["demo_models"] is True
        assert data["auth_required"] is False
        assert data["debug"] is False
        assert data["host"] == "127.0.0.1"

    def test_research_response_is_marked_demo(self):
        data = client.post("/research/EPIC", json={"model_type": "forecast"}).json()
        assert data["demo"] is True
        assert data["model_output"]["diagnostics"]["demo"] is True
