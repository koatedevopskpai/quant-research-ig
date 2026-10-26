"""Credential-free tests; real provider traffic must be mocked explicitly."""

import os

# Set before test modules import configuration. Never discover a developer's .env in CI.
os.environ["QUANT_LOAD_DOTENV"] = "0"

import httpx
import pytest


@pytest.fixture(autouse=True)
def block_provider_http(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("External HTTP is disabled in unit tests; mock the provider boundary")

    async def blocked_async(*args, **kwargs):
        blocked()

    # FastAPI's in-process TestClient uses its own transport and remains usable.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
