"""Credential-free tests; real provider traffic must be mocked explicitly."""

import os
from pathlib import Path

# Set before test modules import configuration. Never discover a developer's .env in CI.
os.environ["QUANT_LOAD_DOTENV"] = "0"

import httpx
import pytest


def pytest_configure(config):
    """Ensure the parent of --basetemp exists.

    WHY: addopts sets --basetemp=test-results/tmp so pytest artifacts stay
    together inside the repository, and .gitignore excludes test-results/. On a
    fresh clone that directory does not exist, so every test that requests
    tmp_path failed at setup with
        FileNotFoundError: .../test-results/tmp
    This reproduced only on a checkout where an earlier --junitxml run had not
    already created the folder, i.e. exactly the fresh-clone case the baseline
    is supposed to guarantee.
    """
    basetemp = config.option.basetemp
    if basetemp:
        Path(basetemp).expanduser().parent.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def block_provider_http(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("External HTTP is disabled in unit tests; mock the provider boundary")

    async def blocked_async(*args, **kwargs):
        blocked()

    # FastAPI's in-process TestClient uses its own transport and remains usable.
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
