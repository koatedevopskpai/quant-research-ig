"""Import-time configuration must not require a provider account."""

import os
import subprocess
import sys

import pytest

from ig_market_data.config import IGSettings


def test_imports_without_credentials():
    env = os.environ.copy()
    for key in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD"):
        env.pop(key, None)
    env["QUANT_LOAD_DOTENV"] = "0"
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import ig_market_data.ingestion; import agentic_finance.run_production",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_missing_credentials_fail_before_http(monkeypatch):
    import ig_market_data.client as client_module
    from ig_market_data.client import IGClient

    settings = IGSettings()
    for attribute in ("api_key", "username", "password"):
        monkeypatch.setattr(settings, attribute, "")
    monkeypatch.setattr(client_module, "ig_settings", settings)
    with pytest.raises(
        RuntimeError, match="Missing IG credentials: IG_API_KEY, IG_USERNAME, IG_PASSWORD"
    ):
        IGClient()


def test_external_http_requires_a_mock():
    import httpx

    with pytest.raises(AssertionError, match="External HTTP is disabled"):
        httpx.get("https://example.invalid")
