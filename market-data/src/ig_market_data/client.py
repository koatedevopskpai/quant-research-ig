"""
client.py — IGClient: authenticated HTTP client for the IG REST API (v3).

Supports OAuth 2.0 Bearer token auth with auto-refresh on expiry.

Endpoints covered:
  - POST /session (v3)           — authentication
  - GET /categories (v1)         — list tradeable categories
  - GET /categories/{id}/instruments (v1) — instruments in a category
  - GET /markets (v1)            — search markets by term
  - GET /markets/{epic} (v3)     — market details for a single epic
  - GET /prices/{epic} (v3)      — historical OHLCV prices

Usage:
    from ig_market_data.client import IGClient
    client = IGClient()
    prices = client.fetch_prices("IX.D.SPTRD.IFD.IP", resolution="DAY", max=10)
"""

import time
import logging
from typing import Any

import httpx

from ig_market_data.config import ig_settings

logger = logging.getLogger(__name__)

TOKEN_REFRESH_SECONDS = 1500  # refresh 5 min before the 30-min expiry


class IGClient:
    def __init__(self) -> None:
        self.settings = ig_settings
        self._client = httpx.Client()
        self.token: str = ""
        self.account_id: str = ""
        self._token_acquired_at: float = 0.0
        self._authenticate()

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------
    def _authenticate(self) -> None:
        headers = {
            "X-IG-API-KEY": self.settings.api_key,
            "Content-Type": "application/json",
            "VERSION": "3",
        }
        body = {"identifier": self.settings.username, "password": self.settings.password}
        resp = self._client.post(
            f"{self.settings.base_url}/session",
            headers=headers,
            json=body,
            timeout=15,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"IG auth failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        self.token = data["oauthToken"]["access_token"]
        self.account_id = data["accountId"]
        self._token_expired_at = time.time() + TOKEN_REFRESH_SECONDS
        logger.info("Authenticated — account=%s", self.account_id)

    def _ensure_auth(self) -> None:
        if time.time() > self._token_expired_at:
            logger.info("Token expired; re-authenticating")
            self._authenticate()

    def _headers(self, version: str = "1") -> dict[str, str]:
        self._ensure_auth()
        return {
            "X-IG-API-KEY": self.settings.api_key,
            "Authorization": f"Bearer {self.token}",
            "IG-ACCOUNT-ID": self.account_id,
            "VERSION": version,
        }

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------
    def list_categories(self) -> list[dict[str, Any]]:
        r = self._client.get(
            f"{self.settings.base_url}/categories",
            headers=self._headers("1"),
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("categories", [])

    def list_instruments(
        self, category_code: str, max_results: int = 50
    ) -> list[dict[str, Any]]:
        r = self._client.get(
            f"{self.settings.base_url}/categories/{category_code}/instruments",
            headers=self._headers("1"),
            params={"max": max_results},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("instruments", [])

    def search_markets(self, term: str) -> list[dict[str, Any]]:
        r = self._client.get(
            f"{self.settings.base_url}/markets",
            headers=self._headers("1"),
            params={"searchTerm": term},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("markets", [])

    def get_market_details(self, epic: str) -> dict[str, Any]:
        r = self._client.get(
            f"{self.settings.base_url}/markets/{epic}",
            headers=self._headers("3"),
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------------
    # Historical prices
    # ------------------------------------------------------------------
    def fetch_prices(
        self,
        epic: str,
        resolution: str = "DAY",
        max_points: int = 10,
    ) -> list[dict[str, Any]]:
        r = self._client.get(
            f"{self.settings.base_url}/prices/{epic}",
            headers=self._headers("3"),
            params={"resolution": resolution, "max": max_points},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("prices", [])

    def __del__(self) -> None:
        self._client.close()