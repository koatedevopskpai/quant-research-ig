"""
client.py — IGClient: authenticated HTTP client for the IG REST API.

WHY THIS EXISTS:
    IG Group provides REST APIs for account management, market discovery,
    and historical price data. The IGClient class wraps all of these into
    a single, reusable Python class that handles authentication, token
    refresh, and consistent header management automatically.

ARCHITECTURE NOTE:
    Bryan Downing's influence (C++, low-latency focus) is most visible in
    the streaming client (stream.py). This REST client follows Quant
    Beckman's philosophy — it is the data ingestion layer that feeds the
    "Research" stage of his Research → Know → Do → Get cycle. Clean data
    in means trustworthy research out.

    The IG REST API has multiple version headers per endpoint:
    - v1: Market discovery (categories, instruments, search)
    - v3: Authentication, market details, pricing (v2 deprecated in 2024)

    Each method specifies the correct version in the VERSION header.

IG API v3 OAuth 2.0 Flow:
    1. POST /session with API key, username, password
    2. Response includes oauthToken.access_token (Bearer token, 1800s TTL)
    3. All subsequent requests include: Authorization: Bearer <token>
    4. Token is refreshed at 1500s (5-minute buffer before 1800s expiry)

CLASS DESIGN (ADR-003):
    Why a class instead of functions?
    1. Connection pooling — one httpx.Client session reused across all calls
       saves approximately 50ms per request in DNS/TLS handshake overhead.
    2. Centralised auth — token lifecycle managed in one place.
    3. Testability — the session can be mocked for unit tests.
"""

import time
import logging
from typing import Any

import httpx

from ig_market_data.config import ig_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token refresh threshold
# ---------------------------------------------------------------------------
# WHY 1500 seconds (25 minutes):
#   IG Bearer tokens expire after 1800 seconds (30 minutes). We refresh at
#   1500 seconds to provide a 300-second safety window. This prevents
#   mid-request auth failures due to clock drift, network latency in the
#   refresh call itself, or race conditions where the token expires between
#   the expiry check and the actual API call.
TOKEN_REFRESH_SECONDS = 1500


class IGClient:
    """
    Authenticated HTTP client for the IG REST API.

    Usage:
        client = IGClient()
        categories = client.list_categories()
        prices = client.fetch_prices("IX.D.SPTRD.IFD.IP", resolution="DAY", max_points=10)

    The client authenticates on construction and refreshes the token
    automatically every 1500 seconds.

    WHY synchronous httpx.Client instead of async httpx.AsyncClient:
        The current usage pattern is sequential — fetch data, store it,
        analyse it. Async would add complexity without throughput benefit
        for this pattern. If we later need concurrent fetching of many
        instruments, we can add an async variant.
    """

    def __init__(self) -> None:
        # WHY storing settings as instance attribute vs. re-importing:
        #   Allows mocking settings in tests by patching the instance after
        #   construction, without affecting other modules that import config.
        self.settings = ig_settings
        self.settings.validate_credentials()

        # WHY a single httpx.Client session:
        #   httpx.Client maintains a connection pool internally. Reusing it
        #   avoids DNS resolution and TLS negotiation on every request,
        #   reducing per-request overhead from ~80ms to ~30ms.
        self._client = httpx.Client()

        # Token state — initialised empty; _authenticate() fills these.
        self.token: str = ""
        self.account_id: str = ""

        # WHY tracking token acquisition time:
        #   time.time() provides monotonic seconds. We compare against this
        #   in _ensure_auth() to avoid calling _authenticate() on every
        #   request. The alternative — checking a boolean flag — would not
        #   handle the case where the token expires between checks.
        self._token_acquired_at: float = 0.0

        # Authenticate immediately so the first API call succeeds without
        # an auth round-trip.
        self._authenticate()

    # ------------------------------------------------------------------
    # Authentication (private)
    # ------------------------------------------------------------------

    def _authenticate(self) -> None:
        """
        Authenticate with IG and obtain a Bearer token.

        POST /session (v3) sends:
            - X-IG-API-KEY: API key (identifies the application)
            - identifier: username
            - password: password

        Response (HTTP 200) includes:
            - oauthToken.access_token: the Bearer token
            - oauthToken.expiry: token TTL in seconds (1800)
            - accountId: the default trading account

        WHY HTTP 200 vs. checking resp.is_success:
            IG may return HTTP 403 for auth failures. Checking for exactly
            200 distinguishes "auth failed" from "rate limited" (429) or
            "server error" (5xx).
        """
        headers = {
            "X-IG-API-KEY": self.settings.api_key,
            "Content-Type": "application/json",
            "VERSION": "3",  # v3 endpoint returns OAuth 2.0 Bearer tokens
        }
        body = {
            "identifier": self.settings.username,
            "password": self.settings.password,
        }
        resp = self._client.post(
            f"{self.settings.base_url}/session",
            headers=headers,
            json=body,
            timeout=15,
        )
        if resp.status_code != 200:
            # WHY including response text in the error:
            #   IG's error response body contains useful diagnostics
            #   (e.g., "invalid credentials", "account locked"). Truncating
            #   to 200 characters prevents leaking large stack traces.
            raise RuntimeError(
                f"IG auth failed: HTTP {resp.status_code} {resp.text[:200]}"
            )
        data = resp.json()
        self.token = data["oauthToken"]["access_token"]
        self.account_id = data["accountId"]
        self._token_expired_at = time.time() + TOKEN_REFRESH_SECONDS
        logger.info("Authenticated — account=%s", self.account_id)

    def _ensure_auth(self) -> None:
        """
        Check if the token needs refreshing and re-authenticate if so.

        WHY checking on every request vs. a background timer:
            Background timers add threading complexity and race conditions
            (what if the timer fires mid-request?). Synchronous checking
            before each request is simpler and sufficient given the 5-minute
            safety window.
        """
        if time.time() > self._token_expired_at:
            logger.info("Token expired; re-authenticating")
            self._authenticate()

    def _headers(self, version: str = "1") -> dict[str, str]:
        """
        Build common headers for IG API requests.

        Args:
            version: The IG API version for this endpoint ("1" or "3").

        WHY version is a parameter:
            Different IG endpoints require different API versions in the
            VERSION header. Using v1 for an endpoint that expects v3 (or
            vice versa) returns HTTP 400 with a confusing error message.
        """
        self._ensure_auth()
        return {
            "X-IG-API-KEY": self.settings.api_key,
            # WHY Bearer token in Authorization header:
            #   IG API v3 uses standard OAuth 2.0 Bearer token format.
            #   The v2 CST/XST headers are deprecated and should not be used.
            "Authorization": f"Bearer {self.token}",
            "IG-ACCOUNT-ID": self.account_id,
            "VERSION": version,
        }

    # ------------------------------------------------------------------
    # Market discovery
    # ------------------------------------------------------------------
    # These methods use IG API v1 for historical compatibility.
    # IG has not yet migrated category/instrument endpoints to v3.

    def list_categories(self) -> list[dict[str, Any]]:
        """
        List all tradeable market categories.

        Returns categories like:
            - INDICES (e.g., US 500, FTSE 100)
            - FX (e.g., EUR/USD, GBP/USD)
            - COMMODITIES (e.g., Gold, Oil)
            - CRYPTOCURRENCY (e.g., Bitcoin, Ethereum)

        WHY categories matter:
            Categories are the top level of IG's market hierarchy.
            Instruments are grouped under categories, and many IG API
            operations require the category code first.
        """
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
        """
        List instruments within a specific category.

        Args:
            category_code: The category code from list_categories().
            max_results: Maximum instruments to return (IG default is 50).

        WHY limiting results:
            Some categories contain hundreds of instruments. Fetching all
            of them in one call can time out or return excessive data.
            Use max_results=50 for exploration, then narrow down.

        Returns instrument metadata including:
            - epic: The unique instrument identifier (e.g., "IX.D.SPTRD.IFD.IP")
            - instrumentName: Human-readable name
            - instrumentType: SHARES, INDICES, FX, etc.
        """
        r = self._client.get(
            f"{self.settings.base_url}/categories/{category_code}/instruments",
            headers=self._headers("1"),
            params={"max": max_results},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("instruments", [])

    def search_markets(self, term: str) -> list[dict[str, Any]]:
        """
        Search for markets by text term.

        WHY a search method exists separately from list_instruments:
            You often know the instrument name ("US 500", "EUR/USD") but not
            its category code or epic. search_markets() finds it directly.

        Example:
            markets = client.search_markets("US 500")
            epic = markets[0]["epic"]  # "IX.D.SPTRD.IFD.IP"
        """
        r = self._client.get(
            f"{self.settings.base_url}/markets",
            headers=self._headers("1"),
            params={"searchTerm": term},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("markets", [])

    def get_market_details(self, epic: str) -> dict[str, Any]:
        """
        Get detailed information about a specific market.

        WHY use v3 for this endpoint:
            The v1 version returns incomplete metadata. The v3 endpoint
            includes additional fields like dealing rules, margin
            requirements, and instrument expiry dates.

        Returns:
            - instrument: name, type, epic, currency
            - dealingRules: min/max deal size, step, leverage
            - margin: deposit requirement for CFD trading
            - expiry: for futures/options, the expiry date
        """
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
        """
        Fetch historical OHLCV price data for a given instrument.

        Args:
            epic: The instrument epic (e.g., "IX.D.SPTRD.IFD.IP").
            resolution: Candle resolution — "DAY", "HOUR", "MINUTE",
                       "SECOND", "TICK" (IG-specific).
            max_points: Maximum number of candles to return.

        WHY max_points defaults to 10:
            Fetching too many points in one call (e.g., max_points=1000 for
            daily data covering 4 years) can be slow and memory-intensive.
            Start small for verification, then increase for research.

        Returns a list of candle dicts, each containing:
            - snapshotTime: ISO timestamp of the candle open
            - openPrice: {"bid": x, "ask": y, "mid": z}
            - closePrice: {"bid": x, "ask": y, "mid": z}
              (plus highPrice, lowPrice)
            - lastTradedVolume: total volume for the period
            - instrumentType: type identifier

        WHY we request JSON format (default):
            IG also supports XML, but JSON integrates directly with Python
            dicts without parsing overhead.
        """
        r = self._client.get(
            f"{self.settings.base_url}/prices/{epic}",
            headers=self._headers("3"),
            params={"resolution": resolution, "max": max_points},
            timeout=15,
        )
        r.raise_for_status()
        return r.json().get("prices", [])

    def __del__(self) -> None:
        """
        Clean up the HTTP session on garbage collection.

        WHY explicit cleanup:
            httpx.Client maintains open connections and thread pool
            resources. While Python's garbage collector will eventually
            release them, explicit cleanup in __del__ ensures deterministic
            resource release, especially in long-running processes like the
            ingestion pipeline (Phase 1).

        WHY not a context manager:
            Context managers are preferred, but IGClient is often used as a
            long-lived singleton. __del__ provides a safety net. A future
            enhancement could add __enter__/__exit__ for use with 'with'.
        """
        client = getattr(self, "_client", None)
        if client is not None:
            client.close()
