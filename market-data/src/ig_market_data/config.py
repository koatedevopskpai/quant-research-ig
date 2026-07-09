"""
config.py — Centralised configuration loaded from environment variables.

WHY THIS EXISTS:
    Every module in this project needs IG API credentials and database
    connection parameters. Rather than hardcoding these values or requiring
    users to set them in every terminal session, we load them from a .env
    file once, validate them at import time, and expose them through typed
    settings objects.

WHY A .env FILE:
    The Twelve-Factor App methodology recommends storing configuration in
    environment variables. A .env file is the developer-friendly local
    equivalent — Git-ignored, never committed, easy to share via .env.template.

WHY DIRECTORY TREE WALKING:
    Users may run scripts from any directory (project root, module folder,
    tests subfolder). Walking up from the config module's own location means
    .env is always found regardless of current working directory.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# .env auto-discovery
# ---------------------------------------------------------------------------
# WHY Path(__file__).resolve():
#   __file__ can be a relative path. .resolve() makes it absolute so parent
#   traversal is deterministic regardless of how Python was invoked.
#
# WHY walking parents:
#   This file lives at:
#     market-data/src/ig_market_data/config.py
#   The .env lives at the repository root:
#     .env
#   Walking parents from config.py finds it in 4 steps up. This works whether
#   the user runs: python -m ig_market_data.verify_ig
#   from the repo root, market-data/, or any other directory.
_start = Path(__file__).resolve()
for _parent in [_start] + list(_start.parents):
    candidate = _parent / ".env"
    if candidate.is_file():
        load_dotenv(dotenv_path=candidate)
        break


class IGSettings:
    """
    IG API connection settings loaded from environment variables.

    WHY Pydantic is NOT used here (vs. standard practice):
        Pydantic would add a dependency for a single config class. These
        settings are validated at the point of use — if IG_API_KEY is missing,
        os.environ[] raises a KeyError with a clear message at import time,
        before any API calls are made.
    """

    # WHY os.environ["VAR"] (no default) vs. os.environ.get("VAR", "default"):
    #   API key, username, and password are always required. Accessing via
    #   bracket notation raises KeyError immediately if missing, preventing
    #   cryptic 401 errors later during API calls.
    api_key: str = os.environ["IG_API_KEY"]
    username: str = os.environ["IG_USERNAME"]
    password: str = os.environ["IG_PASSWORD"]

    # WHY account_type defaults to DEMO:
    #   DEMO accounts use simulated funds — safe for development and testing.
    #   A developer must explicitly set IG_ACCOUNT_TYPE=LIVE and provide live
    #   credentials before real money is at risk. This prevents accidental
    #   live trading during development.
    account_type: str = os.environ.get("IG_ACCOUNT_TYPE", "DEMO").upper()

    @property
    def base_url(self) -> str:
        """
        IG REST API base URL.

        WHY two different hosts:
            IG separates DEMO and LIVE environments at the DNS level.
            Using demo-api.ig.com in DEMO mode prevents accidental live
            orders during development and testing.
        """
        host = "api.ig.com" if self.account_type == "LIVE" else "demo-api.ig.com"
        return f"https://{host}/gateway/deal"

    @property
    def lightstreamer_url(self) -> str:
        """
        Lightstreamer endpoint URL.

        WHY a different host from base_url:
            Lightstreamer uses a separate streaming infrastructure from the
            REST API. IG provides distinct hostnames for DEMO and LIVE
            streaming endpoints.
        """
        host = (
            "apd.marketdatasystems.com"
            if self.account_type == "LIVE"
            else "demo-apd.marketdatasystems.com"
        )
        return f"https://{host}"


class PGSettings:
    """
    PostgreSQL connection settings.

    WHY separate settings class:
        PostgreSQL is needed for Phase 1 (data persistence). Defining the
        settings now means models.py and ingestion.py can import pg_settings
        without circular dependencies. Defaults target local dev PostgreSQL.
    """

    host: str = os.environ.get("PG_HOST", "localhost")
    port: int = int(os.environ.get("PG_PORT", "5432"))
    database: str = os.environ.get("PG_DATABASE", "market_data")
    user: str = os.environ.get("PG_USER", "postgres")
    password: str = os.environ.get("PG_PASSWORD", "postgres")

    @property
    def dsn(self) -> str:
        """
        PostgreSQL connection string in SQLAlchemy format.

        WHY building DSN here vs. in models.py:
            Centralising the DSN construction means if the connection string
            format ever changes (e.g., adding SSL parameters), only this file
            needs updating.
        """
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


# ---------------------------------------------------------------------------
# Singleton instances
# ---------------------------------------------------------------------------
# WHY module-level singletons:
#   Most consumers need a single config instance. Creating them at module
#   level ensures settings are loaded once at import time and shared across
#   all consumers in a process. This also surfaces missing environment
#   variables early — at import time, not at first use.
ig_settings = IGSettings()
pg_settings = PGSettings()
