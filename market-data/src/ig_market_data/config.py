"""
config.py — Centralised configuration loaded from environment variables.

Loads from .env via python-dotenv; all env vars validated with pydantic.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Walk up from this file looking for .env (covers any invocation CWD)
_start = Path(__file__).resolve()
for _parent in [_start] + list(_start.parents):
    candidate = _parent / ".env"
    if candidate.is_file():
        load_dotenv(dotenv_path=candidate)
        break


class IGSettings:
    api_key: str = os.environ["IG_API_KEY"]
    username: str = os.environ["IG_USERNAME"]
    password: str = os.environ["IG_PASSWORD"]
    account_type: str = os.environ.get("IG_ACCOUNT_TYPE", "DEMO").upper()

    @property
    def base_url(self) -> str:
        host = "api.ig.com" if self.account_type == "LIVE" else "demo-api.ig.com"
        return f"https://{host}/gateway/deal"

    @property
    def lightstreamer_url(self) -> str:
        host = "apd.marketdatasystems.com" if self.account_type == "LIVE" else "demo-apd.marketdatasystems.com"
        return f"https://{host}"


class PGSettings:
    host: str = os.environ.get("PG_HOST", "localhost")
    port: int = int(os.environ.get("PG_PORT", "5432"))
    database: str = os.environ.get("PG_DATABASE", "market_data")
    user: str = os.environ.get("PG_USER", "postgres")
    password: str = os.environ.get("PG_PASSWORD", "postgres")

    @property
    def dsn(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


ig_settings = IGSettings()
pg_settings = PGSettings()