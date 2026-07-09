"""
Alembic environment configuration.

WHY THIS EXISTS:
    This file configures Alembic's migration runner. It tells Alembic:
    1. Which database to connect to (via pg_settings.dsn).
    2. Which SQLAlchemy models define the target schema (import Base).
    3. Whether to run in "offline" (SQL script generation) or "online"
       (direct database connection) mode.

WHY loading from config.py instead of alembic.ini:
    The database URL in alembic.ini is a placeholder. The actual URL comes
    from environment variables via config.py (pg_settings.dsn). This means
    migrations automatically target the correct database (DEMO vs. LIVE,
    dev vs. production) based on the current environment configuration.
"""

from logging.config import fileConfig

from alembic import context

# WHY importing config and Base:
#   config.pg_settings.dsn provides the database connection URL.
#   models.Base.metadata tells Alembic what the target schema looks like.
#   Without this import, Alembic would not know which tables to create or
#   modify in migrations.
from ig_market_data.config import pg_settings
from ig_market_data.models import Base

# Alembic Config object — provides access to alembic.ini values.
config = context.config

# WHY overriding sqlalchemy.url here:
#   The alembic.ini file has a placeholder URL. We replace it with the
#   configured DSN so that migrations always target the correct database.
config.set_main_option("sqlalchemy.url", pg_settings.dsn)

# Set up Python logging from alembic.ini settings.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for schema comparison.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode.

    Offline mode generates SQL migration scripts without connecting to a
    database. This is useful for:
    - Reviewing migration SQL before applying it.
    - Applying migrations by a DBA who does not have direct database access.
    - Generating migration scripts for multiple environments.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode.

    Online mode connects directly to the database and applies migrations.
    This is the standard mode for development and production deployments.
    """
    connectable = config.attributes.get("connection", None)

    if connectable is None:
        # WHY creating engine here vs. importing from elsewhere:
        #   We need a fresh engine for migration execution. The ingestion
        #   pipeline's engine pool should not be reused for migrations to
        #   avoid connection interference.
        from sqlalchemy import create_engine

        connectable = create_engine(url=pg_settings.dsn)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
