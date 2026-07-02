"""Database engine for the Pattern Finder DB.

Connection string is fully controlled by the ``PATTERN_FINDER_DB_URL`` env
var so the same code runs unmodified against local SQLite (dev) and Cloud
SQL Postgres (deployed) -- see .agents-cli-spec.md, "Data Sources & Auth".

Defaults to a local SQLite file so `agents-cli install && agents-cli run`
works with zero external setup during development.
"""

import os
from functools import lru_cache

from sqlalchemy import Engine, create_engine

from app.db.models import metadata

_DEFAULT_LOCAL_URL = "sqlite:///./pattern_finder_local.db"


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    db_url = os.environ.get("PATTERN_FINDER_DB_URL", _DEFAULT_LOCAL_URL)
    connect_args = {"check_same_thread": False} if db_url.startswith("sqlite") else {}
    engine = create_engine(db_url, connect_args=connect_args)
    metadata.create_all(engine)
    return engine
