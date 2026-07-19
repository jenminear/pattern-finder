#!/usr/bin/env python3
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Back up the Pattern Finder DB, then clear all its tables.

Always backs up first -- clearing is only safe because a restorable copy
exists afterward. Operates on whatever PATTERN_FINDER_DB_URL (or its
local SQLite default) app/db/engine.py resolves to -- read from the
actual Engine's URL rather than duplicating that resolution logic here,
so this can't silently drift out of sync with what the running app/tests
actually use.

Only local SQLite is backed up by file copy. If PATTERN_FINDER_DB_URL
points at Postgres (e.g. deployed Cloud SQL), this refuses to guess at a
backup strategy -- use `pg_dump` directly instead.

Usage:
    uv run python scripts/reset_db.py
"""

import shutil
from datetime import date
from pathlib import Path

from sqlalchemy import Engine

from app.db import models
from app.db.engine import get_engine

_SQLITE_PREFIX = "sqlite:///"


def _sqlite_path_from_url(url: str) -> Path | None:
    if not url.startswith(_SQLITE_PREFIX):
        return None
    return Path(url[len(_SQLITE_PREFIX) :]).resolve()


def backup_database(engine: Engine) -> Path:
    """Copies the SQLite DB file to <stem>_backup_<YYYY-MM-DD>.db next to
    it (or ..._2.db, ..._3.db, ... if a backup with today's date already
    exists -- never silently overwrites a prior backup)."""
    db_url = str(engine.url)
    db_path = _sqlite_path_from_url(db_url)
    if db_path is None:
        raise NotImplementedError(
            f"backup_database only supports local SQLite; PATTERN_FINDER_DB_URL "
            f"is {db_url!r}. For Postgres, use `pg_dump` directly."
        )
    if not db_path.exists():
        raise FileNotFoundError(f"No database file at {db_path}")

    today = date.today().isoformat()
    backup_path = db_path.with_name(f"{db_path.stem}_backup_{today}{db_path.suffix}")
    suffix = 2
    while backup_path.exists():
        backup_path = db_path.with_name(
            f"{db_path.stem}_backup_{today}_{suffix}{db_path.suffix}"
        )
        suffix += 1

    shutil.copy2(db_path, backup_path)
    return backup_path


def clear_all_tables(engine: Engine) -> dict[str, int]:
    """Deletes every row from every table, in FK-safe order -- the same
    approach tests/integration/test_agent.py's isolated_db fixture already
    uses and has proven safe. Returns the row count deleted per table."""
    counts: dict[str, int] = {}
    with engine.begin() as conn:
        for table in reversed(models.metadata.sorted_tables):
            result = conn.execute(table.delete())
            counts[table.name] = result.rowcount
    return counts


def main() -> None:
    engine = get_engine()
    print(f"Database: {engine.url}")

    backup_path = backup_database(engine)
    print(f"Backed up to: {backup_path}")

    counts = clear_all_tables(engine)
    for table_name, n in counts.items():
        print(f"  Cleared {table_name}: {n} row(s)")
    print("Done.")


if __name__ == "__main__":
    main()
