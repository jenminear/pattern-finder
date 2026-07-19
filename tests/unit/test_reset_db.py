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
"""Unit tests for scripts/reset_db.py -- always run against an isolated
tmp_path SQLite file (via the PATTERN_FINDER_DB_URL override, same as
tests/unit/test_agent_deterministic.py's isolated_db fixture), never the
real project DB. This script deletes data for real, so its tests are
exactly the ones that must not touch anything live.
"""

from datetime import date

import pytest
from sqlalchemy import create_engine

import scripts.reset_db as reset_db
from app.db import models
from app.db.engine import get_engine
from app.mcp_server import db_ops


@pytest.fixture
def isolated_engine(monkeypatch, tmp_path):
    db_path = tmp_path / "isolated.db"
    monkeypatch.setenv("PATTERN_FINDER_DB_URL", f"sqlite:///{db_path}")
    get_engine.cache_clear()
    engine = get_engine()
    yield engine
    get_engine.cache_clear()


class TestBackupDatabase:
    def test_creates_file_with_todays_date_in_name(self, isolated_engine, tmp_path):
        backup_path = reset_db.backup_database(isolated_engine)
        assert backup_path.exists()
        assert date.today().isoformat() in backup_path.name
        assert backup_path.parent == tmp_path

    def test_backup_content_matches_original(self, isolated_engine):
        db_ops.upsert_pattern("a pattern", "x0 + 1")
        backup_path = reset_db.backup_database(isolated_engine)

        backed_up_engine = create_engine(f"sqlite:///{backup_path}")
        with backed_up_engine.connect() as conn:
            rows = conn.execute(models.patterns.select()).all()
        assert len(rows) == 1
        assert rows[0].text_desc == "a pattern"

    def test_second_backup_same_day_gets_distinct_name(self, isolated_engine):
        first = reset_db.backup_database(isolated_engine)
        second = reset_db.backup_database(isolated_engine)
        assert first != second
        assert first.exists() and second.exists()  # neither overwrote the other

    def test_non_sqlite_url_raises_not_implemented(self):
        pg_engine = create_engine("postgresql+psycopg://user:pw@host/db")
        with pytest.raises(NotImplementedError, match="pg_dump"):
            reset_db.backup_database(pg_engine)


class TestClearAllTables:
    def test_clears_seeded_data_from_every_table(self, isolated_engine):
        ids = [
            db_ops.insert_scenario("t", {"a": "1", "b": "2"}, "3"),
            db_ops.insert_scenario("t", {"a": "3", "b": "4"}, "7"),
        ]
        pattern_id = db_ops.upsert_pattern("sum", "x0 + x1")
        db_ops.link_pattern_to_scenarios(
            pattern_id, ids, update_label_set=True, label_names=["a", "b"]
        )

        counts = reset_db.clear_all_tables(isolated_engine)

        assert counts["scenarios"] == 2
        assert counts["patterns"] == 1
        assert counts["labels"] == 2
        assert db_ops.get_all_pattern_descriptions() == []
        assert db_ops.get_scenarios_by_label_set(["a", "b"]) == []

    def test_empty_db_reports_zero_counts_without_erroring(self, isolated_engine):
        counts = reset_db.clear_all_tables(isolated_engine)
        assert all(n == 0 for n in counts.values())
