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
"""Unit tests for app/mcp_server/db_ops.py -- the exact label-set matching
and label-independent-match safety rule are the two trickiest, most
important-to-get-right pieces of Pattern Finder Outline.txt Section VI/VII,
so they're covered thoroughly here rather than only via manual/integration
testing.
"""

import pytest

from app.mcp_server import db_ops


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    """Each test gets its own fresh SQLite file, not the shared dev DB."""
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PATTERN_FINDER_DB_URL", f"sqlite:///{db_path}")
    from app.db.engine import get_engine

    get_engine.cache_clear()
    yield
    get_engine.cache_clear()


def test_get_pattern_by_label_set_no_match_returns_none():
    assert db_ops.get_pattern_by_label_set(["a", "b"]) is None


def test_get_pattern_by_label_set_unknown_labels_returns_none():
    # Labels that don't exist anywhere yet can't have an existing pattern.
    db_ops.insert_scenario("t", {"a": "1", "b": "2"}, "3")
    assert db_ops.get_pattern_by_label_set(["c", "d"]) is None


def test_exact_match_is_order_independent():
    sid = db_ops.insert_scenario("t", {"a": "2", "b": "5"}, "9")
    pid = db_ops.upsert_pattern("desc", "a + b")
    db_ops.link_pattern_to_scenarios(pid, [sid], update_label_set=True, label_names=["a", "b"])

    assert db_ops.get_pattern_by_label_set(["a", "b"]) is not None
    assert db_ops.get_pattern_by_label_set(["b", "a"]) is not None


def test_exact_match_rejects_superset_and_subset():
    sid = db_ops.insert_scenario("t", {"a": "2", "b": "5"}, "9")
    pid = db_ops.upsert_pattern("desc", "a + b")
    db_ops.link_pattern_to_scenarios(pid, [sid], update_label_set=True, label_names=["a", "b"])

    assert db_ops.get_pattern_by_label_set(["a", "b", "c"]) is None
    assert db_ops.get_pattern_by_label_set(["a"]) is None


def test_get_scenarios_by_label_set_returns_only_exact_matches():
    db_ops.insert_scenario("t", {"a": "1", "b": "2"}, "3")
    db_ops.insert_scenario("t", {"a": "10", "b": "20"}, "30")
    db_ops.insert_scenario("t", {"a": "1", "b": "2", "c": "3"}, "6")  # different label set
    db_ops.insert_scenario("t", {"x": "1"}, "1")  # unrelated label set

    results = db_ops.get_scenarios_by_label_set(["a", "b"])
    assert len(results) == 2
    assert {r["inputs"]["a"] for r in results} == {"1", "10"}


def test_insert_scenario_handles_missing_input_slot():
    sid = db_ops.insert_scenario("t", {"a": "1", "b": None}, "consequence")
    results = db_ops.get_scenarios_by_label_set(["a", "b"])
    assert len(results) == 1
    assert results[0]["scenario_id"] == sid
    assert results[0]["inputs"]["b"] is None


def test_upsert_pattern_creates_then_updates():
    pid = db_ops.upsert_pattern("first description", "rule_v1")
    db_ops.upsert_pattern("revised description", "rule_v2", pattern_id=pid)

    patterns = db_ops.get_all_pattern_descriptions()
    assert len(patterns) == 1
    assert patterns[0]["text_desc"] == "revised description"
    assert patterns[0]["rule_or_code_link"] == "rule_v2"


def test_link_pattern_exact_match_links_all_scenarios_with_label_set():
    # Section VI step 9: a newly captured pattern must be linked to EVERY
    # scenario sharing that exact label set, not just the triggering one.
    s1 = db_ops.insert_scenario("t", {"a": "1", "b": "2"}, "3")
    s2 = db_ops.insert_scenario("t", {"a": "10", "b": "20"}, "30")
    pid = db_ops.upsert_pattern("sum pattern", "a + b")

    db_ops.link_pattern_to_scenarios(pid, [s1, s2], update_label_set=True, label_names=["a", "b"])

    scenarios = db_ops.get_scenarios_by_label_set(["a", "b"])
    assert {s["pattern_id"] for s in scenarios} == {pid}


def test_label_independent_match_does_not_update_pattern_label_set():
    # Safety rule (.agents-cli-spec.md, Constraints & Safety Rules): a
    # label-independent match must link the ONE scenario, but must NOT
    # cause the pattern to become associated with the new label set.
    pid = db_ops.upsert_pattern("abstracted pattern", "x * 2")
    new_scenario = db_ops.insert_scenario("t", {"p": "3", "q": "4"}, "6")

    db_ops.link_pattern_to_scenarios(pid, [new_scenario], update_label_set=False)

    # The pattern must still NOT be discoverable via this label set.
    assert db_ops.get_pattern_by_label_set(["p", "q"]) is None
    # But the scenario itself must be linked.
    scenarios = db_ops.get_scenarios_by_label_set(["p", "q"])
    assert scenarios[0]["pattern_id"] == pid


def test_link_pattern_to_scenarios_requires_label_names_when_updating_label_set():
    pid = db_ops.upsert_pattern("desc", "rule")
    with pytest.raises(ValueError):
        db_ops.link_pattern_to_scenarios(pid, [], update_label_set=True, label_names=None)


def test_log_candidate_technique_records_full_row():
    from sqlalchemy import select

    from app.db.engine import get_engine
    from app.db.models import candidate_techniques

    candidate_id = db_ops.log_candidate_technique(
        label_names="a, b, c",
        rule="x0**2 + 6*x1 + 9*x2",
        confidence=0.95,
        trace="tried a quadratic regression, R^2 = 1.0",
        source="fresh discovery",
    )

    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(candidate_techniques).where(
                candidate_techniques.c.candidate_id == candidate_id
            )
        ).one()

    assert row.label_names == "a, b, c"
    assert row.rule == "x0**2 + 6*x1 + 9*x2"
    assert row.confidence == 0.95
    assert row.trace == "tried a quadratic regression, R^2 = 1.0"
    assert row.source == "fresh discovery"
    assert row.created_at is not None


def test_log_candidate_technique_optional_fields_default_none():
    candidate_id = db_ops.log_candidate_technique(label_names="x", rule="x0")

    from sqlalchemy import select

    from app.db.engine import get_engine
    from app.db.models import candidate_techniques

    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(
            select(candidate_techniques).where(
                candidate_techniques.c.candidate_id == candidate_id
            )
        ).one()

    assert row.confidence is None
    assert row.trace is None
    assert row.source is None
