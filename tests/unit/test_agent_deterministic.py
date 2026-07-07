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
"""Unit tests for the deterministic Guess/Learn-phase orchestration in
app/agent.py -- the whole point of the revised architecture (see project
memory / .agents-cli-spec.md) is that most turns resolve WITHOUT an LLM
call, so these paths get direct, fast, no-network coverage here. Cases
needing an actual LLM call (non-numeric comparison, new-pattern
description, applying a text-directive rule, freeform search) are covered
by tests/integration/test_agent.py instead.
"""

import json

import pytest

from app.mcp_server import db_ops


@pytest.fixture(autouse=True)
def isolated_db(monkeypatch, tmp_path):
    db_path = tmp_path / "test.db"
    monkeypatch.setenv("PATTERN_FINDER_DB_URL", f"sqlite:///{db_path}")
    from app.db.engine import get_engine

    get_engine.cache_clear()
    yield
    get_engine.cache_clear()


@pytest.fixture
def agent_module():
    # Imported lazily, after the DB env var is patched, and via a fixture
    # (not module-level import) so app.agent's own module-level side
    # effects (loading .env, resolving GOOGLE_CLOUD_PROJECT) don't run
    # before the test's env patches are in place.
    import app.agent as m

    return m


class TestGuessPhaseContent:
    @pytest.mark.asyncio
    async def test_no_history_hands_off_with_freeform_reason(self, agent_module):
        content, handoff = await agent_module._guess_phase_content(
            {"never_seen_x": "1", "never_seen_y": "2"}, emit_ui=True
        )
        assert content is None
        assert handoff is not None
        assert "no exact pattern exists" in handoff.lower()

    @pytest.mark.asyncio
    async def test_exact_numeric_pattern_resolves_without_handoff(self, agent_module):
        a, b = "ex_a", "ex_b"
        ids = [
            db_ops.insert_scenario("t", {a: str(x), b: str(y)}, str(x + y))
            for x, y in [(1, 2), (3, 4)]
        ]
        pid = db_ops.upsert_pattern("sum", "x0 + x1")
        db_ops.link_pattern_to_scenarios(pid, ids, update_label_set=True, label_names=[a, b])

        content, handoff = await agent_module._guess_phase_content(
            {a: "10", b: "20"}, emit_ui=True
        )
        assert handoff is None
        assert content is not None
        assert "30" in content.parts[0].text
        assert len(content.parts) == 2  # text + A2UI DataPart
        assert content.parts[1].inline_data.data.startswith(b"<a2a_datapart_json>")

    @pytest.mark.asyncio
    async def test_exact_numeric_pattern_emit_ui_off_omits_datapart(self, agent_module):
        a, b = "off_a", "off_b"
        ids = [db_ops.insert_scenario("t", {a: "1", b: "2"}, "3")]
        pid = db_ops.upsert_pattern("sum", "x0 + x1")
        db_ops.link_pattern_to_scenarios(pid, ids, update_label_set=True, label_names=[a, b])

        content, _ = await agent_module._guess_phase_content(
            {a: "1", b: "2"}, emit_ui=False
        )
        assert content is not None
        assert len(content.parts) == 1  # text only, no DataPart

    @pytest.mark.asyncio
    async def test_text_directive_pattern_hands_off_with_apply_known_pattern(
        self, agent_module
    ):
        a, b = "td_a", "td_b"
        ids = [db_ops.insert_scenario("t", {a: "1", b: "2"}, "large")]
        pid = db_ops.upsert_pattern(
            "qualitative size", "the bigger of the two values determines the outcome"
        )
        db_ops.link_pattern_to_scenarios(pid, ids, update_label_set=True, label_names=[a, b])

        content, handoff = await agent_module._guess_phase_content(
            {a: "5", b: "1"}, emit_ui=True
        )
        assert content is None
        assert handoff is not None
        assert str(pid) in handoff
        assert "qualitative size" not in handoff  # only the rule, not the description
        assert "the bigger of the two values" in handoff
        # The handoff must give the LLM the canonical x0/x1/... mapping
        # directly (alphabetical: td_a < td_b) rather than leaving it to
        # infer an order from request context.
        assert f"x0={a}, x1={b}" in handoff

    @pytest.mark.asyncio
    async def test_exact_pattern_apply_is_robust_to_request_label_order(self, agent_module):
        # Regression test: x0/x1/... must be anchored to the label set
        # sorted alphabetically, NOT to whatever order a given request
        # happens to list labels in (there's no order stored in the DB --
        # labels are matched as a set). Names below sort as a, b, c.
        a, b, c = "ro_a", "ro_b", "ro_c"
        ids = [
            db_ops.insert_scenario("t", {a: str(x), b: str(y), c: str(z)}, str(x + y * z))
            for x, y, z in [(1, 2, 3), (4, 5, 6)]
        ]
        # x0=a, x1=b, x2=c (alphabetical) -- rule means a + b*c.
        pid = db_ops.upsert_pattern("combo", "x0 + x1*x2")
        db_ops.link_pattern_to_scenarios(
            pid, ids, update_label_set=True, label_names=[a, b, c]
        )

        # Submit the SAME values but listed in a different order than
        # alphabetical (c first) -- must still compute a + b*c, not
        # silently reassign x0/x1/x2 to the request's own order.
        content, handoff = await agent_module._guess_phase_content(
            {c: "3", a: "1", b: "2"}, emit_ui=True
        )
        assert handoff is None
        assert content is not None
        assert "7" in content.parts[0].text  # 1 + 2*3 = 7, not 3 + 1*2 = 5

    @pytest.mark.asyncio
    async def test_script_finds_confident_polynomial_fit_deterministically(
        self, agent_module
    ):
        # a*b, 6 fully-populated rows -> fully-determined degree-2 fit.
        a, b = "poly_a", "poly_b"
        for x, y in [(2, 3), (4, 5), (6, 7), (1, 1), (0, 5), (3, 3)]:
            db_ops.insert_scenario("t", {a: str(x), b: str(y)}, str(x * y))

        content, handoff = await agent_module._guess_phase_content(
            {a: "10", b: "3"}, emit_ui=True
        )
        assert handoff is None
        assert content is not None
        assert "30" in content.parts[0].text

    @pytest.mark.asyncio
    async def test_insufficient_history_stays_conservative(self, agent_module):
        # Only ONE prior example -- underdetermined, must not auto-answer.
        a, b = "sparse_a", "sparse_b"
        db_ops.insert_scenario("t", {a: "5", b: "2"}, "7")

        content, handoff = await agent_module._guess_phase_content(
            {a: "8", b: "1"}, emit_ui=True
        )
        assert content is None
        assert handoff is not None
        assert "no confident" in handoff.lower()


class TestLearnPhaseContent:
    @pytest.mark.asyncio
    async def test_no_match_records_scenario_without_pattern(self, agent_module):
        content = await agent_module._learn_phase_content(
            {"a": "1", "b": "2"}, "99", None, None, None, "42", True, 0.5
        )
        assert "No pattern was created or linked" in content.parts[0].text
        raw = content.parts[1].inline_data.data[
            len(agent_module.A2A_DATA_PART_START_TAG) : -len(
                agent_module.A2A_DATA_PART_END_TAG
            )
        ]
        payload = json.loads(raw)["data"]
        assert payload["matched"] is False
        assert payload["action"] == "none"

    @pytest.mark.asyncio
    async def test_dont_know_guess_records_without_pattern(self, agent_module):
        content = await agent_module._learn_phase_content(
            {"a": "1", "b": "2"}, "99", None, None, None, "I don't know", True, 0.5
        )
        assert "No pattern was created or linked" in content.parts[0].text

    @pytest.mark.asyncio
    async def test_existing_exact_pattern_links_all_matching_scenarios(self, agent_module):
        a, b = "learn_a", "learn_b"
        seeded = [
            db_ops.insert_scenario("t", {a: str(x), b: str(y)}, str(x + y))
            for x, y in [(1, 2), (3, 4)]
        ]
        pid = db_ops.upsert_pattern("sum", "x0 + x1")
        db_ops.link_pattern_to_scenarios(pid, seeded, update_label_set=True, label_names=[a, b])

        content = await agent_module._learn_phase_content(
            {a: "10", b: "20"}, "30", "exact_pattern", pid, None, "30", True, 0.5
        )
        assert "Correct" in content.parts[0].text
        scenarios = db_ops.get_scenarios_by_label_set([a, b])
        assert len(scenarios) == 3
        assert all(s["pattern_id"] == pid for s in scenarios)

    @pytest.mark.asyncio
    async def test_label_independent_match_links_only_this_scenario(self, agent_module):
        pid = db_ops.upsert_pattern("abstracted pattern", "x0 * 2")
        content = await agent_module._learn_phase_content(
            {"li_a": "3", "li_b": "4"}, "6", "label_independent_match", pid, None, "6", True, 0.5
        )
        assert "Correct" in content.parts[0].text
        # Safety rule: must NOT register this label set against the pattern.
        assert db_ops.get_pattern_by_label_set(["li_a", "li_b"]) is None
        scenarios = db_ops.get_scenarios_by_label_set(["li_a", "li_b"])
        assert scenarios[0]["pattern_id"] == pid
