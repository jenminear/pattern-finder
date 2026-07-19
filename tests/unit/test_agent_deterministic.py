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
        # directly (ascending order: td_a < td_b) rather than leaving it to
        # infer an order from request context.
        assert f"x0={a}, x1={b}" in handoff

    @pytest.mark.asyncio
    async def test_exact_pattern_apply_is_robust_to_request_label_order(self, agent_module):
        # Regression test: x0/x1/... must be anchored to the label set
        # sorted in ascending order, NOT to whatever order a given request
        # happens to list labels in (there's no order stored in the DB --
        # labels are matched as a set). Names below sort as a, b, c.
        a, b, c = "ro_a", "ro_b", "ro_c"
        ids = [
            db_ops.insert_scenario("t", {a: str(x), b: str(y), c: str(z)}, str(x + y * z))
            for x, y, z in [(1, 2, 3), (4, 5, 6)]
        ]
        # x0=a, x1=b, x2=c (ascending order) -- rule means a + b*c.
        pid = db_ops.upsert_pattern("combo", "x0 + x1*x2")
        db_ops.link_pattern_to_scenarios(
            pid, ids, update_label_set=True, label_names=[a, b, c]
        )

        # Submit the SAME values but listed in a different order than
        # ascending (c first) -- must still compute a + b*c, not
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
        content, handoff = await agent_module._learn_phase_content(
            {"a": "1", "b": "2"}, "99", None, None, None, "42", True, 0.5
        )
        assert handoff is None
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
        content, handoff = await agent_module._learn_phase_content(
            {"a": "1", "b": "2"}, "99", None, None, None, "I don't know", True, 0.5
        )
        assert handoff is None
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

        content, handoff = await agent_module._learn_phase_content(
            {a: "10", b: "20"}, "30", "exact_pattern", pid, None, "30", True, 0.5
        )
        assert handoff is None
        assert "Correct" in content.parts[0].text
        scenarios = db_ops.get_scenarios_by_label_set([a, b])
        assert len(scenarios) == 3
        assert all(s["pattern_id"] == pid for s in scenarios)

    @pytest.mark.asyncio
    async def test_label_independent_match_links_only_this_scenario(self, agent_module):
        pid = db_ops.upsert_pattern("abstracted pattern", "x0 * 2")
        content, handoff = await agent_module._learn_phase_content(
            {"li_a": "3", "li_b": "4"}, "6", "label_independent_match", pid, None, "6", True, 0.5
        )
        assert handoff is None
        assert "Correct" in content.parts[0].text
        # Safety rule: must NOT register this label set against the pattern.
        assert db_ops.get_pattern_by_label_set(["li_a", "li_b"]) is None
        scenarios = db_ops.get_scenarios_by_label_set(["li_a", "li_b"])
        assert scenarios[0]["pattern_id"] == pid

    @pytest.mark.asyncio
    async def test_label_independent_match_wrong_guess_is_not_revised(self, agent_module):
        # A label-independent match failing doesn't mean the pattern is
        # wrong for its OWN label set -- just that it didn't generalize to
        # this unrelated one. Must fall through to plain "no pattern"
        # recording, not _attempt_pattern_revision.
        pid = db_ops.upsert_pattern("some pattern", "x0 * 2")
        content, handoff = await agent_module._learn_phase_content(
            {"liw_a": "3"}, "99", "label_independent_match", pid, "x0 * 2", "6", True, 0.5
        )
        assert handoff is None
        assert "No pattern was created or linked" in content.parts[0].text
        unchanged = next(
            p for p in db_ops.get_all_pattern_descriptions() if p["pattern_id"] == pid
        )
        assert unchanged["rule_or_code_link"] == "x0 * 2"


class TestPatternRevision:
    # Section VII's "go back and re-evaluate" requirement: a confident
    # exact_pattern guess just turned out wrong. Both tests below rely on
    # the fact that _learn_phase_content already inserted the new
    # counterexample scenario BEFORE checking `matched`, so by the time
    # revision runs, the counterexample is already part of this label
    # set's history.
    @pytest.mark.asyncio
    async def test_deterministic_revision_replaces_rule_with_no_llm_needed_for_the_fit(
        self, agent_module
    ):
        # Seed a pattern from a single ambiguous example: x0=1 is
        # consistent with BOTH "x0" (identity) and "x0^2". The stored rule
        # happens to be the wrong one ("x0").
        label = "rev_x"
        seed_id = db_ops.insert_scenario("numeric", {label: "1"}, "1")
        pid = db_ops.upsert_pattern("identity", "x0")
        db_ops.link_pattern_to_scenarios(pid, [seed_id], update_label_set=True, label_names=[label])

        # Old rule "x0" predicts guess=2 for x0=2, but the true answer is 4
        # (x0^2) -- now disambiguated by this second data point.
        original = agent_module._narrow_llm_call

        async def fake_narrow_llm_call(prompt, model):
            return "the output is the square of the single input"

        agent_module._narrow_llm_call = fake_narrow_llm_call
        try:
            content, handoff = await agent_module._learn_phase_content(
                {label: "2"}, "4", "exact_pattern", pid, "x0", "2", True, 0.5
            )
        finally:
            agent_module._narrow_llm_call = original

        assert handoff is None
        assert content is not None
        raw = content.parts[1].inline_data.data[
            len(agent_module.A2A_DATA_PART_START_TAG) : -len(
                agent_module.A2A_DATA_PART_END_TAG
            )
        ]
        payload = json.loads(raw)["data"]
        assert payload["matched"] is False
        assert payload["action"] == "revised"
        assert "x^2" in payload["rule_or_code_link"]

        updated = db_ops.get_pattern_by_label_set([label])
        assert updated["pattern_id"] == pid
        assert "x^2" in updated["rule_or_code_link"]
        scenarios = db_ops.get_scenarios_by_label_set([label])
        assert len(scenarios) == 2  # seed + the new counterexample
        assert all(s["pattern_id"] == pid for s in scenarios)

    @pytest.mark.asyncio
    async def test_hands_off_when_deterministic_script_cant_fix_it(self, agent_module):
        # No pre-seeded history at all for this label set -- after
        # _learn_phase_content inserts the new (sole) scenario, history has
        # exactly 1 row, below find_pattern's own 2-row minimum, so the
        # deterministic re-search is guaranteed to come up empty and this
        # must hand off rather than silently keep (or blindly discard) the
        # old rule.
        pid = db_ops.upsert_pattern("some pattern", "x0 + 1")
        content, handoff = await agent_module._learn_phase_content(
            {"nopat_x": "5"}, "99", "exact_pattern", pid, "x0 + 1", "6", True, 0.5
        )
        assert content is None
        assert handoff is not None
        assert str(pid) in handoff
        assert "x0 + 1" in handoff
        assert "6" in handoff  # the wrong guess
        assert "99" in handoff  # the correct consequence
        assert "upsert_pattern" in handoff  # agent saves the revision itself
        # The old rule must be untouched -- nothing confident was found.
        unchanged = next(
            p for p in db_ops.get_all_pattern_descriptions() if p["pattern_id"] == pid
        )
        assert unchanged["rule_or_code_link"] == "x0 + 1"


class TestSanitizeScenarioValues:
    # Regression coverage for a real bug observed live: a value field ended
    # up storing "66I don't know." / "bananaI don't know." in
    # scenario_inputs -- the agent's own "I don't know" phrase leaking into
    # what should be pure user-entered data.
    def test_strips_values_containing_dont_know(self, agent_module):
        result = agent_module._sanitize_scenario_values(
            {"a": "66I don't know.", "b": "bananaI don't know.", "c": "42"}
        )
        assert result == {"a": None, "b": None, "c": "42"}

    def test_leaves_clean_values_and_none_untouched(self, agent_module):
        result = agent_module._sanitize_scenario_values({"a": "apple", "b": None})
        assert result == {"a": "apple", "b": None}

    def test_case_insensitive_and_no_apostrophe_variant(self, agent_module):
        result = agent_module._sanitize_scenario_values(
            {"a": "I DONT KNOW", "b": "Don't Know"}
        )
        assert result == {"a": None, "b": None}


class TestInferScenarioType:
    def test_all_numeric(self, agent_module):
        assert agent_module._infer_scenario_type({"a": "1", "b": "2.5"}) == "numeric"

    def test_all_text(self, agent_module):
        assert agent_module._infer_scenario_type({"a": "apple", "b": "cherry"}) == "text"

    def test_mixed(self, agent_module):
        assert agent_module._infer_scenario_type({"a": "1", "b": "apple"}) == "mixed"

    def test_empty_when_all_none(self, agent_module):
        assert agent_module._infer_scenario_type({"a": None, "b": None}) == "empty"

    def test_ignores_none_slots_when_classifying(self, agent_module):
        # A missing slot shouldn't count against an otherwise-numeric scenario.
        assert agent_module._infer_scenario_type({"a": "1", "b": None}) == "numeric"


class TestLearnPhaseSanitizesAndTypesScenario:
    @pytest.mark.asyncio
    async def test_dont_know_pollution_is_stripped_before_storage(self, agent_module):
        await agent_module._learn_phase_content(
            {"pol_a": "66I don't know.", "pol_b": "9"},
            "99",
            None,
            None,
            None,
            "42",
            True,
            0.5,
        )
        scenarios = db_ops.get_scenarios_by_label_set(["pol_a", "pol_b"])
        assert len(scenarios) == 1
        assert scenarios[0]["inputs"]["pol_a"] is None
        assert scenarios[0]["inputs"]["pol_b"] == "9"
        # Only one non-None value remains ("9", numeric) -- still "numeric",
        # not "mixed", since the polluted slot was dropped rather than kept
        # as a text value.
        assert scenarios[0]["type"] == "numeric"

    @pytest.mark.asyncio
    async def test_type_recorded_as_text_for_non_numeric_scenario(self, agent_module):
        await agent_module._learn_phase_content(
            {"typ_a": "apple", "typ_b": "cherry"}, "banana", None, None, None, "banana", True, 0.5
        )
        scenarios = db_ops.get_scenarios_by_label_set(["typ_a", "typ_b"])
        assert scenarios[0]["type"] == "text"


class TestWritePatternDescriptionPrompt:
    # Regression coverage for a real bug: a bare "xN" rule (no operator --
    # an identity/copy rule meaning "output = that one input's value
    # verbatim") was described by the LLM as "the mathematical double of
    # their corresponding index", because the prompt gave it the rule
    # string with zero explanation of the x0/x1/... convention. Doesn't
    # call the real LLM (that's covered by tests/integration/test_agent.py)
    # -- just asserts the prompt actually carries the disambiguating
    # context, since that's the part that regresses silently.
    @pytest.mark.asyncio
    async def test_prompt_explains_bare_reference_is_not_arithmetic(self, agent_module):
        captured = {}

        async def fake_narrow_llm_call(prompt, model):
            captured["prompt"] = prompt
            return "a description"

        # app.agent is a cached module (re-imported across tests via the
        # agent_module fixture returns the SAME object), so save/restore
        # rather than delete the attribute.
        original = agent_module._narrow_llm_call
        agent_module._narrow_llm_call = fake_narrow_llm_call
        try:
            await agent_module._write_pattern_description("x2", "some-model")
        finally:
            agent_module._narrow_llm_call = original

        prompt = captured["prompt"]
        assert "not exponents or coefficients" in prompt
        assert "does NOT mean" in prompt
        assert "identity/copy rule" in prompt
        assert "The confirmed rule/logic: x2" in prompt


class TestEffortTierConfig:
    # All three tiers now share one model (gemini-3.1-flash-lite) -- a
    # live GCP billing review traced disproportionate spend to
    # gemini-3.5-flash/gemini-2.5-pro combined with an uncapped
    # max_llm_calls (see app/fast_api_app.py's cap and app/agent.py's
    # _FLASH_LITE comment for the full writeup), not to per-call cost on
    # the models themselves. thinking_budget values 0/8192/24576 were all
    # confirmed live against the real API before this change, not assumed.
    def test_all_tiers_share_the_same_model(self, agent_module):
        models = {
            t.model
            for t in (
                agent_module._ECONOMY,
                agent_module._BALANCED,
                agent_module._MAX_QUALITY,
            )
        }
        assert models == {"gemini-3.1-flash-lite"}

    def test_each_tier_uses_thinking_budget_not_level(self, agent_module):
        for tier in (
            agent_module._ECONOMY,
            agent_module._BALANCED,
            agent_module._MAX_QUALITY,
        ):
            assert tier.thinking_level is None, f"{tier.name}: thinking_level should be unused"
            assert tier.thinking_budget is not None, f"{tier.name}: thinking_budget must be set"

    def test_thinking_budget_increases_with_tier(self, agent_module):
        assert (
            agent_module._ECONOMY.thinking_budget
            < agent_module._BALANCED.thinking_budget
            < agent_module._MAX_QUALITY.thinking_budget
        )


class TestConsequenceMatchesNumericRounding:
    # Regression coverage for a real issue found in live testing: numeric
    # correctness was being judged with a 1e-6 tolerance, so ordinary
    # floating-point/division noise several decimal places out (not a
    # genuine miss) was labeled "Incorrect". Comparison now rounds both
    # values to 2 decimal places -- these tests never reach the model
    # param (numeric path is fully deterministic), so a placeholder
    # string is fine there.
    @pytest.mark.asyncio
    async def test_exact_integer_match(self, agent_module):
        assert await agent_module._consequence_matches("30", "30", "unused") is True

    @pytest.mark.asyncio
    async def test_noise_past_two_decimals_still_matches(self, agent_module):
        assert (
            await agent_module._consequence_matches("3.333333333", "3.33", "unused") is True
        )
        assert (
            await agent_module._consequence_matches("12.129999999", "12.13", "unused") is True
        )

    @pytest.mark.asyncio
    async def test_genuine_miss_at_two_decimals_does_not_match(self, agent_module):
        assert await agent_module._consequence_matches("3.34", "3.33", "unused") is False

    @pytest.mark.asyncio
    async def test_rounding_is_symmetric_on_the_correct_value_too(self, agent_module):
        # The correct_consequence side gets rounded too, not just the guess.
        assert await agent_module._consequence_matches("3.33", "3.334", "unused") is True
        assert await agent_module._consequence_matches("3.33", "3.336", "unused") is False

    @pytest.mark.asyncio
    async def test_dont_know_never_matches_regardless_of_closeness(self, agent_module):
        assert await agent_module._consequence_matches("I don't know", "3.33", "unused") is False

    @pytest.mark.asyncio
    async def test_non_numeric_guess_against_numeric_correct_does_not_match(self, agent_module):
        assert await agent_module._consequence_matches("banana", "3.33", "unused") is False


class TestPatternSearchCandidateLogging:
    # pattern_search_agent gets its own narrow McpToolset (only
    # log_candidate_technique) rather than the full db_mcp_toolset
    # root_agent uses -- it has no business calling insert_scenario/
    # upsert_pattern/etc. directly. This is a structural check, not a live
    # MCP call: confirms the wiring is scoped correctly without needing a
    # running MCP subprocess.
    def test_pattern_search_agent_only_has_the_logging_tool(self, agent_module):
        assert agent_module.pattern_search_mcp_toolset in agent_module.pattern_search_agent.tools
        assert agent_module.pattern_search_mcp_toolset.tool_filter == ["log_candidate_technique"]

    def test_root_agent_keeps_the_full_toolset(self, agent_module):
        assert agent_module.db_mcp_toolset in agent_module.root_agent.tools
        # root_agent must NOT also be handed the narrow toolset -- it
        # already has everything via db_mcp_toolset, and giving it both
        # would just be redundant.
        assert agent_module.pattern_search_mcp_toolset not in agent_module.root_agent.tools

    def test_instruction_tells_it_to_log_even_low_confidence_finds(self, agent_module):
        assert "log_candidate_technique" in agent_module._PATTERN_SEARCH_INSTRUCTION
        assert "free parameters" in agent_module._PATTERN_SEARCH_INSTRUCTION
