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
"""Integration tests for the Pattern Finder agent -- exercises the real
agent, real MCP subprocess, and (only where noted) real Gemini calls, per
.agents-cli-spec.md's two-phase Guess/Learn workflow.

Most turns now resolve inside app.agent._before_agent_dispatch before any
LLM call happens at all (see app/agent.py's module docstring) -- so most
tests below exercise the deterministic paths through the REAL Runner/MCP
stack and don't need a live model call or gcloud auth. Only
test_guess_with_no_history_says_dont_know genuinely needs a live call
(there's no deterministic answer for a never-before-seen label set with no
script match), and will fail without valid ADC credentials.
"""

import json

import pytest
from google.adk.a2a.converters.part_converter import (
    A2A_DATA_PART_END_TAG,
    A2A_DATA_PART_START_TAG,
)
from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent
from app.mcp_server import db_ops


@pytest.fixture(scope="module", autouse=True)
def isolated_db():
    """Clears the shared local dev DB before/after this module's tests.

    Can't point this at a per-test tmp file: app.agent.db_mcp_toolset's
    subprocess env is captured once, at module-import time (pytest
    collection, via `from app.agent import root_agent` above) -- which
    happens before any fixture runs, so a later PATTERN_FINDER_DB_URL
    override would never reach the already-configured subprocess. Instead
    this clears the actual default DB file's tables, so this process's
    seeding calls and the subprocess's tool calls agree on the same data.
    Tests stay isolated from each other by using disjoint label sets.
    """
    from app.db import models
    from app.db.engine import get_engine

    get_engine.cache_clear()
    engine = get_engine()
    with engine.begin() as conn:
        for table in reversed(models.metadata.sorted_tables):
            conn.execute(table.delete())
    yield


async def _send(runner: Runner, session_id: str, text: str) -> tuple[str, list[dict]]:
    """Sends one turn, returns (accumulated text, A2UI DataPart payloads)."""
    message = types.Content(role="user", parts=[types.Part.from_text(text=text)])
    texts = []
    data_parts = []
    async for event in runner.run_async(
        new_message=message,
        user_id="test_user",
        session_id=session_id,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        if not (event.content and event.content.parts):
            continue
        for part in event.content.parts:
            if part.text:
                texts.append(part.text)
            elif (
                part.inline_data
                and part.inline_data.data
                and part.inline_data.data.startswith(A2A_DATA_PART_START_TAG)
            ):
                raw = part.inline_data.data[
                    len(A2A_DATA_PART_START_TAG) : -len(A2A_DATA_PART_END_TAG)
                ]
                data_parts.append(json.loads(raw)["data"])
    return "".join(texts), data_parts


@pytest.mark.asyncio
async def test_guess_with_no_history_says_dont_know() -> None:
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="test", user_id="test_user")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    reply, data_parts = await _send(
        runner,
        session.id,
        "PHASE: guess\nlabel=never_seen_before_x value=1\nlabel=never_seen_before_y value=2",
    )

    assert "don't know" in reply.lower() or "do not know" in reply.lower()
    reasoning = next(d for d in data_parts if d.get("surface") == "agent_reasoning")
    assert reasoning["matched_via"] == "none"
    assert reasoning["confidence"] == 0


@pytest.mark.asyncio
async def test_guess_then_learn_applies_and_captures_pattern() -> None:
    # Pre-seed history via direct DB calls (fast/deterministic) rather than
    # real LLM turns -- this test's job is to verify the REAL agent stack's
    # (Runner + before_agent_callback + real MCP subprocess) exact-pattern
    # deterministic fast path end to end (Section VI steps 1-2 and 9), not
    # to re-derive pattern discovery from scratch (already covered in
    # tests/unit/test_db_ops.py and tests/unit/test_agent_deterministic.py).
    # Rule uses the x0/x1 convention (see evaluate_numeric_rule's docstring)
    # -- NOT the literal label names -- since that's what makes the
    # deterministic evaluator (not the LLM) able to apply it.
    label_a, label_b = "seed_x", "seed_y"
    seeded_ids = [
        db_ops.insert_scenario("t", {label_a: str(a), label_b: str(b)}, str(a + b))
        for a, b in [(1, 2), (3, 4), (5, 6)]
    ]
    pattern_id = db_ops.upsert_pattern("sum of the two inputs", "x0 + x1")
    db_ops.link_pattern_to_scenarios(
        pattern_id, seeded_ids, update_label_set=True, label_names=[label_a, label_b]
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="test", user_id="test_user")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    guess_reply, guess_data = await _send(
        runner, session.id, f"PHASE: guess\nlabel={label_a} value=10\nlabel={label_b} value=20"
    )
    assert "30" in guess_reply
    reasoning = next(d for d in guess_data if d.get("surface") == "agent_reasoning")
    assert reasoning["matched_via"] == "exact_pattern"
    assert reasoning["pattern_id"] == pattern_id

    # Echo back what the Guess-phase turn reported, as the real frontend
    # does (frontend/app.js) -- the deterministic Learn-phase orchestrator
    # has no access to the prior turn's reasoning otherwise.
    learn_reply, learn_data = await _send(
        runner,
        session.id,
        "PHASE: learn\nCORRECT_CONSEQUENCE: 30\n"
        f"label={label_a} value=10\nlabel={label_b} value=20\n"
        "GUESS_VALUE: 30\nMATCHED_VIA: exact_pattern\n"
        f"PATTERN_ID: {pattern_id}",
    )
    assert "record" in learn_reply.lower()
    captured = next(d for d in learn_data if d.get("surface") == "pattern_captured")
    assert captured["action"] == "updated_label_set"
    assert captured["matched"] is True

    # Section VI step 9: every scenario sharing this exact label set --
    # the 3 seeded ones plus the new one -- must now carry the pattern_id.
    scenarios = db_ops.get_scenarios_by_label_set([label_a, label_b])
    assert len(scenarios) == 4
    assert all(s["pattern_id"] == pattern_id for s in scenarios)


@pytest.mark.asyncio
async def test_guess_via_seeded_script_needs_no_llm_call() -> None:
    # Section VII's "3rd branch": no exact pattern for this label set, but
    # the seeded pattern_search_script finds a confident fit by itself --
    # _before_agent_dispatch must resolve this WITHOUT invoking the LLM at
    # all. Two rows is enough for try_single_variable_transforms to trust
    # an exact match (see pattern_search_script.py).
    label = "script_only_x"
    for x in (3, 5):
        db_ops.insert_scenario("t", {label: str(x)}, str(x**2))

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="test", user_id="test_user")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    guess_reply, guess_data = await _send(runner, session.id, f"PHASE: guess\nlabel={label} value=7")
    assert "49" in guess_reply
    reasoning = next(d for d in guess_data if d.get("surface") == "agent_reasoning")
    assert reasoning["matched_via"] == "pattern_search"


@pytest.mark.asyncio
async def test_wrong_exact_pattern_guess_triggers_deterministic_revision() -> None:
    # A confident exact_pattern guess turns out wrong -- the deterministic
    # Learn-phase layer must re-search the FULL history for this label set
    # (now including the new counterexample) rather than silently keeping
    # the wrong rule or discarding it. Seeded from a single ambiguous
    # example (x0=1 is consistent with both "x0" and "x0^2"), so the
    # stored rule happens to be the wrong one until a second data point
    # disambiguates it.
    label = "rev_seed_x"
    seed_id = db_ops.insert_scenario("numeric", {label: "1"}, "1")
    pattern_id = db_ops.upsert_pattern("identity", "x0")
    db_ops.link_pattern_to_scenarios(
        pattern_id, [seed_id], update_label_set=True, label_names=[label]
    )

    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name="test", user_id="test_user")
    runner = Runner(agent=root_agent, session_service=session_service, app_name="test")

    guess_reply, guess_data = await _send(runner, session.id, f"PHASE: guess\nlabel={label} value=2")
    assert "2" in guess_reply  # old rule "x0" predicts 2
    reasoning = next(d for d in guess_data if d.get("surface") == "agent_reasoning")
    assert reasoning["matched_via"] == "exact_pattern"

    learn_reply, learn_data = await _send(
        runner,
        session.id,
        f"PHASE: learn\nCORRECT_CONSEQUENCE: 4\nlabel={label} value=2\n"
        f"GUESS_VALUE: 2\nMATCHED_VIA: exact_pattern\nPATTERN_ID: {pattern_id}",
    )
    assert "better-fitting rule" in learn_reply.lower()
    captured = next(d for d in learn_data if d.get("surface") == "pattern_captured")
    assert captured["action"] == "revised"
    assert captured["matched"] is False
    assert "x^2" in captured["rule_or_code_link"]

    updated = db_ops.get_pattern_by_label_set([label])
    assert updated["pattern_id"] == pattern_id
    assert "x^2" in updated["rule_or_code_link"]
    scenarios = db_ops.get_scenarios_by_label_set([label])
    assert len(scenarios) == 2
    assert all(s["pattern_id"] == pattern_id for s in scenarios)
