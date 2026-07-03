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
agent (real Gemini calls, real MCP subprocess) end to end, per
.agents-cli-spec.md's two-phase Guess/Learn workflow.
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
    # real LLM turns -- this test's job is to verify the REAL agent's
    # exact-pattern-match + capture behavior (Section VI steps 1-2 and 9),
    # not to re-derive pattern discovery from scratch (already covered in
    # tests/unit/test_db_ops.py and validated manually against
    # pattern-search specifically).
    label_a, label_b = "seed_x", "seed_y"
    seeded_ids = [
        db_ops.insert_scenario("t", {label_a: str(a), label_b: str(b)}, str(a + b))
        for a, b in [(1, 2), (3, 4), (5, 6)]
    ]
    pattern_id = db_ops.upsert_pattern("sum of the two inputs", f"{label_a} + {label_b}")
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

    learn_reply, learn_data = await _send(
        runner, session.id, "PHASE: learn\nCORRECT_CONSEQUENCE: 30"
    )
    assert "record" in learn_reply.lower()
    captured = next(d for d in learn_data if d.get("surface") == "pattern_captured")
    assert captured["action"] == "updated_label_set"

    # Section VI step 9: every scenario sharing this exact label set --
    # the 3 seeded ones plus the new one -- must now carry the pattern_id.
    scenarios = db_ops.get_scenarios_by_label_set([label_a, label_b])
    assert len(scenarios) == 4
    assert all(s["pattern_id"] == pattern_id for s in scenarios)
