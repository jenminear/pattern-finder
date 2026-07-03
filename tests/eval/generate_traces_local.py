"""Local trace generator, standing in for `agents-cli eval generate`.

`agents-cli eval generate` delegates to Vertex AI's managed inference
(`vertexai._genai.evals.run_inference`), which builds an AgentConfig by
calling `inspect.signature()` on every item in the agent's `tools` list --
this doesn't handle `McpToolset` (a toolset, not a single callable) and
fails with "McpToolset object is not a callable object" for every case.
ADK's own Runner has no such issue (proven repeatedly in
tests/integration/test_agent.py), so this script runs inference locally
through it and writes traces in the exact schema `agents-cli eval grade`
expects -- grading itself is untouched, only inference is local instead of
managed.

Usage:
    uv run python tests/eval/seed_eval_data.py   # once, if not already done
    uv run python tests/eval/generate_traces_local.py
    agents-cli eval grade --traces artifacts/traces/local_traces.json
"""

import asyncio
import json
from pathlib import Path

from google.adk.agents.run_config import RunConfig, StreamingMode
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from app.agent import root_agent

_DATASET_PATH = Path(__file__).parent / "datasets" / "basic-dataset.json"
_OUTPUT_PATH = Path(__file__).parent.parent.parent / "artifacts" / "traces" / "local_traces.json"


def _part_to_dict(part: types.Part) -> dict | None:
    if part.text is not None:
        return {"text": part.text}
    if part.function_call is not None:
        return {
            "function_call": {
                "name": part.function_call.name,
                "args": part.function_call.args,
            }
        }
    if part.function_response is not None:
        return {
            "function_response": {
                "name": part.function_response.name,
                "response": part.function_response.response,
            }
        }
    return None


async def _run_case(runner: Runner, case: dict) -> dict:
    session_service = runner.session_service
    session = await session_service.create_session(
        app_name=runner.app_name, user_id="eval_user"
    )
    user_text = case["prompt"]["parts"][0]["text"]
    message = types.Content(role="user", parts=[types.Part.from_text(text=user_text)])

    events = [
        {"author": "user", "content": {"parts": [{"text": user_text}]}}
    ]
    final_text_parts: list[dict] = []
    async for event in runner.run_async(
        new_message=message,
        user_id="eval_user",
        session_id=session.id,
        run_config=RunConfig(streaming_mode=StreamingMode.SSE),
    ):
        if not (event.content and event.content.parts):
            continue
        parts = [d for p in event.content.parts if (d := _part_to_dict(p)) is not None]
        if parts:
            events.append({"author": root_agent.name, "content": {"parts": parts}})
        text_parts = [p for p in parts if "text" in p]
        if text_parts:
            final_text_parts = text_parts  # keep overwriting -> last one wins

    return {
        "eval_case_id": case["eval_case_id"],
        # Required alongside agent_data for grading (dataset_schema.md):
        # metrics look for a ResponseCandidate here, not derived from
        # agent_data.turns automatically.
        "responses": [{"response": {"role": "model", "parts": final_text_parts}}],
        "agent_data": {
            "agents": {
                root_agent.name: {
                    "agent_id": root_agent.name,
                    "instruction": root_agent.instruction,
                }
            },
            "turns": [{"turn_index": 0, "events": events}],
        },
    }


async def main() -> None:
    dataset = json.loads(_DATASET_PATH.read_text(encoding="utf-8"))
    session_service = InMemorySessionService()
    runner = Runner(agent=root_agent, session_service=session_service, app_name="eval")

    traced_cases = []
    for case in dataset["eval_cases"]:
        print(f"running {case['eval_case_id']}...")
        traced_cases.append(await _run_case(runner, case))

    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _OUTPUT_PATH.write_text(
        json.dumps({"eval_cases": traced_cases}, indent=2), encoding="utf-8"
    )
    print(f"wrote {len(traced_cases)} traces to {_OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
