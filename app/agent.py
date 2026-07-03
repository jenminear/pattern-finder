# ruff: noqa
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

"""Pattern Finder agent.

Implements the two-phase flow from .agents-cli-spec.md / Pattern Finder
Outline.txt as two turns of ONE ADK session, not two agents:

    Turn 1 (Handoff 1, "PHASE: guess"): the deterministic app layer has
    already tried an exact label-set + pattern lookup (see
    app/mcp_server/db_ops.get_pattern_by_label_set, called directly by the
    API layer) and found nothing, so control passes to this agent to
    search for and apply a pattern, or say "I don't know".

    Turn 2 (Handoff 2, "PHASE: learn"), same session: the user has
    confirmed the correct consequence. The agent persists the scenario and
    -- on a correct guess -- captures or links the pattern that produced
    it, per Section VI/VII's label-set-update rules.

Every user message is expected to include a "PHASE: guess" or
"PHASE: learn" line, and may include an "EFFORT_DIAL: <0-1 float>" line
(see _ROOT_INSTRUCTION and _seed_effort_defaults below) -- that's the
contract the API layer (app/fast_api_app.py) is responsible for
constructing.
"""

import ast
import operator
import os
import re
from dataclasses import dataclass
from pathlib import Path

import google.auth
from dotenv import load_dotenv
from a2a.types import DataPart as A2ADataPart
from google.adk.a2a.converters.part_converter import (
    A2A_DATA_PART_END_TAG,
    A2A_DATA_PART_START_TAG,
    A2A_DATA_PART_TEXT_MIME_TYPE,
)
from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.code_executors import BuiltInCodeExecutor
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import AgentTool
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from google.adk.tools.tool_context import ToolContext
from google.genai import types as genai_types
from mcp import StdioServerParameters

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Explicit load: the ADK CLI (`adk run`, `agents-cli run`) loads .env
# internally, but direct invocations (uvicorn, pytest, this preview server)
# don't -- without this, GOOGLE_CLOUD_PROJECT below silently falls through
# to google.auth.default(), which doesn't resolve a project on this
# machine's ADC setup (see project memory on the auth device-flow quirk).
load_dotenv(_PROJECT_ROOT / ".env")

# Prefer an explicitly-set env var (e.g. from .env) over ADC auto-detection --
# on some setups google.auth.default() resolves credentials but not a
# project, which would otherwise crash this module (and therefore every
# `app.*` import, since app/__init__.py imports this module) with
# TypeError: str expected, not NoneType.
if "GOOGLE_CLOUD_PROJECT" not in os.environ:
    _, _detected_project_id = google.auth.default()
    if _detected_project_id:
        os.environ["GOOGLE_CLOUD_PROJECT"] = _detected_project_id
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "True")


# ---------------------------------------------------------------------------
# Effort dial: one user-facing slider drives model tier + thinking budget +
# search persistence + acceptance threshold together (see .agents-cli-spec.md,
# "Effort Dial"). Model IDs confirmed live against this project's Vertex AI
# model list rather than assumed from training data; stable (non-preview)
# picks only, since a capstone project needs to keep working through grading.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EffortTier:
    name: str
    model: str
    thinking_budget: int | None  # set XOR thinking_level
    thinking_level: str | None
    search_persistence: str
    threshold: float


_ECONOMY = EffortTier(
    name="Economy",
    model="gemini-3.1-flash-lite",
    thinking_budget=0,
    thinking_level=None,
    search_persistence=(
        "Run the seeded pattern-search script only. Do not fall back to "
        "freeform reasoning or extra sandboxed regression attempts."
    ),
    threshold=0.75,
)
_BALANCED = EffortTier(
    name="Balanced",
    model="gemini-3.5-flash",
    thinking_budget=None,
    thinking_level="LOW",
    search_persistence=(
        "Run the seeded script; if it finds nothing promising, try a "
        "small number of freeform alternative approaches before giving up."
    ),
    threshold=0.85,
)
_MAX_QUALITY = EffortTier(
    name="Max Quality",
    model="gemini-2.5-pro",
    thinking_budget=None,
    thinking_level="HIGH",
    search_persistence=(
        "Run the seeded script; if it finds nothing promising, thoroughly "
        "explore freeform alternative approaches and extra sandboxed "
        "regression attempts before giving up."
    ),
    threshold=0.92,
)

_DEFAULT_EFFORT_DIAL = 0.5


def _resolve_effort_tier(dial: float) -> EffortTier:
    dial = max(0.0, min(1.0, dial))
    if dial < 1 / 3:
        return _ECONOMY
    if dial < 2 / 3:
        return _BALANCED
    return _MAX_QUALITY


def _effort_description(dial: float) -> str:
    tier = _resolve_effort_tier(dial)
    return (
        f"{tier.name} tier (model: {tier.model}). Acceptance confidence "
        f"threshold: {tier.threshold:.2f} -- below this, say \"I don't "
        f"know\" rather than guessing. Search persistence: "
        f"{tier.search_persistence} State which model answered in your reply."
    )


_EFFORT_DIAL_RE = re.compile(r"EFFORT_DIAL:\s*([0-9]*\.?[0-9]+)")


def _parse_effort_dial_from_content(content: genai_types.Content | None) -> float | None:
    """Reads an "EFFORT_DIAL: <0-1 float>" line from the incoming user
    message, per the text protocol fast_api_app.py constructs (see
    app/fast_api_app.py)."""
    if not content or not content.parts:
        return None
    text = "\n".join(p.text for p in content.parts if p.text)
    match = _EFFORT_DIAL_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


async def _seed_effort_defaults(callback_context: CallbackContext) -> None:
    if "temp:effort_dial" not in callback_context.state:
        dial = _parse_effort_dial_from_content(callback_context.user_content)
        callback_context.state["temp:effort_dial"] = (
            dial if dial is not None else _DEFAULT_EFFORT_DIAL
        )
    callback_context.state["temp:effort_dial_description"] = _effort_description(
        callback_context.state["temp:effort_dial"]
    )


async def _apply_effort_dial(
    callback_context: CallbackContext, llm_request: LlmRequest
) -> LlmResponse | None:
    dial = callback_context.state.get("temp:effort_dial", _DEFAULT_EFFORT_DIAL)
    tier = _resolve_effort_tier(dial)
    llm_request.model = tier.model
    if tier.thinking_budget is not None:
        llm_request.config.thinking_config = genai_types.ThinkingConfig(
            thinking_budget=tier.thinking_budget
        )
    else:
        llm_request.config.thinking_config = genai_types.ThinkingConfig(
            thinking_level=tier.thinking_level
        )
    return None


# ---------------------------------------------------------------------------
# pattern-apply skill: a safe (AST-restricted, not a sandbox escape hatch)
# arithmetic evaluator so numeric rule application doesn't rely on the model
# doing mental math.
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_ALLOWED_UNARYOPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}


def _safe_eval(node: ast.AST, variables: dict[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body, variables)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.Name):
        if node.id not in variables:
            raise ValueError(f"Unknown variable: {node.id}")
        return variables[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        return _ALLOWED_BINOPS[type(node.op)](
            _safe_eval(node.left, variables), _safe_eval(node.right, variables)
        )
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_safe_eval(node.operand, variables))
    raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


def evaluate_numeric_rule(expression: str, variables: dict[str, float]) -> dict:
    """Safely evaluate a numeric pattern rule (the pattern-apply skill).

    Args:
        expression: A pure arithmetic expression using +, -, *, /, **, %,
            and variable names matching keys in `variables` (e.g.
            "x0**2 + 5*x0*x1 + x1**2 + 7"). No function calls, no
            attribute/name access beyond `variables`, no other syntax --
            this is a restricted AST walk, not a code sandbox.
        variables: Maps variable names used in `expression` to their
            numeric values for this scenario.

    Returns:
        {"result": <float>} on success, or {"error": <message>} if the
        expression can't be safely evaluated.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _safe_eval(tree, variables)}
    except Exception as exc:  # noqa: BLE001 -- surfaced to the agent, not raised
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# pattern-search skill: a sub-agent (not a plain function) because it's an
# iterative reasoning loop with sandboxed code execution, seeded from
# app/skills/pattern_search_script.py -- mirrors the data-science reference
# sample's analytics_agent + VertexAiCodeExecutor + AgentTool pattern.
# ---------------------------------------------------------------------------

_SEARCH_SCRIPT_SOURCE = (
    _PROJECT_ROOT / "app" / "skills" / "pattern_search_script.py"
).read_text(encoding="utf-8")


def get_pattern_search_script() -> str:
    """Return the pattern-search skill's existing, already-validated script.

    The script's source can't be embedded directly in this agent's
    instruction text -- ADK's instruction templating scans for `{...}`
    substrings as state-variable references, and Python source (f-strings,
    dict/set literals) is full of those. Returning it as a tool result
    instead sidesteps that: only the static instruction string is
    template-processed, not tool return values.
    """
    return _SEARCH_SCRIPT_SOURCE


_PATTERN_SEARCH_INSTRUCTION = """You search for a rule mapping labeled
scenario inputs to an output value, across a set of same-labeled example
scenarios (Pattern Finder Outline.txt, Section VII).

You have sandboxed Python code execution. Your FIRST action for every
request must be to call get_pattern_search_script to retrieve the
pattern-search skill's existing, already-validated script, then execute its
returned source VERBATIM in a code cell to define its functions.

Then call `analyze(data)` where `data` is a list of (inputs_tuple, output)
pairs built from the scenarios you were given (numeric values coerced to
float; use None for a genuinely absent input slot, keeping input position
consistent across all rows).

Read analyze()'s printed output carefully:
- "EXACT FIT" or "STRONG MATCH" means high confidence.
- "Near-exact fit", underdetermined warnings, or no clean fit at all mean
  low-to-no confidence -- do not report these as confident matches.

If analyze() finds nothing with real confidence, do not give up
immediately: think about other approaches that could fit the data (the
outline's examples: numerical equations, logical constructions, qualitative
scales, and other structures entirely) and test them by writing and running
new code against the same data -- to the extent your current
search-persistence setting allows: {temp:effort_dial_description}

Report back:
- Whether you found a candidate rule, and if so, its exact formula/logic in
  a form the pattern-apply skill could execute (a pure arithmetic
  expression) or interpret (a text directive).
- Your confidence (0-1) and why.
- A short trace of what you tried, in order.
- Whether the winning approach came from the seeded script or from your own
  freeform reasoning, so the caller knows whether the script is a candidate
  for updating (Section VII: "the script should be updated to include the
  new approach"). This build does not auto-edit the script file; flag it in
  your trace instead so a developer can fold it in.
"""

pattern_search_agent = Agent(
    name="pattern_search",
    model=_BALANCED.model,  # overridden per-request by _apply_effort_dial
    instruction=_PATTERN_SEARCH_INSTRUCTION,
    description=(
        "Searches for a pattern/rule across same-labeled historical "
        "scenarios: runs the existing pattern-search script first, then "
        "freeform reasoning with sandboxed code execution if needed."
    ),
    # Model-internal sandbox: no separate cloud resource to provision, no
    # network call at import time (unlike VertexAiCodeExecutor, which
    # creates/loads a Vertex AI Extension resource on construction -- see
    # git history for why that was tried first and reverted).
    tools=[get_pattern_search_script],
    code_executor=BuiltInCodeExecutor(),
    # AgentTool-invoked sub-agents don't inherit the parent's `temp:` state
    # (confirmed empirically -- it's scoped to the invocation that sets it,
    # and this sub-agent's call is its own nested invocation), so it needs
    # its own copy of the seeding callback, not just before_model_callback.
    before_agent_callback=_seed_effort_defaults,
    before_model_callback=_apply_effort_dial,
)

pattern_search_tool = AgentTool(pattern_search_agent)


# ---------------------------------------------------------------------------
# Pattern Finder DB, via MCP only (no raw SQL from agent code -- see
# .agents-cli-spec.md, Constraints & Safety Rules). Local dev: stdio to our
# own server module. Deployed: swap for SseConnectionParams pointing at the
# MCP server's Cloud Run URL (see google-agents-cli-deploy when we get there).
# ---------------------------------------------------------------------------

db_mcp_toolset = McpToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command="uv",
            args=["run", "python", "-m", "app.mcp_server.server"],
            cwd=str(_PROJECT_ROOT),
            # The mcp client's default (env=None) only forwards a small
            # allowlist of "safe" vars (see mcp.client.stdio.
            # get_default_environment) -- NOT the full parent environment.
            # Without this, PATTERN_FINDER_DB_URL (and in production, any
            # Cloud SQL connection vars) silently never reach the
            # subprocess, which then falls back to the default local
            # SQLite path instead of the configured DB. Since this
            # subprocess is our own trusted MCP server, not a third-party
            # one, forwarding the full environment is safe here.
            env=dict(os.environ),
        ),
    ),
)


# ---------------------------------------------------------------------------
# A2UI surfaces: Agent Reasoning (Guess Phase) and Pattern Captured (Learn
# Phase). The agent reports these via dedicated tool calls (structured,
# typed args) rather than us regex-scraping its prose reply. An
# after_tool_callback stashes the call's args in session state; a paired
# after_model_callback appends them to the final text response as an A2A
# DataPart, once the response actually has text (not a bare tool-call step).
#
# The DataPart wire format is ADK's, verified directly against
# google.adk.a2a.converters.part_converter: a genai Part whose inline_data
# is JSON-serialized a2a.types.DataPart, wrapped in A2A_DATA_PART_*_TAG and
# tagged with A2A_DATA_PART_TEXT_MIME_TYPE -- convert_genai_part_to_a2a_part
# unwraps exactly this shape into a real a2a DataPart on the wire.
# ---------------------------------------------------------------------------


def emit_agent_reasoning(
    guess: str, confidence: float, matched_via: str, trace: str
) -> dict:
    """Record the Agent Reasoning UI surface. Call this as your LAST action
    in PHASE: guess, once you've decided your guess (or "I don't know") --
    it supplements your normal text reply, it doesn't replace it.

    Args:
        guess: Your guessed consequence, or the literal string "I don't know".
        confidence: Your confidence in [0, 1]. 0 if you don't know.
        matched_via: One of "exact_pattern", "pattern_search",
            "label_independent_match", or "none".
        trace: A short human-readable trace of what you tried, in order.
    """
    return {"recorded": True}


def emit_pattern_captured(
    action: str,
    text_desc: str | None = None,
    rule_or_code_link: str | None = None,
    scenarios_linked: int = 0,
) -> dict:
    """Record the Pattern Captured UI surface. Call this as your LAST action
    in PHASE: learn, IF a pattern was created, updated, or linked. Skip
    calling this if no pattern was touched (wrong guess or "I don't know").

    Args:
        action: One of "created", "updated_label_set", or "linked_only"
            (matching guess-step 2, guess-step 1, and guess-step 3
            respectively).
        text_desc: The pattern's abstracted, label-free description.
        rule_or_code_link: The pattern's rule/code.
        scenarios_linked: How many scenarios got linked to this pattern.
    """
    return {"recorded": True}


# Maps each emit tool to the A2UI catalog ID declared on the AgentCard
# (see app/fast_api_app.py, A2UI_EXTENSION) -- these must match.
_A2UI_EMIT_TOOL_SURFACES = {
    "emit_agent_reasoning": "agent_reasoning",
    "emit_pattern_captured": "pattern_captured",
}


async def _capture_a2ui_payload(
    tool, args: dict, tool_context: ToolContext, tool_response: dict
) -> dict | None:
    surface = _A2UI_EMIT_TOOL_SURFACES.get(tool.name)
    if surface:
        tool_context.state["temp:a2ui_surface"] = surface
        tool_context.state["temp:a2ui_payload"] = args
    return None


def _build_a2ui_data_part(surface: str, payload: dict) -> genai_types.Part:
    envelope = A2ADataPart(data={"surface": surface, **payload})
    tagged = (
        A2A_DATA_PART_START_TAG
        + envelope.model_dump_json(by_alias=True, exclude_none=True).encode("utf-8")
        + A2A_DATA_PART_END_TAG
    )
    return genai_types.Part(
        inline_data=genai_types.Blob(data=tagged, mime_type=A2A_DATA_PART_TEXT_MIME_TYPE)
    )


async def _attach_a2ui_surface(
    callback_context: CallbackContext, llm_response: LlmResponse
) -> LlmResponse | None:
    payload = callback_context.state.get("temp:a2ui_payload")
    if not payload:
        return None
    if not llm_response.content or not llm_response.content.parts:
        return None
    if not any(p.text for p in llm_response.content.parts):
        return None  # only attach once the response actually has user-facing text
    surface = callback_context.state.get("temp:a2ui_surface", "unknown")
    llm_response.content.parts.append(_build_a2ui_data_part(surface, payload))
    callback_context.state["temp:a2ui_payload"] = None
    return llm_response


# ---------------------------------------------------------------------------
# Root agent: the two-phase Guess/Learn workflow.
# ---------------------------------------------------------------------------

_ROOT_INSTRUCTION = """You are the Pattern Finder agent. Users give you a
"scenario" made of up to 5 labeled inputs (label -> value; values can be
numeric or textual, and a slot may be intentionally absent). Your job is to
guess the scenario's "consequence" by finding and reusing patterns from
past scenarios, and -- once told the correct answer -- to capture what you
learned so the same pattern helps next time.

Every user message includes a line "PHASE: guess" or "PHASE: learn" (an
"EFFORT_DIAL:" line may also be present -- that's handled automatically,
you don't need to act on it directly). Follow the matching procedure below.
Never skip the label-set lookup order to guess early; never fabricate a
pattern match without genuine confidence.

====================
PHASE: guess
====================
Input: the scenario's labeled fields.

1. Call get_pattern_by_label_set with this scenario's exact label set. If it
   returns a pattern, apply its rule_or_code_link to this scenario's values
   -- call evaluate_numeric_rule if it's a computable expression (don't do
   the arithmetic by hand), or apply it by reasoning if it's a text
   directive -- and report that as your guess.
2. Otherwise, call get_scenarios_by_label_set with this scenario's exact
   label set to gather every past scenario sharing it, then use the
   pattern_search tool with that data to look for a rule (include a line
   "EFFORT_DIAL: <value>" matching this conversation's dial position in
   your request to it, so it searches at the same effort level). If it
   returns a candidate with strong confidence, apply it and report your
   guess.
3. Otherwise (or if step 2's confidence is weak), call
   get_all_pattern_descriptions and check whether this scenario's values
   plausibly match any EXISTING pattern's abstracted description --
   independent of labels. If one matches with real confidence, apply it.
4. If nothing from steps 1-3 clears your current confidence bar, say
   plainly that you don't know rather than guessing. Do not guess "just to
   have an answer." Your confidence bar and how hard to search:
   {temp:effort_dial_description}

Always state, in your reply: your guess (or "I don't know"), your
confidence, which of steps 1-4 produced it, and which model answered. As
your LAST action this phase, call emit_agent_reasoning with that same
information (see its docstring for the exact fields).

====================
PHASE: learn
====================
Input: "CORRECT_CONSEQUENCE: <value>" -- the confirmed correct answer for
the scenario from your immediately preceding PHASE: guess turn in this same
conversation. If you have no record of a prior guess in this conversation,
say so and ask the user to resubmit the scenario via PHASE: guess first.

1. Call insert_scenario to persist that scenario (its labels/values from the
   guess turn, plus this consequence) -- this always happens, correct guess
   or not.
2. Compare the correct consequence to your prior guess.
   - If they DON'T match (or you said "I don't know"): do NOT create or
     link any pattern for this scenario. Just confirm the scenario was
     recorded.
   - If they DO match:
     a. If you used an EXISTING pattern (guess step 1, an exact label-set
        match): call link_pattern_to_scenarios with update_label_set=True,
        that pattern's label set, and every scenario_id
        get_scenarios_by_label_set returns for this label set (not just
        this one) -- every scenario sharing this label set must get the
        pattern_id.
     b. If pattern_search found a genuinely NEW pattern (guess step 2): call
        upsert_pattern to create it. Write text_desc as a LABEL-FREE
        abstraction (e.g. "data fit a parabolic pattern when labels are
        sorted in ascending order") -- never mention this scenario's actual
        label names. Then call link_pattern_to_scenarios the same way as
        (a), update_label_set=True.
     c. If you matched via guess step 3 (label-independent match): call
        link_pattern_to_scenarios with update_label_set=False and only this
        one scenario_id. Do NOT add this label set to the pattern.

Always confirm in your reply what was recorded: the scenario, and (if
applicable) whether a pattern was created, updated, or just linked. As your
LAST action this phase, IF a pattern was created, updated, or linked, call
emit_pattern_captured with that information (see its docstring for the
exact fields) -- skip this call entirely if no pattern was touched.
"""

root_agent = Agent(
    name="pattern_finder",
    model=_BALANCED.model,  # overridden per-request by _apply_effort_dial
    instruction=_ROOT_INSTRUCTION,
    description=(
        "Guesses the consequence of a labeled scenario by finding and "
        "reusing patterns, then learns from the confirmed correct answer."
    ),
    tools=[
        db_mcp_toolset,
        pattern_search_tool,
        evaluate_numeric_rule,
        emit_agent_reasoning,
        emit_pattern_captured,
    ],
    before_agent_callback=_seed_effort_defaults,
    before_model_callback=_apply_effort_dial,
    after_tool_callback=_capture_a2ui_payload,
    after_model_callback=_attach_a2ui_surface,
)

app = App(name="pattern-finder", root_agent=root_agent)
