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
Outline.txt as two turns of ONE ADK session, not two agents -- and, as of
this revision, most of both turns resolve WITHOUT the LLM at all. A single
`before_agent_callback` (_before_agent_dispatch) intercepts every turn:

    PHASE: guess -- tries a deterministic exact-label-set lookup, then (if
    the rule is a plain numeric expression) applies it directly; failing
    that, runs the seeded pattern-search script deterministically and
    applies it if confident. Only when BOTH come up empty does it return
    None, handing off to the full agent (Handoff 1) for either "interpret
    a known text-directive rule" or "freeform pattern-search reasoning".

    PHASE: learn -- always resolves in the callback. insert_scenario is
    always deterministic; comparing the guess to the correct answer is
    deterministic for numeric answers and a narrow (non-agent) LLM call
    otherwise; capturing/linking a pattern is deterministic when an
    existing pattern was reused, and only a narrow LLM call (to write the
    label-free description) when pattern-search found something new.

Returning `genai_types.Content` from `before_agent_callback` sets
`ctx.end_invocation = True` in ADK's BaseAgent.run_async, which skips the
LLM/tool loop entirely -- confirmed directly against
google.adk.agents.base_agent before building this.

Every user message is expected to include a "PHASE: guess" or
"PHASE: learn" line, an "EFFORT_DIAL: <0-1 float>" line, and may include
"EMIT_UI: on|off". PHASE: learn messages additionally carry
"CORRECT_CONSEQUENCE:", and -- since the deterministic callback has no
access to "what the agent was thinking" during the prior guess turn --
"MATCHED_VIA:", "PATTERN_ID:", "APPLIED_RULE:", and "GUESS_VALUE:" lines
echoing back what the Guess-phase turn reported, plus the scenario's
"label=X value=Y" lines again. That's the contract the API layer
(app/fast_api_app.py) and frontend (frontend/app.js) are responsible for
constructing -- see the Agent Reasoning card's payload fields, which is
where the frontend reads these values from.
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
from google import genai
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

from app.mcp_server import db_ops
from app.skills import pattern_search_script

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
        "Try only what pattern_search reports on the first attempt. Do not "
        "fall back to additional freeform alternatives."
    ),
    threshold=0.75,
)
_BALANCED = EffortTier(
    name="Balanced",
    model="gemini-3.5-flash",
    thinking_budget=None,
    thinking_level="LOW",
    search_persistence=(
        "Try a small number of freeform alternative approaches before "
        "giving up."
    ),
    threshold=0.85,
)
_MAX_QUALITY = EffortTier(
    name="Max Quality",
    # gemini-2.5-pro (unlike the 3.x models used by the other two tiers)
    # only supports the older thinking_budget field, not thinking_level --
    # confirmed live (it 400s on thinking_level, and even on
    # thinking_budget=0, since 2.5-pro can't disable thinking entirely).
    # 32768 is that model's max budget, the closest explicit equivalent to
    # "HIGH".
    model="gemini-2.5-pro",
    thinking_budget=32768,
    thinking_level=None,
    search_persistence=(
        "Thoroughly explore freeform alternative approaches and extra "
        "sandboxed regression attempts before giving up."
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


# ---------------------------------------------------------------------------
# Message text protocol parsing. Everything the API layer/frontend encodes
# as "KEY: value" lines or "label=X value=Y" lines -- see module docstring.
# ---------------------------------------------------------------------------


def _get_text(content: genai_types.Content | None) -> str:
    if not content or not content.parts:
        return ""
    return "\n".join(p.text for p in content.parts if p.text)


def _parse_field(text: str, key: str) -> str | None:
    match = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    return match.group(1).strip() if match else None


def _parse_effort_dial_from_text(text: str) -> float | None:
    value = _parse_field(text, "EFFORT_DIAL")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


_LABEL_VALUE_RE = re.compile(r"^label=(\S+)\s+value=(.*)$", re.MULTILINE)


def _parse_label_values(text: str) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for match in _LABEL_VALUE_RE.finditer(text):
        label, value = match.group(1), match.group(2).strip()
        values[label] = value if value else None
    return values


async def _seed_effort_defaults(callback_context: CallbackContext) -> None:
    """Used by pattern_search_agent (a plain sub-agent, not the dispatcher
    below) -- unchanged from before this revision."""
    if "temp:effort_dial" not in callback_context.state:
        text = _get_text(callback_context.user_content)
        dial = _parse_effort_dial_from_text(text)
        callback_context.state["temp:effort_dial"] = (
            dial if dial is not None else _DEFAULT_EFFORT_DIAL
        )
    callback_context.state["temp:effort_dial_description"] = _effort_description(
        callback_context.state["temp:effort_dial"]
    )


# Per-invocation dynamic instruction text (e.g. Guess-phase handoff
# context), keyed by invocation_id. NOT ADK session state: empirically,
# state written in before_agent_callback was NOT reliably visible to
# instruction templating on this ADK version in this project's testing --
# confirmed via a minimal repro (has_delta() True inside the callback, but
# the yielded Event's actions.state_delta came back empty, and the
# following turn's session.state ended up with none of it). Appending
# directly to LlmRequest from before_model_callback sidesteps that
# entirely -- proven reliable all session for the model/thinking-config
# override below, which uses the exact same mechanism.
_handoff_instructions: dict[str, str] = {}


async def _cleanup_handoff_instructions(callback_context: CallbackContext) -> None:
    _handoff_instructions.pop(callback_context.invocation_id, None)


async def _apply_effort_dial_and_context(
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

    handoff_text = _handoff_instructions.get(callback_context.invocation_id)
    if handoff_text:
        llm_request.append_instructions([handoff_text])
    return None


# ---------------------------------------------------------------------------
# pattern-apply skill: a safe (AST-restricted, not a sandbox escape hatch)
# arithmetic evaluator so numeric rule application doesn't rely on the model
# doing mental math. Used by both the deterministic pre-check below and, for
# the cases the agent still handles, as a callable tool.
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

    Convention: rules ALWAYS use positional variable names x0, x1, x2, ...
    for the scenario's labeled inputs IN ORDER -- never the actual label
    names. This is what keeps a stored rule label-independent (the same
    property text_desc already has), so it stays applicable if it's later
    reused via a label-independent match against a scenario with entirely
    different label names. When WRITING a new rule (pattern_search,
    emit_agent_reasoning's `rule` field), use x0/x1/... too.

    Args:
        expression: A pure arithmetic expression using +, -, *, /, **, %,
            and variable names x0, x1, ... matching keys in `variables`
            (e.g. "x0**2 + 5*x0*x1 + x1**2 + 7"). No function calls, no
            attribute/name access beyond `variables`, no other syntax --
            this is a restricted AST walk, not a code sandbox.
        variables: Maps x0, x1, ... to their numeric values for this
            scenario. "Order" here always means the scenario's labels
            sorted alphabetically by name (x0 = first alphabetically),
            NOT the order labels happen to appear in any given request --
            the labels table has no stored order, and requests can list
            the same label set in any order (whichever row the user typed
            each label into), so alphabetical order is the only stable
            anchor for what x0/x1/... mean across requests.

    Returns:
        {"result": <float>} on success, or {"error": <message>} if the
        expression can't be safely evaluated.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return {"result": _safe_eval(tree, variables)}
    except Exception as exc:  # noqa: BLE001 -- surfaced to the agent, not raised
        return {"error": str(exc)}


def _looks_numeric_expression(rule: str) -> bool:
    try:
        ast.parse(rule, mode="eval")
        return True
    except SyntaxError:
        return False


def _try_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _format_number(value: float) -> str:
    rounded = round(value)
    return str(rounded) if abs(value - rounded) < 1e-6 else str(round(value, 4))


# ---------------------------------------------------------------------------
# A2UI surfaces: Agent Reasoning (Guess Phase) and Pattern Captured (Learn
# Phase). The LLM-driven paths report these via dedicated tool calls
# (structured, typed args); the deterministic paths below construct the
# same payload shape directly, no tool call needed.
#
# The DataPart wire format is ADK's, verified directly against
# google.adk.a2a.converters.part_converter: a genai Part whose inline_data
# is JSON-serialized a2a.types.DataPart, wrapped in A2A_DATA_PART_*_TAG and
# tagged with A2A_DATA_PART_TEXT_MIME_TYPE -- convert_genai_part_to_a2a_part
# unwraps exactly this shape into a real a2a DataPart on the wire.
# ---------------------------------------------------------------------------


def emit_agent_reasoning(
    guess: str,
    confidence: float,
    matched_via: str,
    trace: str,
    pattern_id: int | None = None,
    rule: str | None = None,
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
        pattern_id: The existing pattern's id, if matched_via is
            "exact_pattern" or "label_independent_match". Omit otherwise.
        rule: The exact rule/logic you applied, as a string (a pure
            arithmetic expression using x0, x1, x2, ... for the labeled
            inputs -- never the actual label names -- if computable,
            otherwise a short text directive). x0/x1/... always refer to
            this scenario's labels sorted ALPHABETICALLY by name (x0 =
            first alphabetically), regardless of what order they were
            given to you in -- use the x0=..., x1=... mapping in the
            context you were given rather than re-deriving it. Needed
            later to capture a NEW pattern if this guess turns out
            correct -- omit only for "I don't know".
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


def _compose_content(
    text: str, surface: str | None, payload: dict | None, emit_ui: bool
) -> genai_types.Content:
    parts = [genai_types.Part(text=text)]
    if emit_ui and surface and payload is not None:
        parts.append(_build_a2ui_data_part(surface, payload))
    return genai_types.Content(role="model", parts=parts)


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
# Narrow, single-purpose LLM calls -- used only where the Learn-phase
# orchestrator genuinely needs judgment (non-numeric answer comparison,
# writing a new pattern's label-free description). Deliberately NOT routed
# through the full root_agent (with its whole tool/instruction surface) --
# these are small, fast, single-turn calls via the raw client.
# ---------------------------------------------------------------------------

_narrow_llm_client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"),
)


async def _narrow_llm_call(prompt: str, model: str) -> str:
    response = await _narrow_llm_client.aio.models.generate_content(
        model=model, contents=prompt
    )
    return response.text or ""


async def _consequence_matches(guess: str | None, correct: str, model: str) -> bool:
    """Section VI step 7's comparison. Deterministic whenever the correct
    consequence is purely numeric (per user direction); a narrow LLM call
    only for non-numeric consequences that aren't an exact string match."""
    guess_norm = (guess or "").strip().lower()
    if not guess_norm or "don't know" in guess_norm or "dont know" in guess_norm:
        return False

    numeric_correct = _try_float(correct.strip())
    if numeric_correct is not None:
        numeric_guess = _try_float(guess_norm)
        return numeric_guess is not None and abs(numeric_guess - numeric_correct) < 1e-6

    if guess_norm == correct.strip().lower():
        return True

    prompt = (
        "Do these two short answers describe the same real-world outcome, "
        "even if worded differently? Reply with exactly one word: yes or no.\n\n"
        f"Answer A: {guess}\nAnswer B: {correct}"
    )
    reply = await _narrow_llm_call(prompt, model)
    return reply.strip().lower().startswith("y")


async def _write_pattern_description(rule: str, model: str) -> str:
    prompt = (
        "You are writing an abstracted, label-free description of a data "
        "pattern just confirmed correct, for reuse on future unrelated "
        "scenarios (Pattern Finder Outline.txt, Section VII). Do not "
        "mention any specific label names -- describe the STRUCTURE of the "
        'pattern only, e.g. "data fit a parabolic pattern when labels are '
        'sorted in ascending order".\n\n'
        "Context on notation: x0, x1, x2, ... are VARIABLE REFERENCES to "
        "the scenario's labeled inputs, positionally sorted alphabetically "
        "by label name (x0 = first alphabetically, x1 = second, etc.) -- "
        "they are not exponents or coefficients. A rule that is a bare "
        "reference like \"x2\" with no operator means the output directly "
        "copies/equals that one specific input's value verbatim (an "
        "identity/copy rule, e.g. \"output = 5\" when x2=5, or "
        '"output = \'cherry\'" when x2=\'cherry\') -- it does NOT mean '
        '"x squared" or "x times 2". Only treat it as arithmetic if it '
        "actually contains an operator (+, -, *, /, **, %).\n\n"
        f"The confirmed rule/logic: {rule}\n\n"
        "Reply with ONLY the one-sentence description, nothing else."
    )
    reply = await _narrow_llm_call(prompt, model)
    return reply.strip()


# ---------------------------------------------------------------------------
# Deterministic Guess-phase pre-check (Handoff 1's "cheap deterministic
# lookup" from .agents-cli-spec.md, extended per the revised guess-flow
# design: also runs the seeded script deterministically before handing off
# to the agent). Returns Content when it can answer outright; returns None
# (after seeding handoff context) when the agent must take over.
# ---------------------------------------------------------------------------


def _scenarios_to_training_data(
    scenarios: list[dict], label_names: list[str]
) -> list[tuple[tuple[float | None, ...], float]]:
    rows = []
    for s in scenarios:
        inputs = tuple(_try_float(s["inputs"].get(name)) for name in label_names)
        out = _try_float(s["consequence"])
        if out is None:
            continue
        rows.append((inputs, out))
    return rows


def _pattern_rule_string(found: dict) -> str:
    """A rule string for a find_pattern() result: a clean x0/x1/... formula
    when the kind is directly re-appliable via evaluate_numeric_rule
    (polynomial); otherwise a text directive describing what to do, which
    correctly routes future re-application through the agent's own
    interpretation path (_looks_numeric_expression rejects it as pure
    arithmetic) rather than the deterministic evaluator."""
    if found["kind"] == "polynomial":
        return pattern_search_script.format_polynomial(found["names"], found["coef"])
    return found["summary"]


def _format_history_for_prompt(scenarios: list[dict], label_names: list[str]) -> str:
    if not scenarios:
        return "(no historical scenarios with this label set)"
    lines = []
    for s in scenarios:
        pairs = ", ".join(f"{name}={s['inputs'].get(name)}" for name in label_names)
        lines.append(f"- {pairs} -> {s['consequence']}")
    return "\n".join(lines)


async def _guess_phase_content(
    label_values: dict[str, str | None], emit_ui: bool
) -> tuple[genai_types.Content | None, str | None]:
    """Returns (resolved_content, handoff_instruction) -- exactly one is
    non-None. resolved_content when answerable deterministically;
    handoff_instruction (extra instruction text for the agent, describing
    exactly why it was invoked and what's already been ruled out) when the
    full agent must take over.

    Deliberately a pure function (no callback_context/state mutation): the
    caller (_before_agent_dispatch) hands the returned instruction text to
    the model callback to append directly via LlmRequest.append_instructions
    -- see that function's docstring for why state wasn't used for this.
    """
    # Canonical order: x0, x1, x2, ... always refer to this label set's
    # labels sorted ALPHABETICALLY -- never the order labels happen to
    # appear in this particular request's text (which varies by which row
    # the user typed each label into, or how a test/LLM happens to build
    # the dict). The labels table has no stored order of its own, so
    # alphabetical is the only anchor stable across requests -- without
    # it, a rule learned as "x0 + x1*x2" from one label order would
    # silently compute something different when the same label set is
    # later submitted in a different order. See evaluate_numeric_rule's
    # docstring.
    sorted_names = sorted(label_values.keys())
    pattern = db_ops.get_pattern_by_label_set(sorted_names)

    if pattern:
        rule = pattern.get("rule_or_code_link") or ""
        numeric_values = {
            f"x{i}": _try_float(label_values[name]) for i, name in enumerate(sorted_names)
        }
        if (
            rule
            and all(v is not None for v in numeric_values.values())
            and _looks_numeric_expression(rule)
        ):
            result = evaluate_numeric_rule(rule, numeric_values)
            if "result" in result:
                guess = _format_number(result["result"])
                payload = {
                    "guess": guess,
                    "confidence": 1.0,
                    "matched_via": "exact_pattern",
                    "pattern_id": pattern["pattern_id"],
                    "rule": rule,
                    "trace": (
                        f"Deterministic exact label-set match "
                        f"(pattern_id={pattern['pattern_id']}); applied "
                        f"{rule} directly -- no LLM needed."
                    ),
                }
                text = (
                    f"Guess: {guess}\n\n- Confidence: 1.00\n"
                    f"- Matched via: exact pattern (pattern_id={pattern['pattern_id']})\n"
                    f"- Model: none -- resolved deterministically"
                )
                return _compose_content(text, "agent_reasoning", payload, emit_ui), None
        # Known pattern, but not a plain numeric expression (or an input
        # isn't numeric) -- hand off to the agent to interpret/apply it.
        mapping = ", ".join(f"x{i}={name}" for i, name in enumerate(sorted_names))
        values_text = ", ".join(f"{name}={label_values[name]}" for name in sorted_names)
        handoff = (
            f"Why you were invoked: an existing pattern for this exact "
            f"label set was found (pattern_id {pattern['pattern_id']}), but "
            f"its rule is not a plain numeric expression (or an input "
            f"wasn't numeric), so it needs your judgment to interpret and "
            f"apply: {rule}\nIf it uses x0, x1, x2, ..., those always refer "
            f"to this label set's labels sorted alphabetically -- "
            f"{mapping} (values: {values_text}) -- not the order given to "
            f"you here. Apply it to this scenario's values and report your "
            f"guess. Do not re-call get_pattern_by_label_set -- you "
            f"already have the pattern above."
        )
        return None, handoff

    scenarios = db_ops.get_scenarios_by_label_set(sorted_names)
    history = _scenarios_to_training_data(scenarios, sorted_names)
    found = pattern_search_script.find_pattern(history) if len(history) >= 2 else None
    if found:
        try:
            new_inputs = tuple(_try_float(label_values[name]) for name in sorted_names)
            result = pattern_search_script.apply_pattern(found, new_inputs)
            guess = _format_number(float(result))
            payload = {
                "guess": guess,
                "confidence": 1.0,
                "matched_via": "pattern_search",
                "rule": _pattern_rule_string(found),
                "trace": (
                    f"Deterministic run of the seeded pattern-search script "
                    f"found: {found['summary']} -- no LLM needed."
                ),
            }
            text = (
                f"Guess: {guess}\n\n- Confidence: 1.00\n"
                f"- Matched via: seeded pattern-search script\n"
                f"- Model: none -- resolved deterministically"
            )
            return _compose_content(text, "agent_reasoning", payload, emit_ui), None
        except (ValueError, KeyError, TypeError):
            pass  # couldn't apply to this new scenario's shape; fall through

    # No deterministic answer. Hand off for freeform search, with
    # everything already fetched so neither the agent nor pattern_search
    # need to redo this work.
    script_summary = (
        found["summary"]
        if found
        else "no confident structural, single-variable, or fully-determined "
        "polynomial match"
    )
    history_text = _format_history_for_prompt(scenarios, sorted_names)
    mapping = ", ".join(f"x{i}={name}" for i, name in enumerate(sorted_names))
    handoff = (
        f"Why you were invoked: no exact pattern exists, and the seeded "
        f"script (run deterministically, before you were invoked) found: "
        f"{script_summary}\nHistorical scenarios sharing this label set:\n"
        f"{history_text}\nCall the pattern_search tool, including this "
        f"history and the script's result in your request to it (it "
        f"already knows not to re-run the script). If it returns a "
        f"candidate with strong confidence, apply it and report your "
        f"guess. If you or pattern_search report a computable rule, write "
        f"it using x0, x1, x2, ... where those always refer to this label "
        f"set's labels sorted alphabetically -- {mapping} -- so the rule "
        f"stays reusable regardless of what order a future request lists "
        f"these labels in. If not, call get_all_pattern_descriptions and "
        f"check whether this scenario's values plausibly match any "
        f"EXISTING pattern's abstracted description -- independent of "
        f"labels. If nothing clears your confidence bar, say plainly that "
        f"you don't know rather than guessing."
    )
    return None, handoff


# ---------------------------------------------------------------------------
# Deterministic Learn-phase orchestrator. Always resolves -- never falls
# through to the full agent (see module docstring).
# ---------------------------------------------------------------------------


async def _learn_phase_content(
    label_values: dict[str, str | None],
    correct_consequence: str,
    matched_via: str | None,
    pattern_id: int | None,
    applied_rule: str | None,
    guess_value: str | None,
    emit_ui: bool,
    effort_dial: float,
) -> genai_types.Content:
    # Sorted for consistency with _guess_phase_content's canonical x0/x1/...
    # order -- not load-bearing here (get_scenarios_by_label_set/
    # link_pattern_to_scenarios treat label sets as sets, order doesn't
    # affect correctness), just keeps the convention uniform.
    label_names = sorted(label_values.keys())
    scenario_id = db_ops.insert_scenario("t", label_values, correct_consequence)
    model = _resolve_effort_tier(effort_dial).model

    # `matched` always goes into the payload (when emit_ui) even when
    # action="none" -- the frontend's scorecard needs the backend's actual
    # comparison result (numeric tolerance / narrow LLM judgment), not a
    # naive client-side re-derivation that wouldn't match this logic.
    matched = await _consequence_matches(guess_value, correct_consequence, model)
    if not matched:
        text = (
            f"Recorded scenario {scenario_id}. No pattern was created or "
            f"linked (the guess didn't match, or was \"I don't know\")."
        )
        payload = {"matched": False, "action": "none", "scenarios_linked": 0}
        return _compose_content(text, "pattern_captured", payload, emit_ui)

    if matched_via in ("exact_pattern", "label_independent_match") and pattern_id is not None:
        update_label_set = matched_via == "exact_pattern"
        if update_label_set:
            linked_ids = [
                s["scenario_id"] for s in db_ops.get_scenarios_by_label_set(label_names)
            ]
        else:
            linked_ids = [scenario_id]
        db_ops.link_pattern_to_scenarios(
            pattern_id,
            linked_ids,
            update_label_set=update_label_set,
            label_names=label_names if update_label_set else None,
        )
        payload = {
            "matched": True,
            "action": "updated_label_set" if update_label_set else "linked_only",
            "scenarios_linked": len(linked_ids),
        }
        text = (
            f"Recorded scenario {scenario_id}. Correct! Linked to existing "
            f"pattern_id={pattern_id} ({len(linked_ids)} scenario(s))."
        )
        return _compose_content(text, "pattern_captured", payload, emit_ui)

    if matched_via == "pattern_search" and applied_rule:
        text_desc = await _write_pattern_description(applied_rule, model)
        new_pattern_id = db_ops.upsert_pattern(text_desc, applied_rule)
        linked_ids = [
            s["scenario_id"] for s in db_ops.get_scenarios_by_label_set(label_names)
        ]
        db_ops.link_pattern_to_scenarios(
            new_pattern_id, linked_ids, update_label_set=True, label_names=label_names
        )
        payload = {
            "matched": True,
            "action": "created",
            "text_desc": text_desc,
            "rule_or_code_link": applied_rule,
            "scenarios_linked": len(linked_ids),
        }
        text = (
            f"Recorded scenario {scenario_id}. Correct! Created new "
            f"pattern_id={new_pattern_id}: {text_desc}"
        )
        return _compose_content(text, "pattern_captured", payload, emit_ui)

    # Matched, but missing/garbled context about how -- conservative:
    # record the scenario only, don't guess at a pattern capture action.
    text = (
        f"Recorded scenario {scenario_id}. Correct, but I don't have enough "
        f"context about how this was matched to safely capture a pattern."
    )
    payload = {"matched": True, "action": "none", "scenarios_linked": 0}
    return _compose_content(text, "pattern_captured", payload, emit_ui)


# ---------------------------------------------------------------------------
# pattern-search skill: a sub-agent (not a plain function) because it's an
# iterative freeform-reasoning loop with sandboxed code execution. Only
# ever invoked in freeform-only mode now -- the seeded script itself
# already ran deterministically (see _guess_phase_content above) before
# this sub-agent is reached, so it doesn't re-fetch or re-run it; the root
# agent's own request to this tool carries what the script found.
# ---------------------------------------------------------------------------

_PATTERN_SEARCH_INSTRUCTION = """You search for a rule mapping labeled
scenario inputs to an output value, across a set of same-labeled example
scenarios (Pattern Finder Outline.txt, Section VII).

You are only ever invoked AFTER the seeded pattern-search script has
already run directly (deterministically, outside your own reasoning) and
found nothing trustworthy -- the request you receive states the historical
data and what the script tried and found. Do not re-derive or re-run that
script; move straight to freeform reasoning: think about other approaches
that could fit the data (numerical equations, logical constructions,
qualitative scales, or other structures entirely) and test them using your
sandboxed code execution -- to the extent your current search-persistence
setting allows: {temp:effort_dial_description}

Report back:
- Whether you found a candidate rule, and if so, its exact formula/logic. If
  it's a computable expression, write it using x0, x1, x2, ... for the
  scenario's labeled inputs -- never the actual label names. Use the
  x0=..., x1=... mapping given to you in the request (the labels sorted
  alphabetically) rather than re-deriving your own order -- this keeps the
  rule reusable later regardless of what order a future request lists
  these same labels in, and label-independent (like an abstracted pattern
  description already is) if reused against a scenario with entirely
  different label names. If it's not a pure expression, give a text
  directive instead.
- Your confidence (0-1) and why.
- A short trace of what you tried, in order.
- Since this is by definition a new approach the seeded script doesn't
  cover, note that a developer should consider folding it into the script
  (Section VII) -- this build does not auto-edit the script file.
"""

pattern_search_agent = Agent(
    name="pattern_search",
    model=_BALANCED.model,  # overridden per-request by _apply_effort_dial_and_context
    instruction=_PATTERN_SEARCH_INSTRUCTION,
    description=(
        "Freeform-reasoning fallback for pattern discovery, used only "
        "after the seeded script has already been tried deterministically "
        "and found nothing confident."
    ),
    # Model-internal sandbox: no separate cloud resource to provision, no
    # network call at import time (unlike VertexAiCodeExecutor, which
    # creates/loads a Vertex AI Extension resource on construction -- see
    # git history for why that was tried first and reverted).
    code_executor=BuiltInCodeExecutor(),
    # AgentTool-invoked sub-agents don't inherit the parent's `temp:` state
    # (confirmed empirically -- it's scoped to the invocation that sets it,
    # and this sub-agent's call is its own nested invocation), so it needs
    # its own copy of the seeding callback, not just before_model_callback.
    before_agent_callback=_seed_effort_defaults,
    before_model_callback=_apply_effort_dial_and_context,
)

pattern_search_tool = AgentTool(pattern_search_agent)


# ---------------------------------------------------------------------------
# Pattern Finder DB, via MCP only from the AGENT's side (no raw SQL from
# agent code -- see .agents-cli-spec.md, Constraints & Safety Rules). The
# deterministic pre-check functions above call app.mcp_server.db_ops
# directly, in-process -- no MCP round trip, since that code is ours, not
# agent-generated. Local dev: stdio to our own server module. Deployed:
# swap for SseConnectionParams pointing at the MCP server's Cloud Run URL
# (see google-agents-cli-deploy when we get there).
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
# The dispatcher: intercepts every turn before the LLM/tool loop. Returns
# Content to resolve deterministically (or via a narrow LLM call), or None
# to hand off to the full agent (Guess phase only -- Learn phase always
# resolves here). See module docstring for the full design.
# ---------------------------------------------------------------------------


async def _before_agent_dispatch(
    callback_context: CallbackContext,
) -> genai_types.Content | None:
    text = _get_text(callback_context.user_content)

    dial = _parse_effort_dial_from_text(text)
    effort_dial = dial if dial is not None else _DEFAULT_EFFORT_DIAL
    # Still state-based (unlike the handoff instructions below): this exact
    # pattern (before_agent_callback sets temp:effort_dial, before_model_
    # callback reads it) is what _apply_effort_dial_and_context already
    # relies on for the model/thinking-config override, proven reliable all
    # session -- no reason to change a working mechanism.
    callback_context.state["temp:effort_dial"] = effort_dial

    emit_ui_field = _parse_field(text, "EMIT_UI")
    emit_ui = (emit_ui_field or "on").strip().lower() != "off"
    emit_ui_instruction = (
        "As your LAST action, call emit_agent_reasoning with your guess, "
        "confidence, matched_via, and trace, plus pattern_id and rule where "
        "applicable (see its docstring for the exact fields) -- these let "
        "the app layer capture the right pattern later without re-deriving "
        "your reasoning. Write `rule` using x0, x1, x2, ... for the "
        "labeled inputs in order, never the actual label names, if it's a "
        "computable expression."
        if emit_ui
        else "Do NOT call emit_agent_reasoning this turn -- just reply with text."
    )

    phase = (_parse_field(text, "PHASE") or "").lower()

    if phase == "guess":
        label_values = _parse_label_values(text)
        if not label_values:
            return None  # malformed; let the agent do its best
        content, handoff = await _guess_phase_content(label_values, emit_ui)
        if content is not None:
            return content
        _handoff_instructions[callback_context.invocation_id] = (
            f"{handoff}\n\nYour confidence bar and how hard to search: "
            f"{_effort_description(effort_dial)}\n\n{emit_ui_instruction}"
        )
        return None

    if phase == "learn":
        correct_consequence = _parse_field(text, "CORRECT_CONSEQUENCE")
        if correct_consequence is None:
            return None  # malformed; let the agent do its best
        label_values = _parse_label_values(text)
        matched_via = _parse_field(text, "MATCHED_VIA")
        pattern_id_field = _parse_field(text, "PATTERN_ID")
        applied_rule = _parse_field(text, "APPLIED_RULE")
        guess_value = _parse_field(text, "GUESS_VALUE")
        return await _learn_phase_content(
            label_values,
            correct_consequence,
            matched_via,
            int(pattern_id_field) if pattern_id_field and pattern_id_field.isdigit() else None,
            applied_rule,
            guess_value,
            emit_ui,
            effort_dial,
        )

    return None


# ---------------------------------------------------------------------------
# Root agent. Reached only for Guess-phase turns the deterministic
# pre-check couldn't resolve (Learn phase always resolves in the
# dispatcher above and never reaches this instruction in practice).
# ---------------------------------------------------------------------------

_ROOT_INSTRUCTION = """You are the Pattern Finder agent. Users give you a
"scenario" made of up to 5 labeled inputs (label -> value; values can be
numeric or textual, and a slot may be intentionally absent). Your job is to
guess the scenario's "consequence" by finding and reusing patterns from
past scenarios.

You are only ever invoked for a guess AFTER a deterministic pre-check has
already ruled out the fast, no-LLM-needed cases (an exact label-set match
with a plain numeric rule, or the seeded pattern-search script finding a
confident match by itself). Every message you receive this turn is
appended with the specific reason you were invoked, what's already been
ruled out, and any historical data already fetched for you -- read that
appended context carefully before acting, and don't redo work it says is
already done (e.g. don't re-call get_pattern_by_label_set if you're told a
pattern was already found for you, and don't re-run the seeded script if
you're told it already ran and what it found).

Always state, in your reply: your guess (or "I don't know"), your
confidence, how you reached it, and which model answered. Then follow the
appended instruction about whether to call emit_agent_reasoning.

If you ever receive a "PHASE: learn" message directly, that's unexpected
(the deterministic layer normally handles Learn phase entirely) --
acknowledge that the correct consequence was noted, without fabricating
any pattern-capture action you can't verify actually happened.
"""

root_agent = Agent(
    name="pattern_finder",
    model=_BALANCED.model,  # overridden per-request by _apply_effort_dial_and_context
    instruction=_ROOT_INSTRUCTION,
    description=(
        "Guesses the consequence of a labeled scenario by finding and "
        "reusing patterns, then learns from the confirmed correct answer. "
        "Most turns are resolved deterministically before reaching this "
        "agent at all -- see app.agent._before_agent_dispatch."
    ),
    tools=[
        db_mcp_toolset,
        pattern_search_tool,
        evaluate_numeric_rule,
        emit_agent_reasoning,
        emit_pattern_captured,
    ],
    before_agent_callback=_before_agent_dispatch,
    before_model_callback=_apply_effort_dial_and_context,
    after_tool_callback=_capture_a2ui_payload,
    after_model_callback=_attach_a2ui_surface,
    after_agent_callback=_cleanup_handoff_instructions,
)

app = App(name="pattern-finder", root_agent=root_agent)
