"""Pattern Finder MCP server.

Exposes the six DB operations from db_ops.py as MCP tools -- these are the
*only* way the agent (or anything else) touches the Pattern Finder DB (see
.agents-cli-spec.md, "Constraints & Safety Rules": no raw SQL from agent
code). Each tool wraps a fixed, validated operation; there is no
execute_sql-style escape hatch.

Run locally over stdio (what McpToolset's StdioConnectionParams expects):
    uv run python -m app.mcp_server.server

Deployed, this would run over SSE/HTTP behind its own Cloud Run service,
mirroring the genmedia-for-commerce sample's McpToolset wiring (see
.agents-cli-spec.md, Reference Samples).
"""

from fastmcp import FastMCP

from app.mcp_server import db_ops

mcp = FastMCP("pattern-finder-db")


@mcp.tool
def get_pattern_by_label_set(label_names: list[str]) -> dict | None:
    """Look up whether this exact set of labels is already tied to a known pattern.

    Args:
        label_names: The scenario's input labels (order doesn't matter).

    Returns:
        {pattern_id, text_desc, rule_or_code_link} if an exact label-set
        match exists, else None.
    """
    return db_ops.get_pattern_by_label_set(label_names)


@mcp.tool
def get_scenarios_by_label_set(label_names: list[str]) -> list[dict]:
    """Fetch all historical scenarios whose label set exactly matches label_names.

    Args:
        label_names: The scenario's input labels (order doesn't matter).

    Returns:
        List of {scenario_id, type, consequence, pattern_id, inputs} --
        inputs is {label: value}. Used by pattern-search to find a rule.
    """
    return db_ops.get_scenarios_by_label_set(label_names)


@mcp.tool
def get_all_pattern_descriptions() -> list[dict]:
    """Fetch every known pattern's abstracted (label-free) description.

    Returns:
        List of {pattern_id, text_desc, rule_or_code_link}. Used by the
        label-independent-match fallback when no label-based pattern applies.
    """
    return db_ops.get_all_pattern_descriptions()


@mcp.tool
def insert_scenario(
    scenario_type: str | None, inputs: dict[str, str | None], consequence: str
) -> int:
    """Persist a new scenario together with its user-confirmed consequence.

    Args:
        scenario_type: Optional free-text scenario type/category.
        inputs: {label: value} for up to 5 input fields. A value may be
            None to represent a deliberately absent input slot.
        consequence: The correct consequence the user provided.

    Returns:
        The new scenario_id.
    """
    return db_ops.insert_scenario(scenario_type, inputs, consequence)


@mcp.tool
def upsert_pattern(
    text_desc: str,
    rule_or_code_link: str | None = None,
    pattern_id: int | None = None,
) -> int:
    """Create a new pattern, or update an existing one's description/rule.

    Args:
        text_desc: Abstracted, label-free description of the pattern (e.g.
            "data fit a parabolic pattern when labels are sorted ascending").
        rule_or_code_link: A rule, code snippet, or link to one, applied by
            the pattern-apply skill.
        pattern_id: If given, updates that pattern instead of creating a new one.

    Returns:
        The pattern_id (new or updated).
    """
    return db_ops.upsert_pattern(text_desc, rule_or_code_link, pattern_id)


@mcp.tool
def link_pattern_to_scenarios(
    pattern_id: int,
    scenario_ids: list[int],
    update_label_set: bool,
    label_names: list[str] | None = None,
) -> None:
    """Attach a pattern to one or more scenarios, per Section VI/VII of the outline.

    Args:
        pattern_id: The pattern being attached.
        scenario_ids: Scenario(s) to link to this pattern. On the exact
            label-set match path, this is every scenario sharing that label
            set; on the label-independent path, just the one scenario.
        update_label_set: True ONLY on the exact label-set match path --
            registers this pattern's label set (pattern_labels). Must be
            False on the label-independent-match path, per the spec's
            safety rule that such matches must not alter which label sets a
            pattern is known for.
        label_names: Required when update_label_set=True.
    """
    db_ops.link_pattern_to_scenarios(
        pattern_id, scenario_ids, update_label_set, label_names
    )


@mcp.tool
def log_candidate_technique(
    label_names: str,
    rule: str,
    confidence: float | None = None,
    trace: str | None = None,
    source: str | None = None,
) -> int:
    """Record a candidate rule/technique your freeform search discovered,
    for a developer to review later -- not the same as reporting your
    answer for this turn. Call this whenever you found a candidate rule,
    even one you weren't confident enough to actually use.

    Args:
        label_names: Copy this VERBATIM from the "LABEL_NAMES: ..." line in
            your request -- do not reconstruct or paraphrase it from
            anything else (x0/x1/... indices are not a reliable source).
            If your request has no such line, do not call this tool.
        rule: The candidate rule/logic you found, in the same form you'd
            report it (x0/x1/... convention if it's a computable expression).
        confidence: Your confidence in it (0-1), if you had one.
        trace: A short trace of how you found it.
        source: Why you were searching -- e.g. "fresh discovery" or
            "pattern revision after a wrong guess".

    Returns:
        The new candidate_id.
    """
    return db_ops.log_candidate_technique(label_names, rule, confidence, trace, source)


if __name__ == "__main__":
    mcp.run()
