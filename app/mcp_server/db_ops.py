"""Pattern Finder DB operations.

Plain Python functions (no MCP/transport concerns) implementing the six
operations named in .agents-cli-spec.md's "Tools Required" section. Kept
separate from server.py so they're directly unit-testable, and so the
deterministic app layer (the exact-label-set lookup before Handoff 1) can
call them without going through the MCP transport.

Exact label-set matching (used by get_pattern_by_label_set and
get_scenarios_by_label_set) means the *set* of labels attached to a
scenario/pattern equals the target set exactly -- no more, no fewer. See
Pattern Finder Outline.txt, Section VI, step 2.
"""

from __future__ import annotations

from sqlalchemy import Engine, Table, insert, select, update

from app.db.engine import get_engine
from app.db.models import candidate_techniques as candidate_techniques_t
from app.db.models import labels as labels_t
from app.db.models import pattern_labels as pattern_labels_t
from app.db.models import patterns as patterns_t
from app.db.models import scenario_inputs as scenario_inputs_t
from app.db.models import scenarios as scenarios_t


def _label_ids_for(engine: Engine, label_names: list[str]) -> dict[str, int]:
    """Map label name -> label_id for labels that already exist.

    Names not yet present in the `labels` table are simply absent from the
    returned dict (a label that doesn't exist yet can't be part of any
    existing scenario/pattern's label set).
    """
    with engine.connect() as conn:
        rows = conn.execute(
            select(labels_t.c.label_id, labels_t.c.label).where(
                labels_t.c.label.in_(label_names)
            )
        ).all()
    return {row.label: row.label_id for row in rows}


def _get_or_create_label_ids(engine: Engine, label_names: list[str]) -> dict[str, int]:
    """Like _label_ids_for, but creates any missing labels."""
    with engine.begin() as conn:
        existing = {
            row.label: row.label_id
            for row in conn.execute(
                select(labels_t.c.label_id, labels_t.c.label).where(
                    labels_t.c.label.in_(label_names)
                )
            ).all()
        }
        missing = [name for name in label_names if name not in existing]
        for name in missing:
            result = conn.execute(insert(labels_t).values(label=name))
            existing[name] = result.inserted_primary_key[0]
    return existing


def _exact_label_set_owner_ids(
    engine: Engine,
    assoc_table: Table,
    owner_col: str,
    target_label_ids: set[int],
) -> list[int]:
    """Find owner ids (scenario_id or pattern_id) whose associated label-id
    set in *assoc_table* exactly equals *target_label_ids*.

    Done in Python rather than dialect-specific SQL (e.g. FILTER/array_agg)
    so the same code works against SQLite (dev) and Postgres (prod).
    """
    if not target_label_ids:
        return []
    owner_column = assoc_table.c[owner_col]
    with engine.connect() as conn:
        # Candidates: owners with at least one row matching a target label.
        candidate_ids = {
            row[0]
            for row in conn.execute(
                select(owner_column).where(
                    assoc_table.c.label_id.in_(target_label_ids)
                )
            ).all()
        }
        if not candidate_ids:
            return []
        # Now fetch each candidate's FULL label-id set (not just the
        # matching ones) to confirm it's an exact match, not a superset.
        rows = conn.execute(
            select(owner_column, assoc_table.c.label_id).where(
                owner_column.in_(candidate_ids)
            )
        ).all()
    owner_to_labels: dict[int, set[int]] = {}
    for owner_id, label_id in rows:
        owner_to_labels.setdefault(owner_id, set()).add(label_id)
    return [
        owner_id
        for owner_id, label_set in owner_to_labels.items()
        if label_set == target_label_ids
    ]


def get_pattern_by_label_set(label_names: list[str]) -> dict | None:
    """Section VI, step 2: is this exact label set already tied to a pattern?"""
    engine = get_engine()
    label_ids = _label_ids_for(engine, label_names)
    if len(label_ids) != len(set(label_names)):
        return None  # some labels don't exist yet -> no existing pattern possible
    target_ids = set(label_ids.values())
    pattern_ids = _exact_label_set_owner_ids(
        engine, pattern_labels_t, "pattern_id", target_ids
    )
    if not pattern_ids:
        return None
    # An exact label set should map to at most one pattern by construction
    # (see link_pattern_to_scenarios), but guard defensively.
    pattern_id = pattern_ids[0]
    with engine.connect() as conn:
        row = conn.execute(
            select(patterns_t).where(patterns_t.c.pattern_id == pattern_id)
        ).one()
    return {
        "pattern_id": row.pattern_id,
        "text_desc": row.text_desc,
        "rule_or_code_link": row.rule_or_code_link,
    }


def get_scenarios_by_label_set(label_names: list[str]) -> list[dict]:
    """Section VI, step 3: gather same-labeled scenarios for pattern-search."""
    engine = get_engine()
    label_ids = _label_ids_for(engine, label_names)
    if len(label_ids) != len(set(label_names)):
        return []
    target_ids = set(label_ids.values())
    scenario_ids = _exact_label_set_owner_ids(
        engine, scenario_inputs_t, "scenario_id", target_ids
    )
    if not scenario_ids:
        return []
    id_to_label = {v: k for k, v in label_ids.items()}
    with engine.connect() as conn:
        scenario_rows = conn.execute(
            select(scenarios_t).where(scenarios_t.c.scenario_id.in_(scenario_ids))
        ).all()
        input_rows = conn.execute(
            select(scenario_inputs_t).where(
                scenario_inputs_t.c.scenario_id.in_(scenario_ids)
            )
        ).all()
    inputs_by_scenario: dict[int, dict[str, str | None]] = {}
    for row in input_rows:
        inputs_by_scenario.setdefault(row.scenario_id, {})[
            id_to_label.get(row.label_id, str(row.label_id))
        ] = row.value
    return [
        {
            "scenario_id": row.scenario_id,
            "type": row.type,
            "consequence": row.consequence,
            "pattern_id": row.pattern_id,
            "inputs": inputs_by_scenario.get(row.scenario_id, {}),
        }
        for row in scenario_rows
    ]


def get_all_pattern_descriptions() -> list[dict]:
    """For the label-independent-match fallback (Section VI, step 5)."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(select(patterns_t)).all()
    return [
        {
            "pattern_id": row.pattern_id,
            "text_desc": row.text_desc,
            "rule_or_code_link": row.rule_or_code_link,
        }
        for row in rows
    ]


def insert_scenario(
    scenario_type: str | None,
    inputs: dict[str, str | None],
    consequence: str,
) -> int:
    """Section VI, step 8: persist a scenario with its confirmed consequence."""
    engine = get_engine()
    label_ids = _get_or_create_label_ids(engine, list(inputs.keys()))
    with engine.begin() as conn:
        result = conn.execute(
            insert(scenarios_t).values(type=scenario_type, consequence=consequence)
        )
        scenario_id = result.inserted_primary_key[0]
        if inputs:
            conn.execute(
                insert(scenario_inputs_t),
                [
                    {
                        "scenario_id": scenario_id,
                        "label_id": label_ids[label],
                        "value": value,
                    }
                    for label, value in inputs.items()
                ],
            )
    return scenario_id


def upsert_pattern(
    text_desc: str,
    rule_or_code_link: str | None = None,
    pattern_id: int | None = None,
) -> int:
    """Create a new pattern, or update text_desc/rule_or_code_link on an
    existing one when *pattern_id* is given."""
    engine = get_engine()
    with engine.begin() as conn:
        if pattern_id is not None:
            conn.execute(
                update(patterns_t)
                .where(patterns_t.c.pattern_id == pattern_id)
                .values(text_desc=text_desc, rule_or_code_link=rule_or_code_link)
            )
            return pattern_id
        result = conn.execute(
            insert(patterns_t).values(
                text_desc=text_desc, rule_or_code_link=rule_or_code_link
            )
        )
        return result.inserted_primary_key[0]


def link_pattern_to_scenarios(
    pattern_id: int,
    scenario_ids: list[int],
    update_label_set: bool,
    label_names: list[str] | None = None,
) -> None:
    """Section VI step 9 / Section VII: attach a pattern to scenario(s).

    update_label_set must be True only on the exact-label-set match path --
    it registers/refreshes which label set this pattern is known for
    (pattern_labels). On the label-independent-match path it must be False:
    the matching scenario gets linked, but the pattern's label set is left
    untouched (spec Constraints & Safety Rules).
    """
    engine = get_engine()
    with engine.begin() as conn:
        if scenario_ids:
            conn.execute(
                update(scenarios_t)
                .where(scenarios_t.c.scenario_id.in_(scenario_ids))
                .values(pattern_id=pattern_id)
            )
        if update_label_set:
            if not label_names:
                raise ValueError("label_names is required when update_label_set=True")
            label_ids = _get_or_create_label_ids(engine, label_names)
            existing = {
                row.label_id
                for row in conn.execute(
                    select(pattern_labels_t.c.label_id).where(
                        pattern_labels_t.c.pattern_id == pattern_id
                    )
                ).all()
            }
            new_rows = [
                {"pattern_id": pattern_id, "label_id": lid}
                for lid in label_ids.values()
                if lid not in existing
            ]
            if new_rows:
                conn.execute(insert(pattern_labels_t), new_rows)


def log_candidate_technique(
    label_names: str,
    rule: str,
    confidence: float | None = None,
    trace: str | None = None,
    source: str | None = None,
) -> int:
    """Records a technique pattern_search's freeform reasoning discovered
    that the seeded script didn't already cover -- see candidate_techniques'
    table comment in app/db/models.py. Purely a log: never read by any
    deterministic code path, only by a developer reviewing the table
    directly.
    """
    engine = get_engine()
    with engine.begin() as conn:
        result = conn.execute(
            insert(candidate_techniques_t).values(
                label_names=label_names,
                rule=rule,
                confidence=confidence,
                trace=trace,
                source=source,
            )
        )
        return result.inserted_primary_key[0]
