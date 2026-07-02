"""SQLAlchemy Core schema for the Pattern Finder DB.

This is the single source of truth for the schema described in
`Pattern Finder Outline.txt` (Section IV). It is written with SQLAlchemy
Core (not the ORM) so the same table definitions produce correct DDL on
both SQLite (local dev, see app/db/engine.py) and Cloud SQL Postgres
(deployed) via ``MetaData.create_all(engine)`` -- no hand-maintained,
dialect-specific schema files to keep in sync.
"""

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    UniqueConstraint,
)

metadata = MetaData()

labels = Table(
    "labels",
    metadata,
    Column("label_id", Integer, primary_key=True, autoincrement=True),
    Column("label", Text, nullable=False),
    UniqueConstraint("label", name="uq_labels_label"),
)

patterns = Table(
    "patterns",
    metadata,
    Column("pattern_id", Integer, primary_key=True, autoincrement=True),
    # Abstracted, label-free description, e.g. "data fit a parabolic
    # pattern when labels are sorted in ascending order" (Section VII).
    Column("text_desc", Text, nullable=False),
    # A rule (plain text directive), a code snippet, or a link to one --
    # applied by the pattern-apply skill.
    Column("rule_or_code_link", Text, nullable=True),
)

scenarios = Table(
    "scenarios",
    metadata,
    Column("scenario_id", Integer, primary_key=True, autoincrement=True),
    Column("type", Text, nullable=True),
    # Filled in once the user provides the correct consequence (Section
    # VI, step 7). Null between the guess and the user's confirmation.
    Column("consequence", Text, nullable=True),
    Column(
        "pattern_id",
        Integer,
        ForeignKey("patterns.pattern_id"),
        nullable=True,
    ),
)

scenario_inputs = Table(
    "scenario_inputs",
    metadata,
    Column(
        "scenario_id",
        Integer,
        ForeignKey("scenarios.scenario_id"),
        primary_key=True,
    ),
    Column(
        "label_id", Integer, ForeignKey("labels.label_id"), primary_key=True
    ),
    # Nullable: a scenario can have up to 5 inputs, some slots absent --
    # this is itself a pattern signal (see pattern_finder.py's structural
    # presence-pattern check).
    Column("value", Text, nullable=True),
)

pattern_labels = Table(
    "pattern_labels",
    metadata,
    Column(
        "pattern_id",
        Integer,
        ForeignKey("patterns.pattern_id"),
        primary_key=True,
    ),
    Column(
        "label_id", Integer, ForeignKey("labels.label_id"), primary_key=True
    ),
)
