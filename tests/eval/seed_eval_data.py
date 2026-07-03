"""Seeds DB state the eval dataset's cases depend on.

Run once before `agents-cli eval generate` (which runs the agent against
the SAME default local DB this script writes to -- see app/db/engine.py):

    uv run python tests/eval/seed_eval_data.py
    agents-cli eval generate
    agents-cli eval grade

Idempotent: safe to run more than once (checks before inserting).

Cases and why each needs what:
  - "no_history_says_dont_know" (basic-dataset.json) needs NO seeding --
    it uses label names guaranteed not to collide with anything below.
  - "exact_pattern_match_applies_correctly" needs a captured pattern
    (Section VI step 1: exact label-set match) to apply.
  - "insufficient_data_stays_honest" needs exactly ONE historical
    scenario -- enough to tempt a pattern-search guess, not enough to be
    confident about it (Pattern Finder Outline.txt Section VII: an
    underdetermined system's "exact fit" is one of infinitely many, and
    shouldn't be trusted as THE answer).
"""

from app.mcp_server import db_ops

EXACT_MATCH_LABELS = ["eval_side_a", "eval_side_b"]
INSUFFICIENT_DATA_LABELS = ["eval_trial_x", "eval_trial_y"]


def seed_exact_pattern_match() -> None:
    if db_ops.get_pattern_by_label_set(EXACT_MATCH_LABELS):
        print("exact_pattern_match: already seeded, skipping")
        return
    a, b = EXACT_MATCH_LABELS
    ids = [
        db_ops.insert_scenario("t", {a: str(x), b: str(y)}, str(x * y))
        for x, y in [(2, 3), (4, 5), (6, 7)]
    ]
    pattern_id = db_ops.upsert_pattern("product of the two inputs", f"{a} * {b}")
    db_ops.link_pattern_to_scenarios(
        pattern_id, ids, update_label_set=True, label_names=EXACT_MATCH_LABELS
    )
    print(f"exact_pattern_match: seeded pattern_id={pattern_id}")


def seed_insufficient_data() -> None:
    existing = db_ops.get_scenarios_by_label_set(INSUFFICIENT_DATA_LABELS)
    if existing:
        print("insufficient_data: already seeded, skipping")
        return
    x, y = INSUFFICIENT_DATA_LABELS
    db_ops.insert_scenario("t", {x: "5", y: "2"}, "7")
    print("insufficient_data: seeded 1 scenario")


if __name__ == "__main__":
    seed_exact_pattern_match()
    seed_insufficient_data()
