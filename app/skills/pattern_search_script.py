#!/usr/bin/env python3
"""
pattern_search_script.py

v1 of the `pattern-search` skill's script (see .agents-cli-spec.md,
Architecture & Sub-Agents / Reference Samples). Seeded from a script
already validated on real puzzles in an earlier session; the
pattern_search_agent (app/agent.py) is instructed to run this FIRST,
before falling back to freeform reasoning, per Section VII of
Pattern Finder Outline.txt:

    "the agent should first call an existing script it maintains for
    searching for patterns in the data. If the script does not yield any
    promising results, the agent should think about other ways to
    approach the pattern and test them on the available data."

    "When a successful pattern has been identified outside of the
    existing script for hunting for patterns, the script should be
    updated to include the new approach."

This file IS the "existing script" -- it is the thing the agent is meant
to extend over time. Editing this file on disk from within a running
agent (true self-modifying skill) is not implemented in this build; see
the project README for that scope boundary. For now, extending the
script is a manual step a developer performs based on what the agent
discovers, same as any other code review.

Known limitation carried over from the seed version: this script is
numeric-only (np.linalg.lstsq over floats). The outline allows textual /
qualitative input values too -- handling those is the first gap the
pattern-search skill's freeform-reasoning fallback needs to cover, and
a natural first extension to make here.

Strategy, in order (cheapest/most-general first):
    1. Structural check: does the output depend only on *which* inputs are
       present (their position pattern), ignoring values? Catches puzzles
       like the "missing input -> constant output" case.
    2. Polynomial least-squares fit: for a fixed number of inputs, build a
       feature matrix of degree-<=2 monomials (a^2, b^2, ab, a, b, 1, etc.)
       and solve via least squares. Reports an exact fit if residuals are
       ~0. This catches linear/quadratic combos like a^2+b^2+5ab+7.
    3. Single-variable transforms: for one-input puzzles, try a battery of
       common scalar functions (digit sum, digit product, reversal, mod,
       etc.) before falling back to polynomial fit on the raw value.
    4. Report best candidates ranked by fit quality, not just the first hit.

This won't replace careful reasoning about *why* a formula makes sense, but
it should get to candidate formulas much faster than manual trial and error.
"""

import itertools
import numpy as np


# ---------------------------------------------------------------------------
# 1. STRUCTURAL / POSITIONAL PATTERN CHECK
# ---------------------------------------------------------------------------

def check_structural_pattern(data):
    """
    Check whether output depends only on which input slots are present
    (i.e. ignore actual values, just look at the None/non-None pattern).
    Returns a dict {presence_pattern: set_of_outputs_seen} so you can see
    if each presence pattern maps to a single consistent output.
    """
    buckets = {}
    for inputs, output in data:
        pattern = tuple(x is not None for x in inputs)
        buckets.setdefault(pattern, set()).add(output)

    is_structural = all(len(outs) == 1 for outs in buckets.values())
    return is_structural, buckets


# ---------------------------------------------------------------------------
# 2. POLYNOMIAL LEAST-SQUARES FIT (degree <= 2, handles N inputs)
# ---------------------------------------------------------------------------

def build_poly_features(inputs, degree=2):
    """
    Build monomial feature names + values up to given degree for a tuple of
    numeric inputs, e.g. for (a, b) degree 2:
        1, a, b, a^2, ab, b^2
    """
    n = len(inputs)
    names = []
    values = []

    # degree 0
    names.append("1")
    values.append(1.0)

    # degree 1
    for i in range(n):
        names.append(f"x{i}")
        values.append(inputs[i])

    # degree 2
    if degree >= 2:
        for i, j in itertools.combinations_with_replacement(range(n), 2):
            label = f"x{i}*x{j}" if i != j else f"x{i}^2"
            names.append(label)
            values.append(inputs[i] * inputs[j])

    return names, values


def fit_polynomial(data, degree=2, tol=1e-6):
    """
    Fit output = linear combination of monomials in the inputs, via least
    squares. Only uses rows where ALL inputs are present (no None).
    Returns (names, coefficients, max_abs_residual, underdetermined_flag)
    or None if no usable rows.

    underdetermined_flag is True when there are fewer rows than free
    coefficients -- in that case an "exact fit" is meaningless (infinitely
    many exact fits exist) and should not be trusted as THE answer.
    """
    rows = [(inp, out) for inp, out in data if all(v is not None for v in inp)]
    if len(rows) < 2:
        return None

    n_inputs = len(rows[0][0])
    if any(len(inp) != n_inputs for inp, _ in rows):
        return None  # inconsistent input lengths; skip

    names = None
    A = []
    y = []
    for inp, out in rows:
        names, feats = build_poly_features(inp, degree=degree)
        A.append(feats)
        y.append(out)

    A = np.array(A, dtype=float)
    y = np.array(y, dtype=float)

    n_coef = A.shape[1]
    n_rows = A.shape[0]
    underdetermined = n_rows < n_coef

    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = A.dot(coef)
    max_resid = float(np.max(np.abs(pred - y)))

    return names, coef, max_resid, underdetermined


def format_polynomial(names, coef, tol=1e-6):
    terms = []
    for name, c in zip(names, coef):
        if abs(c) < tol:
            continue
        c_rounded = round(c)
        c_disp = c_rounded if abs(c - c_rounded) < 1e-4 else round(c, 4)
        if name == "1":
            terms.append(f"{c_disp}")
        else:
            terms.append(f"{c_disp}*{name}")
    return " + ".join(terms) if terms else "0"


# ---------------------------------------------------------------------------
# 3. SINGLE-VARIABLE TRANSFORM BATTERY (for one-input puzzles)
# ---------------------------------------------------------------------------

def digit_sum(n):
    return sum(int(d) for d in str(abs(int(n))))


def digit_product(n):
    p = 1
    for d in str(abs(int(n))):
        p *= int(d)
    return p


def digit_reverse(n):
    s = str(abs(int(n)))
    return int(s[::-1]) * (1 if n >= 0 else -1)


def try_single_variable_transforms(data):
    """
    For single-input puzzles, try a battery of named scalar transforms and
    see if any matches all rows exactly.
    """
    rows = [(inp[0], out) for inp, out in data if len(inp) == 1 and inp[0] is not None]
    if len(rows) < 2:
        return []

    candidates = {
        "x": lambda x: x,
        "-x": lambda x: -x,
        "x^2": lambda x: x ** 2,
        "digit_sum(x)": lambda x: digit_sum(x),
        "digit_product(x)": lambda x: digit_product(x),
        "digit_reverse(x)": lambda x: digit_reverse(x),
        "x - digit_sum(x)": lambda x: x - digit_sum(x),
        "digit_sum(x) - x": lambda x: digit_sum(x) - x,
        "digit_product(x) - digit_sum(x)": lambda x: digit_product(x) - digit_sum(x),
        "100 - x": lambda x: 100 - x,
        "x mod 100": lambda x: x % 100,
        "x mod 97": lambda x: x % 97,  # arbitrary prime, cheap to test
    }

    hits = []
    for label, fn in candidates.items():
        try:
            if all(fn(x) == out for x, out in rows):
                hits.append(label)
        except Exception:
            continue
    return hits


# ---------------------------------------------------------------------------
# 4. DRIVER
# ---------------------------------------------------------------------------

def fit_sparse_polynomial(data, degree=2):
    """
    When the system is underdetermined, least squares gives one of many
    exact fits -- usually an ugly one. Instead, search over small subsets
    of monomials (preferring fewer terms, then small integer-ish
    coefficients) for an exact fit using exhaustive subset search. Only
    practical for small numbers of candidate monomials, so this is a
    best-effort heuristic, not a guarantee of the "true" formula.
    """
    rows = [(inp, out) for inp, out in data if all(v is not None for v in inp)]
    if len(rows) < 2:
        return None

    n_inputs = len(rows[0][0])
    names, _ = build_poly_features(rows[0][0], degree=degree)
    n_feats = len(names)

    A_full = []
    y = []
    for inp, out in rows:
        _, feats = build_poly_features(inp, degree=degree)
        A_full.append(feats)
        y.append(out)
    A_full = np.array(A_full, dtype=float)
    y = np.array(y, dtype=float)

    n_rows = len(rows)
    best = None  # (n_terms, coef_roughness, names_subset, coef_subset)

    # Try subsets of size 1..n_rows (can't exactly solve more unknowns than
    # equations), smallest first so we prefer the simplest exact fit.
    max_subset_size = min(n_feats, n_rows)
    for k in range(1, max_subset_size + 1):
        found_at_this_size = False
        for combo in itertools.combinations(range(n_feats), k):
            sub_A = A_full[:, combo]
            # Solve in least-squares sense; check if it's actually exact.
            coef, _, rank, _ = np.linalg.lstsq(sub_A, y, rcond=None)
            if rank < k:
                continue  # underdetermined subset, skip
            pred = sub_A.dot(coef)
            resid = np.max(np.abs(pred - y))
            if resid < 1e-6:
                roughness = float(np.sum(np.abs(coef - np.round(coef))))
                candidate = (k, roughness, [names[i] for i in combo], coef)
                if best is None or candidate[:2] < best[:2]:
                    best = candidate
                found_at_this_size = True
        if found_at_this_size:
            break  # don't bother with larger subsets once we have a fit

    return best


def analyze(data, degree=2):
    print("=" * 70)
    print(f"Analyzing {len(data)} rows")
    print("=" * 70)

    # --- Structural check ---
    is_structural, buckets = check_structural_pattern(data)
    print("\n[1] Structural (presence-pattern) check:")
    for pattern, outs in buckets.items():
        present_idx = [i for i, p in enumerate(pattern) if p]
        print(f"    inputs present at {present_idx}: outputs seen = {outs}")
    structural_match = is_structural and len(buckets) > 1
    if structural_match:
        print("    -> STRONG MATCH: output depends only on which inputs are present.")
        print("    (Skipping polynomial fit -- it would just be fitting noise")
        print("     around a pattern that's already explained structurally.)")
    elif is_structural and len(buckets) == 1:
        print("    -> Inconclusive (only one presence-pattern in data).")
    else:
        print("    -> No match: same presence-pattern gives different outputs.")

    if structural_match:
        print()
        return

    # --- Single-variable transform battery ---
    sv_hits = try_single_variable_transforms(data)
    if sv_hits:
        print("\n[2] Single-variable transform matches (exact fit on all rows):")
        for h in sv_hits:
            print(f"    -> {h}")

    # --- Polynomial fit (only on fully-populated rows) ---
    result = fit_polynomial(data, degree=degree)
    if result:
        names, coef, max_resid, underdetermined = result
        print(f"\n[3] Polynomial fit (degree<={degree}, complete rows only):")
        if underdetermined:
            print("    WARNING: fewer data rows than free coefficients --")
            print("    this system is underdetermined. The least-squares fit")
            print("    below is just ONE of infinitely many exact fits and is")
            print("    likely NOT the true formula. Get more data, or see the")
            print("    sparse-fit search below for a simpler candidate.")
        formula = format_polynomial(names, coef)
        print(f"    least-squares formula: output ~= {formula}")
        print(f"    max |residual| across rows = {max_resid:.6g}")
        if max_resid < 1e-6 and not underdetermined:
            print("    -> EXACT FIT (system was fully/over-determined: trustworthy).")
        elif max_resid < 1e-6 and underdetermined:
            print("    -> Exact but UNTRUSTWORTHY (see warning above).")
        elif max_resid < 1e-2:
            print("    -> Near-exact fit (check rounding).")
        else:
            print("    -> Not a clean fit; formula is likely wrong or non-polynomial.")

        if underdetermined:
            sparse = fit_sparse_polynomial(data, degree=degree)
            if sparse:
                k, roughness, sub_names, sub_coef = sparse
                sparse_formula = format_polynomial(sub_names, sub_coef)
                print(f"\n    Simplest exact-fit candidate found ({k} term(s)):")
                print(f"        output ~= {sparse_formula}")
                print("    (Still treat this as a candidate, not a certainty,")
                print("     until verified against new data.)")
    else:
        print("\n[3] Polynomial fit: not enough complete rows to attempt.")

    print()
