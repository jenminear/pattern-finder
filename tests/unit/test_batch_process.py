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
"""Unit tests for scripts/batch_process.py's pure helpers -- no agent or
session dependency, so these run fast with no network/LLM calls. The
full pipeline (real agent, real spreadsheet I/O) is covered by
tests/integration/test_batch_process.py instead.
"""

import inspect

import scripts.batch_process as bp


class TestConsequenceIsolation:
    # The user's core requirement: the Guess-phase turn must never see a
    # row's ground-truth "Consequence" value. Verified at the function
    # SIGNATURE level, not just by reading the implementation -- if anyone
    # ever adds a consequence parameter to this function, this test fails
    # immediately, regardless of whether the new parameter actually gets
    # used.
    def test_build_guess_message_has_no_consequence_parameter(self):
        params = set(inspect.signature(bp.build_guess_message).parameters)
        assert params == {"label_values", "effort_dial"}
        assert not any("conseq" in p.lower() for p in params)

    def test_build_guess_message_output_never_contains_consequence_text(self):
        # Belt-and-suspenders: even if a future refactor renamed the
        # parameter, the actual message text sent to the agent must not
        # contain a value that's only ever passed as a consequence.
        msg = bp.build_guess_message({"a": "1", "b": "2"}, 0.5)
        assert "CORRECT_CONSEQUENCE" not in msg
        assert "PHASE: guess" in msg


class TestSliderValueToEffortDial:
    def test_maps_tier_1_to_economy_midpoint(self):
        assert bp.slider_value_to_effort_dial(1) == 1 / 6

    def test_maps_tier_2_to_balanced_midpoint(self):
        assert bp.slider_value_to_effort_dial(2) == 0.5

    def test_maps_tier_3_to_max_quality_midpoint(self):
        assert bp.slider_value_to_effort_dial(3) == 5 / 6

    def test_string_digit_is_accepted(self):
        assert bp.slider_value_to_effort_dial("2") == 0.5

    def test_blank_falls_back_to_balanced(self):
        assert bp.slider_value_to_effort_dial(None) == 0.5

    def test_out_of_range_falls_back_to_balanced(self):
        assert bp.slider_value_to_effort_dial(99) == 0.5

    def test_non_numeric_falls_back_to_balanced(self):
        assert bp.slider_value_to_effort_dial("thorough") == 0.5


class TestSanitizeLabel:
    def test_leaves_clean_label_untouched(self):
        assert bp.sanitize_label("product_name") == "product_name"

    def test_collapses_internal_whitespace(self):
        assert bp.sanitize_label("product name") == "product_name"

    def test_strips_surrounding_whitespace(self):
        assert bp.sanitize_label("  product  ") == "product"


class TestRowToLabelValues:
    def test_skips_slots_with_blank_labels(self):
        labels = ["a", "", None, "d", "e"]
        values = ["1", "2", "3", "4", None]
        result = bp.row_to_label_values(labels, values)
        assert result == {"a": "1", "d": "4", "e": None}

    def test_converts_numeric_cell_values_to_strings(self):
        result = bp.row_to_label_values(["a"], [34])
        assert result == {"a": "34"}

    def test_all_blank_labels_yields_empty_dict(self):
        assert bp.row_to_label_values([None, "", None], [1, 2, 3]) == {}


class TestBuildLearnMessage:
    def test_echoes_reasoning_fields(self):
        reasoning = {
            "guess": "42",
            "matched_via": "exact_pattern",
            "pattern_id": 7,
            "rule": "x0 + x1",
        }
        msg = bp.build_learn_message({"a": "1"}, 0.5, "42", reasoning)
        assert "CORRECT_CONSEQUENCE: 42" in msg
        assert "GUESS_VALUE: 42" in msg
        assert "MATCHED_VIA: exact_pattern" in msg
        assert "PATTERN_ID: 7" in msg
        assert "APPLIED_RULE: x0 + x1" in msg

    def test_missing_reasoning_omits_optional_fields(self):
        msg = bp.build_learn_message({"a": "1"}, 0.5, "42", None)
        assert "CORRECT_CONSEQUENCE: 42" in msg
        assert "GUESS_VALUE" not in msg
        assert "MATCHED_VIA" not in msg


class TestDetermineResult:
    def test_dont_know_guess(self):
        assert bp.determine_result("I don't know", None) == "I don't know"
        assert bp.determine_result(None, None) == "I don't know"

    def test_matched_is_correct(self):
        assert bp.determine_result("42", {"matched": True}) == "Correct"

    def test_unmatched_is_incorrect(self):
        assert bp.determine_result("42", {"matched": False}) == "Incorrect"
        assert bp.determine_result("42", None) == "Incorrect"


class TestSummarizeCaptured:
    def test_none_action_reports_no_pattern(self):
        assert bp.summarize_captured({"action": "none"}) == "(no pattern captured)"
        assert bp.summarize_captured(None) == "(no pattern captured)"

    def test_revised_action_includes_all_fields(self):
        summary = bp.summarize_captured(
            {
                "action": "revised",
                "text_desc": "square of the input",
                "rule_or_code_link": "x0^2",
                "scenarios_linked": 3,
            }
        )
        assert "Action: revised" in summary
        assert "square of the input" in summary
        assert "x0^2" in summary
        assert "3" in summary
