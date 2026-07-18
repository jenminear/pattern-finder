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
"""Tests for app/fast_api_app.py's _capped_request_converter -- the fix for
a live GCP billing review that found Vertex AI spend traced to ADK's
RunConfig.max_llm_calls defaulting to 500 and never being overridden on
the live UI path (see app/agent.py's _FLASH_LITE comment for the full
writeup).

In tests/integration/ (not tests/unit/) because importing app.fast_api_app
directly pulls in real GCP client setup (telemetry, Cloud Logging) that
takes ~12s -- heavier than this project's unit tests, but a one-time
module-import cost, and the only way to test this wrapper without running
a real subprocess server (tests/integration/test_server_e2e.py's approach,
which can't easily assert on an internal RunConfig object).
"""

from google.adk.a2a.converters.request_converter import AgentRunRequest
from google.adk.agents.run_config import RunConfig

import app.fast_api_app as fast_api_app


def test_capped_request_converter_lowers_max_llm_calls(monkeypatch):
    # Stub out the real converter (which needs a live a2a RequestContext)
    # -- this test is only about _capped_request_converter's OWN logic:
    # does it take whatever run_config the real converter built and lower
    # its max_llm_calls, regardless of what that run_config looked like.
    fake_run_request = AgentRunRequest(
        user_id="u", session_id="s", run_config=RunConfig(max_llm_calls=500)
    )
    monkeypatch.setattr(
        fast_api_app,
        "convert_a2a_request_to_agent_run_request",
        lambda request, part_converter: fake_run_request,
    )

    result = fast_api_app._capped_request_converter(object(), object())

    assert result is fake_run_request
    assert result.run_config.max_llm_calls == fast_api_app._MAX_LLM_CALLS_PER_INVOCATION
    assert result.run_config.max_llm_calls < 500


def test_capped_request_converter_handles_missing_run_config(monkeypatch):
    # Defensive case: if the real converter ever returns a request with no
    # run_config at all, the wrapper must not crash trying to set an
    # attribute on None.
    fake_run_request = AgentRunRequest(user_id="u", session_id="s", run_config=None)
    monkeypatch.setattr(
        fast_api_app,
        "convert_a2a_request_to_agent_run_request",
        lambda request, part_converter: fake_run_request,
    )

    result = fast_api_app._capped_request_converter(object(), object())

    assert result.run_config is None  # left alone, not fabricated
