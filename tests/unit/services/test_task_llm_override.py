# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-task LLM override: request plumbing and endpoint resolution precedence.

Precedence is task override > ``artemis.jsonc`` node config > built-in default,
and the override lives on the per-task context, so nothing global is mutated.
"""

from types import SimpleNamespace

from artemis.llm.router import ModelProvider
from artemis.services.llm import _resolve_endpoint
from artemis.sdk.builders.task_request_builder import TaskRequestBuilder


def _node(provider: str = "google", model: str = "gemini-3.8-flash", **extra):
    return SimpleNamespace(provider=provider, model=model, temperature=0.0, **extra)


def _ctx(node, llm_model=None, llm_provider=None):
    """Minimal stand-in for the per-task ArtemisContext."""
    return SimpleNamespace(
        llm_config=SimpleNamespace(
            get_agent=lambda name: node,
            get_utils=lambda name: node,
        ),
        llm_model=llm_model,
        llm_provider=llm_provider,
    )


def test_override_wins_over_node_config():
    endpoint = _resolve_endpoint(
        _ctx(_node(), llm_model="gpt-5.1", llm_provider="openai"),
        "operator",
    )

    assert endpoint.provider == ModelProvider.OPENAI
    assert endpoint.model_name == "gpt-5.1"


def test_node_config_used_when_no_override():
    endpoint = _resolve_endpoint(_ctx(_node(provider="anthropic", model="claude-x")), "operator")

    assert endpoint.provider == ModelProvider.ANTHROPIC
    assert endpoint.model_name == "claude-x"


def test_model_only_override_keeps_node_provider():
    endpoint = _resolve_endpoint(
        _ctx(_node(provider="google"), llm_model="gemini-3.8-pro"), "planner"
    )

    assert endpoint.provider == ModelProvider.GOOGLE
    assert endpoint.model_name == "gemini-3.8-pro"


def test_blank_override_falls_back_to_node_config():
    endpoint = _resolve_endpoint(_ctx(_node(), llm_model="  ", llm_provider=""), "planner")

    assert endpoint.provider == ModelProvider.GOOGLE
    assert endpoint.model_name == "gemini-3.8-flash"


def test_override_also_applies_to_resolved_fallback():
    node = _node(fallback=_node(model="gemini-3.7-flash"))

    endpoint = _resolve_endpoint(
        _ctx(node, llm_model="gpt-5.1", llm_provider="openai"),
        "operator",
        use_fallback=True,
    )

    assert endpoint.provider == ModelProvider.OPENAI
    assert endpoint.model_name == "gpt-5.1"


def test_builder_carries_override_on_the_task_request():
    builder = TaskRequestBuilder(goal="Audit checkout")

    request = builder.with_llm_override(model="  gpt-5.1  ", provider=" OpenAI ").build()

    assert request.llm_model == "gpt-5.1"
    assert request.llm_provider == "openai"


def test_builder_blank_override_leaves_request_unset():
    builder = TaskRequestBuilder(goal="Open Settings")

    request = builder.with_llm_override(model="  ").build()

    assert request.llm_model is None
    assert request.llm_provider is None
