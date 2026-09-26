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

import json
from unittest.mock import patch, MagicMock

import pytest

from apps.admin_console.services.model_service import ModelService


def test_get_active_model_info_pro_architecture():
    """Verify that pro profile returns Pro architecture while keeping real LLM model."""
    info = ModelService.get_active_model_info("pro")
    assert info["name"] == "Pro"
    assert info["architecture"] == "ARTEMIS Pro"
    assert info["provider"] == "google"
    assert "id" in info


def test_get_active_model_info_flash_architecture():
    """Verify that flash profile returns Flash architecture."""
    info = ModelService.get_active_model_info("flash")
    assert info["name"] == "Flash"
    assert info["architecture"] == "ARTEMIS Flash"
    assert info["provider"] == "google"


def test_resolve_session_profile_from_device_info():
    """Verify profile resolution from device_info JSON."""
    row = {"device_info": json.dumps({"profile": "pro"})}
    assert ModelService.resolve_session_profile(row) == "pro"

    row_flash = {"device_info": json.dumps({"profile": "flash"})}
    assert ModelService.resolve_session_profile(row_flash) == "flash"


def test_resolve_session_profile_from_agent_names():
    """Verify profile resolution from agent/trace names."""
    row = {"device_info": None}
    assert ModelService.resolve_session_profile(row, agent_names=["planner", "operator"]) == "pro"
    assert ModelService.resolve_session_profile(row, agent_names=["flashrunner"]) == "flash"


# The configured (global) LLM is patched so these tests never read artemis.jsonc.
_CONFIGURED = ("google", "gemini-3.8-flash")


@patch.object(ModelService, "_get_llm_provider_and_model", return_value=_CONFIGURED)
def test_get_active_model_info_prefers_the_recorded_override(_configured):
    """A per-task override wins the model id and, when given, the provider."""
    info = ModelService.get_active_model_info("pro", "gpt-5.1", "openai")

    assert info["id"] == "gpt-5.1"
    assert info["provider"] == "openai"
    assert info["name"] == "Pro"
    assert info["architecture"] == "ARTEMIS Pro"


@patch.object(ModelService, "_get_llm_provider_and_model", return_value=_CONFIGURED)
def test_get_active_model_info_falls_back_to_config_without_override(_configured):
    """Rows without the override keys keep reporting the configured model."""
    info = ModelService.get_active_model_info("flash")

    assert info["id"] == "gemini-3.8-flash"
    assert info["provider"] == "google"
    assert info["name"] == "Flash"


@patch.object(ModelService, "_get_llm_provider_and_model", return_value=_CONFIGURED)
def test_get_active_model_info_keeps_node_provider_when_override_has_none(_configured):
    """A model-only override leaves the configured provider in place."""
    info = ModelService.get_active_model_info("pro", "gemini-3.8-pro")

    assert info["id"] == "gemini-3.8-pro"
    assert info["provider"] == "google"


def test_resolve_session_llm_override_reads_device_info():
    row = {"device_info": json.dumps({"llm_model": " gpt-5.1 ", "llm_provider": " OpenAI "})}
    assert ModelService.resolve_session_llm_override(row) == ("gpt-5.1", "openai")


def test_resolve_session_llm_override_accepts_a_parsed_device_info():
    row = {"device_info": {"llm_model": "gpt-5.1", "llm_provider": "openai"}}
    assert ModelService.resolve_session_llm_override(row) == ("gpt-5.1", "openai")


def test_resolve_session_llm_override_is_empty_for_legacy_and_broken_rows():
    # No device_info at all, a pre-override row, and malformed JSON.
    assert ModelService.resolve_session_llm_override({}) == (None, None)
    assert ModelService.resolve_session_llm_override({"device_info": None}) == (None, None)
    assert ModelService.resolve_session_llm_override(
        {"device_info": json.dumps({"profile": "pro"})}
    ) == (
        None,
        None,
    )
    assert ModelService.resolve_session_llm_override({"device_info": "{not json"}) == (None, None)
    assert ModelService.resolve_session_llm_override({"device_info": '"a string"'}) == (None, None)


def test_resolve_session_llm_override_treats_blank_values_as_unset():
    row = {"device_info": json.dumps({"llm_model": "  ", "llm_provider": ""})}
    assert ModelService.resolve_session_llm_override(row) == (None, None)
