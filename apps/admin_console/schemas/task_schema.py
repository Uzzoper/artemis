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

from typing import Any, get_args

from pydantic import BaseModel, field_validator, model_validator

from artemis.config.constants import LLMProvider

# Providers accepted by the per-task override. Single source of truth: the same
# literal types the LLM config validates against.
SUPPORTED_LLM_PROVIDERS: tuple[str, ...] = get_args(LLMProvider)


class RunRequest(BaseModel):
    goal: str | None = None
    goals: list[str] | None = None
    profile: str | None = "flash"
    expected_output: str | None = None
    enable_outputter: bool | None = None
    # Pro-profile tuning (ignored by the Flash profile): a coarse Checker preset
    # ('off' | 'final' | 'checkpoints' | 'strict') and the Explorer perception
    # version used by the Operator ('flash' | 'pro' | 'ultra').
    verification_level: str | None = None
    explorer_mode: str | None = None
    # Per-task LLM override: pins every model of this one task to `llm_model`
    # (optionally on `llm_provider`) without touching artemis.jsonc or any
    # global state. Omitted/blank means "use the configured nodes".
    llm_model: str | None = None
    llm_provider: str | None = None
    locked_app_package: str | None = None
    app_path: str | None = None
    device_serial: str | None = None
    ingress: str | None = "frontend"
    session_id: str | None = None
    conversation_id: str | None = None

    @field_validator("llm_model", "llm_provider", mode="before")
    @classmethod
    def _blank_override_is_unset(cls, value: Any) -> Any:
        """Treat a blank/whitespace override as 'not requested'."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("llm_provider")
    @classmethod
    def _known_provider(cls, value: str | None) -> str | None:
        """Reject an unknown provider up front instead of failing mid-task."""
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(
                f"Unknown llm_provider {value!r}. Supported providers: "
                + ", ".join(SUPPORTED_LLM_PROVIDERS)
            )
        return normalized

    @model_validator(mode="after")
    def _provider_requires_model(self) -> "RunRequest":
        """Reject a provider without a model (FastAPI answers 422).

        A provider only says where to send the model, so accepting it alone
        would enqueue a task that dies mid-run on a broken endpoint.
        """
        if self.llm_provider and not self.llm_model:
            raise ValueError(
                "llm_provider requires llm_model: pass llm_model too, or drop llm_provider."
            )
        return self


class ReplayRequest(BaseModel):
    device_id: str
    user_submits: dict
    tool_name: str = "ask_explorer"
    replay_id: str | None = None


class StopRequest(BaseModel):
    session_id: str | None = None
    device_id: str | None = None
    all: bool = False
