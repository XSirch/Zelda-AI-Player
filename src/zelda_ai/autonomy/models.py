from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


IntentMode = Literal[
    "explore",
    "navigate",
    "interact",
    "combat",
    "dialogue",
    "menu",
    "observe",
]

IntentDirection = Literal["forward", "back", "left", "right", "up", "down"]


class AgentIntent(BaseModel):
    """High-level intent only.

    The model never chooses a controller skill or a raw button.  The local
    controller owns moment-to-moment input and may keep acting while a new
    AgentIntent is being inferred.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    objective: str = Field(min_length=1, max_length=240)
    summary: str = Field(min_length=1, max_length=320)
    mode: IntentMode = "explore"
    target_actor_id: int | None = Field(default=None, ge=-32768, le=32767)
    target_actor_params: int | None = Field(default=None, ge=-32768, le=32767)
    target_actor_uid: str | None = Field(default=None, max_length=96)
    target_position: tuple[float, float, float] | None = None
    direction: IntentDirection | None = None
    choice_index: int | None = Field(default=None, ge=0, le=2)
    horizon_ms: int = Field(default=15000, ge=2000, le=60000)

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        properties = schema.get("properties", {})
        if properties:
            # OpenAI/OpenRouter strict structured output requires every property
            # to be present. Nullable fields remain nullable, but are not omitted.
            schema["required"] = list(properties)
            for value in properties.values():
                value.pop("default", None)
        return schema

    @field_validator("target_position")
    @classmethod
    def finite_target(cls, value):
        if value is None:
            return value
        import math
        if not all(math.isfinite(part) for part in value):
            raise ValueError("target_position must be finite")
        return value

    @model_validator(mode="after")
    def coherent_target(self):
        if self.mode == "navigate" and self.target_position is None and self.target_actor_id is None:
            # Navigation without a concrete target is equivalent to exploration.
            self.mode = "explore"
        if self.mode == "interact" and self.target_actor_id is None and self.target_position is None:
            # Contextual A-actions can still be handled locally; keep the mode.
            pass
        if self.mode == "dialogue" and self.choice_index is not None and self.choice_index > 2:
            raise ValueError("choice_index must be between 0 and 2")
        return self

    @classmethod
    def bootstrap(cls) -> "AgentIntent":
        return cls(
            objective="Discover the controls and make progress from the current game state.",
            summary="I am exploring continuously while I build a control and world model.",
            mode="explore",
            horizon_ms=10000,
        )
