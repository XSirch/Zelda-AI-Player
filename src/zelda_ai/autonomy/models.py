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

CompletionKind = Literal[
    "equipment",
    "inventory_item",
    "quest_item",
    "story_flag",
    "scene",
    "scene_room",
    "leave_scene_room",
    "rupees_at_least",
    "heart_pieces_at_least",
    "skull_tokens_at_least",
    "small_keys_at_least",
    "magic_acquired",
    "dialogue_actor",
    "event_kind",
    "game_completed",
    "manual",
]


class ObjectiveCompletion(BaseModel):
    """Machine-verifiable completion contract for one sticky objective."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    kind: CompletionKind
    name: str | None = Field(default=None, max_length=120)
    item_id: int | None = Field(default=None, ge=0, le=255)
    flag: str | None = Field(default=None, max_length=96)
    scene: int | None = Field(default=None, ge=-1, le=65535)
    room: int | None = Field(default=None, ge=-1, le=255)
    threshold: int | None = Field(default=None, ge=0, le=99999)
    actor_id: int | None = Field(default=None, ge=-32768, le=32767)
    actor_name: str | None = Field(default=None, max_length=96)
    event_kind: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def coherent_completion(self):
        if self.kind == "equipment" and not self.name and self.item_id is None:
            raise ValueError("equipment completion needs name or item_id")
        if self.kind == "inventory_item" and not self.name and self.item_id is None:
            raise ValueError("inventory_item completion needs name or item_id")
        if self.kind == "quest_item" and not self.name:
            raise ValueError("quest_item completion needs name")
        if self.kind == "story_flag" and not self.flag:
            raise ValueError("story_flag completion needs flag")
        if self.kind == "scene" and self.scene is None and not self.name:
            raise ValueError("scene completion needs scene or name")
        if self.kind == "scene_room" and self.room is None:
            raise ValueError("scene_room completion needs room")
        if self.kind.endswith("_at_least") and self.threshold is None:
            raise ValueError(f"{self.kind} completion needs threshold")
        if (
            self.kind == "dialogue_actor"
            and self.actor_id is None
            and not self.actor_name
        ):
            raise ValueError("dialogue_actor completion needs actor_id or actor_name")
        if self.kind == "event_kind" and not self.event_kind:
            raise ValueError("event_kind completion needs event_kind")
        return self


class AgentIntent(BaseModel):
    """High-level intent only.

    The model never chooses a controller skill or a raw button.  The local
    controller owns moment-to-moment input and may keep acting while a new
    AgentIntent is being inferred.
    """

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    objective: str = Field(min_length=1, max_length=240)
    completion: ObjectiveCompletion = Field(
        default_factory=lambda: ObjectiveCompletion(kind="manual")
    )
    summary: str = Field(min_length=1, max_length=320)
    mode: IntentMode = "explore"
    target_actor_id: int | None = Field(default=None, ge=-32768, le=32767)
    target_actor_params: int | None = Field(default=None, ge=-32768, le=32767)
    target_actor_uid: str | None = Field(default=None, max_length=96)
    target_position: tuple[float, float, float] | None = None
    target_item_id: int | None = Field(default=None, ge=0, le=255)
    direction: IntentDirection | None = None
    choice_index: int | None = Field(default=None, ge=0, le=2)
    horizon_ms: int = Field(default=15000, ge=2000, le=60000)

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        schema = super().model_json_schema(*args, **kwargs)
        properties = schema.get("properties", {})
        def strictify(node):
            if isinstance(node, dict):
                properties = node.get("properties")
                if isinstance(properties, dict) and properties:
                    node["required"] = list(properties)
                    for value in properties.values():
                        if isinstance(value, dict):
                            value.pop("default", None)
                for value in node.values():
                    strictify(value)
            elif isinstance(node, list):
                for value in node:
                    strictify(value)

        if properties:
            # OpenAI/OpenRouter strict structured output requires every property
            # to be present, including nested completion-contract fields.
            strictify(schema)

            # Pydantic represents a fixed-length tuple with prefixItems. Codex /
            # OpenAI structured output expects a regular array schema with items.
            # Runtime validation still enforces the Python 3-tuple below.
            properties["target_position"] = {
                "anyOf": [
                    {
                        "type": "array",
                        "items": {"type": "number"},
                        "minItems": 3,
                        "maxItems": 3,
                    },
                    {"type": "null"},
                ],
                "title": "Target Position",
            }
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
        has_actor_target = (
            self.target_actor_uid is not None or self.target_actor_id is not None
        )
        if self.mode == "navigate" and self.target_position is None and not has_actor_target:
            # Navigation without a concrete target is equivalent to exploration.
            self.mode = "explore"
        if self.mode == "interact" and not has_actor_target and self.target_position is None:
            # Contextual A-actions can still be handled locally; keep the mode.
            pass
        if self.mode == "dialogue" and self.choice_index is not None and self.choice_index > 2:
            raise ValueError("choice_index must be between 0 and 2")
        return self

    @classmethod
    def bootstrap(cls) -> "AgentIntent":
        return cls(
            objective="Discover the controls and make progress from the current game state.",
            completion=ObjectiveCompletion(kind="manual"),
            summary="Discover the controls and make progress from the current game state.",
            mode="explore",
            horizon_ms=10000,
        )
