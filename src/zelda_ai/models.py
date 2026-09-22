"""Versioned wire contracts. Missing usage/cost is unknown, never a measured zero."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProviderId = Literal["codex", "openrouter", "demo"]
Effort = str  # Actual accepted values are validated against the provider catalog.
Skill = Literal["move", "turn", "interact", "attack", "defend", "target", "use_item", "wait"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PlayerState(StrictModel):
    position: tuple[float, float, float]
    yaw: int = Field(ge=-32768, le=32767)
    health: int = Field(ge=0, le=320)  # Native OoT units: 16 = one heart.
    max_health: int = Field(ge=0, le=320)
    rupees: int = Field(ge=0, le=9999)
    magic: int = Field(default=0, ge=0, le=255)
    age: Literal["child", "adult"] = "child"


class GameEvent(StrictModel):
    id: str = Field(max_length=96)
    kind: Literal["scene_changed", "item_received", "enemy_defeated", "boss_defeated", "health_changed"]
    detail: str = Field(default="", max_length=200)


class GameState(StrictModel):
    protocol: Literal[1] = 1
    source: Literal["soh", "simulator"]
    instance_id: str = Field(min_length=1, max_length=80)
    seq: int = Field(ge=0)
    scene_epoch: int = Field(ge=0)
    scene: int = Field(ge=-1, le=65535)
    room: int = Field(ge=-1, le=255)
    in_game: bool
    player: PlayerState | None
    camera_eye: tuple[float, float, float] | None = None
    camera_at: tuple[float, float, float] | None = None
    inventory: list[int] = Field(default_factory=list, max_length=32)
    # SoH ItemEquips.buttonItems[8]: B + 3 C-buttons + 4 D-pad slots.
    # Keep accepting older four-slot payloads without truncating native telemetry.
    equipped: list[int] = Field(default_factory=list, max_length=8)
    message_id: int | None = None
    paused: bool = False
    events: list[GameEvent] = Field(default_factory=list, max_length=16)
    last_command_seq: int = 0
    upstream_revision: str = Field(default="unknown", max_length=64)

    @field_validator("camera_eye", "camera_at")
    @classmethod
    def finite_vector(cls, value):
        if value is not None and not all(math.isfinite(v) for v in value):
            raise ValueError("Non-finite position")
        return value


class SkillArgs(StrictModel):
    # Required nullable keys keep the schema compatible with strict JSON outputs.
    direction: Literal["forward", "back", "left", "right"] | None
    duration_ms: int = Field(ge=50, le=2000)
    strength: float = Field(ge=0, le=1)
    slot: Literal["left", "down", "right"] | None


class Decision(StrictModel):
    goal: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=280)
    skill: Skill
    args: SkillArgs
    memory_note: str | None = Field(max_length=400)

    @model_validator(mode="after")
    def skill_arguments(self):
        if self.skill in {"move", "turn"} and self.args.direction is None:
            raise ValueError(f"{self.skill} requires direction")
        if self.skill == "turn" and self.args.direction not in {"left", "right"}:
            raise ValueError("turn requires left or right")
        if self.skill == "use_item" and self.args.slot is None:
            raise ValueError("use_item requires an equipped C-button slot")
        return self


class Usage(StrictModel):
    input_tokens: int | None = Field(default=None, ge=0)
    cached_input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    reasoning_output_tokens: int | None = Field(default=None, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    generation_id: str | None = None
    actual_model: str | None = None
    upstream_provider: str | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.input_tokens is None or self.output_tokens is None:
            return None
        # Cached input and reasoning output are subsets, NOT additional tokens.
        return self.input_tokens + self.output_tokens


class ModelInfo(StrictModel):
    id: str
    name: str
    efforts: list[str] = Field(default_factory=list)
    effort_source: str = "unsupported"
    default_effort: str | None = None
    structured_output: bool = True
    pricing: dict[str, str] = Field(default_factory=dict)


class RunConfig(StrictModel):
    provider: ProviderId
    model: str = Field(min_length=1, max_length=200)
    effort: Effort | None = None
    goal: str = Field(default="Explore o ambiente e avance no jogo.", min_length=1, max_length=400)
    memory_mode: Literal["isolated", "adaptive"] = "adaptive"
    max_calls: int = Field(default=100, ge=1, le=10000)
    max_tokens: int = Field(default=100000, ge=1000, le=10000000)
    max_cost_usd: float = Field(default=2, gt=0, le=1000)
    max_output_tokens: int = Field(default=2048, ge=256, le=16384)
    max_runtime_s: int = Field(default=3600, ge=10, le=86400)
    checkpoint_label: str = Field(default="manual", max_length=120)


class SwitchConfig(StrictModel):
    provider: ProviderId
    model: str = Field(min_length=1, max_length=200)
    effort: Effort | None = None
