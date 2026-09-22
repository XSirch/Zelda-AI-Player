"""Versioned wire contracts. Missing usage/cost is unknown, never a measured zero."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProviderId = Literal["codex", "openrouter", "demo"]
Effort = str  # Actual accepted values are validated against the provider catalog.
Skill = Literal[
    "move", "turn", "interact", "attack", "defend", "target", "use_item", "wait",
    "advance_dialogue", "choose_dialogue", "camera_center", "roll", "backflip",
    "sidestep", "jump_attack", "pause_toggle", "menu_move", "menu_confirm",
    "menu_cancel", "menu_assign", "continue_gameover", "play_song",
    "navigate_to", "approach_actor", "follow_actor", "talk_to_actor", "interact_with_actor",
    "equip_item", "equip_gear", "aim_at", "face_target", "shield_face",
    "fight_enemy", "explore_area", "manipulate_object",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PlayerState(StrictModel):
    position: tuple[float, float, float]
    yaw: int = Field(ge=-32768, le=32767)
    speed_xz: float = 0.0
    floor_height: float = 0.0
    wall_yaw: int = Field(default=0, ge=-32768, le=32767)
    bg_check_flags: int = Field(default=0, ge=0, le=65535)
    y_dist_to_water: float = 0.0
    health: int = Field(ge=0, le=320)  # Native OoT units: 16 = one heart.
    max_health: int = Field(ge=0, le=320)
    rupees: int = Field(ge=0, le=9999)
    magic: int = Field(default=0, ge=0, le=255)
    age: Literal["child", "adult"] = "child"


class InventoryObservation(StrictModel):
    slot: int = Field(ge=0, le=31)
    item_id: int = Field(ge=0, le=255)
    name: str = Field(min_length=1, max_length=96)
    ammo: int | None = Field(default=None, ge=-1, le=127)


class ActorObservation(StrictModel):
    actor_id: int = Field(ge=-32768, le=32767)
    name: str = Field(default="", max_length=96)
    description: str = Field(default="", max_length=200)
    category: int = Field(ge=0, le=255)
    params: int = Field(ge=-32768, le=32767)
    position: tuple[float, float, float]
    focus_position: tuple[float, float, float] | None = None
    distance: float = Field(ge=0)
    targeted: bool = False
    drawn: bool = False
    text_id: int | None = Field(default=None, ge=0, le=65535)

    @field_validator("position", "focus_position")
    @classmethod
    def finite_position(cls, value):
        if value is not None and not all(math.isfinite(v) for v in value):
            raise ValueError("Non-finite actor position")
        return value


class DialogueState(StrictModel):
    active: bool = False
    text_id: int | None = Field(default=None, ge=0, le=65535)
    text: str = Field(default="", max_length=4096)
    state: str = Field(default="none", max_length=40)
    state_code: int = Field(default=0, ge=0, le=255)
    message_mode: int = Field(default=0, ge=0, le=255)
    can_advance: bool = False
    choice_count: int = Field(default=0, ge=0, le=3)
    choice_index: int = Field(default=0, ge=0, le=2)
    choices: list[str] = Field(default_factory=list, max_length=3)
    speaker: ActorObservation | None = None


class PauseMenuState(StrictModel):
    active: bool = False
    ready: bool = False
    state: int = Field(default=0, ge=0, le=65535)
    transition_state: int = Field(default=0, ge=0, le=65535)
    page_index: int = Field(default=0, ge=0, le=4)
    cursor_special_pos: int = Field(default=0, ge=-32768, le=32767)
    cursor_point: list[int] = Field(default_factory=list, max_length=5)
    cursor_item: list[int] = Field(default_factory=list, max_length=4)
    cursor_slot: list[int] = Field(default_factory=list, max_length=4)
    named_item: int | None = Field(default=None, ge=0, le=65535)
    prompt_choice: int = Field(default=0, ge=-32768, le=32767)


class EquipmentObservation(StrictModel):
    item_id: int = Field(ge=0, le=255)
    name: str = Field(min_length=1, max_length=96)
    equipment_type: Literal["sword", "shield", "tunic", "boots"]
    value: int = Field(ge=1, le=4)
    equipped: bool = False


class ProgressState(StrictModel):
    quest_items: list[str] = Field(default_factory=list, max_length=24)
    owned_equipment: list[str] = Field(default_factory=list, max_length=12)
    equipment: list[EquipmentObservation] = Field(default_factory=list, max_length=12)
    upgrade_levels: dict[str, int] = Field(default_factory=dict, max_length=8)
    heart_pieces: int = Field(default=0, ge=0, le=15)
    skull_tokens: int = Field(default=0, ge=0, le=999)
    magic_acquired: bool = False
    double_magic: bool = False
    double_defense: bool = False
    map_index: int = Field(default=0, ge=0, le=65535)
    dungeon_items: list[str] = Field(default_factory=list, max_length=3)
    small_keys: int = Field(default=0, ge=0, le=99)


class ContextAction(StrictModel):
    code: int = Field(default=10, ge=0, le=255)
    label: str = Field(default="none", max_length=32)


class GameEvent(StrictModel):
    id: str = Field(max_length=96)
    # Keep event kinds extensible so a newer native observer cannot invalidate the whole heartbeat.
    kind: str = Field(min_length=1, max_length=64)
    detail: str = Field(default="", max_length=1000)


class GameState(StrictModel):
    protocol: Literal[1] = 1
    source: Literal["soh", "simulator"]
    instance_id: str = Field(min_length=1, max_length=80)
    seq: int = Field(ge=0)
    scene_epoch: int = Field(ge=0)
    scene: int = Field(ge=-1, le=65535)
    scene_name: str = Field(default="", max_length=96)
    room: int = Field(ge=-1, le=255)
    entrance_index: int = Field(default=-1, ge=-1, le=2147483647)
    day_time: int = Field(default=0, ge=0, le=65535)
    is_night: bool = False
    in_game: bool
    player: PlayerState | None
    camera_eye: tuple[float, float, float] | None = None
    camera_at: tuple[float, float, float] | None = None
    inventory: list[int] = Field(default_factory=list, max_length=32)
    inventory_named: list[InventoryObservation] = Field(default_factory=list, max_length=24)
    # SoH ItemEquips.buttonItems[8]: B + 3 C-buttons + 4 D-pad slots.
    # Keep accepting older four-slot payloads without truncating native telemetry.
    equipped: list[int] = Field(default_factory=list, max_length=8)
    message_id: int | None = None  # Deprecated compatibility mirror; prefer dialogue.text_id.
    dialogue: DialogueState = Field(default_factory=DialogueState)
    progress: ProgressState = Field(default_factory=ProgressState)
    context_action: ContextAction = Field(default_factory=ContextAction)
    pause_menu: PauseMenuState = Field(default_factory=PauseMenuState)
    game_over_state: int = Field(default=0, ge=0, le=65535)
    ocarina_mode: int = Field(default=0, ge=0, le=65535)
    ocarina_action: int = Field(default=0, ge=0, le=65535)
    last_played_song: int = Field(default=0, ge=0, le=65535)
    target_actor: ActorObservation | None = None
    nearby_actors: list[ActorObservation] = Field(default_factory=list, max_length=24)
    cutscene_active: bool = False
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
    direction: Literal["forward", "back", "left", "right", "up", "down"] | None
    duration_ms: int = Field(ge=50, le=12000)
    strength: float = Field(ge=0, le=1)
    slot: Literal["left", "down", "right"] | None
    choice_index: int | None = Field(ge=0, le=2)
    song: Literal["minuet", "bolero", "serenade", "requiem", "nocturne", "prelude",
        "sarias", "eponas", "lullaby", "suns", "time", "storms"] | None
    target_actor_id: int | None = Field(ge=-32768, le=32767)
    target_actor_params: int | None = Field(ge=-32768, le=32767)
    target_position: list[float] | None = Field(min_length=3, max_length=3)
    stop_distance: float | None = Field(ge=12, le=600)
    item_id: int | None = Field(ge=0, le=255)

    @field_validator("target_position")
    @classmethod
    def finite_target_position(cls, value):
        if value is not None and not all(math.isfinite(v) for v in value):
            raise ValueError("Non-finite target position")
        return value


class Decision(StrictModel):
    goal: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=280)
    skill: Skill
    args: SkillArgs
    memory_note: str | None = Field(max_length=400)

    @model_validator(mode="after")
    def skill_arguments(self):
        if self.skill in {"move", "turn", "sidestep", "menu_move"} and self.args.direction is None:
            raise ValueError(f"{self.skill} requires direction")
        if self.skill == "move" and self.args.direction not in {"forward", "back", "left", "right"}:
            raise ValueError("move requires forward, back, left or right")
        if self.skill in {"turn", "sidestep"} and self.args.direction not in {"left", "right"}:
            raise ValueError(f"{self.skill} requires left or right")
        if self.skill == "menu_move" and self.args.direction not in {"up", "down", "left", "right"}:
            raise ValueError("menu_move requires up, down, left or right")
        if self.skill == "use_item" and self.args.slot is None:
            raise ValueError("use_item requires an equipped C-button slot")
        if self.skill == "choose_dialogue" and self.args.choice_index is None:
            raise ValueError("choose_dialogue requires choice_index")
        if self.skill == "menu_assign" and self.args.slot is None:
            raise ValueError("menu_assign requires a C-button slot")
        if self.skill not in {"navigate_to", "approach_actor", "follow_actor", "talk_to_actor", "interact_with_actor",
                              "equip_item", "equip_gear", "aim_at", "face_target", "shield_face",
                              "fight_enemy", "explore_area", "manipulate_object"} and self.args.duration_ms > 2000:
            raise ValueError("primitive skills are limited to 2000 ms")
        if self.skill == "play_song" and self.args.song is None:
            raise ValueError("play_song requires song")
        if self.skill == "navigate_to" and self.args.target_position is None:
            raise ValueError("navigate_to requires target_position")
        if self.skill in {"approach_actor", "follow_actor", "talk_to_actor", "interact_with_actor", "fight_enemy",
                           "manipulate_object"} and self.args.target_actor_id is None:
            raise ValueError(f"{self.skill} requires target_actor_id")
        if self.skill == "equip_item" and (self.args.item_id is None or self.args.slot is None):
            raise ValueError("equip_item requires item_id and C-button slot")
        if self.skill == "equip_gear" and self.args.item_id is None:
            raise ValueError("equip_gear requires item_id")
        if self.skill == "aim_at" and (self.args.slot is None or
                (self.args.target_actor_id is None and self.args.target_position is None)):
            raise ValueError("aim_at requires a C-button slot and actor or position target")
        if self.skill in {"face_target", "shield_face"} and (
                self.args.target_actor_id is None and self.args.target_position is None):
            raise ValueError(f"{self.skill} requires actor or position target")
        if self.skill == "manipulate_object" and self.args.direction not in {"forward", "back"}:
            raise ValueError("manipulate_object requires forward or back direction")
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
    goal: str = Field(default="Complete Ocarina of Time autonomously and defeat final Ganon.", min_length=1, max_length=400)
    memory_mode: Literal["isolated", "adaptive"] = "adaptive"
    max_calls: int = Field(default=5000, ge=1, le=10000)
    max_tokens: int = Field(default=5000000, ge=1000, le=10000000)
    max_cost_usd: float = Field(default=2, gt=0, le=1000)
    max_output_tokens: int = Field(default=2048, ge=256, le=16384)
    max_runtime_s: int = Field(default=43200, ge=10, le=86400)
    checkpoint_label: str = Field(default="manual", max_length=120)


class SwitchConfig(StrictModel):
    provider: ProviderId
    model: str = Field(min_length=1, max_length=200)
    effort: Effort | None = None
