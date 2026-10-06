"""Versioned wire contracts. Missing usage/cost is unknown, never a measured zero."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProviderId = Literal["codex", "openrouter", "demo"]
RunMode = Literal["train", "evaluation"]
Effort = str  # Actual accepted values are validated against the provider catalog.
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class PlayerState(StrictModel):
    position: tuple[float, float, float]
    yaw: int = Field(ge=-32768, le=32767)
    speed_xz: float = 0.0
    floor_height: float = 0.0
    floor_exit_index: int = Field(default=0, ge=0, le=31)
    wall_yaw: int = Field(default=0, ge=-32768, le=32767)
    bg_check_flags: int = Field(default=0, ge=0, le=65535)
    y_dist_to_water: float = 0.0
    wall_flags: int = Field(default=0, ge=0, le=65535)
    state_flags_1: int = Field(default=0, ge=0, le=4294967295)
    state_flags_2: int = Field(default=0, ge=0, le=4294967295)
    z_target_active_timer: int = Field(default=0, ge=0, le=2147483647)
    melee_weapon_animation: int = Field(default=0, ge=-128, le=127)
    melee_weapon_state: int = Field(default=0, ge=-128, le=127)
    invincibility_timer: int = Field(default=0, ge=-128, le=127)
    control_stick_direction: int = Field(default=-1, ge=-1, le=3)
    hop_direction: int | None = Field(default=None, ge=0, le=3)
    climbing_ladder: bool = False
    hanging_ledge: bool = False
    climbing_ledge: bool = False
    can_climb: bool = False
    can_down: bool = False
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


class NavigationProbe(StrictModel):
    direction: Literal["forward", "forward_right", "right", "back_right",
        "back", "back_left", "left", "forward_left"]
    distance: float = Field(gt=0, le=500)
    floor_found: bool = False
    floor_y: float | None = None
    delta_y: float | None = None
    floor_type: int | None = Field(default=None, ge=0, le=255)
    wall_hit: bool = False
    wall_distance: float | None = Field(default=None, ge=0, le=500)
    wall_flags: int = Field(default=0, ge=0, le=65535)


class TraversalAffordanceObservation(StrictModel):
    """Local collision-derived vertical route candidate; not hidden map knowledge."""
    kind: Literal["stairs_or_slope_up", "stairs_or_slope_down", "ledge_down", "ledge_up",
                  "ladder_up", "ladder_down", "climbable_wall_up"]
    direction: Literal["up", "down"]
    approach_position: tuple[float, float, float]
    target_position: tuple[float, float, float]
    distance: float = Field(ge=0, le=500)
    height_delta: float = Field(ge=-500, le=500)
    wall_flags: int = Field(default=0, ge=0, le=65535)

    @field_validator("approach_position", "target_position")
    @classmethod
    def finite_positions(cls, value):
        if not all(math.isfinite(v) for v in value):
            raise ValueError("Non-finite traversal affordance position")
        return value


class SceneExitObservation(StrictModel):
    """Currently observed collision surface that triggers a scene/entrance transition."""
    exit_index: int = Field(ge=1, le=31)
    entrance_index: int = Field(ge=-32768, le=65535)
    position: tuple[float, float, float]
    samples: int = Field(ge=1, le=10000)
    direct_reachable: bool = False

    @field_validator("position")
    @classmethod
    def finite_position(cls, value):
        if not all(math.isfinite(v) for v in value):
            raise ValueError("Non-finite scene-exit position")
        return value


class NavigationMeshSnapshot(StrictModel):
    """Compact local walkable graph produced from SoH collision on full snapshots."""
    refinement_request_id: int = Field(default=0, ge=0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    step: float = Field(default=0.0, ge=0.0, le=500.0)
    half_extent: int = Field(default=0, ge=0, le=8)
    cells: list[tuple[int, int, float, int]] = Field(default_factory=list, max_length=289)
    # QA counter over this snapshot's bounded local rays; never a route or
    # solution flag. Old adapters retain the explicit zero default.
    backface_rejections: int = Field(default=0, ge=0, le=8192)
    lower_band_rejections: int = Field(default=0, ge=0, le=8192)
    refined_component_cells: int = Field(default=0, ge=0, le=4)
    # Native query time, including a coarse retry when refined. Unknown on old adapters.
    query_us: int | None = Field(default=None, ge=0, le=60_000_000)

    @property
    def available(self) -> bool:
        return self.step > 0 and bool(self.cells)

    @field_validator("origin")
    @classmethod
    def finite_origin(cls, value):
        if not all(math.isfinite(v) for v in value):
            raise ValueError("Non-finite navmesh origin")
        return value

    @model_validator(mode="after")
    def valid_cells(self):
        if self.cells and self.step <= 0:
            raise ValueError("NavMesh cells require a positive step")
        seen: set[tuple[int, int]] = set()
        for gx, gz, floor_y, links in self.cells:
            if abs(gx) > self.half_extent or abs(gz) > self.half_extent:
                raise ValueError("NavMesh cell outside declared extent")
            if not math.isfinite(floor_y):
                raise ValueError("Non-finite navmesh floor")
            if links < 0 or links > 255:
                raise ValueError("Invalid navmesh link mask")
            key = (gx, gz)
            if key in seen:
                raise ValueError("Duplicate navmesh cell")
            seen.add(key)
        return self


class ActorObservation(StrictModel):
    actor_uid: str | None = Field(default=None, max_length=96)
    yaw: int | None = Field(default=None, ge=-32768, le=32767)
    actor_id: int = Field(ge=-32768, le=32767)
    name: str = Field(default="", max_length=96)
    description: str = Field(default="", max_length=200)
    category: int = Field(ge=0, le=255)
    category_name: str = Field(default="", max_length=32)
    room: int = Field(default=-1, ge=-1, le=255)
    params: int = Field(ge=-32768, le=32767)
    position: tuple[float, float, float]
    focus_position: tuple[float, float, float] | None = None
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    speed_xz: float = 0.0
    xz_distance: float = Field(default=0.0, ge=0)
    collision_health_hint: int | None = Field(default=None, ge=0, le=255)
    freeze_timer: int = Field(default=0, ge=0, le=65535)
    color_filter_timer: int = Field(default=0, ge=0, le=255)
    actor_flags: int = Field(default=0, ge=0, le=4294967295)
    distance: float = Field(ge=0)
    targeted: bool = False
    drawn: bool = False
    container_lid_rotation_z: int | None = Field(default=None, ge=-32768, le=32767)
    container_lid_pose: Literal["open", "closed", "unknown"] = "unknown"
    container_lid_closed_rotation_z: int | None = Field(default=None, ge=-32768, le=32767)
    container_lid_open_rotation_z: int | None = Field(default=None, ge=-32768, le=32767)
    text_id: int | None = Field(default=None, ge=0, le=65535)

    @field_validator("position", "focus_position", "velocity")
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
    text_visible: bool | None = None
    visible_bytes: int | None = Field(default=None, ge=0, le=4096)
    can_advance: bool = False
    choice_count: int = Field(default=0, ge=0, le=3)
    choice_index: int = Field(default=0, ge=0, le=2)
    choices: list[str] = Field(default_factory=list, max_length=3)
    speaker: ActorObservation | None = None

    @model_validator(mode="after")
    def undisplayed_buffer_is_not_observation(self):
        # These pinned native phases do not display a decoded current page.
        # Older adapters can retain a previous buffer here. Preserve modal
        # ownership/ID, but never present those bytes as text or a choice.
        if self.message_mode in {1, 2, 3, 4, 5, 54, 55}:
            self.text_visible = False
        if self.text_visible is False:
            self.text = ""
            self.choices = []
            self.choice_count = 0
            self.can_advance = False
        return self


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
    story_flags: dict[str, bool] = Field(default_factory=dict, max_length=32)


class AutosaveState(StrictModel):
    pending: bool = False
    target_scene: int = Field(default=-1, ge=-1, le=65535)
    last_scene: int = Field(default=-1, ge=-1, le=65535)
    count: int = Field(default=0, ge=0)
    last_saved_at_ms: int = Field(default=0, ge=0)
    in_flight: bool = False
    requested_count: int = Field(default=0, ge=0)
    pending_reasons: list[Literal["scene", "resource_gain", "item_gain", "chest_opened"]] = Field(default_factory=list, max_length=4)
    last_reasons: list[Literal["scene", "resource_gain", "item_gain", "chest_opened"]] = Field(default_factory=list, max_length=4)
    last_rupees: int | None = Field(default=None, ge=0, le=9999)


class ContextAction(StrictModel):
    code: int = Field(default=10, ge=0, le=255)
    label: str = Field(default="none", max_length=32)


class InputReceipt(StrictModel):
    seq: int = Field(ge=1)
    owner_epoch: int = Field(ge=0)
    status: Literal["accepted", "consumed", "completed", "superseded", "cancelled", "rejected"]
    first_tick: int = Field(ge=0)
    last_tick: int = Field(ge=0)
    pressed: int = Field(ge=0, le=65535)
    released: int = Field(ge=0, le=65535)
    apply_latency_ms: float | None = Field(default=None, ge=0)
    client_to_consume_ms: float | None = Field(default=None, ge=0)
    reason: str = Field(default="", max_length=80)


class GameEvent(StrictModel):
    actor_uid: str | None = Field(default=None, max_length=96)
    id: str = Field(max_length=96)
    # Keep event kinds extensible so a newer native observer cannot invalidate the whole heartbeat.
    kind: str = Field(min_length=1, max_length=64)
    detail: str = Field(default="", max_length=1000)


class StartupState(StrictModel):
    phase: int = Field(default=0, ge=0, le=4)  # Unknown/title/file-select/confirm/busy.
    cursor: int = Field(default=-1, ge=-1, le=16)
    selected_slot: int = Field(default=-1, ge=-1, le=2)
    existing_slots: list[bool] = Field(default_factory=lambda: [False] * 3, min_length=3, max_length=3)


class GameState(StrictModel):
    protocol: Literal[1, 3] = 1
    kind: Literal["full"] = "full"
    capabilities: list[str] = Field(default_factory=list, max_length=32)
    bridge_build: str = Field(default="legacy", max_length=80)
    full_seq: int = Field(default=0, ge=0)
    capture_tick: int = Field(default=0, ge=0)
    context_epoch: int = Field(default=0, ge=0)
    input_tick: int = Field(default=0, ge=0)
    owner_epoch: int = Field(default=0, ge=0)
    last_received_seq: int = Field(default=0, ge=0)
    last_applied_command_seq: int = Field(default=0, ge=0)
    input_receipts: list[InputReceipt] = Field(default_factory=list, max_length=64)
    event_floor: int = Field(default=0, ge=0)
    event_seq: int = Field(default=0, ge=0)
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
    startup: StartupState = Field(default_factory=StartupState)
    player: PlayerState | None
    camera_eye: tuple[float, float, float] | None = None
    camera_at: tuple[float, float, float] | None = None
    camera_input_yaw: int | None = Field(default=None, ge=-32768, le=32767)
    mirrored_world: bool = False
    inventory: list[int] = Field(default_factory=list, max_length=32)
    inventory_named: list[InventoryObservation] = Field(default_factory=list, max_length=24)
    # SoH ItemEquips.buttonItems[8]: B + 3 C-buttons + 4 D-pad slots.
    # Keep accepting older four-slot payloads without truncating native telemetry.
    equipped: list[int] = Field(default_factory=list, max_length=8)
    message_id: int | None = None  # Deprecated compatibility mirror; prefer dialogue.text_id.
    dialogue: DialogueState = Field(default_factory=DialogueState)
    progress: ProgressState = Field(default_factory=ProgressState)
    context_action: ContextAction = Field(default_factory=ContextAction)
    context_actor: ActorObservation | None = None
    pause_menu: PauseMenuState = Field(default_factory=PauseMenuState)
    game_over_state: int = Field(default=0, ge=0, le=65535)
    ocarina_mode: int = Field(default=0, ge=0, le=65535)
    ocarina_action: int = Field(default=0, ge=0, le=65535)
    last_played_song: int = Field(default=0, ge=0, le=65535)
    target_actor: ActorObservation | None = None
    target_candidate: ActorObservation | None = None
    nearby_actors: list[ActorObservation] = Field(default_factory=list, max_length=24)
    room_actors: list[ActorObservation] = Field(default_factory=list, max_length=64)
    room_actor_count: int = Field(default=0, ge=0, le=4096)
    room_actors_truncated: bool = False
    navigation_probes: list[NavigationProbe] = Field(default_factory=list, max_length=16)
    traversal_affordances: list[TraversalAffordanceObservation] = Field(default_factory=list, max_length=24)
    scene_exits: list[SceneExitObservation] = Field(default_factory=list, max_length=31)
    navmesh: NavigationMeshSnapshot = Field(default_factory=NavigationMeshSnapshot)
    autosave: AutosaveState = Field(default_factory=AutosaveState)
    cutscene_active: bool = False
    paused: bool = False
    events: list[GameEvent] = Field(default_factory=list, max_length=64)
    last_command_seq: int = 0
    upstream_revision: str = Field(default="unknown", max_length=64)

    @field_validator("camera_eye", "camera_at")
    @classmethod
    def finite_vector(cls, value):
        if value is not None and not all(math.isfinite(v) for v in value):
            raise ValueError("Non-finite position")
        return value


class RealtimeState(StrictModel):
    protocol: Literal[3]
    kind: Literal["fast"]
    source: Literal["soh"]
    instance_id: str = Field(min_length=1, max_length=80)
    seq: int = Field(ge=1)
    full_seq: int = Field(ge=1)
    capture_tick: int = Field(ge=0)
    scene_epoch: int = Field(ge=0)
    context_epoch: int = Field(ge=0)
    scene: int = Field(ge=-1, le=65535)
    room: int = Field(ge=-1, le=255)
    in_game: bool
    startup: StartupState = Field(default_factory=StartupState)
    player: PlayerState | None
    camera_eye: tuple[float, float, float] | None
    camera_at: tuple[float, float, float] | None
    camera_input_yaw: int | None = Field(default=None, ge=-32768, le=32767)
    mirrored_world: bool = False
    paused: bool
    cutscene_active: bool
    game_over_state: int = Field(ge=0, le=65535)
    ocarina_mode: int = Field(ge=0, le=65535)
    dialogue: DialogueState
    context_action: ContextAction
    context_actor: ActorObservation | None
    target_actor: ActorObservation | None
    target_candidate: ActorObservation | None
    room_actors: list[ActorObservation] = Field(max_length=64)
    room_actor_count: int = Field(ge=0, le=4096)
    room_actors_truncated: bool
    navigation_probes: list[NavigationProbe] = Field(max_length=16)
    input_tick: int = Field(ge=0)
    owner_epoch: int = Field(ge=0)
    last_received_seq: int = Field(ge=0)
    last_command_seq: int = Field(ge=0)
    last_applied_command_seq: int = Field(ge=0)
    input_receipts: list[InputReceipt] = Field(max_length=64)
    events: list[GameEvent] = Field(max_length=64)
    event_floor: int = Field(ge=0)
    event_seq: int = Field(ge=0)


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
    run_mode: RunMode = "train"
    champion_id: str | None = Field(default=None, max_length=80)
    goal: str = Field(default="Complete Ocarina of Time autonomously and defeat final Ganon.", min_length=1, max_length=400)
    max_calls: int = Field(default=5000, ge=0, le=10000, description="Run call limit; 0 disables this limit.")
    max_tokens: int = Field(default=5000000, ge=0, le=10000000, description="Run token limit; 0 disables this limit.")
    max_cost_usd: float = Field(default=2, ge=0, le=1000, description="Run USD limit; 0 disables this limit.")
    max_output_tokens: int = Field(default=2048, ge=256, le=16384)
    max_runtime_s: int = Field(default=43200, ge=0, le=86400, description="Run duration limit; 0 disables this limit.")


class SwitchConfig(StrictModel):
    provider: ProviderId
    model: str = Field(min_length=1, max_length=200)
    effort: Effort | None = None
