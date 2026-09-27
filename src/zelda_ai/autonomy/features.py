from __future__ import annotations

import math

from ..models import ActorObservation, GameState
from .models import AgentIntent

BASE_FEATURE_DIM = 208
STACK_FRAMES = 4
FEATURE_DIM = BASE_FEATURE_DIM * STACK_FRAMES
BUTTON_NAMES = ("A", "B", "Z", "START", "R", "C_UP", "C_LEFT", "C_DOWN", "C_RIGHT")
MODE_NAMES = ("explore", "navigate", "interact", "combat", "dialogue", "menu", "observe")
PROBE_NAMES = (
    "forward", "forward_right", "right", "back_right",
    "back", "back_left", "left", "forward_left",
)


def _squash(value: float, scale: float) -> float:
    return math.tanh(float(value) / scale)


def _relative(player, point):
    if player is None or point is None:
        return (0.0, 0.0, 0.0, 0.0)
    dx = point[0] - player.position[0]
    dy = point[1] - player.position[1]
    dz = point[2] - player.position[2]
    return (
        _squash(dx, 250.0),
        _squash(dy, 120.0),
        _squash(dz, 250.0),
        _squash(math.hypot(dx, dz), 400.0),
    )


def _target_actor(game: GameState, intent: AgentIntent) -> ActorObservation | None:
    if intent.target_actor_id is None and not intent.target_actor_uid:
        return None
    candidates = list(game.room_actors)
    if game.target_actor is not None:
        candidates.insert(0, game.target_actor)
    matches = []
    for actor in candidates:
        if intent.target_actor_uid and actor.actor_uid != intent.target_actor_uid:
            continue
        if intent.target_actor_id is not None and actor.actor_id != intent.target_actor_id:
            continue
        if intent.target_actor_params is not None and actor.params != intent.target_actor_params:
            continue
        matches.append(actor)
    return min(matches, key=lambda row: row.distance) if matches else None


def target_point(game: GameState, intent: AgentIntent):
    actor = _target_actor(game, intent)
    if actor is not None:
        return actor.focus_position or actor.position
    return intent.target_position


def encode_state(
    game: GameState,
    intent: AgentIntent,
    *,
    last_stick: tuple[float, float] = (0.0, 0.0),
    last_buttons: tuple[float, ...] | list[float] = (),
) -> list[float]:
    f: list[float] = []
    player = game.player

    # Absolute scene-local context lets the persistent network learn specific
    # routes instead of only generic obstacle avoidance.
    day_angle = game.day_time * (2.0 * math.pi / 65536.0)
    f.extend([
        _squash(game.scene, 128.0),
        _squash(game.room, 16.0),
        _squash(game.entrance_index, 1024.0),
        math.sin(day_angle),
        math.cos(day_angle),
        1.0 if game.is_night else 0.0,
        _squash(player.position[0], 2000.0) if player else 0.0,
        _squash(player.position[1], 1000.0) if player else 0.0,
        _squash(player.position[2], 2000.0) if player else 0.0,
    ])

    # Intent is context, not an action command.  The policy still emits raw controls.
    f.extend(1.0 if intent.mode == name else 0.0 for name in MODE_NAMES)
    f.extend([
        _squash(intent.horizon_ms, 5000.0),
        1.0 if intent.direction == "up" else -1.0 if intent.direction == "down" else 0.0,
        1.0 if intent.direction == "forward" else -1.0 if intent.direction == "back" else 0.0,
        1.0 if intent.direction == "right" else -1.0 if intent.direction == "left" else 0.0,
    ])

    if player is None:
        f.extend([0.0] * 21)
    else:
        yaw = player.yaw * math.pi / 32768.0
        f.extend([
            player.health / max(1.0, float(player.max_health)),
            player.magic / 255.0,
            _squash(player.rupees, 200.0),
            _squash(player.speed_xz, 8.0),
            math.sin(yaw),
            math.cos(yaw),
            _squash(player.y_dist_to_water, 100.0),
            _squash(player.wall_yaw, 16384.0),
            1.0 if player.climbing_ladder else 0.0,
            1.0 if player.hanging_ledge else 0.0,
            1.0 if player.climbing_ledge else 0.0,
            1.0 if player.can_climb else 0.0,
            1.0 if player.can_down else 0.0,
            1.0 if player.z_target_active_timer > 0 else 0.0,
            _squash(player.melee_weapon_animation, 12.0),
            _squash(player.melee_weapon_state, 4.0),
            _squash(player.invincibility_timer, 20.0),
            float(player.control_stick_direction) / 3.0 if player.control_stick_direction >= 0 else -1.0,
            1.0 if player.hop_direction is not None else 0.0,
            _squash(player.floor_height - player.position[1], 50.0),
            1.0 if player.age == "adult" else 0.0,
        ])

    f.extend([
        1.0 if game.in_game else 0.0,
        1.0 if game.paused else 0.0,
        1.0 if game.cutscene_active else 0.0,
        1.0 if game.game_over_state else 0.0,
        1.0 if game.dialogue.active else 0.0,
        1.0 if game.dialogue.can_advance else 0.0,
        game.dialogue.choice_count / 3.0,
        game.dialogue.choice_index / 2.0,
        1.0 if game.pause_menu.active else 0.0,
        1.0 if game.pause_menu.ready else 0.0,
        1.0 if game.context_action.label != "none" else 0.0,
        _squash(game.context_action.code, 32.0),
        _squash(game.ocarina_mode, 8.0),
        _squash(game.ocarina_action, 8.0),
    ])

    target = target_point(game, intent)
    f.extend(_relative(player, target))
    f.append(1.0 if target is not None else 0.0)

    # Nearest actors are deliberately represented by raw observed properties.
    # No actor ID -> semantic action mapping is encoded here.
    actors = sorted(game.room_actors, key=lambda row: row.distance)[:6]
    for index in range(6):
        if index >= len(actors):
            f.extend([0.0] * 13)
            continue
        actor = actors[index]
        f.extend(_relative(player, actor.focus_position or actor.position))
        f.extend([
            _squash(actor.category, 12.0),
            _squash(actor.actor_id, 600.0),
            _squash(actor.speed_xz, 8.0),
            1.0 if actor.drawn else 0.0,
            1.0 if actor.targeted else 0.0,
            1.0 if actor.text_id is not None else 0.0,
            _squash(actor.velocity[0], 8.0),
            _squash(actor.velocity[2], 8.0),
            (actor.collision_health_hint / 255.0) if actor.collision_health_hint is not None else -1.0,
        ])

    probe_by_name = {}
    for probe in game.navigation_probes:
        current = probe_by_name.get(probe.direction)
        if current is None or abs(probe.distance - 70.0) < abs(current.distance - 70.0):
            probe_by_name[probe.direction] = probe
    for name in PROBE_NAMES:
        probe = probe_by_name.get(name)
        if probe is None:
            f.extend([0.0] * 5)
        else:
            f.extend([
                1.0 if probe.floor_found else 0.0,
                _squash(probe.delta_y or 0.0, 60.0),
                1.0 if probe.wall_hit else 0.0,
                _squash(probe.wall_distance or 0.0, 100.0),
                _squash(probe.wall_flags, 1024.0),
            ])

    # Progress/inventory is intentionally coarse.  It lets the policy notice that
    # its behaviour changed the game without hard-coding how any item is obtained.
    f.extend([
        _squash(len(game.inventory_named), 12.0),
        _squash(len(game.progress.quest_items), 12.0),
        _squash(len(game.progress.owned_equipment), 8.0),
        _squash(game.progress.heart_pieces, 4.0),
        _squash(game.progress.skull_tokens, 20.0),
        _squash(game.progress.small_keys, 5.0),
        1.0 if game.progress.magic_acquired else 0.0,
        1.0 if game.progress.double_magic else 0.0,
        1.0 if game.progress.double_defense else 0.0,
        _squash(len(game.scene_exits), 4.0),
        _squash(len(game.traversal_affordances), 8.0),
    ])

    f.extend([max(-1.0, min(1.0, float(last_stick[0]))),
              max(-1.0, min(1.0, float(last_stick[1])))])
    buttons = list(last_buttons)[: len(BUTTON_NAMES)]
    buttons += [0.0] * (len(BUTTON_NAMES) - len(buttons))
    f.extend(1.0 if value > 0.5 else 0.0 for value in buttons)

    if len(f) > BASE_FEATURE_DIM:
        raise RuntimeError(
            f"base feature vector grew to {len(f)} > BASE_FEATURE_DIM={BASE_FEATURE_DIM}"
        )
    f.extend([0.0] * (BASE_FEATURE_DIM - len(f)))
    return f


def stack_frames(history: list[list[float]]) -> list[float]:
    """Flatten the latest structured frames, left-padding a new episode with zeros."""
    frames = [list(row) for row in history[-STACK_FRAMES:]]
    padding = [[0.0] * BASE_FEATURE_DIM for _ in range(STACK_FRAMES - len(frames))]
    combined = [value for row in [*padding, *frames] for value in row]
    if len(combined) != FEATURE_DIM:
        raise RuntimeError("invalid stacked observation size")
    return combined
