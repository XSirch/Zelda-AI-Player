from __future__ import annotations

import math

from ..models import ActorObservation, GameState
from .models import AgentIntent

BASE_FEATURE_DIM = 288
STACK_FRAMES = 4
FEATURE_DIM = BASE_FEATURE_DIM * STACK_FRAMES
BUTTON_NAMES = ("A", "B", "Z", "START", "R", "C_UP", "C_LEFT", "C_DOWN", "C_RIGHT")
MODE_NAMES = ("explore", "navigate", "interact", "combat", "dialogue", "menu", "observe")
PROBE_NAMES = (
    "forward", "forward_right", "right", "back_right",
    "back", "back_left", "left", "forward_left",
)

_NOVELTY_INTENT = AgentIntent(
    objective="Environment novelty encoding",
    summary="Environment novelty encoding",
    mode="observe",
    target_actor_id=None,
    target_actor_params=None,
    target_actor_uid=None,
    target_position=None,
    target_item_id=None,
    direction=None,
    choice_index=None,
    horizon_ms=2000,
)



def _squash(value: float, scale: float) -> float:
    return math.tanh(float(value) / scale)


def _item_feature(value: int | None) -> float:
    if value is None or value == 0xFF:
        return -1.0
    return max(-1.0, min(1.0, float(value) / 127.0 - 1.0))


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


def goal_guidance(game: GameState, intent: AgentIntent) -> dict:
    """Convert a structured high-level goal into camera-relative analog guidance.

    This is not a route solver and does not choose buttons. It only makes an
    observed point/direction actionable for the goal-conditioned ML policy.
    The PPO distribution remains stochastic in training and learns residual
    corrections around this prior.
    """
    player = game.player
    if player is None or intent.mode in {"combat", "dialogue", "menu"}:
        return {
            "active": False,
            "stick": (0.0, 0.0),
            "strength": 0.0,
            "button_quiet": 0.0,
            "distance": None,
            "source": "none",
            "target": None,
        }

    point = target_point(game, intent)
    source = (
        "target_actor"
        if intent.target_actor_uid is not None or intent.target_actor_id is not None
        else "target_position"
    )
    if point is None and intent.direction in {"up", "down"}:
        candidates = [
            row for row in game.traversal_affordances
            if row.direction == intent.direction
        ]
        if candidates:
            affordance = min(candidates, key=lambda row: row.distance)
            approach_distance = math.dist(
                player.position,
                affordance.approach_position,
            )
            if approach_distance > 85.0:
                point = affordance.approach_position
                source = f"traversal:{affordance.kind}:approach"
            else:
                point = affordance.target_position
                source = f"traversal:{affordance.kind}:target"

    if point is not None:
        dx = point[0] - player.position[0]
        dz = point[2] - player.position[2]
        horizontal = math.hypot(dx, dz)
        distance = math.dist(player.position, point)
        if horizontal < 1e-4:
            stick = (0.0, 0.0)
        else:
            camera_raw = (
                game.camera_input_yaw
                if game.camera_input_yaw is not None
                else player.yaw
            )
            camera_yaw = camera_raw * math.pi / 32768.0
            forward_x = math.sin(camera_yaw)
            forward_z = math.cos(camera_yaw)
            right_x = math.cos(camera_yaw)
            right_z = -math.sin(camera_yaw)
            forward = dx * forward_x + dz * forward_z
            right = dx * right_x + dz * right_z
            norm = max(1e-6, math.hypot(right, forward))
            stick = (
                max(-1.0, min(1.0, right / norm)),
                max(-1.0, min(1.0, forward / norm)),
            )

        base_strength = {
            "navigate": 0.86,
            "interact": 0.78,
            "explore": 0.74,
            "observe": 0.62,
        }.get(intent.mode, 0.68)
        # Fade the steering prior near the waypoint so learned interaction/
        # traversal behaviour can take over instead of orbiting the point.
        proximity = max(0.0, min(1.0, (horizontal - 45.0) / 120.0))
        strength = base_strength * proximity
        return {
            "active": strength > 0.01,
            "stick": stick,
            "strength": strength,
            "button_quiet": (
                strength * 0.9
                if intent.mode in {"navigate", "explore", "observe"}
                else strength * 0.25
            ),
            "distance": distance,
            "source": source,
            "target": tuple(point),
        }

    direction_sticks = {
        "forward": (0.0, 1.0),
        "back": (0.0, -1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
    }
    stick = direction_sticks.get(intent.direction)
    if stick is not None:
        return {
            "active": True,
            "stick": stick,
            "strength": 0.58,
            "button_quiet": 0.48,
            "distance": None,
            "source": f"direction:{intent.direction}",
            "target": None,
        }

    return {
        "active": False,
        "stick": (0.0, 0.0),
        "strength": 0.0,
        "button_quiet": 0.0,
        "distance": None,
        "source": "none",
        "target": None,
    }


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
        1.0 if intent.choice_index is not None else 0.0,
        (intent.choice_index / 2.0) if intent.choice_index is not None else 0.0,
        1.0 if intent.target_item_id is not None else 0.0,
        _item_feature(intent.target_item_id),
    ])

    if player is None:
        f.extend([0.0] * 27)
    else:
        yaw = player.yaw * math.pi / 32768.0
        camera_raw = game.camera_input_yaw if game.camera_input_yaw is not None else player.yaw
        camera_yaw = camera_raw * math.pi / 32768.0
        relative_yaw = ((player.yaw - camera_raw + 32768) % 65536 - 32768) * math.pi / 32768.0
        f.extend([
            player.health / max(1.0, float(player.max_health)),
            player.magic / 255.0,
            _squash(player.rupees, 200.0),
            _squash(player.speed_xz, 8.0),
            math.sin(yaw),
            math.cos(yaw),
            math.sin(camera_yaw),
            math.cos(camera_yaw),
            math.sin(relative_yaw),
            math.cos(relative_yaw),
            1.0 if game.camera_input_yaw is not None else 0.0,
            1.0 if game.mirrored_world else 0.0,
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

    raw_inventory = list(game.inventory[:32])
    raw_inventory += [0xFF] * (32 - len(raw_inventory))
    f.extend(_item_feature(item) for item in raw_inventory)

    equipped = list(game.equipped[:8])
    equipped += [0xFF] * (8 - len(equipped))
    f.extend(_item_feature(item) for item in equipped)

    gear = {kind: [] for kind in ("sword", "shield", "tunic", "boots")}
    for row in game.progress.equipment:
        gear[row.equipment_type].append(row)
    for kind in ("sword", "shield", "tunic", "boots"):
        rows = gear[kind]
        equipped_value = next((row.value for row in rows if row.equipped), 0)
        max_owned = max((row.value for row in rows), default=0)
        f.extend([equipped_value / 4.0, max_owned / 4.0])

    pause = game.pause_menu
    f.extend([
        _squash(pause.state, 32.0),
        _squash(pause.transition_state, 16.0),
        pause.page_index / 4.0,
        _squash(pause.cursor_special_pos, 4.0),
        _item_feature(pause.named_item),
        _squash(pause.prompt_choice, 2.0),
    ])
    cursor_point = list(pause.cursor_point[:5]) + [0] * (5 - len(pause.cursor_point[:5]))
    cursor_item = list(pause.cursor_item[:4]) + [0xFF] * (4 - len(pause.cursor_item[:4]))
    cursor_slot = list(pause.cursor_slot[:4]) + [0] * (4 - len(pause.cursor_slot[:4]))
    f.extend(_squash(value, 16.0) for value in cursor_point)
    f.extend(_item_feature(value) for value in cursor_item)
    f.extend(_squash(value, 16.0) for value in cursor_slot)
    f.append(_squash(game.last_played_song, 12.0))

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


def encode_novelty_state(game: GameState) -> list[float]:
    """Environment-only frame for RND; excludes changing intent and previous action."""
    return encode_state(
        game,
        _NOVELTY_INTENT,
        last_stick=(0.0, 0.0),
        last_buttons=(),
    )


def stack_frames(history: list[list[float]]) -> list[float]:
    """Flatten the latest structured frames, left-padding a new episode with zeros."""
    frames = [list(row) for row in history[-STACK_FRAMES:]]
    padding = [[0.0] * BASE_FEATURE_DIM for _ in range(STACK_FRAMES - len(frames))]
    combined = [value for row in [*padding, *frames] for value in row]
    if len(combined) != FEATURE_DIM:
        raise RuntimeError("invalid stacked observation size")
    return combined
