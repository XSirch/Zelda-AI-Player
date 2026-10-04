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


_PROBE_YAW_OFFSETS = {
    "forward": 0.0,
    "forward_left": math.pi / 4.0,
    "left": math.pi / 2.0,
    "back_left": 3.0 * math.pi / 4.0,
    "back": math.pi,
    "back_right": -3.0 * math.pi / 4.0,
    "right": -math.pi / 2.0,
    "forward_right": -math.pi / 4.0,
}


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def camera_world_yaw(game: GameState, player=None) -> float | None:
    """Return the world-space heading that N64 stick-forward currently means.

    SoH exposes Camera_GetInputDirYaw directly.  The view vector is a fallback
    for transient/legacy payloads where that field is unavailable; player yaw is
    only the final fallback because fixed indoor cameras are not player-relative.
    """
    if game.camera_input_yaw is not None:
        return game.camera_input_yaw * math.pi / 32768.0

    if game.camera_eye is not None and game.camera_at is not None:
        dx = float(game.camera_at[0]) - float(game.camera_eye[0])
        dz = float(game.camera_at[2]) - float(game.camera_eye[2])
        if math.hypot(dx, dz) > 1e-4:
            return math.atan2(dx, dz)

    if player is not None:
        return player.yaw * math.pi / 32768.0
    return None


def camera_relative_stick(
    game: GameState,
    player,
    world_yaw: float,
) -> tuple[float, float]:
    camera_yaw = camera_world_yaw(game, player)
    if camera_yaw is None:
        camera_yaw = player.yaw * math.pi / 32768.0
    relative = _wrap_angle(world_yaw - camera_yaw)
    # Pinned SoH consumes Math_Atan2S(relY, -relX), and flips relX in
    # mirrored worlds. Invert that native transform rather than reflecting
    # the desired world-space heading across the current camera axis.
    horizontal_sign = 1.0 if game.mirrored_world else -1.0
    return (
        max(-1.0, min(1.0, horizontal_sign * math.sin(relative))),
        max(-1.0, min(1.0, math.cos(relative))),
    )


def _probe_is_walkable(probe) -> bool:
    if not probe.floor_found:
        return False
    if probe.wall_hit:
        return False
    delta_y = probe.delta_y
    if delta_y is not None and (delta_y > 70.0 or delta_y < -90.0):
        return False
    return True


def _collision_detour(game: GameState, player, target_yaw: float) -> tuple[tuple[float, float] | None, dict]:
    """Pick a locally collision-safe heading when the direct heading is blocked.

    Navigation probes are scene collision observations relative to Link's yaw.
    This is deliberately local/reactive: it does not solve a route or encode any
    Zelda-specific destination knowledge.
    """
    probes = {}
    for probe in game.navigation_probes:
        current = probes.get(probe.direction)
        if current is None or abs(probe.distance - 70.0) < abs(current.distance - 70.0):
            probes[probe.direction] = probe
    if not probes:
        return None, {
            "blocked": False,
            "detour": None,
            "direct_probe": None,
        }

    player_yaw = player.yaw * math.pi / 32768.0
    relative_target = _wrap_angle(target_yaw - player_yaw)
    direct_name = min(
        _PROBE_YAW_OFFSETS,
        key=lambda name: abs(_wrap_angle(relative_target - _PROBE_YAW_OFFSETS[name])),
    )
    direct_probe = probes.get(direct_name)
    if direct_probe is None or _probe_is_walkable(direct_probe):
        return None, {
            "blocked": False,
            "detour": None,
            "direct_probe": direct_name,
        }

    candidates = []
    for name, offset in _PROBE_YAW_OFFSETS.items():
        probe = probes.get(name)
        if probe is None or not _probe_is_walkable(probe):
            continue
        angular_error = abs(_wrap_angle(relative_target - offset))
        alignment = math.cos(angular_error)
        # Avoid choosing a reverse heading unless every forward/side option is
        # unavailable.  Small floor-height changes are preferred.
        if alignment < -0.25:
            continue
        height_penalty = min(0.2, abs(float(probe.delta_y or 0.0)) / 350.0)
        candidates.append((alignment - height_penalty, -angular_error, name, offset))

    if not candidates:
        return None, {
            "blocked": True,
            "detour": None,
            "direct_probe": direct_name,
        }

    _, _, selected_name, selected_offset = max(candidates)
    selected_yaw = _wrap_angle(player_yaw + selected_offset)
    return camera_relative_stick(game, player, selected_yaw), {
        "blocked": True,
        "detour": selected_name,
        "direct_probe": direct_name,
    }


def goal_guidance(
    game: GameState,
    intent: AgentIntent,
    *,
    local_dwell_seconds: float = 0.0,
    route_hint: dict | None = None,
) -> dict:
    """Convert a structured high-level goal into collision-aware analog guidance.

    The target remains cognition-provided and observed.  Local collision probes
    can bend the steering prior around an immediate obstacle, but they do not
    create a route or select controller buttons.  Long residence in the same
    area also fades the prior so a stale target cannot become a permanent magnet.
    """
    player = game.player
    modal_source = (
        "dialogue_hold"
        if game.dialogue.active
        else "menu_hold"
        if game.pause_menu.active
        else "none"
    )
    if (
        player is None
        or game.dialogue.active
        or game.pause_menu.active
        or intent.mode in {"combat", "dialogue", "menu"}
    ):
        return {
            "active": False,
            "stick": (0.0, 0.0),
            "strength": 0.0,
            "button_quiet": 0.0,
            "distance": None,
            "source": modal_source,
            "target": None,
            "blocked": False,
            "detour": None,
            "direct_probe": None,
            "stuck_scale": 1.0,
            "route_active": False,
            "route_path_nodes": 0,
            "route_confidence": 0.0,
            "route_target_gap": None,
            "route_waypoint": None,
            "route_waypoint_id": None,
            "route_partial": False,
            "route_edge_key": None,
            "route_edge_failures": 0,
            "frontier_active": False,
            "frontier_direction": None,
            "frontier_stable": False,
            "frontier_age_s": None,
            "exit_active": False,
            "exit_index": None,
            "exit_direct_reachable": False,
            "forced_escape": False,
            "remembered_escape": False,
            "escape_key": None,
        }

    point = target_point(game, intent)
    final_point = point
    source = (
        "target_actor"
        if intent.target_actor_uid is not None or intent.target_actor_id is not None
        else "target_position"
    )
    route_active = False
    route_path_nodes = 0
    route_confidence = 0.0
    route_target_gap = None
    route_waypoint = None
    route_waypoint_id = None
    route_partial = False
    route_edge_key = None
    route_edge_failures = 0
    frontier_active = False
    frontier_direction = None
    frontier_stable = False
    frontier_age_s = None
    exit_active = False
    exit_index = None
    exit_direct_reachable = False
    traversal_route_active = False
    traversal_route_kind = None
    traversal_route_direction = None
    traversal_route_phase = None
    forced_escape = False
    is_remembered_escape = False
    escape_key = None
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

    if (
        route_hint
        and intent.mode in {"navigate", "explore", "observe"}
        and not source.startswith("traversal:")
        and (
            point is not None
            or bool(route_hint.get("frontier"))
            or bool(route_hint.get("exit"))
            or bool(route_hint.get("traversal"))
        )
    ):
        waypoint = route_hint.get("waypoint")
        if (
            isinstance(waypoint, (list, tuple))
            and len(waypoint) == 3
            and all(isinstance(value, (int, float)) and math.isfinite(value) for value in waypoint)
        ):
            is_frontier = bool(route_hint.get("frontier"))
            is_exit = bool(route_hint.get("exit"))
            is_traversal_route = bool(route_hint.get("traversal"))
            is_door_escape = bool(route_hint.get("door"))
            is_remembered_escape = bool(route_hint.get("remembered"))
            forced_escape = bool(route_hint.get("forced_escape"))
            raw_escape_key = route_hint.get("escape_key")
            escape_key = (
                str(raw_escape_key)[:240]
                if raw_escape_key is not None
                else None
            )
            route_active = not route_hint.get("local_path") and not is_frontier and not is_exit and int(
                route_hint.get("path_nodes") or 0
            ) > 1
            frontier_active = is_frontier
            exit_active = is_exit
            traversal_route_active = is_traversal_route
            traversal_route_kind = (
                str(route_hint.get("traversal_kind"))[:64]
                if is_traversal_route
                and route_hint.get("traversal_kind") is not None
                else None
            )
            traversal_route_direction = (
                str(route_hint.get("traversal_direction"))[:16]
                if is_traversal_route
                and route_hint.get("traversal_direction") is not None
                else None
            )
            traversal_route_phase = (
                str(route_hint.get("traversal_phase"))[:16]
                if is_traversal_route
                and route_hint.get("traversal_phase") is not None
                else None
            )
            exit_index = (
                int(route_hint.get("exit_index"))
                if is_exit and route_hint.get("exit_index") is not None
                else None
            )
            exit_direct_reachable = bool(
                route_hint.get("direct_reachable")
            ) if is_exit else False
            frontier_direction = (
                str(route_hint.get("direction"))[:32]
                if is_frontier and route_hint.get("direction") is not None
                else None
            )
            frontier_stable = bool(route_hint.get("stable")) if is_frontier else False
            age_value = route_hint.get("age_s")
            frontier_age_s = (
                float(age_value)
                if is_frontier and isinstance(age_value, (int, float))
                else None
            )
            route_waypoint = tuple(float(value) for value in waypoint)
            waypoint_id = route_hint.get("waypoint_id")
            route_waypoint_id = (
                str(waypoint_id)[:160]
                if waypoint_id is not None
                else None
            )
            point = route_waypoint
            if is_frontier:
                source = "observed_frontier"
            elif is_traversal_route:
                traversal_prefix = (
                    "escape_traversal"
                    if forced_escape
                    else "observed_traversal"
                )
                source = (
                    f"{traversal_prefix}:{traversal_route_kind}:"
                    f"{traversal_route_phase}"
                )
                final_escape = route_hint.get("escape_final_target")
                if (
                    isinstance(final_escape, (list, tuple))
                    and len(final_escape) == 3
                    and all(
                        isinstance(value, (int, float))
                        and math.isfinite(value)
                        for value in final_escape
                    )
                ):
                    final_point = tuple(float(value) for value in final_escape)
            elif is_exit:
                path_nodes = int(route_hint.get("path_nodes") or 0)
                if is_remembered_escape:
                    memory_kind = str(route_hint.get("memory_kind") or "exit")
                    source = (
                        f"remembered_{memory_kind}"
                        if path_nodes <= 1
                        else f"remembered_{memory_kind}_route"
                    )
                elif is_door_escape:
                    source = (
                        "observed_door"
                        if path_nodes <= 1
                        else "observed_door_route"
                    )
                else:
                    source = (
                        "scene_exit"
                        if path_nodes <= 1
                        else "scene_exit_route"
                    )
                final_exit = route_hint.get("exit_position")
                if (
                    isinstance(final_exit, (list, tuple))
                    and len(final_exit) == 3
                    and all(
                        isinstance(value, (int, float))
                        and math.isfinite(value)
                        for value in final_exit
                    )
                ):
                    final_point = tuple(
                        float(value) for value in final_exit
                    )
            else:
                source = "observed_local_path" if route_hint.get("local_path") else "learned_route"
            route_path_nodes = max(0, int(route_hint.get("path_nodes") or 0))
            route_confidence = max(
                0.0, min(1.0, float(route_hint.get("confidence") or 0.0))
            )
            gap = route_hint.get("target_gap")
            route_target_gap = float(gap) if isinstance(gap, (int, float)) else None
            route_partial = bool(route_hint.get("partial"))
            edge_key = route_hint.get("edge_key")
            route_edge_key = (
                str(edge_key)[:240] if edge_key is not None else None
            )
            route_edge_failures = max(
                0,
                int(route_hint.get("edge_failures") or 0),
            )

    if point is not None:
        dx = point[0] - player.position[0]
        dz = point[2] - player.position[2]
        horizontal = math.hypot(dx, dz)
        distance = math.dist(
            player.position,
            final_point if final_point is not None else point,
        )
        blocked = False
        detour = None
        direct_probe = None
        steering_yaw = None
        if horizontal < 1e-4:
            stick = (0.0, 0.0)
        else:
            target_yaw = math.atan2(dx, dz)
            steering_yaw = target_yaw
            stick = camera_relative_stick(game, player, target_yaw)
            # Explicit vertical affordances (ladder/climbable wall/stairs) may
            # intentionally terminate at collision.  Do not steer away from the
            # very surface the observer identified as the traversal target.
            if (
                intent.mode in {"navigate", "explore", "observe"}
                and not source.startswith("traversal:")
                and not source.startswith("escape_traversal:")
                and not source.startswith("observed_traversal:")
                and not (
                    source.startswith("scene_exit")
                    and exit_direct_reachable
                )
                and not (
                    (
                        source.startswith("observed_door")
                        or source.startswith("remembered_door")
                    )
                    and horizontal <= 120.0
                )
            ):
                detour_stick, detour_info = _collision_detour(
                    game,
                    player,
                    target_yaw,
                )
                blocked = bool(detour_info["blocked"])
                detour = detour_info["detour"]
                direct_probe = detour_info["direct_probe"]
                if detour_stick is not None:
                    stick = detour_stick
                    if detour is not None:
                        steering_yaw = _wrap_angle(
                            player.yaw * math.pi / 32768.0
                            + _PROBE_YAW_OFFSETS[detour]
                        )
                    source = f"{source}:detour:{detour}"

        if traversal_route_active:
            # Observed stairs/ladders/ledges are physical navigation primitives,
            # not PPO exploration suggestions. Give the low-level controller
            # deterministic authority while traversing them.
            base_strength = 1.0
        elif route_active:
            # A route actually traversed by Link should be authoritative enough
            # to replay, while still leaving a residual channel for corrections.
            base_strength = (
                0.78 + 0.16 * route_confidence
            ) * (0.85 if route_partial else 1.0)
        elif exit_active:
            if forced_escape:
                # Once room recovery explicitly commits to an escape, PPO
                # residual steering must not pull Link back into the same room.
                base_strength = 1.0
            else:
                # Normal observed scene-exit guidance keeps the historical mix;
                # only explicit recovery/room-memory escape takes full authority.
                base_strength = 0.92 if exit_direct_reachable else 0.84
        elif frontier_active:
            # Targetless exploration should still have a stable local heading.
            # The waypoint comes from an observed open probe, not hidden map data.
            base_strength = 0.80
        else:
            base_strength = {
                "navigate": 0.86,
                "interact": 0.78,
                "explore": 0.74,
                "observe": 0.62,
            }.get(intent.mode, 0.68)
        # Direct cognition targets may be stale/unreachable, so they keep
        # the anti-attractor fade. Empirically learned routes, observed frontiers,
        # and native scene exits already have their own validity/expiry logic and
        # must remain authoritative until their current waypoint is actually hit.
        proximity = max(0.0, min(1.0, (horizontal - 45.0) / 120.0))
        if (
            route_active
            or frontier_active
            or exit_active
            or traversal_route_active
        ):
            proximity = 1.0

        dwell = max(0.0, float(local_dwell_seconds or 0.0))
        if (
            route_active
            or frontier_active
            or exit_active
            or traversal_route_active
        ):
            # A partial learned route is retired by route-memory endpoint
            # exhaustion; frontiers are recomputed from current probes; exits
            # are explicit native transition surfaces. Weakening any of these
            # because local dwell is large hands control back to PPO residual
            # exactly when structured navigation is most needed.
            stuck_scale = 1.0
        elif dwell <= 90.0:
            stuck_scale = 1.0
        else:
            stuck_scale = max(
                0.25,
                1.0 - min(1.0, (dwell - 90.0) / 180.0) * 0.75,
            )
        obstacle_scale = (
            1.0
            if forced_escape
            and (exit_active or traversal_route_active)
            and (not blocked or detour is not None)
            else 0.82
            if detour is not None
            else 0.30
            if blocked
            else 1.0
        )
        strength = base_strength * proximity * stuck_scale * obstacle_scale
        if traversal_route_active:
            quiet_multiplier = 0.95
        elif exit_active:
            # Explicit room recovery suppresses policy button noise; normal
            # exit guidance preserves the previous lower quieting behaviour.
            quiet_multiplier = 0.95 if forced_escape else 0.12
        else:
            quiet_multiplier = (
                0.45 if blocked else 0.9
            ) if intent.mode in {"navigate", "explore", "observe"} else 0.25
        return {
            "active": strength > 0.01,
            "stick": stick,
            "strength": strength,
            "button_quiet": strength * quiet_multiplier,
            "distance": distance,
            "source": source,
            "target": tuple(point),
            "world_yaw": steering_yaw,
            "camera_yaw": camera_world_yaw(game, player),
            "blocked": blocked,
            "detour": detour,
            "direct_probe": direct_probe,
            "stuck_scale": stuck_scale,
            "route_active": route_active,
            "local_path_active": bool(route_hint and route_hint.get("local_path")),
            "route_path_nodes": route_path_nodes,
            "route_confidence": route_confidence,
            "route_target_gap": route_target_gap,
            "route_waypoint": route_waypoint,
            "route_waypoint_id": route_waypoint_id,
            "route_partial": route_partial,
            "route_edge_key": route_edge_key,
            "route_edge_failures": route_edge_failures,
            "frontier_active": frontier_active,
            "frontier_direction": frontier_direction,
            "frontier_stable": frontier_stable,
            "frontier_age_s": frontier_age_s,
            "exit_active": exit_active,
            "exit_index": exit_index,
            "exit_direct_reachable": exit_direct_reachable,
            "traversal_route_active": traversal_route_active,
            "traversal_route_kind": traversal_route_kind,
            "traversal_route_direction": traversal_route_direction,
            "traversal_route_phase": traversal_route_phase,
            "forced_escape": forced_escape if point is not None else False,
            "remembered_escape": is_remembered_escape if point is not None else False,
            "escape_key": escape_key if point is not None else None,
        }

    direction_sticks = {
        "forward": (0.0, 1.0),
        "back": (0.0, -1.0),
        "left": (-1.0, 0.0),
        "right": (1.0, 0.0),
    }
    stick = direction_sticks.get(intent.direction)
    if stick is not None:
        dwell = max(0.0, float(local_dwell_seconds or 0.0))
        stuck_scale = 1.0 if dwell <= 90.0 else max(
            0.25,
            1.0 - min(1.0, (dwell - 90.0) / 180.0) * 0.75,
        )
        return {
            "active": True,
            "stick": stick,
            "strength": 0.58 * stuck_scale,
            "button_quiet": 0.48 * stuck_scale,
            "distance": None,
            "source": f"direction:{intent.direction}",
            "target": None,
            "blocked": False,
            "detour": None,
            "direct_probe": None,
            "stuck_scale": stuck_scale,
            "route_active": False,
            "route_path_nodes": 0,
            "route_confidence": 0.0,
            "route_target_gap": None,
            "route_waypoint": None,
            "route_waypoint_id": None,
            "route_partial": False,
            "route_edge_key": None,
            "route_edge_failures": 0,
            "frontier_active": False,
            "frontier_direction": None,
            "frontier_stable": False,
            "frontier_age_s": None,
            "exit_active": False,
            "exit_index": None,
            "exit_direct_reachable": False,
        }

    return {
        "active": False,
        "stick": (0.0, 0.0),
        "strength": 0.0,
        "button_quiet": 0.0,
        "distance": None,
        "source": "none",
        "target": None,
        "blocked": False,
        "detour": None,
        "direct_probe": None,
        "stuck_scale": 1.0,
        "route_active": False,
        "route_path_nodes": 0,
        "route_confidence": 0.0,
        "route_target_gap": None,
        "route_waypoint": None,
        "route_waypoint_id": None,
        "route_partial": False,
        "route_edge_key": None,
        "route_edge_failures": 0,
        "frontier_active": False,
        "frontier_direction": None,
        "frontier_stable": False,
        "frontier_age_s": None,
        "exit_active": False,
        "exit_index": None,
        "exit_direct_reachable": False,
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
