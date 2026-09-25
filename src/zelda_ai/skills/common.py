"""Local common controller helpers."""
from __future__ import annotations

import asyncio
import math
import time
from ..bridge import Bridge
from ..control.feedback import consumed, is_realtime
from ..models import Decision, GameState

def _matching_actor(game: GameState, actor_id: int, params: int | None, actor_uid: str | None = None):
    # room_actors is the authoritative current-room set. nearby_actors remains a smaller
    # rendered/proximity subset for compact UI/telemetry compatibility.
    candidates = list(game.room_actors) + list(game.nearby_actors)
    if game.target_actor is not None:
        candidates.append(game.target_actor)
    matches = [actor for actor in candidates
        if actor.actor_id == actor_id and (params is None or actor.params == params)
        and (actor_uid is None or actor.actor_uid == actor_uid)]
    return min(matches, key=lambda actor: actor.distance) if matches else None


def _is_door_actor(actor) -> bool:
    return actor is not None and (actor.category == 10 or actor.category_name == "door")


def _probe_stick(direction: str, game: GameState | None = None) -> tuple[int, int]:
    """Convert a Link-relative terrain-probe direction into SoH raw stick input."""
    if game is not None and game.player is not None:
        if "probe_yaw_v2" in game.capabilities:
            offsets = {
                "forward": 0x0000,
                "forward_left": 0x2000,
                "left": 0x4000,
                "back_left": 0x6000,
                "back": 0x8000,
                "back_right": -0x6000,
                "right": -0x4000,
                "forward_right": -0x2000,
            }
        else:
            # v2.5 and earlier mislabeled +yaw samples as "right".
            offsets = {
                "forward": 0x0000,
                "forward_right": 0x2000,
                "right": 0x4000,
                "back_right": 0x6000,
                "back": 0x8000,
                "back_left": -0x6000,
                "left": -0x4000,
                "forward_left": -0x2000,
            }
        return _world_yaw_stick(game, game.player.yaw + offsets[direction], 48)

    # Compatibility fallback for callers without a GameState.
    scale = 48
    diagonal = 34
    return {
        "forward": (0, scale),
        "forward_left": (-diagonal, diagonal),
        "left": (-scale, 0),
        "back_left": (-diagonal, -diagonal),
        "back": (0, -scale),
        "back_right": (diagonal, -diagonal),
        "right": (scale, 0),
        "forward_right": (diagonal, diagonal),
    }[direction]


def _world_yaw_stick(game: GameState, desired_world_yaw: float,
                     magnitude: int) -> tuple[int, int]:
    """Invert SoH Player_ProcessControlStick for an exact desired world yaw."""
    magnitude = max(0, min(80, int(magnitude)))
    if game.camera_input_yaw is not None:
        stick_angle = ((desired_world_yaw - game.camera_input_yaw + 32768.0) % 65536.0) - 32768.0
        angle = stick_angle * math.pi / 32768.0
        # func_80077D10 computes Math_Atan2S(relY, -relX).
        x = round(-math.sin(angle) * magnitude)
        y = round(math.cos(angle) * magnitude)
        if game.mirrored_world:
            x = -x
        return max(-80, min(80, x)), max(-80, min(80, y))

    # Legacy fallback: infer the camera input yaw from its look vector. This
    # path is only for bridges that do not expose Camera_GetInputDirYaw.
    if game.camera_eye is not None and game.camera_at is not None:
        fx = game.camera_at[0] - game.camera_eye[0]
        fz = game.camera_at[2] - game.camera_eye[2]
        if math.hypot(fx, fz) >= 1e-4:
            camera_yaw = math.atan2(fx, fz) * 32768.0 / math.pi
            stick_angle = ((desired_world_yaw - camera_yaw + 32768.0) % 65536.0) - 32768.0
            angle = stick_angle * math.pi / 32768.0
            x = round(-math.sin(angle) * magnitude)
            y = round(math.cos(angle) * magnitude)
            return max(-80, min(80, x)), max(-80, min(80, y))

    # Last-resort Link-relative fallback if camera orientation is unavailable.
    stick_angle = ((desired_world_yaw - game.player.yaw + 32768.0) % 65536.0) - 32768.0 if game.player else 0.0
    angle = stick_angle * math.pi / 32768.0
    return (
        max(-80, min(80, round(-math.sin(angle) * magnitude))),
        max(-80, min(80, round(math.cos(angle) * magnitude))),
    )


def _steer_to(game: GameState, target: tuple[float, float, float], strength: float) -> tuple[int, int, float]:
    """Convert a world-space target into the raw N64 stick consumed by OoT.

    Player_ProcessControlStick computes:
        world_yaw = Camera_GetInputDirYaw() + Math_Atan2S(stick_y, -stick_x)
    so the inverse transform must negate the sine term for raw stick X.
    """
    if not game.player:
        return 0, 0, float("inf")
    px, _, pz = game.player.position
    dx, dz = target[0] - px, target[2] - pz
    distance = math.hypot(dx, dz)
    if distance < 1e-6:
        return 0, 0, 0.0

    desired_yaw = math.atan2(dx, dz)
    if game.camera_input_yaw is not None:
        camera_yaw = game.camera_input_yaw * math.pi / 32768.0
    elif game.camera_eye is not None and game.camera_at is not None:
        fx = game.camera_at[0] - game.camera_eye[0]
        fz = game.camera_at[2] - game.camera_eye[2]
        if math.hypot(fx, fz) >= 1e-4:
            camera_yaw = math.atan2(fx, fz)
        else:
            camera_yaw = game.player.yaw * math.pi / 32768.0
    else:
        camera_yaw = game.player.yaw * math.pi / 32768.0

    stick_angle = (desired_yaw - camera_yaw + math.pi) % (2 * math.pi) - math.pi
    scale = max(28, min(80, round(80 * max(0.35, strength))))
    stick_x = round(-math.sin(stick_angle) * scale)
    stick_y = round(math.cos(stick_angle) * scale)
    # Shipwright mirrors relX before deriving stick angle, so pre-mirror here
    # to preserve the same requested world-space target.
    if game.mirrored_world:
        stick_x = -stick_x
    return max(-80, min(80, stick_x)), max(-80, min(80, stick_y)), distance


PLAYER_STATE2_HOPPING = 1 << 19
_DODGE_DIRECTION = {"left": 1, "back": 2, "right": 3}


def _player_relative_stick(game: GameState, direction: str, magnitude: int = 70) -> tuple[int, int]:
    """Convert a Link-relative direction into the raw camera-relative N64 stick used by OoT."""
    if not game.player or direction not in {"forward", "back", "left", "right"}:
        return 0, 0
    offsets = {"forward": 0x0000, "left": 0x4000, "back": 0x8000, "right": -0x4000}
    desired_world_yaw = game.player.yaw + offsets[direction]
    return _world_yaw_stick(game, desired_world_yaw, magnitude)


def _rotate_stick_quadrants(stick_x: int, stick_y: int, steps: int) -> tuple[int, int]:
    """Rotate raw stick in OoT's positive direction: forward -> left -> back -> right."""
    x, y = stick_x, stick_y
    for _ in range(steps % 4):
        x, y = -y, x
    return x, y


async def _prime_dodge_direction(bridge: Bridge, observation: GameState, direction: str,
                                 magnitude: int = 70, timeout: float = .5) -> tuple[GameState | None, int, int]:
    """Hold Z + stick until Player_ProcessControlStick reports the requested direction."""
    expected = _DODGE_DIRECTION[direction]
    current = bridge.state or observation
    stick_x, stick_y = _player_relative_stick(current, direction, magnitude)
    deadline = time.monotonic() + timeout
    while current and time.monotonic() < deadline:
        actual = current.player.control_stick_direction if current.player else -1
        if actual == expected:
            return current, stick_x, stick_y
        if actual in {0, 1, 2, 3}:
            # Close the loop on OoT's own classification instead of trusting camera math blindly.
            stick_x, stick_y = _rotate_stick_quadrants(
                stick_x, stick_y, (expected - actual) % 4)
        else:
            stick_x, stick_y = _player_relative_stick(current, direction, magnitude)
        bridge.send(buttons=0x2000, stick_x=stick_x, stick_y=stick_y, lease_ms=260)
        try:
            current = await bridge.next_state(current.seq, timeout=min(.2, max(.001, deadline-time.monotonic())))
        except RuntimeError:
            return None, stick_x, stick_y
    if current and current.player and current.player.control_stick_direction == expected:
        return current, stick_x, stick_y
    return None, stick_x, stick_y


async def _observe_dodge_effect(bridge: Bridge, start: GameState, expected_direction: int,
                                timeout: float = .8) -> dict:
    current = bridge.state
    hopping_seen = False
    hop_direction = None
    max_distance = 0.0
    expected_distance = 0.0
    if not start.player:
        return {"confirmed": False, "hopping_seen": False, "hop_direction": None,
                "max_distance": 0.0, "expected_distance": 0.0}
    yaw = start.player.yaw * math.pi / 32768.0
    forward = (math.sin(yaw), math.cos(yaw))
    right = (math.cos(yaw), -math.sin(yaw))
    start_x, _, start_z = start.player.position
    deadline = time.monotonic() + timeout
    while current and time.monotonic() < deadline:
        if current.player and current.scene_epoch == start.scene_epoch:
            dx = current.player.position[0] - start_x
            dz = current.player.position[2] - start_z
            max_distance = max(max_distance, math.hypot(dx, dz))
            if expected_direction == 2:
                projected = -(dx * forward[0] + dz * forward[1])
            elif expected_direction == 1:
                projected = -(dx * right[0] + dz * right[1])
            else:
                projected = dx * right[0] + dz * right[1]
            expected_distance = max(expected_distance, projected)
            if current.player.state_flags_2 & PLAYER_STATE2_HOPPING:
                hopping_seen = True
                if current.player.hop_direction is not None:
                    hop_direction = current.player.hop_direction
                    if hop_direction == expected_direction:
                        break
        try:
            current = await bridge.next_state(current.seq, timeout=min(.2, max(.001, deadline-time.monotonic())))
        except RuntimeError:
            break
    return {"confirmed": hopping_seen and hop_direction == expected_direction,
            "hopping_seen": hopping_seen, "hop_direction": hop_direction,
            "max_distance": round(max_distance, 2), "expected_distance": round(expected_distance, 2)}


def _dodge_direction_safe(game: GameState, direction: str) -> bool:
    probe_direction = direction
    if "probe_yaw_v2" not in game.capabilities:
        probe_direction = {"left": "right", "right": "left"}.get(direction, direction)
    probes = [p for p in game.navigation_probes
              if p.direction == probe_direction and p.distance <= 70.0 and p.floor_found and p.delta_y is not None]
    if not probes:
        return False
    probe = min(probes, key=lambda p: p.distance)
    return abs(probe.delta_y) <= 22.0 and not probe.wall_hit


async def _perform_dodge(bridge: Bridge, observation: GameState, direction: str,
                         magnitude: int = 70) -> dict:
    expected = _DODGE_DIRECTION[direction]
    if not _dodge_direction_safe(observation, direction):
        return {"reason": "dodge_terrain_not_safe", "receipt": None,
                "confirmed": False, "hopping_seen": False, "hop_direction": None,
                "expected_hop_direction": expected, "max_distance": 0.0, "expected_distance": 0.0,
                "stick_x": 0, "stick_y": 0}
    if ("player_relative_dodge_state" not in observation.capabilities or
            "control_stick_direction" not in observation.capabilities):
        return {"reason": "bridge_upgrade_required_for_dodge_confirmation", "receipt": None,
                "confirmed": False, "hopping_seen": False, "hop_direction": None,
                "expected_hop_direction": expected, "max_distance": 0.0, "expected_distance": 0.0,
                "stick_x": 0, "stick_y": 0}
    primed, stick_x, stick_y = await _prime_dodge_direction(bridge, observation, direction, magnitude)
    if primed is None:
        actual = bridge.state.player.control_stick_direction if bridge.state and bridge.state.player else None
        return {"reason": "dodge_direction_not_established", "receipt": None, "confirmed": False,
                "hopping_seen": False, "hop_direction": None, "expected_hop_direction": expected,
                "control_stick_direction": actual,
                "max_distance": 0.0, "expected_distance": 0.0,
                "stick_x": stick_x, "stick_y": stick_y}
    receipt = await bridge.sequence_receipt([
        {"buttons": 0x2000, "stick_x": stick_x, "stick_y": stick_y, "ticks": 1},
        {"buttons": 0xA000, "stick_x": stick_x, "stick_y": stick_y, "ticks": 1},
        {"buttons": 0x2000, "stick_x": stick_x, "stick_y": stick_y, "ticks": 2},
        {"buttons": 0x2000, "stick_x": 0, "stick_y": 0, "ticks": 1},
    ], baseline_buttons=0x2000, edge_buttons=0x8000)
    delivered = bool(receipt and receipt.first_tick > 0 and
                     receipt.status not in {"rejected", "cancelled", "superseded"})
    effect = await _observe_dodge_effect(bridge, observation, expected) if delivered else {
        "confirmed": False, "hopping_seen": False, "hop_direction": None,
        "max_distance": 0.0, "expected_distance": 0.0}
    reason = ("dodge_confirmed" if effect["confirmed"] else
              ("wrong_dodge_direction" if effect["hopping_seen"] and effect["hop_direction"] is not None else
               ("dodge_effect_not_observed" if delivered else "input_not_consumed")))
    return {"reason": reason, "receipt": receipt, "expected_hop_direction": expected,
            "control_stick_direction": primed.player.control_stick_direction if primed.player else None,
            "stick_x": stick_x, "stick_y": stick_y, **effect}


async def _pulse(bridge: Bridge, *, buttons: int = 0, stick_x: int = 0, stick_y: int = 0,
                 hold_ms: int = 100, settle_s: float = 0.12) -> bool:
    if is_realtime(bridge):
        tick_ms = getattr(bridge, "input_tick_ms", 50.0)
        ticks = max(1, min(8, math.ceil(hold_ms / tick_ms)))
        return await bridge.pulse(buttons=buttons, stick_x=stick_x, stick_y=stick_y, hold_ticks=ticks)
    command_id = bridge.send(buttons=buttons, stick_x=stick_x, stick_y=stick_y,
        lease_ms=max(50, min(300, hold_ms)))
    await asyncio.sleep(max(0.05, hold_ms / 1000))
    bridge.release()
    await asyncio.sleep(settle_s)
    current = bridge.state
    return bool(current and consumed(bridge, command_id))


def _target_position(game: GameState, decision: Decision) -> tuple[float, float, float] | None:
    if decision.args.target_actor_id is not None:
        actor = _matching_actor(game, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
        if actor is None:
            return None
        return actor.focus_position or actor.position
    return decision.args.target_position


def _yaw_to_target(player_position: tuple[float, float, float],
                   target: tuple[float, float, float]) -> int:
    dx = target[0] - player_position[0]
    dz = target[2] - player_position[2]
    return int(round(math.atan2(dx, dz) * 32768 / math.pi))


def _yaw_error_units(current_yaw: int, desired_yaw: int) -> int:
    return ((desired_yaw - current_yaw + 32768) % 65536) - 32768


def _aim_error(game: GameState, target: tuple[float, float, float]) -> tuple[float, float] | None:
    if game.camera_eye is None or game.camera_at is None:
        return None
    eye = game.camera_eye
    at = game.camera_at
    current = (at[0] - eye[0], at[1] - eye[1], at[2] - eye[2])
    desired = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
    cur_h = math.hypot(current[0], current[2])
    dst_h = math.hypot(desired[0], desired[2])
    if cur_h < 1e-4 or dst_h < 1e-4:
        return None
    current_yaw = math.atan2(current[0], current[2])
    desired_yaw = math.atan2(desired[0], desired[2])
    yaw_error = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
    current_pitch = math.atan2(current[1], cur_h)
    desired_pitch = math.atan2(desired[1], dst_h)
    return yaw_error, desired_pitch - current_pitch


def _aim_stick(error: float, sign: int) -> int:
    degrees = math.degrees(error)
    if abs(degrees) < 1.2:
        return 0
    magnitude = max(12, min(55, round(abs(degrees) * 2.2)))
    return magnitude * (1 if degrees > 0 else -1) * sign
