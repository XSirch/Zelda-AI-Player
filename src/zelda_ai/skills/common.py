"""Local common controller helpers."""
from __future__ import annotations

import asyncio
import math
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


def _probe_stick(direction: str) -> tuple[int, int]:
    scale = 48
    diagonal = 36
    return {
        "forward": (0, scale),
        "forward_right": (diagonal, diagonal),
        "right": (scale, 0),
        "back_right": (diagonal, -diagonal),
        "back": (0, -scale),
        "back_left": (-diagonal, -diagonal),
        "left": (-scale, 0),
        "forward_left": (-diagonal, diagonal),
    }[direction]


def _steer_to(game: GameState, target: tuple[float, float, float], strength: float) -> tuple[int, int, float]:
    if not game.player:
        return 0, 0, float("inf")
    px, _, pz = game.player.position
    dx, dz = target[0] - px, target[2] - pz
    distance = math.hypot(dx, dz)
    if distance < 1e-6:
        return 0, 0, 0.0

    if game.camera_eye is not None and game.camera_at is not None:
        fx = game.camera_at[0] - game.camera_eye[0]
        fz = game.camera_at[2] - game.camera_eye[2]
        flen = math.hypot(fx, fz)
    else:
        flen = 0.0
    if flen < 1e-4:
        angle = game.player.yaw * math.pi / 32768.0
        fx, fz = math.sin(angle), math.cos(angle)
    else:
        fx, fz = fx / flen, fz / flen

    # Camera-relative right vector. N64 stick X is right, Y is forward.
    rx, rz = fz, -fx
    ux, uz = dx / distance, dz / distance
    local_x = ux * rx + uz * rz
    local_y = ux * fx + uz * fz
    scale = max(28, min(80, round(80 * max(0.35, strength))))
    return (
        max(-80, min(80, round(local_x * scale))),
        max(-80, min(80, round(local_y * scale))),
        distance,
    )


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
