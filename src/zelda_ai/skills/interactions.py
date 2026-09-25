"""Local interactions controller helpers."""
from __future__ import annotations

import asyncio
import math
import time
from ..bridge import Bridge
from ..control.feedback import consumed
from ..models import Decision, GameState
from ..navmesh import waypoint_probe_safe
from .catalog import BUTTONS
from .common import _is_door_actor, _matching_actor, _pulse, _yaw_error_units, _yaw_to_target
from .navigation import _backtrack_recovery, _navigate_local

async def _wait_interaction_evidence(bridge: Bridge, observation: GameState, before_event_ids: set[str],
                                     timeout_s: float = 1.5) -> tuple[str | None, dict]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        sample = bridge.state
        if not sample:
            await asyncio.sleep(0.06)
            continue
        if (sample.instance_id, sample.scene_epoch) != (observation.instance_id, observation.scene_epoch):
            return "world_transition_after_interaction", {}
        if sample.dialogue.active:
            return "dialogue_opened", {}
        if sample.cutscene_active:
            return "interaction_started", {}
        outcome = next((event for event in sample.events
            if event.id not in before_event_ids and event.kind in {
                "item_received", "scene_flag_set", "scene_flag_unset"}), None)
        if outcome is not None:
            return outcome.kind, {"detail": outcome.detail}
        await asyncio.sleep(0.06)
    return None, {}


def _door_step_safe(game: GameState, actor, step_distance: float = 28.0) -> bool:
    """Validate only the next short advance; the closed door itself may be the final wall."""
    if "local_navmesh" not in game.capabilities:
        return True
    if not game.player:
        return False
    dx = actor.position[0] - game.player.position[0]
    dz = actor.position[2] - game.player.position[2]
    distance = math.hypot(dx, dz)
    if distance < 1e-4:
        return True
    step = min(step_distance, distance)
    target = (
        game.player.position[0] + dx / distance * step,
        game.player.position[1],
        game.player.position[2] + dz / distance * step,
    )
    return waypoint_probe_safe(game, target, wall_clearance=36.0)


async def _interact_with_door(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    """Door-specific servo: face the door, center camera, approach straight, then press A."""
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    actor = _matching_actor(before, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
    if not _is_door_actor(actor):
        return {"status": "failed", "reason": "target_is_not_door", "skill": decision.skill}

    start_position = before.player.position
    deadline = time.monotonic() + min(8.0, max(2.0, decision.args.duration_ms / 1000))
    acknowledged = False

    # A room-global door can be observed even when walls stand between Link and
    # it. Use the collision-aware NavMesh for the long approach, then hand over
    # to the precise face/center/straight door servo only for the final meters.
    if actor.distance > 150.0 and "local_navmesh" in before.capabilities:
        approach_ms = min(3500, max(800, decision.args.duration_ms // 2))
        approach_args = decision.args.model_copy(update={
            "duration_ms": approach_ms,
            "stop_distance": 135.0,
        })
        approach_decision = decision.model_copy(update={"skill": "approach_actor", "args": approach_args})
        approach = await _navigate_local(bridge, approach_decision, observation, actor_mode=True)
        acknowledged |= bool(approach.get("acknowledged"))
        current = bridge.state
        actor = (_matching_actor(current, decision.args.target_actor_id,
                    decision.args.target_actor_params, decision.args.target_actor_uid)
                 if current else None)
        if approach.get("status") != "completed":
            return {"status": approach.get("status", "failed"),
                "reason": f"door_approach_failed:{approach.get('reason', 'unknown')}",
                "distance": math.dist(start_position, current.player.position)
                    if current and current.player else None,
                "acknowledged": acknowledged, "skill": decision.skill}
        if actor is None:
            return {"status": "failed", "reason": "door_actor_lost_after_approach",
                "distance": math.dist(start_position, current.player.position)
                    if current and current.player else None,
                "acknowledged": acknowledged, "skill": decision.skill}
        before = current or before

    attempts = 0
    best_distance = actor.distance
    turn_sign = 1
    turn_flipped = False
    previous_yaw_error = None

    try:
        while time.monotonic() < deadline and attempts < 8:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "completed", "reason": "world_transition_after_interaction",
                    "distance": math.dist(start_position, current.player.position) if current.player else None,
                    "acknowledged": acknowledged, "skill": decision.skill}
            if not current.in_game or not current.player:
                return {"status": "interrupted", "reason": "game_not_ready", "skill": decision.skill}
            if current.paused or current.dialogue.active or current.cutscene_active or current.game_over_state != 0:
                return {"status": "completed" if (current.dialogue.active or current.cutscene_active) else "interrupted",
                    "reason": "interaction_started" if (current.dialogue.active or current.cutscene_active)
                        else "gameplay_state_changed",
                    "acknowledged": acknowledged, "skill": decision.skill}

            actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
            if actor is None:
                return {"status": "failed", "reason": "door_actor_not_observed", "skill": decision.skill}

            before_event_ids = {event.id for event in current.events}
            context_matches = bool(current.context_actor and
                current.context_actor.actor_id == actor.actor_id and
                (decision.args.target_actor_params is None or current.context_actor.params == actor.params))
            if current.context_action.label in {"open", "enter"} and context_matches:
                acknowledged |= await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=120, settle_s=0.05)
                evidence, extra = await _wait_interaction_evidence(
                    bridge, observation, before_event_ids, timeout_s=1.6)
                if evidence:
                    return {"status": "completed", "reason": evidence, "target_distance": actor.distance,
                        "distance": math.dist(start_position, (bridge.state or current).player.position)
                            if (bridge.state or current).player else None,
                        "acknowledged": acknowledged, "skill": decision.skill, **extra}

            # Do not use camera-relative orbiting near a wall-mounted target. First make Link face it
            # using native yaw feedback, then put the camera behind Link and advance on a straight line.
            desired_yaw = _yaw_to_target(current.player.position, actor.position)
            yaw_error = _yaw_error_units(current.player.yaw, desired_yaw)
            abs_yaw_error = abs(yaw_error)
            if (previous_yaw_error is not None and abs_yaw_error > previous_yaw_error + 350 and not turn_flipped):
                # Camera/control sign can be inverted by the current view. Calibrate once from observed yaw feedback.
                turn_sign *= -1
                turn_flipped = True
            previous_yaw_error = abs_yaw_error

            if abs_yaw_error > 1200:
                magnitude = max(32, min(66, round(abs_yaw_error / 260)))
                stick_x = magnitude * (1 if yaw_error > 0 else -1) * turn_sign
                command_id = bridge.send(stick_x=stick_x, stick_y=8, lease_ms=180)
                await asyncio.sleep(0.18)
                bridge.release()
                await asyncio.sleep(0.06)
                sample = bridge.state
                acknowledged |= bool(sample and consumed(bridge, command_id))
                attempts += 1
                continue

            acknowledged |= await _pulse(bridge, buttons=BUTTONS["Z"], hold_ms=90, settle_s=0.07)
            current = bridge.state or current
            actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid) or actor

            if actor.distance <= 115.0:
                # A slight forward bias helps the engine establish doorActor/context action on the final step,
                # but never cross an unvalidated wall on the way there.
                if not _door_step_safe(current, actor, 20.0):
                    return {"status": "failed", "reason": "door_final_path_blocked",
                        "target_distance": actor.distance,
                        "distance": math.dist(start_position, current.player.position),
                        "acknowledged": acknowledged, "skill": decision.skill}
                command_id = bridge.send(buttons=BUTTONS["A"], stick_y=24, lease_ms=180)
                await asyncio.sleep(0.18)
                bridge.release()
                await asyncio.sleep(0.05)
                sample = bridge.state
                acknowledged |= bool(sample and consumed(bridge, command_id))
                evidence, extra = await _wait_interaction_evidence(
                    bridge, observation, before_event_ids, timeout_s=1.4)
                if evidence:
                    return {"status": "completed", "reason": evidence, "target_distance": actor.distance,
                        "distance": math.dist(start_position, (bridge.state or current).player.position)
                            if (bridge.state or current).player else None,
                        "acknowledged": acknowledged, "skill": decision.skill, **extra}

            # Straight approach after Z-centering; validate a short world-space step first so
            # an intervening wall cannot be mistaken for the closed door at the target.
            if not _door_step_safe(current, actor):
                return {"status": "failed", "reason": "door_final_path_blocked",
                    "target_distance": actor.distance,
                    "distance": math.dist(start_position, current.player.position),
                    "acknowledged": acknowledged, "skill": decision.skill}
            hold_ms = 220 if actor.distance > 180 else 140
            command_id = bridge.send(stick_x=0, stick_y=56, lease_ms=hold_ms)
            await asyncio.sleep(hold_ms / 1000)
            bridge.release()
            await asyncio.sleep(0.06)
            sample = bridge.state
            if sample:
                acknowledged |= consumed(bridge, command_id)
                updated = _matching_actor(sample, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
                if updated:
                    if updated.distance + 3.0 < best_distance:
                        best_distance = updated.distance
                        previous_yaw_error = None
                        turn_flipped = False
                        turn_sign = 1
                    elif updated.distance >= best_distance - 1.0 and attempts >= 3:
                        # Create a little clearance and retry the face/center/straight sequence.
                        recovery = await _backtrack_recovery(bridge, sample, attempts)
                        acknowledged |= recovery["acknowledged"]
            attempts += 1
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "failed", "reason": "door_interaction_no_transition",
        "target_distance": (_matching_actor(after, decision.args.target_actor_id,
            decision.args.target_actor_params, decision.args.target_actor_uid).distance if after and
            _matching_actor(after, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid) else None),
        "distance": math.dist(start_position, after.player.position) if after and after.player else None,
        "attempts": attempts, "acknowledged": acknowledged, "skill": decision.skill}


async def _manipulate_object(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    actor = _matching_actor(before, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
    if actor is None:
        return {"status": "failed", "reason": "target_actor_not_observed", "skill": decision.skill}

    approach = await _navigate_local(bridge, decision, observation, actor_mode=True)
    if approach.get("status") != "completed":
        return {"status": approach.get("status", "failed"),
            "reason": f"approach_failed:{approach.get('reason', 'unknown')}",
            "skill": decision.skill}

    current = bridge.state
    if not current or not current.player:
        return {"status": "failed", "reason": "state_lost_after_approach", "skill": decision.skill}
    actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
    if actor is None:
        return {"status": "failed", "reason": "target_actor_lost_after_approach", "skill": decision.skill}

    start_actor_position = actor.position
    start_player_position = current.player.position
    before_events = {event.id for event in current.events}
    start_epoch = current.scene_epoch
    start_context = current.context_action.label
    acknowledged = await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=180, settle_s=0.10)
    first_move_command = None
    direction_y = 62 if decision.args.direction == "forward" else -62
    deadline = time.monotonic() + min(4.0, max(0.6, decision.args.duration_ms / 1000))

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not current.player:
                return {"status": "interrupted", "reason": "game_state_lost", "skill": decision.skill}
            if current.scene_epoch != start_epoch:
                return {"status": "completed", "reason": "world_transition_after_manipulation",
                    "acknowledged": acknowledged, "skill": decision.skill}
            if current.dialogue.active or current.cutscene_active:
                return {"status": "completed", "reason": "interaction_started",
                    "acknowledged": acknowledged, "skill": decision.skill}

            outcome = next((event for event in current.events
                if event.id not in before_events and event.kind in {
                    "item_received", "scene_flag_set", "scene_flag_unset"}), None)
            if outcome is not None:
                return {"status": "completed", "reason": outcome.kind, "detail": outcome.detail,
                    "acknowledged": acknowledged, "skill": decision.skill}

            if first_move_command is not None:
                acknowledged |= consumed(bridge, first_move_command)
            actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
            if actor is not None:
                displacement = math.dist(start_actor_position, actor.position)
                if displacement >= 8.0:
                    return {"status": "completed", "reason": "object_displaced",
                        "object_distance": displacement,
                        "player_distance": math.dist(start_player_position, current.player.position),
                        "acknowledged": acknowledged, "skill": decision.skill}
            if current.context_action.label in {"throw", "drop"} and current.context_action.label != start_context:
                return {"status": "completed", "reason": "object_grabbed",
                    "context_action": current.context_action.label,
                    "acknowledged": acknowledged, "skill": decision.skill}

            command_id = bridge.send(buttons=BUTTONS["A"], stick_y=direction_y, lease_ms=260)
            if first_move_command is None:
                first_move_command = command_id
            await asyncio.sleep(0.12)
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "failed", "reason": "object_manipulation_unconfirmed",
        "player_distance": math.dist(start_player_position, after.player.position)
            if after and after.player else None,
        "acknowledged": acknowledged, "skill": decision.skill}
