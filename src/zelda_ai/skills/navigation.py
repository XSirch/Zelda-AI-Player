"""Local feedback steering and bounded recovery; not a global navmesh."""
from __future__ import annotations

import asyncio
import math
import time
from ..bridge import Bridge
from ..control.feedback import consumed, feedback, is_realtime
from ..models import Decision, GameState
from ..navmesh import plan_navmesh, waypoint_probe_safe
from .catalog import BUTTONS
from .common import _matching_actor, _pulse, _steer_to, _target_position, _yaw_error_units, _yaw_to_target

def _recovery_inputs(attempt: int) -> list[tuple[int, int, int, int]]:
    """Deterministic wall escape: create clearance backwards before attempting another heading."""
    side = 1 if attempt % 2 == 0 else -1
    return [
        (0, 0, -62, 380),                 # Back away from the obstacle first.
        (0, 52 * side, -36, 300),         # Continue reversing on an arc to create turning room.
        (0, 68 * side, 8, 280),           # Rotate toward a different heading without charging forward.
        (BUTTONS["Z"], 0, 0, 100),        # Recenter camera after the escape.
    ]


def _safe_escape_target(game: GameState, attempt: int) -> tuple[float, float, float] | None:
    """A conservative local retreat, not a navmesh or a general path-safety proof."""
    if not game.player:
        return None
    preferred = ["back", "back_right", "back_left"] if attempt % 2 == 0 else ["back", "back_left", "back_right"]
    offsets = ({"back": 32768, "back_right": -24576, "back_left": 24576}
        if "probe_yaw_v2" in game.capabilities
        else {"back": 32768, "back_right": 24576, "back_left": -24576})
    for direction in preferred:
        probes = [p for p in game.navigation_probes if p.direction == direction and p.distance <= 75]
        probe = min(probes, key=lambda p: p.distance, default=None)
        if (not probe or not probe.floor_found or probe.delta_y is None or
                abs(probe.delta_y) > 20 or probe.wall_hit):
            continue
        # Step less than the probed radius and recheck every new state. No blind reverse arc.
        yaw = (game.player.yaw + offsets[direction]) * math.pi / 32768
        x, y, z = game.player.position
        return x + math.sin(yaw)*25, y, z + math.cos(yaw)*25
    return None


async def _backtrack_recovery(bridge: Bridge, observation: GameState, attempt: int = 0) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"acknowledged": False, "distance": 0.0, "yaw_delta": 0, "world_changed": False}
    start_position = before.player.position
    start_yaw = before.player.yaw
    first_command = None
    acknowledged = False

    if is_realtime(bridge):
        deadline = time.monotonic() + .8
        try:
            while time.monotonic() < deadline:
                current = bridge.state
                if (not current or not current.player or not current.in_game or current.paused or
                        current.dialogue.active or current.cutscene_active or current.game_over_state):
                    break
                if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                    return {"acknowledged": acknowledged, "distance": 0.0, "yaw_delta": 0, "world_changed": True}
                target = _safe_escape_target(current, attempt)
                if target is None:
                    break
                x, y, _ = _steer_to(current, target, .45)
                command = bridge.send(stick_x=x, stick_y=y, lease_ms=150)
                sample = await feedback(bridge, current)
                acknowledged |= consumed(bridge, command)
                if sample and sample.player and math.dist(start_position, sample.player.position) >= 20:
                    break
        finally:
            bridge.release()
        after = bridge.state
        return {"acknowledged": acknowledged,
            "distance": math.dist(start_position, after.player.position) if after and after.player else 0.0,
            "yaw_delta": ((after.player.yaw-start_yaw+32768)%65536)-32768 if after and after.player else 0,
            "world_changed": bool(after and after.scene_epoch != observation.scene_epoch)}

    try:
        for buttons, stick_x, stick_y, hold_ms in _recovery_inputs(attempt):
            current = bridge.state
            if not current or not bridge.connected:
                break
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"acknowledged": acknowledged, "distance": 0.0, "yaw_delta": 0,
                    "world_changed": True}
            if (not current.in_game or not current.player or current.paused or current.dialogue.active or
                    current.cutscene_active or current.game_over_state != 0):
                break
            command_id = bridge.send(buttons=buttons, stick_x=stick_x, stick_y=stick_y,
                lease_ms=max(50, min(500, hold_ms)))
            if first_command is None:
                first_command = command_id
            await asyncio.sleep(hold_ms / 1000)
            bridge.release()
            await asyncio.sleep(0.05)
            sample = bridge.state
            if sample and first_command is not None:
                acknowledged |= consumed(bridge, first_command)
    finally:
        bridge.release()

    after = bridge.state
    if not after or not after.player:
        return {"acknowledged": acknowledged, "distance": 0.0, "yaw_delta": 0, "world_changed": False}
    if (after.instance_id, after.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"acknowledged": acknowledged, "distance": 0.0, "yaw_delta": 0, "world_changed": True}
    return {"acknowledged": acknowledged,
        "distance": math.dist(start_position, after.player.position),
        "yaw_delta": ((after.player.yaw - start_yaw + 32768) % 65536) - 32768,
        "world_changed": False}


def _resolve_scene_exit(game: GameState, target_position: list[float] | tuple[float, float, float] | None):
    """Match a model-selected coordinate back to an exit surface observed by the engine."""
    if not game.scene_exits or target_position is None:
        return None
    target = tuple(float(v) for v in target_position)
    nearest = min(game.scene_exits, key=lambda row: math.dist(row.position, target))
    tolerance = max(120.0, (game.navmesh.step * 2.0) if game.navmesh.available else 120.0)
    return nearest if math.dist(nearest.position, target) <= tolerance else None


def _gap_target_score(point: tuple[float, float, float],
                      target: tuple[float, float, float] | list[float]) -> float:
    return (math.hypot(point[0] - float(target[0]), point[2] - float(target[2])) +
            abs(point[1] - float(target[1])) * 0.5)


def _best_gap_link(game: GameState, target: tuple[float, float, float] | list[float]):
    """Pick a reachable off-mesh link that makes meaningful target progress."""
    if not game.player or not game.navigation_links:
        return None
    current_score = _gap_target_score(game.player.position, target)
    step = game.navmesh.step if game.navmesh.available else 70.0
    candidates = []
    for link in game.navigation_links:
        landing_score = _gap_target_score(link.landing_position, target)
        if landing_score >= current_score - max(25.0, step * 0.35):
            continue

        takeoff_xz = math.hypot(
            link.takeoff_position[0] - game.player.position[0],
            link.takeoff_position[2] - game.player.position[2],
        )
        takeoff_y = abs(link.takeoff_position[1] - game.player.position[1])
        if takeoff_xz <= 45.0 and takeoff_y <= 24.0:
            approach_cost = 0.0
        elif game.navmesh.available:
            plan = plan_navmesh(game, link.takeoff_position)
            if not plan or not plan.exact_goal_reachable:
                continue
            if plan.target_distance > max(75.0, step * 1.1):
                continue
            approach_cost = plan.cost + plan.target_distance
        else:
            continue

        # Prefer short, level gaps and meaningful progress toward the real target.
        score = (approach_cost + link.gap_distance * 1.5 +
                 abs(link.height_delta) * 0.5 + landing_score)
        candidates.append((score, link))
    return min(candidates, key=lambda row: row[0])[1] if candidates else None


def _refresh_gap_link(game: GameState, original):
    if not game.navigation_links:
        return None
    candidates = [row for row in game.navigation_links if row.kind == original.kind]
    if not candidates:
        return None
    best = min(candidates, key=lambda row: (
        math.dist(row.takeoff_position, original.takeoff_position) +
        math.dist(row.landing_position, original.landing_position)))
    if (math.dist(best.takeoff_position, original.takeoff_position) <= 80.0 and
            math.dist(best.landing_position, original.landing_position) <= 100.0):
        return best
    return None


async def _execute_gap_link(bridge: Bridge, observation: GameState, original, skill: str) -> dict:
    """Run toward a native-observed landing and let OoT trigger its own auto-jump."""
    current = bridge.state
    if not current or not current.player:
        return {"status": "failed", "reason": "player_state_unavailable",
            "acknowledged": False}
    link = _refresh_gap_link(current, original)
    if link is None:
        return {"status": "failed", "reason": "offmesh_link_lost",
            "acknowledged": False}

    takeoff_distance = math.hypot(
        link.takeoff_position[0] - current.player.position[0],
        link.takeoff_position[2] - current.player.position[2],
    )
    if takeoff_distance > 85.0 or abs(link.takeoff_position[1] - current.player.position[1]) > 30.0:
        return {"status": "failed", "reason": "offmesh_takeoff_not_reached",
            "acknowledged": False}

    start = current.player.position
    start_health = current.player.health
    seen_jump = False
    seen_air = False
    acknowledged = False
    first_command = None
    deadline = time.monotonic() + 2.4

    bridge.set_navigation_debug(
        skill=skill, status="gap_link_takeoff",
        target_position=list(link.landing_position),
        waypoint=list(link.takeoff_position),
        path_cells=0, probe_safe=True, navmesh_used=True,
        target_distance=_gap_target_score(current.player.position, link.landing_position),
        offmesh_kind=link.kind, offmesh_gap=link.gap_distance)

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not current.player or not bridge.connected:
                return {"status": "failed", "reason": "bridge_disconnected",
                    "acknowledged": acknowledged}
            if (current.instance_id, current.scene_epoch) != (
                    observation.instance_id, observation.scene_epoch):
                return {"status": "interrupted", "reason": "world_changed",
                    "acknowledged": acknowledged}
            if (current.paused or current.dialogue.active or current.cutscene_active or
                    current.game_over_state or not current.in_game):
                return {"status": "interrupted", "reason": "game_not_ready",
                    "acknowledged": acknowledged}

            jumping = bool(current.player.state_flags_1 & (1 << 18))
            freefall = bool(current.player.state_flags_1 & (1 << 19))
            seen_jump |= jumping
            seen_air |= jumping or freefall

            landing_xz = math.hypot(
                current.player.position[0] - link.landing_position[0],
                current.player.position[2] - link.landing_position[2],
            )
            landing_y = abs(current.player.position[1] - link.landing_position[1])
            if seen_jump and landing_xz <= 45.0 and landing_y <= 45.0 and not (jumping or freefall):
                bridge.set_navigation_debug(
                    skill=skill, status="gap_link_landed",
                    target_position=list(link.landing_position), waypoint=None,
                    path_cells=0, probe_safe=True, navmesh_used=True,
                    target_distance=landing_xz, offmesh_kind=link.kind,
                    offmesh_gap=link.gap_distance)
                return {"status": "completed", "reason": "offmesh_jump_landed",
                    "acknowledged": acknowledged,
                    "distance": math.dist(start, current.player.position),
                    "health_lost": max(0, start_health - current.player.health)}

            if current.player.position[1] < link.landing_position[1] - 110.0:
                return {"status": "failed", "reason": "offmesh_jump_fell",
                    "acknowledged": acknowledged,
                    "distance": math.dist(start, current.player.position)}

            stick_x, stick_y, _ = _steer_to(current, link.landing_position, 1.0)
            command = bridge.send(stick_x=stick_x, stick_y=stick_y, lease_ms=180)
            if first_command is None:
                first_command = command
            sample = await feedback(bridge, current, .10)
            acknowledged |= consumed(bridge, command)
            if sample and sample.player:
                jumping_sample = bool(sample.player.state_flags_1 & (1 << 18))
                freefall_sample = bool(sample.player.state_flags_1 & (1 << 19))
                seen_jump |= jumping_sample
                seen_air |= jumping_sample or freefall_sample

            if (not seen_air and time.monotonic() > deadline - 1.1 and
                    math.dist(start, current.player.position) < 20.0):
                return {"status": "failed", "reason": "offmesh_jump_not_triggered",
                    "acknowledged": acknowledged}
    finally:
        bridge.release()

    return {"status": "failed",
        "reason": "offmesh_jump_timeout" if seen_air else "offmesh_jump_not_triggered",
        "acknowledged": acknowledged,
        "distance": math.dist(start, bridge.state.player.position)
            if bridge.state and bridge.state.player else None}


async def _navigate_local(bridge: Bridge, decision: Decision, observation: GameState,
                          *, actor_mode: bool, talk: bool = False, interact: bool = False,
                          exit_mode: bool = False) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}

    if exit_mode:
        if "scene_exit_surfaces" not in before.capabilities or "local_navmesh" not in before.capabilities:
            return {"status": "failed", "reason": "scene_exit_bridge_upgrade_required",
                "skill": decision.skill}
        selected_exit = _resolve_scene_exit(before, decision.args.target_position)
        if selected_exit is None:
            return {"status": "failed", "reason": "scene_exit_not_observed",
                "skill": decision.skill}
        selected_exit_index = selected_exit.exit_index
        selected_entrance_index = selected_exit.entrance_index
    else:
        selected_exit = None
        selected_exit_index = None
        selected_entrance_index = None

    deadline = time.monotonic() + min(8.0, max(0.25, decision.args.duration_ms / 1000))
    start_position = before.player.position
    stop_distance = 4.0 if exit_mode else decision.args.stop_distance
    interaction_mode = talk or interact
    if stop_distance is None:
        stop_distance = 90.0 if interaction_mode else (70.0 if actor_mode else 35.0)

    first_command = None
    acknowledged = False
    best_distance = float("inf")
    stagnant_samples = 0
    recentered = False
    recovery_attempts = 0
    last_actor = None
    last_progress_seq = -1
    last_progress_at = time.monotonic()
    progress_mode = "target"
    navmesh_blocked_samples = 0
    navmesh_used = False
    last_path_cells = 0

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "completed" if (interaction_mode or exit_mode) else "interrupted",
                    "reason": ("scene_exit_traversed" if exit_mode else
                        ("world_transition_after_interaction" if interaction_mode else "world_changed")),
                    "exit_index": selected_exit_index, "entrance_index": selected_entrance_index,
                    "skill": decision.skill}
            if not current.in_game or not current.player:
                return {"status": "interrupted", "reason": "game_not_ready", "skill": decision.skill}
            if current.paused:
                return {"status": "interrupted", "reason": "pause_menu_opened", "skill": decision.skill}
            if current.dialogue.active:
                return {"status": "completed" if interaction_mode else "interrupted",
                    "reason": "dialogue_opened", "skill": decision.skill,
                    "target_distance": last_actor.distance if last_actor else None}
            if current.cutscene_active:
                return {"status": "completed" if (interaction_mode or exit_mode) else "interrupted",
                    "reason": ("scene_exit_transition_started" if exit_mode else
                        ("interaction_started" if interaction_mode else "cutscene_started")),
                    "exit_index": selected_exit_index, "entrance_index": selected_entrance_index,
                    "skill": decision.skill}

            if exit_mode and current.player.floor_exit_index == selected_exit_index:
                # The engine now reports the selected SceneExitIndex under Link's
                # actual floor polygon. Stop steering and give
                # Player_HandleExitsAndVoids a frame to start the transition.
                bridge.set_navigation_debug(
                    skill=decision.skill, status="exit_surface_contact",
                    target_position=list(selected_exit.position) if selected_exit else None,
                    waypoint=None, path_cells=last_path_cells, probe_safe=True,
                    navmesh_used=navmesh_used, target_distance=0.0,
                    exit_index=selected_exit_index,
                    entrance_index=selected_entrance_index)
                bridge.release()
                await feedback(bridge, current, .12)
                continue

            if actor_mode:
                actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
                if actor is None:
                    return {"status": "failed", "reason": "target_actor_not_observed", "skill": decision.skill}
                last_actor = actor
                target = actor.position
            elif exit_mode:
                exit_row = next((row for row in current.scene_exits
                    if row.exit_index == selected_exit_index), None)
                if exit_row is None:
                    return {"status": "failed", "reason": "scene_exit_lost",
                        "exit_index": selected_exit_index, "skill": decision.skill}
                selected_exit = exit_row
                target = exit_row.position
            else:
                target = decision.args.target_position

            _, _, target_distance = _steer_to(current, target, decision.args.strength)
            direct_exit_proven = False
            if exit_mode and selected_exit and current.navmesh.available:
                px, _, pz = current.player.position
                ox, _, oz = current.navmesh.origin
                proof_age_distance = math.hypot(px - ox, pz - oz)
                direct_exit_proven = bool(
                    selected_exit.direct_reachable and proof_age_distance <= 18.0)
            if target_distance <= stop_distance and abs(current.player.position[1] - target[1]) > 45:
                return {"status": "failed", "reason": "target_on_different_floor",
                    "target_distance": target_distance, "skill": decision.skill}
            if target_distance <= stop_distance:
                bridge.set_navigation_debug(
                    skill=decision.skill, status="reached", target_position=list(target),
                    waypoint=None, path_cells=last_path_cells, probe_safe=True,
                    navmesh_used=navmesh_used, target_distance=target_distance)
                if not interaction_mode and not exit_mode:
                    return {"status": "completed", "reason": "target_reached",
                        "target_distance": target_distance,
                        "distance": math.dist(start_position, current.player.position),
                        "navmesh_used": navmesh_used, "path_cells": last_path_cells,
                        "acknowledged": acknowledged, "skill": decision.skill}
                if exit_mode:
                    # The coordinate is guaranteed to lie on a collision polygon whose
                    # SceneExitIndex is non-zero. Do not stop short: crossing onto that
                    # floor polygon is what makes Player_HandleExitsAndVoids transition.
                    bridge.set_navigation_debug(
                        skill=decision.skill, status="entering_exit_surface",
                        target_position=list(target), waypoint=list(target),
                        path_cells=last_path_cells, probe_safe=True, navmesh_used=True,
                        target_distance=target_distance, exit_index=selected_exit_index,
                        entrance_index=selected_entrance_index)

                if not interaction_mode:
                    # Exit mode continues through the normal NavMesh steering below.
                    pass
                else:
                    before_event_ids = {event.id for event in current.events}
                    before_epoch = current.scene_epoch
                    acknowledged |= await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=90, settle_s=0.12)
                    interaction_deadline = time.monotonic() + 1.2
                    while time.monotonic() < interaction_deadline:
                        sample = bridge.state
                        if not sample:
                            break
                        if sample.scene_epoch != before_epoch:
                            return {"status": "completed", "reason": "world_transition_after_interaction",
                                "target_distance": target_distance, "acknowledged": acknowledged,
                                "skill": decision.skill}
                        if sample.dialogue.active:
                            return {"status": "completed", "reason": "dialogue_opened",
                                "target_distance": target_distance,
                                "distance": math.dist(start_position, sample.player.position) if sample.player else None,
                                "acknowledged": acknowledged, "skill": decision.skill}
                        if sample.cutscene_active:
                            return {"status": "completed", "reason": "interaction_started",
                                "target_distance": target_distance, "acknowledged": acknowledged,
                                "skill": decision.skill}
                        outcome = next((event for event in sample.events
                            if event.id not in before_event_ids and event.kind in {
                                "item_received", "scene_flag_set", "scene_flag_unset"}), None)
                        if outcome is not None:
                            return {"status": "completed", "reason": outcome.kind,
                                "detail": outcome.detail, "target_distance": target_distance,
                                "acknowledged": acknowledged, "skill": decision.skill}
                        await asyncio.sleep(0.08)
                    return {"status": "failed",
                        "reason": "talk_interaction_not_started" if talk else "interaction_unconfirmed",
                        "target_distance": target_distance, "acknowledged": acknowledged,
                        "skill": decision.skill}

            steer_target = target
            progress_distance = target_distance
            next_progress_mode = "target"
            gap_link = None
            gap_approach = False
            if "local_navmesh" in current.capabilities:
                plan = plan_navmesh(current, target)

                if (not exit_mode and "offmesh_jump_links_v1" in current.capabilities and
                        (plan is None or not plan.exact_goal_reachable)):
                    gap_link = _best_gap_link(current, target)
                    if gap_link is not None:
                        takeoff_distance = math.hypot(
                            gap_link.takeoff_position[0] - current.player.position[0],
                            gap_link.takeoff_position[2] - current.player.position[2],
                        )
                        takeoff_height = abs(
                            gap_link.takeoff_position[1] - current.player.position[1])
                        if takeoff_distance <= 70.0 and takeoff_height <= 30.0:
                            jump = await _execute_gap_link(
                                bridge, observation, gap_link, decision.skill)
                            acknowledged |= bool(jump.get("acknowledged"))
                            if jump.get("status") == "completed":
                                navmesh_blocked_samples = 0
                                best_distance = float("inf")
                                progress_mode = "target"
                                last_progress_at = time.monotonic()
                                continue
                            return {
                                **jump,
                                "skill": decision.skill,
                                "target_distance": target_distance,
                                "navmesh_used": True,
                                "offmesh_kind": gap_link.kind,
                            }

                        approach_plan = plan_navmesh(current, gap_link.takeoff_position)
                        if approach_plan and approach_plan.exact_goal_reachable:
                            plan = approach_plan
                            gap_approach = True
                            next_progress_mode = "gap_takeoff"
                            progress_distance = takeoff_distance

                if progress_mode != next_progress_mode:
                    progress_mode = next_progress_mode
                    best_distance = float("inf")
                    stagnant_samples = 0
                    last_progress_at = time.monotonic()

                if plan is None:
                    final_exit_direct = bool(
                        exit_mode and direct_exit_proven and target_distance <= 70.0 and
                        waypoint_probe_safe(current, target, wall_clearance=30.0,
                            known_floor_target=True))
                    if final_exit_direct:
                        navmesh_used = True
                        last_path_cells = 1
                        steer_target = target
                        navmesh_blocked_samples = 0
                        bridge.set_navigation_debug(
                            skill=decision.skill, status="entering_exit_surface",
                            target_position=list(target), waypoint=list(target),
                            path_cells=1, probe_safe=True, navmesh_used=True,
                            target_distance=target_distance,
                            exit_index=selected_exit_index,
                            entrance_index=selected_entrance_index)
                    else:
                        bridge.set_navigation_debug(
                            skill=decision.skill, status="no_path", target_position=list(target),
                            waypoint=None, path_cells=0, probe_safe=False, navmesh_used=True,
                            target_distance=target_distance,
                            exit_index=selected_exit_index if exit_mode else None,
                            entrance_index=selected_entrance_index if exit_mode else None)
                        navmesh_blocked_samples += 1
                        if navmesh_blocked_samples >= 5:
                            return {"status": "failed", "reason": "navigation_no_path",
                                "target_distance": target_distance,
                                "distance": math.dist(start_position, current.player.position),
                                "navmesh_used": True, "acknowledged": acknowledged,
                                "skill": decision.skill}
                        bridge.release()
                        await feedback(bridge, current, .10)
                        continue
                else:
                    navmesh_used = True
                    last_path_cells = len(plan.path)
                    steer_target = plan.waypoint
                    final_exit_waypoint = bool(
                        exit_mode and direct_exit_proven and
                        math.dist(tuple(steer_target), tuple(target)) <= 1e-3)
                    final_takeoff_waypoint = bool(
                        gap_approach and gap_link is not None and
                        math.dist(tuple(steer_target), tuple(gap_link.takeoff_position)) <= 1e-3)
                    probe_safe = waypoint_probe_safe(
                        current, steer_target,
                        known_floor_target=(final_exit_waypoint or final_takeoff_waypoint))
                    debug_status = (
                        "gap_link_approach" if gap_approach else
                        ("active" if probe_safe else "blocked"))
                    bridge.set_navigation_debug(
                        skill=decision.skill, status=debug_status if probe_safe else "blocked",
                        target_position=list(target), waypoint=list(plan.waypoint),
                        path_cells=last_path_cells, probe_safe=probe_safe, navmesh_used=True,
                        target_distance=target_distance, plan_target_distance=plan.target_distance,
                        plan_cost=round(plan.cost, 2), exact_goal_reachable=plan.exact_goal_reachable,
                        exit_index=selected_exit_index if exit_mode else None,
                        entrance_index=selected_entrance_index if exit_mode else None,
                        offmesh_kind=gap_link.kind if gap_link else None,
                        offmesh_takeoff=list(gap_link.takeoff_position) if gap_link else None,
                        offmesh_landing=list(gap_link.landing_position) if gap_link else None)
                    if not probe_safe:
                        navmesh_blocked_samples += 1
                        if navmesh_blocked_samples >= 5:
                            return {"status": "failed", "reason": "navigation_path_blocked",
                                "target_distance": target_distance,
                                "distance": math.dist(start_position, current.player.position),
                                "path_cells": last_path_cells, "navmesh_used": True,
                                "acknowledged": acknowledged, "skill": decision.skill}
                        bridge.release()
                        await feedback(bridge, current, .10)
                        continue
                    navmesh_blocked_samples = 0

            if "local_navmesh" not in current.capabilities:
                bridge.set_navigation_debug(
                    skill=decision.skill, status="legacy_steering", target_position=list(target),
                    waypoint=list(steer_target), path_cells=0, probe_safe=None,
                    navmesh_used=False, target_distance=target_distance)
            stick_x, stick_y, _ = _steer_to(current, steer_target, decision.args.strength)

            if current.seq != last_progress_seq:
                last_progress_seq = current.seq
                if progress_distance + 2.0 < best_distance:
                    progress_gain = best_distance - progress_distance
                    best_distance = progress_distance
                    stagnant_samples = 0
                    last_progress_at = time.monotonic()
                    if math.isfinite(progress_gain) and progress_gain >= 12.0:
                        recovery_attempts = 0
                        recentered = False
                else:
                    stagnant_samples += 1

            if is_realtime(bridge):
                # Preserve the original ~5 Hz windows without counting fast samples as failures.
                stagnant_samples = int((time.monotonic() - last_progress_at) / .2)
            wall_contact = bool(current.player.bg_check_flags & 0x008)
            obstructed = (wall_contact and stagnant_samples >= 2) or stagnant_samples >= 4
            if obstructed and recovery_attempts < 2:
                # Back up first. A sidestep alone often keeps Link pinned against an interior wall/corner.
                recovery = await _backtrack_recovery(bridge, current, recovery_attempts)
                acknowledged |= recovery["acknowledged"]
                recovery_attempts += 1
                stagnant_samples = 0
                last_progress_at = time.monotonic()
                best_distance = float("inf")
                recentered = False
                if recovery["world_changed"]:
                    return {"status": "interrupted", "reason": "world_changed_during_recovery",
                        "acknowledged": acknowledged, "skill": decision.skill}
                continue
            if stagnant_samples >= 5 and not recentered:
                bridge.release()
                acknowledged |= await _pulse(bridge, buttons=BUTTONS["Z"], hold_ms=80, settle_s=0.15)
                recentered = True
                stagnant_samples = 0
                last_progress_at = time.monotonic()
                continue
            if stagnant_samples >= 7:
                return {"status": "failed", "reason": "navigation_no_progress",
                    "target_distance": target_distance,
                    "distance": math.dist(start_position, current.player.position),
                    "recovery_attempts": recovery_attempts, "navmesh_used": navmesh_used,
                    "path_cells": last_path_cells,
                    "acknowledged": acknowledged, "skill": decision.skill}

            command_id = bridge.send(stick_x=stick_x, stick_y=stick_y, lease_ms=250)
            if first_command is None:
                first_command = command_id
            acknowledged |= consumed(bridge, first_command)
            await feedback(bridge, current, .10)
    finally:
        bridge.release()

    after = bridge.state
    if after and first_command is not None:
        acknowledged |= consumed(bridge, first_command)
    bridge.set_navigation_debug(
        skill=decision.skill, status="timeout",
        target_position=list(last_actor.position) if last_actor else (
            list(decision.args.target_position) if decision.args.target_position else None),
        waypoint=None, path_cells=last_path_cells, probe_safe=None,
        navmesh_used=navmesh_used,
        target_distance=(last_actor.distance if last_actor else None))
    return {"status": "failed", "reason": "scene_exit_not_triggered" if exit_mode else "navigation_timeout",
        "exit_index": selected_exit_index, "entrance_index": selected_entrance_index,
        "target_distance": (last_actor.distance if last_actor else None),
        "distance": math.dist(start_position, after.player.position) if after and after.player else None,
        "navmesh_used": navmesh_used, "path_cells": last_path_cells,
        "acknowledged": acknowledged, "skill": decision.skill}


async def _face_target(bridge: Bridge, decision: Decision, observation: GameState,
                       *, shield: bool = False) -> dict:
    current = bridge.state
    if not current or not current.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    if current.paused or current.dialogue.active or current.cutscene_active or current.game_over_state != 0:
        return {"status": "stale", "reason": "gameplay_state_blocks_facing", "skill": decision.skill}

    deadline = time.monotonic() + min(6.0, max(0.4, decision.args.duration_ms / 1000))
    sign = 1
    flipped = False
    previous_abs = None
    previous_stick = 0
    stable = 0
    first_command = None
    acknowledged = False
    last_error = None

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not current.player:
                return {"status": "interrupted", "reason": "game_state_lost", "skill": decision.skill}
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
            target = _target_position(current, decision)
            if target is None:
                return {"status": "failed", "reason": "face_target_not_observed", "skill": decision.skill}

            desired = _yaw_to_target(current.player.position, target)
            error = _yaw_error_units(current.player.yaw, desired)
            last_error = error
            abs_error = abs(error)
            if first_command is not None:
                acknowledged |= consumed(bridge, first_command)

            if previous_abs is not None and previous_stick != 0 and abs_error > previous_abs + 350 and not flipped:
                sign *= -1
                flipped = True

            if abs_error <= 900:
                stable += 1
                if stable >= 2:
                    bridge.release()
                    if shield:
                        hold_ms = max(250, min(1500, int(max(0.25, deadline - time.monotonic()) * 1000)))
                        acknowledged |= await _pulse(bridge, buttons=BUTTONS["R"], hold_ms=min(300, hold_ms),
                            settle_s=0.05)
                        remaining = hold_ms - 300
                        while remaining > 0:
                            chunk = min(300, remaining)
                            command_id = bridge.send(buttons=BUTTONS["R"], lease_ms=chunk)
                            if first_command is None:
                                first_command = command_id
                            await asyncio.sleep(chunk / 1000)
                            remaining -= chunk
                        bridge.release()
                    return {"status": "completed",
                        "reason": "shield_oriented" if shield else "target_faced",
                        "yaw_error": error, "yaw_error_deg": round(error * 180 / 32768, 2),
                        "acknowledged": acknowledged, "skill": decision.skill}
            else:
                stable = 0

            previous_abs = abs_error
            magnitude = max(28, min(70, round(abs_error / 260)))
            stick_x = magnitude * (1 if error > 0 else -1) * sign
            previous_stick = stick_x
            command_id = bridge.send(stick_x=stick_x, stick_y=10, lease_ms=240)
            if first_command is None:
                first_command = command_id
            await feedback(bridge, current, .10)
    finally:
        bridge.release()

    return {"status": "failed", "reason": "face_timeout", "yaw_error": last_error,
        "yaw_error_deg": round(last_error * 180 / 32768, 2) if last_error is not None else None,
        "acknowledged": acknowledged, "skill": decision.skill}


async def _follow_actor(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    desired = decision.args.stop_distance or 150.0
    deadline = time.monotonic() + min(12.0, max(1.0, decision.args.duration_ms / 1000))
    start_position = before.player.position
    start_health = before.player.health
    first_command = None
    acknowledged = False
    missing = 0
    last_distance = None

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "completed", "reason": "transition_while_following",
                    "distance": math.dist(start_position, current.player.position) if current.player else None,
                    "acknowledged": acknowledged, "skill": decision.skill}
            if not current.in_game or not current.player:
                return {"status": "interrupted", "reason": "game_not_ready", "skill": decision.skill}
            if current.dialogue.active or current.cutscene_active or current.paused or current.game_over_state != 0:
                return {"status": "completed", "reason": "follow_sequence_progressed",
                    "dialogue": current.dialogue.active, "cutscene": current.cutscene_active,
                    "acknowledged": acknowledged, "skill": decision.skill}
            if start_health - current.player.health >= 16:
                return {"status": "failed", "reason": "danger_detected",
                    "health_lost": start_health - current.player.health,
                    "acknowledged": acknowledged, "skill": decision.skill}

            if first_command is not None:
                acknowledged |= consumed(bridge, first_command)
            actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
            if actor is None:
                missing += 1
                if missing >= 6:
                    return {"status": "failed", "reason": "follow_target_lost",
                        "last_target_distance": last_distance,
                        "acknowledged": acknowledged, "skill": decision.skill}
                await feedback(bridge, current, .10)
                continue
            missing = 0
            last_distance = actor.distance

            if actor.distance > desired + 35:
                steer_target = actor.position
                if "local_navmesh" in current.capabilities:
                    plan = plan_navmesh(current, actor.position)
                    probe_safe = bool(plan and waypoint_probe_safe(current, plan.waypoint))
                    bridge.set_navigation_debug(
                        skill=decision.skill, status="active" if probe_safe else ("blocked" if plan else "no_path"),
                        target_position=list(actor.position),
                        waypoint=list(plan.waypoint) if plan else None,
                        path_cells=len(plan.path) if plan else 0, probe_safe=probe_safe,
                        navmesh_used=True, target_distance=actor.distance,
                        plan_cost=round(plan.cost, 2) if plan else None,
                        exact_goal_reachable=plan.exact_goal_reachable if plan else False)
                    if plan is None or not probe_safe:
                        bridge.release()
                        await feedback(bridge, current, .10)
                        continue
                    steer_target = plan.waypoint
                stick_x, stick_y, _ = _steer_to(current, steer_target, max(0.55, decision.args.strength))
                command_id = bridge.send(stick_x=stick_x, stick_y=stick_y, lease_ms=240)
            elif actor.distance < max(45.0, desired - 50):
                if "local_navmesh" in current.capabilities:
                    dx = current.player.position[0] - actor.position[0]
                    dz = current.player.position[2] - actor.position[2]
                    length = math.hypot(dx, dz)
                    if length < 1e-4:
                        command_id = bridge.send(lease_ms=160)
                    else:
                        away_target = (
                            current.player.position[0] + dx / length * 120.0,
                            current.player.position[1],
                            current.player.position[2] + dz / length * 120.0,
                        )
                        plan = plan_navmesh(current, away_target)
                        if plan is None or not waypoint_probe_safe(current, plan.waypoint):
                            command_id = bridge.send(lease_ms=160)
                        else:
                            stick_x, stick_y, _ = _steer_to(
                                current, plan.waypoint, max(0.4, decision.args.strength))
                            command_id = bridge.send(stick_x=stick_x, stick_y=stick_y, lease_ms=180)
                else:
                    stick_x, stick_y, _ = _steer_to(current, actor.position, max(0.4, decision.args.strength))
                    command_id = bridge.send(stick_x=-stick_x, stick_y=-max(20, stick_y), lease_ms=180)
            else:
                command_id = bridge.send(lease_ms=160)

            if first_command is None:
                first_command = command_id
            await feedback(bridge, current, .10)
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "completed", "reason": "follow_window_complete",
        "last_target_distance": last_distance,
        "distance": math.dist(start_position, after.player.position) if after and after.player else None,
        "acknowledged": bool(after and first_command is not None and
            consumed(bridge, first_command)) or acknowledged,
        "skill": decision.skill}


def _reachable_traversal_affordances(game: GameState):
    if not game.player:
        return []
    result = []
    for row in game.traversal_affordances:
        horizontal = math.hypot(
            row.approach_position[0] - game.player.position[0],
            row.approach_position[2] - game.player.position[2],
        )
        if horizontal <= 30.0:
            result.append(row)
            continue
        if not game.navmesh.available:
            continue
        plan = plan_navmesh(game, row.approach_position)
        if plan and plan.exact_goal_reachable:
            result.append(row)
    return result


async def _explore_area(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    if before.paused or before.dialogue.active or before.cutscene_active or before.game_over_state != 0:
        return {"status": "stale", "reason": "gameplay_state_blocks_exploration", "skill": decision.skill}

    baseline_actors = {(actor.actor_id, actor.params) for actor in before.room_actors}
    baseline_affordances = {
        (row.kind, row.direction,
         round(row.approach_position[0] / 35.0),
         round(row.approach_position[2] / 35.0))
        for row in _reachable_traversal_affordances(before)
    }
    baseline_context = before.context_action.label
    start_health = before.player.health
    start_position = before.player.position
    deadline = time.monotonic() + min(12.0, max(1.0, decision.args.duration_ms / 1000))
    last_position = start_position
    last_seq = before.seq
    last_progress_at = time.monotonic()
    stagnant = 0
    turn_ticks = 0
    turn_side = 1
    recovery_attempts = 0
    first_command = None
    acknowledged = False
    affordance_full_seq = -1
    reachable_affordances_cache = []

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "completed", "reason": "transition_discovered",
                    "from": [observation.scene, observation.room],
                    "to": [current.scene, current.room],
                    "distance": math.dist(start_position, current.player.position) if current.player else None,
                    "acknowledged": acknowledged, "skill": decision.skill}
            if not current.in_game or not current.player:
                return {"status": "interrupted", "reason": "game_not_ready", "skill": decision.skill}
            if current.paused or current.dialogue.active or current.cutscene_active or current.game_over_state != 0:
                return {"status": "completed", "reason": "interactive_state_discovered",
                    "dialogue": current.dialogue.active, "cutscene": current.cutscene_active,
                    "acknowledged": acknowledged, "skill": decision.skill}

            if first_command is not None:
                acknowledged |= consumed(bridge, first_command)
            if current.full_seq != affordance_full_seq:
                reachable_affordances_cache = _reachable_traversal_affordances(current)
                affordance_full_seq = current.full_seq
            new_actors = [actor for actor in current.room_actors
                if (actor.actor_id, actor.params) not in baseline_actors]
            if new_actors:
                actor = min(new_actors, key=lambda row: row.distance)
                return {"status": "completed", "reason": "actor_discovered",
                    "actor": actor.model_dump(), "distance": math.dist(start_position, current.player.position),
                    "acknowledged": acknowledged, "skill": decision.skill}
            new_affordances = [row for row in reachable_affordances_cache
                if (row.kind, row.direction,
                    round(row.approach_position[0] / 35.0),
                    round(row.approach_position[2] / 35.0)) not in baseline_affordances]
            if new_affordances:
                affordance = min(new_affordances, key=lambda row: row.distance)
                return {"status": "completed", "reason": "traversal_affordance_discovered",
                    "affordance": {
                        "kind": affordance.kind, "direction": affordance.direction,
                        "approach_position": list(affordance.approach_position),
                        "target_position": list(affordance.target_position),
                        "height_delta": affordance.height_delta,
                    },
                    "distance": math.dist(start_position, current.player.position),
                    "acknowledged": acknowledged, "skill": decision.skill}
            if current.context_action.label != "none" and current.context_action.label != baseline_context:
                return {"status": "completed", "reason": "context_action_discovered",
                    "context_action": current.context_action.model_dump(),
                    "distance": math.dist(start_position, current.player.position),
                    "acknowledged": acknowledged, "skill": decision.skill}
            if start_health - current.player.health >= 16:
                return {"status": "failed", "reason": "danger_detected",
                    "health_lost": start_health - current.player.health,
                    "distance": math.dist(start_position, current.player.position),
                    "acknowledged": acknowledged, "skill": decision.skill}

            if current.seq != last_seq:
                moved = math.dist(last_position, current.player.position)
                if moved >= 1.2:
                    last_position = current.player.position
                    last_progress_at = time.monotonic()
                last_seq = current.seq
                if moved < 1.2:
                    stagnant += 1
                else:
                    stagnant = max(0, stagnant - 1)
                    if moved >= 4.0:
                        recovery_attempts = 0

            if is_realtime(bridge):
                stagnant = int((time.monotonic() - last_progress_at) / .2)
            wall_contact = bool(current.player.bg_check_flags & 0x008)
            obstructed = (wall_contact and stagnant >= 1) or stagnant >= 3
            if obstructed and recovery_attempts < 3:
                # Unknown rooms need clearance before another probe. Alternate reverse arcs on each retry.
                recovery = await _backtrack_recovery(bridge, current, recovery_attempts)
                acknowledged |= recovery["acknowledged"]
                recovery_attempts += 1
                stagnant = 0
                last_progress_at = time.monotonic()
                sample = bridge.state
                if recovery["world_changed"] and sample:
                    return {"status": "completed", "reason": "transition_discovered_during_recovery",
                        "from": [observation.scene, observation.room],
                        "to": [sample.scene, sample.room],
                        "distance": math.dist(start_position, sample.player.position) if sample.player else None,
                        "acknowledged": acknowledged, "skill": decision.skill}
                if sample and sample.player:
                    last_position = sample.player.position
                    last_seq = sample.seq
                continue

            if obstructed and recovery_attempts >= 3:
                turn_ticks = max(turn_ticks, 6)
                stagnant = 0
                last_progress_at = time.monotonic()
                turn_side *= -1

            if "local_navmesh" in current.capabilities:
                base_angle = current.player.yaw * math.pi / 32768.0
                angle = base_angle
                if turn_ticks > 0:
                    # Choose a connected side frontier instead of rotating with
                    # forward bias, which can still scrape a nearby wall.
                    angle += turn_side * math.pi / 2.0
                    turn_ticks -= 1
                explore_target = (
                    current.player.position[0] + math.sin(angle) * 350.0,
                    current.player.position[1],
                    current.player.position[2] + math.cos(angle) * 350.0,
                )
                plan = plan_navmesh(current, explore_target)
                probe_safe = bool(plan and waypoint_probe_safe(current, plan.waypoint))
                bridge.set_navigation_debug(
                    skill=decision.skill, status="active" if probe_safe else ("blocked" if plan else "no_path"),
                    target_position=list(explore_target),
                    waypoint=list(plan.waypoint) if plan else None,
                    path_cells=len(plan.path) if plan else 0, probe_safe=probe_safe,
                    navmesh_used=True, target_distance=None,
                    plan_cost=round(plan.cost, 2) if plan else None,
                    exact_goal_reachable=plan.exact_goal_reachable if plan else False)
                if plan is None or not probe_safe:
                    if reachable_affordances_cache:
                        affordance = min(reachable_affordances_cache, key=lambda row: row.distance)
                        return {"status": "completed", "reason": "traversal_affordance_available",
                            "affordance": {
                                "kind": affordance.kind, "direction": affordance.direction,
                                "approach_position": list(affordance.approach_position),
                                "target_position": list(affordance.target_position),
                                "height_delta": affordance.height_delta,
                            },
                            "distance": math.dist(start_position, current.player.position),
                            "acknowledged": acknowledged, "skill": decision.skill}
                    # Stay neutral; try the opposite connected side on the next
                    # sample rather than issuing any unvalidated movement.
                    buttons, stick_x, stick_y = 0, 0, 0
                    turn_side *= -1
                    turn_ticks = max(turn_ticks, 2)
                else:
                    buttons = 0
                    stick_x, stick_y, _ = _steer_to(
                        current, plan.waypoint, max(.55, decision.args.strength))
            elif turn_ticks > 0:
                buttons, stick_x, stick_y = 0, 58 * turn_side, 8
                turn_ticks -= 1
            else:
                buttons, stick_x, stick_y = 0, 0, 58

            command_id = bridge.send(buttons=buttons, stick_x=stick_x, stick_y=stick_y, lease_ms=240)
            if first_command is None:
                first_command = command_id
            await feedback(bridge, current, .10)
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "completed", "reason": "exploration_window_complete",
        "distance": math.dist(start_position, after.player.position) if after and after.player else None,
        "acknowledged": bool(after and first_command is not None and
            consumed(bridge, first_command)) or acknowledged,
        "skill": decision.skill}
