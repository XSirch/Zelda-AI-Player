"""Local traversal controller helpers."""
from __future__ import annotations

import asyncio
import math
import time
from ..bridge import Bridge
from ..control.feedback import consumed
from ..models import Decision, GameState
from ..navmesh import plan_navmesh
from .catalog import BUTTONS
from .common import _probe_stick, _pulse

def _best_traversal_probe(game: GameState, direction: str):
    probes = [p for p in game.navigation_probes if p.floor_found and p.delta_y is not None]
    if direction == "down":
        candidates = [p for p in probes if -240.0 <= p.delta_y <= -8.0]
        if not candidates:
            return None
        # Prefer a modest safe descent over the largest drop; stairs/ladder landings win naturally.
        return min(candidates, key=lambda p: (abs(p.delta_y + 45.0), p.distance))
    candidates = [p for p in probes if 8.0 <= p.delta_y <= 120.0]
    if not candidates:
        return None
    return min(candidates, key=lambda p: (p.delta_y, p.distance))


def _best_climb_surface_probe(game: GameState, direction: str):
    # OoT wall flags: LADDER=0x02, LADDER_TOP=0x04, CLIMBABLE=0x08.
    candidates = [p for p in game.navigation_probes
        if p.wall_hit and p.wall_distance is not None and (p.wall_flags & 0x0E)]
    if direction == "down":
        tops = [p for p in candidates if p.wall_flags & 0x04]
        if tops:
            return min(tops, key=lambda p: (p.wall_distance, p.distance))
        ladders = [p for p in candidates if p.wall_flags & 0x02]
        return min(ladders, key=lambda p: (p.wall_distance, p.distance)) if ladders else None
    climbable = [p for p in candidates if p.wall_flags & (0x02 | 0x08)]
    return min(climbable, key=lambda p: (p.wall_distance, p.distance)) if climbable else None


def _local_traversal_evidence(game: GameState, direction: str,
                              *, kind: str | None = None) -> bool:
    """Fast evidence that Link is already at the requested vertical feature."""
    if not game.player:
        return False
    player = game.player
    climb_kind = bool(kind and (kind.startswith("ladder_") or kind.startswith("climbable_wall")))
    if direction == "down":
        if (player.climbing_ladder or player.hanging_ledge or player.can_down or
                game.context_action.label == "down" or (player.wall_flags & 0x06)):
            return True
    else:
        if (player.climbing_ladder or player.climbing_ledge or player.can_climb or
                game.context_action.label == "climb" or (player.wall_flags & 0x0A)):
            return True

    if _best_climb_surface_probe(game, direction) is not None:
        return True
    # A ladder/climb-wall affordance cannot be replaced by an arbitrary floor
    # delta after the slow scan recenters.
    if climb_kind:
        return False
    return _best_traversal_probe(game, direction) is not None


def _fast_revalidation_matches_original(game: GameState, original) -> bool:
    """Require fast traversal evidence to be physically near the original target."""
    if not game.player:
        return False
    step = game.navmesh.step if game.navmesh.available else 70.0
    max_distance = max(150.0, step * 2.25)
    if math.dist(game.player.position, original.target_position) > max_distance:
        return False
    return _local_traversal_evidence(
        game, original.direction, kind=original.kind)


def _refresh_traversal_affordance(game: GameState, original):
    """Re-identify a moving/recentered affordance by nearby physical geometry."""
    if not game.player:
        return None
    candidates = [row for row in game.traversal_affordances
                  if row.direction == original.direction]
    if not candidates:
        return None

    player_pos = game.player.position
    step = game.navmesh.step if game.navmesh.available else 70.0
    same_kind = [row for row in candidates if row.kind == original.kind]
    if same_kind:
        best = min(same_kind, key=lambda row: (
            math.dist(row.target_position, original.target_position),
            math.dist(row.approach_position, player_pos)))
        if (math.dist(best.target_position, original.target_position) <= max(160.0, step * 2.5) and
                math.dist(best.approach_position, player_pos) <= max(120.0, step * 1.75)):
            return best

    # The same ladder/ledge can be reclassified after Link reaches its top/base.
    # Only accept a same-direction replacement if it is both physically near Link
    # and near the original target geometry.
    best = min(candidates, key=lambda row: (
        math.dist(row.approach_position, player_pos),
        math.dist(row.target_position, original.target_position)))
    if (math.dist(best.approach_position, player_pos) <= max(80.0, step * 1.25) and
            math.dist(best.target_position, original.target_position) <= max(140.0, step * 2.0)):
        return best
    return None


def _player_traversal_state_evidence(game: GameState, direction: str) -> bool:
    """Native player/context state proving Link is already on the vertical feature."""
    if not game.player:
        return False
    player = game.player
    if direction == "down":
        return bool(
            player.climbing_ladder or player.hanging_ledge or player.can_down or
            game.context_action.label == "down" or (player.wall_flags & 0x06)
        )
    return bool(
        player.climbing_ladder or player.climbing_ledge or player.can_climb or
        game.context_action.label == "climb" or (player.wall_flags & 0x0A)
    )


def _immediate_traversal_evidence(game: GameState, direction: str) -> bool:
    """Only evidence close enough to justify immediate local traversal."""
    if not game.player:
        return False
    player = game.player
    if direction == "down":
        if (player.climbing_ladder or player.hanging_ledge or player.can_down or
                game.context_action.label == "down" or (player.wall_flags & 0x06)):
            return True
        if any(p.wall_hit and p.wall_distance is not None and p.wall_distance <= 80.0 and
               (p.wall_flags & 0x06) for p in game.navigation_probes):
            return True
        return any(p.distance <= 75.0 and p.floor_found and p.delta_y is not None and
                   -70.0 <= p.delta_y <= -8.0 for p in game.navigation_probes)

    if (player.climbing_ladder or player.climbing_ledge or player.can_climb or
            game.context_action.label == "climb" or (player.wall_flags & 0x0A)):
        return True
    if any(p.wall_hit and p.wall_distance is not None and p.wall_distance <= 80.0 and
           (p.wall_flags & (0x02 | 0x08)) for p in game.navigation_probes):
        return True
    return any(p.distance <= 75.0 and p.floor_found and p.delta_y is not None and
               8.0 <= p.delta_y <= 120.0 for p in game.navigation_probes)


def _best_auto_traversal_affordance(game: GameState, direction: str):
    """Choose the best currently reachable vertical route without another model call."""
    if not game.player:
        return None
    candidates = [row for row in game.traversal_affordances if row.direction == direction]
    if not candidates:
        return None

    penalties = {
        "ladder_down": 0.0,
        "ladder_up": 0.0,
        "climbable_wall_up": 10.0,
        "stairs_or_slope_down": 20.0,
        "stairs_or_slope_up": 20.0,
        "ledge_down": 300.0,
    }
    scored = []
    step = game.navmesh.step if game.navmesh.available else 70.0
    for row in candidates:
        horizontal = math.hypot(
            row.approach_position[0] - game.player.position[0],
            row.approach_position[2] - game.player.position[2],
        )
        approach_vertical = abs(row.approach_position[1] - game.player.position[1])
        if horizontal <= 30.0 and approach_vertical <= 24.0:
            route_cost = 0.0
        elif game.navmesh.available:
            plan = plan_navmesh(game, row.approach_position)
            if not plan or not plan.exact_goal_reachable:
                continue
            if plan.target_distance > max(85.0, step * 1.25):
                continue
            route_cost = plan.cost + plan.target_distance
        else:
            continue
        scored.append((
            route_cost + penalties.get(row.kind, 50.0),
            horizontal,
            abs(row.height_delta),
            row,
        ))
    return min(scored, key=lambda item: item[:3])[3] if scored else None


async def _traverse_auto(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    """High-level vertical navigation: local traverse or automatic A* to a discovered route."""
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    direction = decision.args.direction
    if _player_traversal_state_evidence(before, direction):
        result = await _traverse_local(bridge, decision, observation, direction)
        result["auto_navpath"] = False
        result["navpath_mode"] = "native_state"
        return result

    affordance = _best_auto_traversal_affordance(before, direction)
    if affordance is None:
        # No slow-scan route is currently reachable. A nearby fast probe can
        # still support immediate local traversal (e.g. stairs just entering
        # the 70u safety window) without another model call.
        if _immediate_traversal_evidence(before, direction):
            result = await _traverse_local(bridge, decision, observation, direction)
            result["auto_navpath"] = False
            result["navpath_mode"] = "fast_probe_fallback"
            return result
        return {"status": "failed", "reason": "no_reachable_traversal_affordance",
            "direction": direction, "auto_navpath": True, "skill": decision.skill}

    internal_args = decision.args.model_copy(update={
        "target_position": list(affordance.approach_position),
        "duration_ms": max(8000, decision.args.duration_ms),
    })
    internal = decision.model_copy(update={
        "skill": "traverse_to",
        "summary": "Automatic NavPath to the best reachable vertical route.",
        "args": internal_args,
    })
    result = await _traverse_to_affordance(bridge, internal, observation)
    result["requested_skill"] = decision.skill
    result["controller_skill"] = "traverse_to"
    result["auto_navpath"] = True
    result["selected_affordance"] = {
        "kind": affordance.kind,
        "direction": affordance.direction,
        "approach_position": list(affordance.approach_position),
        "target_position": list(affordance.target_position),
    }
    result["skill"] = decision.skill
    return result


def _resolve_traversal_affordance(game: GameState, direction: str,
                                  target_position: list[float] | tuple[float, float, float] | None,
                                  *, kind: str | None = None):
    if not game.traversal_affordances or target_position is None:
        return None
    target = tuple(float(v) for v in target_position)
    candidates = [row for row in game.traversal_affordances
                  if row.direction == direction and (kind is None or row.kind == kind)]
    if not candidates:
        return None
    nearest = min(candidates, key=lambda row: math.dist(row.approach_position, target))
    tolerance = max(100.0, (game.navmesh.step * 1.5) if game.navmesh.available else 100.0)
    return nearest if math.dist(nearest.approach_position, target) <= tolerance else None


async def _traverse_to_affordance(bridge: Bridge, decision: Decision,
                                  observation: GameState) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    if "traversal_affordances_v1" not in before.capabilities:
        return {"status": "failed", "reason": "traversal_affordance_bridge_upgrade_required",
            "skill": decision.skill}

    affordance = _resolve_traversal_affordance(
        before, decision.args.direction, decision.args.target_position)
    if affordance is None:
        return {"status": "failed", "reason": "traversal_affordance_not_observed",
            "direction": decision.args.direction, "skill": decision.skill}

    total_budget_ms = min(12000, max(2500, decision.args.duration_ms))
    approach_budget_ms = min(5000, max(1800, total_budget_ms // 2))
    approach_args = decision.args.model_copy(update={
        "direction": None,
        "duration_ms": approach_budget_ms,
        "strength": max(0.55, decision.args.strength),
        "target_position": list(affordance.approach_position),
        "stop_distance": 20.0,
    })
    approach_decision = decision.model_copy(update={
        "skill": "navigate_to",
        "summary": "Approach observed traversal affordance.",
        "args": approach_args,
    })

    # Import lazily to keep traversal/navigation modules acyclic.
    from .navigation import _navigate_local
    started = time.monotonic()
    approach = await _navigate_local(
        bridge, approach_decision, observation, actor_mode=False)
    if approach.get("status") != "completed":
        return {"status": "failed", "reason": "traversal_approach_failed",
            "approach_reason": approach.get("reason"), "affordance_kind": affordance.kind,
            "direction": decision.args.direction, "approach": approach, "skill": decision.skill}

    current = bridge.state
    if not current or not current.player:
        return {"status": "interrupted", "reason": "game_not_ready", "skill": decision.skill}
    refreshed = _refresh_traversal_affordance(current, affordance)
    local_evidence = _fast_revalidation_matches_original(current, affordance)
    if refreshed is None and not local_evidence:
        return {"status": "failed", "reason": "traversal_affordance_lost",
            "affordance_kind": affordance.kind, "direction": decision.args.direction,
            "skill": decision.skill}
    # Once fast collision evidence confirms the ladder/ledge under Link, do not
    # require the slow recentered scan to preserve an identical label.
    active_affordance = refreshed or affordance

    elapsed_ms = int((time.monotonic() - started) * 1000)
    remaining_ms = total_budget_ms - elapsed_ms
    if remaining_ms < 1000:
        return {"status": "failed", "reason": "traversal_approach_timeout",
            "affordance_kind": active_affordance.kind, "direction": decision.args.direction,
            "skill": decision.skill}
    local_args = decision.args.model_copy(update={
        "duration_ms": remaining_ms,
        "target_position": None,
    })
    local_decision = decision.model_copy(update={"args": local_args})
    result = await _traverse_local(
        bridge, local_decision, current, decision.args.direction)
    result["affordance_kind"] = active_affordance.kind
    result["approach_position"] = list(active_affordance.approach_position)
    result["revalidated_by"] = "slow_scan" if refreshed is not None else "fast_collision"
    return result


async def _traverse_local(bridge: Bridge, decision: Decision, observation: GameState,
                          direction: str) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    start_y = before.player.position[1]
    start_position = before.player.position
    start_health = before.player.health
    deadline = time.monotonic() + min(10.0, max(2.0, decision.args.duration_ms / 1000))
    acknowledged = False
    probe_attempts = 0
    last_probe = None

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "completed", "reason": "world_changed_during_traversal",
                    "direction": direction, "acknowledged": acknowledged, "skill": decision.skill}
            if not current.in_game or not current.player:
                return {"status": "interrupted", "reason": "game_not_ready", "skill": decision.skill}
            if current.paused or current.dialogue.active or current.cutscene_active or current.game_over_state != 0:
                return {"status": "interrupted", "reason": "gameplay_state_changed", "skill": decision.skill}

            vertical = current.player.position[1] - start_y
            grounded = bool(current.player.bg_check_flags & 0x001)
            if direction == "down" and vertical <= -35.0 and not current.player.climbing_ladder and grounded:
                return {"status": "completed", "reason": "descended",
                    "vertical_distance": vertical,
                    "distance": math.dist(start_position, current.player.position),
                    "health_lost": max(0, start_health - current.player.health),
                    "acknowledged": acknowledged, "skill": decision.skill}
            if direction == "up" and vertical >= 35.0 and not current.player.climbing_ladder and grounded:
                return {"status": "completed", "reason": "ascended",
                    "vertical_distance": vertical,
                    "distance": math.dist(start_position, current.player.position),
                    "acknowledged": acknowledged, "skill": decision.skill}

            # Once latched to a ladder, vertical stick is the reliable control. Do not press A:
            # in OoT A can dismount/drop instead of climbing.
            if current.player.climbing_ladder:
                stick_y = -58 if direction == "down" else 58
                command_id = bridge.send(stick_y=stick_y, lease_ms=300)
                await asyncio.sleep(0.26)
                bridge.release()
                await asyncio.sleep(0.05)
                sample = bridge.state
                acknowledged |= bool(sample and consumed(bridge, command_id))
                continue

            # Hanging at a ledge while descending: A-button "Down" releases to the lower surface.
            if direction == "down" and current.player.hanging_ledge and current.context_action.label == "down":
                acknowledged |= await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=100, settle_s=0.12)
                continue

            surface_probe = _best_climb_surface_probe(current, direction)
            if surface_probe is not None and not current.player.climbing_ladder:
                # Unlike floor deltas, this is direct collision-surface evidence of a ladder/climb wall.
                # Center behind Link, then move toward the probed surface until OoT latches the climb state.
                acknowledged |= await _pulse(bridge, buttons=BUTTONS["Z"], hold_ms=70, settle_s=0.04)
                stick_x, stick_y = _probe_stick(surface_probe.direction, current)
                command_id = bridge.send(stick_x=stick_x, stick_y=stick_y, lease_ms=220)
                await asyncio.sleep(0.20)
                bridge.release()
                await asyncio.sleep(0.08)
                sample = bridge.state
                acknowledged |= bool(sample and consumed(bridge, command_id))
                last_probe = surface_probe.direction
                probe_attempts += 1
                if sample and sample.player and sample.player.climbing_ladder:
                    continue
                # A climbable wall approached from below may require the explicit Climb affordance.
                if direction == "up" and sample and sample.player and (
                        sample.player.can_climb or sample.context_action.label == "climb"):
                    acknowledged |= await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=100, settle_s=0.08)
                continue

            # At a LADDER_TOP wall, OoT attaches Link by moving into the wall; A would be the wrong input.
            if direction == "down" and (current.player.wall_flags & 0x04):
                acknowledged |= await _pulse(bridge, buttons=BUTTONS["Z"], hold_ms=70, settle_s=0.04)
                command_id = bridge.send(stick_y=50, lease_ms=220)
                await asyncio.sleep(0.20)
                bridge.release()
                await asyncio.sleep(0.08)
                sample = bridge.state
                acknowledged |= bool(sample and consumed(bridge, command_id))
                continue

            # A climb affordance is explicit engine evidence. Start it only for upward traversal.
            if direction == "up" and (current.player.can_climb or current.context_action.label == "climb"):
                acknowledged |= await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=110, settle_s=0.10)
                continue

            probe = _best_traversal_probe(current, direction)
            if probe is None:
                # If touching a ladder/climbable wall but not attached, upward A can latch it.
                if direction == "up" and (current.player.wall_flags & 0x0E):
                    acknowledged |= await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=110, settle_s=0.10)
                    continue
                return {"status": "failed", "reason": "no_traversal_affordance_observed",
                    "direction": direction, "vertical_distance": vertical,
                    "probe_attempts": probe_attempts, "acknowledged": acknowledged, "skill": decision.skill}

            last_probe = probe.direction
            # Z puts the camera behind Link so the relative terrain probe maps predictably to the stick.
            acknowledged |= await _pulse(bridge, buttons=BUTTONS["Z"], hold_ms=80, settle_s=0.06)
            stick_x, stick_y = _probe_stick(probe.direction, current)
            # Approach drops cautiously; stairs can be traversed continuously but ledges should be probed.
            hold_ms = 260 if abs(probe.delta_y or 0) <= 70 else 180
            command_id = bridge.send(stick_x=stick_x, stick_y=stick_y, lease_ms=hold_ms)
            await asyncio.sleep(hold_ms / 1000)
            bridge.release()
            await asyncio.sleep(0.08)
            sample = bridge.state
            acknowledged |= bool(sample and consumed(bridge, command_id))
            probe_attempts += 1

            if sample and sample.player and start_health - sample.player.health >= 16:
                return {"status": "failed", "reason": "descent_caused_damage",
                    "direction": direction, "health_lost": start_health - sample.player.health,
                    "vertical_distance": sample.player.position[1] - start_y,
                    "probe": last_probe, "acknowledged": acknowledged, "skill": decision.skill}
    finally:
        bridge.release()

    after = bridge.state
    vertical = after.player.position[1] - start_y if after and after.player else 0.0
    progressed = (direction == "down" and vertical <= -20.0) or (direction == "up" and vertical >= 20.0)
    return {"status": "completed" if progressed else "failed",
        "reason": "traversal_progress" if progressed else "traversal_timeout",
        "direction": direction, "vertical_distance": vertical, "probe": last_probe,
        "probe_attempts": probe_attempts, "acknowledged": acknowledged, "skill": decision.skill}
