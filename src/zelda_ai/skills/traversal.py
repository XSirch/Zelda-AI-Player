"""Local traversal controller helpers."""
from __future__ import annotations

import asyncio
import math
import time
from ..bridge import Bridge
from ..control.feedback import consumed
from ..models import Decision, GameState
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
                stick_x, stick_y = _probe_stick(surface_probe.direction)
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
            stick_x, stick_y = _probe_stick(probe.direction)
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
