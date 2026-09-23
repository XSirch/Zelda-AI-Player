"""Generic observed-target policy. Enemy-specific animation timing is not inferred."""
from __future__ import annotations

import time
from ..bridge import Bridge
from ..control.feedback import consumed, feedback, is_realtime
from ..models import Decision, GameState
from .catalog import BUTTONS
from .common import _matching_actor, _steer_to

async def _fight_enemy(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    actor = _matching_actor(before, decision.args.target_actor_id, decision.args.target_actor_params,
                            decision.args.target_actor_uid)
    if actor is None or actor.category not in {5, 9}:
        return {"status": "failed", "reason": "target_is_not_observed_enemy", "skill": decision.skill}
    target_uid = actor.actor_uid
    start_health = before.player.health
    deadline = time.monotonic() + min(10., max(.5, decision.args.duration_ms / 1000))
    events_before = {e.id for e in before.events}
    first_command = None
    next_attack_at = 0.0
    missing_since = None
    phase = "acquire_target"
    acknowledged = False
    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
            if (not current.in_game or not current.player or current.dialogue.active or current.cutscene_active
                    or current.paused or current.game_over_state):
                return {"status": "interrupted", "reason": "gameplay_state_changed", "skill": decision.skill}
            acknowledged |= consumed(bridge, first_command)
            defeated = next((e for e in current.events if e.id not in events_before
                and e.kind in {"enemy_defeated", "boss_defeated"}
                and (e.actor_uid == target_uid if target_uid else e.detail == str(decision.args.target_actor_id))), None)
            if defeated:
                return {"status": "completed", "reason": defeated.kind, "effect_confirmed": True,
                    "target_actor_uid": target_uid, "health_lost": max(0, start_health-current.player.health),
                    "acknowledged": acknowledged, "skill": decision.skill}
            actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params, target_uid)
            if actor is None:
                missing_since = missing_since or time.monotonic()
                bridge.release()
                if time.monotonic()-missing_since > .6:
                    return {"status": "failed", "reason": "enemy_lost_without_defeat_event", "skill": decision.skill}
                await feedback(bridge, current, .1)
                continue
            missing_since = None
            locked = bool(current.target_actor and current.target_actor.targeted and
                (current.target_actor.actor_uid == target_uid if target_uid else
                 current.target_actor.actor_id == actor.actor_id))
            shield = any(g.equipment_type == "shield" and g.equipped for g in current.progress.equipment)
            # Unknown equipment is not a guarantee of protection. No reflex can bypass game animation locks.
            guard = BUTTONS["Z"] | (BUTTONS["R"] if shield else 0)
            if actor.distance > 120:
                phase = "approach"
                x, y, _ = _steer_to(current, actor.position, min(.7, max(.4, decision.args.strength)))
                command = bridge.send(buttons=BUTTONS["Z"], stick_x=x, stick_y=y, lease_ms=300)
            elif not locked:
                phase = "acquire_target"
                command = bridge.send(buttons=guard, lease_ms=300)
            elif time.monotonic() < next_attack_at:
                phase = "guard_recovery"
                command = bridge.send(buttons=guard, lease_ms=300)
            else:
                phase = "attack"
                if is_realtime(bridge):
                    acknowledged |= await bridge.pulse(buttons=BUTTONS["Z"] | BUTTONS["B"],
                        baseline_buttons=BUTTONS["Z"], edge_buttons=BUTTONS["B"], hold_ticks=1)
                    # A bounded cooldown is tactical policy, not transport latency or an inferred opening.
                    next_attack_at = time.monotonic() + .35
                    continue
                command = bridge.send(buttons=BUTTONS["Z"] | BUTTONS["B"], lease_ms=120)
                next_attack_at = time.monotonic() + .4
            if first_command is None:
                first_command = command
            await feedback(bridge, current, .1)
    finally:
        bridge.release()
    after = bridge.state
    return {"status": "failed", "reason": "combat_timeout", "phase": phase,
        "target_actor_uid": target_uid, "health_lost": max(0, start_health-after.player.health)
            if after and after.player else None, "acknowledged": acknowledged, "skill": decision.skill}
