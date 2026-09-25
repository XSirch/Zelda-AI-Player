"""Per-enemy learned combat policy. Motor primitives are generic; strategy is persistent per opponent."""
from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass

from ..bridge import Bridge
from ..combat_learning import combat_state_key, choose_action, empty_policy, update_policy
from ..control.feedback import consumed, feedback, is_realtime
from ..models import ActorObservation, Decision, GameState
from .catalog import BUTTONS
from .common import (
    _dodge_direction_safe,
    _matching_actor,
    _perform_dodge,
    _steer_to,
    _yaw_error_units,
    _yaw_to_target,
)


@dataclass(frozen=True)
class CombatAssessment:
    distance: float
    locked: bool
    facing_player: bool
    closing_rate: float
    disabled: bool
    recent_damage: bool
    threat_score: int


def _sword_equipped(game: GameState) -> bool:
    return any(row.equipment_type == "sword" and row.equipped for row in game.progress.equipment)


def _shield_equipped(game: GameState) -> bool:
    return any(row.equipment_type == "shield" and row.equipped for row in game.progress.equipment)


def _locked_to(game: GameState, actor: ActorObservation) -> bool:
    target = game.target_actor
    if target is None or not target.targeted:
        return False
    if actor.actor_uid and target.actor_uid:
        return target.actor_uid == actor.actor_uid
    return target.actor_id == actor.actor_id and target.params == actor.params


def _candidate_is(game: GameState, actor: ActorObservation) -> bool:
    candidate = game.target_candidate
    if candidate is None:
        return False
    if actor.actor_uid and candidate.actor_uid:
        return candidate.actor_uid == actor.actor_uid
    return candidate.actor_id == actor.actor_id and candidate.params == actor.params


def _facing_player(game: GameState, actor: ActorObservation) -> bool:
    if not game.player or actor.yaw is None:
        return False
    desired = _yaw_to_target(actor.position, game.player.position)
    return abs(_yaw_error_units(actor.yaw, desired)) <= 7600


def _assessment(game: GameState, actor: ActorObservation, previous_distance: float | None,
                elapsed_s: float, recent_damage: bool) -> CombatAssessment:
    closing = 0.0
    if previous_distance is not None and elapsed_s > 1e-4:
        closing = max(-600.0, min(600.0, (previous_distance - actor.distance) / elapsed_s))
    facing = _facing_player(game, actor)
    disabled = actor.freeze_timer > 0
    threat = 0
    if actor.distance <= 85:
        threat += 2
    elif actor.distance <= 125:
        threat += 1
    if closing >= 80:
        threat += 2
    elif closing >= 30:
        threat += 1
    if abs(actor.speed_xz) >= 3:
        threat += 1
    if facing:
        threat += 1
    if recent_damage:
        threat += 4
    if disabled:
        threat = max(0, threat - 3)
    return CombatAssessment(actor.distance, _locked_to(game, actor), facing, closing,
                            disabled, recent_damage, threat)


def _available_actions(game: GameState, assessment: CombatAssessment, shield: bool) -> list[str]:
    if not assessment.locked:
        return ["acquire"]
    if assessment.distance > 160:
        return ["approach"]
    actions = ["hold"]
    if assessment.distance > 105:
        actions.append("approach")
    else:
        actions.append("attack")
    if shield:
        actions.append("guard")
    for direction in ("back", "left", "right"):
        if _dodge_direction_safe(game, direction):
            actions.append(f"dodge_{direction}")
    return actions


def _priors(assessment: CombatAssessment, shield: bool) -> dict[str, float]:
    priors: dict[str, float] = {}
    if not assessment.locked:
        priors["acquire"] = .4
    if assessment.distance > 105:
        priors["approach"] = .22
    elif assessment.threat_score < 2:
        priors["attack"] = .18
    if assessment.threat_score >= 4:
        if shield:
            priors["guard"] = .28
        priors["dodge_back"] = .18
        priors["dodge_left"] = .12
        priors["dodge_right"] = .12
        priors["attack"] = -.12
    if assessment.disabled:
        priors["attack"] = .32
        priors["approach"] = .18
    if assessment.distance < 48:
        priors["dodge_back"] = priors.get("dodge_back", 0) + .14
    return priors


async def _wait_melee_cycle(bridge: Bridge, start: GameState, timeout: float = .65) -> tuple[bool, GameState]:
    current = bridge.state or start
    started = bool(current.player and current.player.melee_weapon_state != 0)
    deadline = time.monotonic() + timeout
    while current and time.monotonic() < deadline:
        if current.player:
            if current.player.melee_weapon_state != 0:
                started = True
            elif started:
                break
        try:
            current = await bridge.next_state(current.seq, timeout=min(.2, max(.001, deadline-time.monotonic())))
        except RuntimeError:
            break
    return started, current or start


async def _do_action(bridge: Bridge, action: str, game: GameState, actor: ActorObservation,
                     strength: float, shield: bool) -> dict:
    if action == "acquire":
        if _candidate_is(game, actor):
            command = bridge.send(buttons=BUTTONS["Z"], lease_ms=220)
        else:
            x, y, _ = _steer_to(game, actor.position, .35)
            command = bridge.send(stick_x=x, stick_y=y, lease_ms=180)
        await feedback(bridge, game, .1)
        after = bridge.state or game
        return {"ok": consumed(bridge, command), "detail": "lock_acquire",
                "locked": _locked_to(after, actor), "after": after}

    if action == "approach":
        x, y, _ = _steer_to(game, actor.position, min(.68, max(.34, strength)))
        command = bridge.send(buttons=BUTTONS["Z"] if _locked_to(game, actor) else 0,
                              stick_x=x, stick_y=y, lease_ms=200)
        await feedback(bridge, game, .1)
        return {"ok": consumed(bridge, command), "detail": "approach", "after": bridge.state or game}

    if action == "hold":
        command = bridge.send(buttons=BUTTONS["Z"], lease_ms=180)
        await feedback(bridge, game, .1)
        return {"ok": consumed(bridge, command), "detail": "hold_lock", "after": bridge.state or game}

    if action == "guard":
        command = bridge.send(buttons=BUTTONS["Z"] | BUTTONS["R"], lease_ms=220)
        await feedback(bridge, game, .12)
        return {"ok": consumed(bridge, command), "detail": "guard", "after": bridge.state or game}

    if action.startswith("dodge_"):
        direction = action.removeprefix("dodge_")
        dodge = await _perform_dodge(bridge, game, direction)
        return {"ok": bool(dodge.get("confirmed")), "detail": dodge.get("reason", "dodge"),
                "dodge": dodge, "after": bridge.state or game}

    if action == "attack":
        if not is_realtime(bridge):
            command = bridge.send(buttons=BUTTONS["Z"] | BUTTONS["B"], lease_ms=120)
            await feedback(bridge, game, .12)
            return {"ok": consumed(bridge, command), "detail": "legacy_attack",
                    "attack_started": False, "after": bridge.state or game}
        delivered = await bridge.pulse(buttons=BUTTONS["Z"] | BUTTONS["B"],
            baseline_buttons=BUTTONS["Z"], edge_buttons=BUTTONS["B"], hold_ticks=1)
        started, after = await _wait_melee_cycle(bridge, game) if delivered else (False, bridge.state or game)
        return {"ok": delivered and started, "detail": "melee_swing" if started else "swing_not_observed",
                "attack_started": started, "after": after}

    return {"ok": False, "detail": "unknown_combat_action", "after": game}


def _reward(action: str, before: GameState, after: GameState, actor_before: ActorObservation,
            actor_after: ActorObservation | None, action_result: dict, assessment: CombatAssessment) -> float:
    reward = 0.0
    if before.player and after.player:
        hearts_lost = max(0, before.player.health - after.player.health) / 16.0
        reward -= hearts_lost * 2.6
    if actor_after is not None:
        if (actor_before.collision_health_hint is not None and actor_after.collision_health_hint is not None
                and actor_after.collision_health_hint < actor_before.collision_health_hint):
            reward += min(2.4, (actor_before.collision_health_hint - actor_after.collision_health_hint) * 1.2)
        if actor_after.color_filter_timer > actor_before.color_filter_timer:
            reward += .35
    if not action_result.get("ok"):
        reward -= .45
    if action == "attack" and action_result.get("attack_started"):
        reward += .12
    elif action.startswith("dodge_"):
        reward += .38 if action_result.get("ok") else -.18
    elif action == "guard" and assessment.threat_score >= 3 and before.player and after.player:
        if after.player.health == before.player.health:
            reward += .16
    elif action == "approach" and actor_after is not None:
        delta = actor_before.distance - actor_after.distance
        reward += .14 if delta >= 8 else (-.06 if delta <= -5 else 0)
    elif action == "acquire":
        reward += .18 if action_result.get("locked") else -.08
    elif action == "hold" and assessment.threat_score >= 3 and before.player and after.player:
        reward += .06 if after.player.health == before.player.health else -.1
    return round(max(-5.0, min(5.0, reward)), 4)


def _finish(reason: str, outcome: str, start_health: int, current: GameState | None, trace: list[dict],
            *, actor_uid: str | None, attacks: int, confirmed_hits: int, dodges: int, guards: int,
            started_at: float, skill: str) -> dict:
    if trace:
        if outcome == "win":
            trace[-1]["reward"] = round(min(10.0, trace[-1]["reward"] + 3.0), 4)
        elif outcome == "loss":
            trace[-1]["reward"] = round(max(-10.0, trace[-1]["reward"] - 3.0), 4)
    health = current.player.health if current and current.player else start_health
    return {"status": "completed" if outcome == "win" else ("failed" if outcome == "loss" else "interrupted"),
        "reason": reason, "outcome": outcome, "effect_confirmed": outcome == "win",
        "target_actor_uid": actor_uid, "attacks": attacks, "confirmed_hits": confirmed_hits,
        "dodges": dodges, "guards": guards, "health_lost": max(0, start_health-health),
        "duration_ms": round((time.monotonic()-started_at)*1000, 2), "learning_trace": trace,
        "skill": skill}


async def _fight_enemy(bridge: Bridge, decision: Decision, observation: GameState,
                       profile: dict | None = None) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "outcome": "incomplete",
                "learning_trace": [], "skill": decision.skill}
    actor = _matching_actor(before, decision.args.target_actor_id, decision.args.target_actor_params,
                            decision.args.target_actor_uid)
    if actor is None or actor.category not in {5, 9}:
        return {"status": "failed", "reason": "target_is_not_observed_enemy", "outcome": "incomplete",
                "learning_trace": [], "skill": decision.skill}
    if not _sword_equipped(before):
        return {"status": "failed", "reason": "melee_weapon_not_equipped", "outcome": "incomplete",
                "learning_trace": [], "skill": decision.skill}
    if "combat_learning_state" not in before.capabilities:
        return {"status": "failed", "reason": "bridge_upgrade_required_for_combat_learning",
                "outcome": "incomplete", "learning_trace": [], "skill": decision.skill}

    target_uid = actor.actor_uid
    start_health = before.player.health
    started_at = time.monotonic()
    deadline = started_at + min(12.0, max(1.0, decision.args.duration_ms / 1000))
    events_before = {e.id for e in before.events}
    previous_distance = actor.distance
    previous_at = started_at
    recent_damage_until = 0.0
    previous_player_health = start_health
    trace: list[dict] = []
    working_profile = deepcopy(profile or {"enemy_key": "new", "encounters": 0, "policy": empty_policy()})
    working_profile["policy"] = deepcopy(working_profile.get("policy") or empty_policy())
    attacks = confirmed_hits = dodges = guards = 0
    step = 0
    missing_since = None

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                return _finish("bridge_disconnected", "incomplete", start_health, current, trace,
                    actor_uid=target_uid, attacks=attacks, confirmed_hits=confirmed_hits,
                    dodges=dodges, guards=guards, started_at=started_at, skill=decision.skill)
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return _finish("world_changed", "incomplete", start_health, current, trace,
                    actor_uid=target_uid, attacks=attacks, confirmed_hits=confirmed_hits,
                    dodges=dodges, guards=guards, started_at=started_at, skill=decision.skill)
            if current.player and (current.player.health == 0 or current.game_over_state):
                return _finish("player_defeated", "loss", start_health, current, trace,
                    actor_uid=target_uid, attacks=attacks, confirmed_hits=confirmed_hits,
                    dodges=dodges, guards=guards, started_at=started_at, skill=decision.skill)
            if (not current.in_game or not current.player or current.dialogue.active or current.cutscene_active
                    or current.paused):
                return _finish("gameplay_state_changed", "incomplete", start_health, current, trace,
                    actor_uid=target_uid, attacks=attacks, confirmed_hits=confirmed_hits,
                    dodges=dodges, guards=guards, started_at=started_at, skill=decision.skill)

            defeated = next((e for e in current.events if e.id not in events_before
                and e.kind in {"enemy_defeated", "boss_defeated"}
                and (e.actor_uid == target_uid if target_uid else e.detail == str(decision.args.target_actor_id))), None)
            if defeated:
                return _finish(defeated.kind, "win", start_health, current, trace,
                    actor_uid=target_uid, attacks=attacks, confirmed_hits=confirmed_hits,
                    dodges=dodges, guards=guards, started_at=started_at, skill=decision.skill)

            actor = _matching_actor(current, decision.args.target_actor_id,
                                    decision.args.target_actor_params, target_uid)
            if actor is None:
                missing_since = missing_since or time.monotonic()
                bridge.release()
                if time.monotonic() - missing_since > .65:
                    return _finish("enemy_lost_without_defeat_event", "incomplete", start_health, current, trace,
                        actor_uid=target_uid, attacks=attacks, confirmed_hits=confirmed_hits,
                        dodges=dodges, guards=guards, started_at=started_at, skill=decision.skill)
                await feedback(bridge, current, .05)
                continue
            missing_since = None

            now = time.monotonic()
            if current.player.health < previous_player_health:
                recent_damage_until = now + .9
            assessment = _assessment(current, actor, previous_distance,
                max(.001, now-previous_at), now < recent_damage_until)
            state = combat_state_key(distance=actor.distance, threat_score=assessment.threat_score,
                locked=assessment.locked, disabled=assessment.disabled,
                closing_rate=assessment.closing_rate, facing_player=assessment.facing_player,
                recent_damage=assessment.recent_damage)
            shield = _shield_equipped(current)
            available = _available_actions(current, assessment, shield)

            # Motor constraints are not strategy: don't queue B during an active swing.
            if current.player.melee_weapon_state != 0:
                available = ["guard"] if shield and "guard" in available else ["hold"]
            action = choose_action(working_profile, state, available,
                priors=_priors(assessment, shield), step=step)

            before_action = current
            actor_before = actor
            action_result = await _do_action(bridge, action, current, actor, decision.args.strength, shield)
            after = action_result.get("after") or bridge.state or current
            actor_after = _matching_actor(after, decision.args.target_actor_id,
                                          decision.args.target_actor_params, target_uid) if after else None
            reward = _reward(action, before_action, after, actor_before, actor_after, action_result, assessment)
            trace.append({"state": state, "action": action, "reward": reward,
                          "detail": action_result.get("detail", "")})
            working_profile["policy"] = update_policy(working_profile["policy"], state, action, reward)

            attacks += int(action == "attack")
            confirmed_hits += int(actor_after is not None and
                actor_before.collision_health_hint is not None and actor_after.collision_health_hint is not None and
                actor_after.collision_health_hint < actor_before.collision_health_hint)
            dodges += int(action.startswith("dodge_") and action_result.get("ok"))
            guards += int(action == "guard")
            previous_player_health = after.player.health if after.player else previous_player_health
            previous_distance = actor_after.distance if actor_after else actor.distance
            previous_at = time.monotonic()
            step += 1
    finally:
        bridge.release()

    return _finish("combat_window_complete", "incomplete", start_health, bridge.state, trace,
        actor_uid=target_uid, attacks=attacks, confirmed_hits=confirmed_hits, dodges=dodges,
        guards=guards, started_at=started_at, skill=decision.skill)
