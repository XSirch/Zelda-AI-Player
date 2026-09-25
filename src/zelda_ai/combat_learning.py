"""Per-enemy combat learning shared by runtime, controller and persistence."""
from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Iterable

from .models import ActorObservation, GameState


def enemy_key(game: GameState, actor: ActorObservation) -> str:
    age = game.player.age if game.player else "unknown"
    # Actor ID identifies the enemy class. Params are retained per encounter but do not
    # fragment learning across spawn switches/room-specific parameters.
    return f"{age}:{actor.category}:{actor.actor_id}"


def empty_policy() -> dict:
    return {"states": {}, "best_by_state": {}}


def combat_state_key(*, distance: float, threat_score: int, locked: bool,
                     disabled: bool = False, closing_rate: float = 0.0,
                     facing_player: bool = False, recent_damage: bool = False) -> str:
    if distance > 160:
        band = "far"
    elif distance > 105:
        band = "mid"
    elif distance >= 48:
        band = "melee"
    else:
        band = "point_blank"
    threat = "high" if threat_score >= 4 else ("medium" if threat_score >= 2 else "low")
    if closing_rate >= 80:
        motion = "rush"
    elif closing_rate >= 25:
        motion = "closing"
    elif closing_rate <= -25:
        motion = "retreating"
    else:
        motion = "steady"
    facing = "facing" if facing_player else "away"
    damage = "hurt_recently" if recent_damage else "clean"
    return f"{band}|{threat}|{motion}|{facing}|{damage}|{'locked' if locked else 'unlocked'}|{'disabled' if disabled else 'active'}"


def _stat(policy: dict, state: str, action: str) -> dict:
    return ((policy or {}).get("states") or {}).get(state, {}).get(action, {})


def choose_action(profile: dict | None, state: str, available: Iterable[str],
                  *, priors: dict[str, float] | None = None, step: int = 0) -> str:
    """Deterministic UCB-like exploration. Safety is enforced by the available-action set."""
    actions = list(dict.fromkeys(available))
    if not actions:
        return "hold"
    policy = (profile or {}).get("policy") or empty_policy()
    encounters = int((profile or {}).get("encounters") or 0)
    priors = priors or {}
    scored = []
    for action in actions:
        stat = _stat(policy, state, action)
        n = int(stat.get("n") or 0)
        mean = float(stat.get("mean") or 0.0)
        exploration = 0.55 / math.sqrt(n + 1)
        novelty = 0.22 if n == 0 else 0.0
        digest = hashlib.sha256(f"{(profile or {}).get('enemy_key','new')}|{state}|{action}|{encounters}|{step}".encode()).digest()
        tie_break = int.from_bytes(digest[:2], "big") / 65535 * 0.01
        scored.append((mean + exploration + novelty + float(priors.get(action, 0.0)) + tie_break, action))
    return max(scored)[1]


def update_policy(policy: dict | None, state: str, action: str, reward: float) -> dict:
    result = deepcopy(policy or empty_policy())
    states = result.setdefault("states", {})
    actions = states.setdefault(state, {})
    stat = dict(actions.get(action) or {})
    n = int(stat.get("n") or 0) + 1
    total = float(stat.get("reward_sum") or 0.0) + max(-10.0, min(10.0, float(reward)))
    stat.update({"n": n, "reward_sum": round(total, 5), "mean": round(total / n, 5)})
    actions[action] = stat
    result["best_by_state"] = best_actions(result)
    return result


def best_actions(policy: dict | None) -> dict[str, dict]:
    states = (policy or {}).get("states") or {}
    output: dict[str, dict] = {}
    for state, actions in states.items():
        candidates = [(float(stat.get("mean") or 0.0), int(stat.get("n") or 0), action)
                      for action, stat in actions.items() if int(stat.get("n") or 0) > 0]
        if candidates:
            mean, n, action = max(candidates)
            output[state] = {"action": action, "mean": round(mean, 4), "samples": n}
    return output


def compact_profile(profile: dict | None) -> dict | None:
    if not profile:
        return None
    return {
        "id": profile.get("id"),
        "enemy_key": profile.get("enemy_key"),
        "actor_id": profile.get("actor_id"),
        "enemy_name": profile.get("enemy_name") or "",
        "category": profile.get("category"),
        "encounters": profile.get("encounters") or 0,
        "wins": profile.get("wins") or 0,
        "losses": profile.get("losses") or 0,
        "incomplete": profile.get("incomplete") or 0,
        "damage_taken": profile.get("damage_taken") or 0,
        "best_by_state": ((profile.get("policy") or {}).get("best_by_state") or {}),
        "updated_at": profile.get("updated_at"),
    }
