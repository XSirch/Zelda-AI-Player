"""Legacy read-only intent helpers retained for import compatibility; not dispatched."""
from __future__ import annotations

import math
import re
from ..models import Decision, GameState
from .common import _is_door_actor, _matching_actor

def _door_intent_actor(game: GameState, decision: Decision):
    doors = [actor for actor in game.room_actors if _is_door_actor(actor)]
    if not doors:
        return None

    if decision.args.target_actor_id is not None:
        target = _matching_actor(game, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
        if _is_door_actor(target):
            return target

    tokens = set(re.findall(r"[a-zà-ÿ]+", f"{decision.goal} {decision.summary}".lower()))
    exit_intent = bool(tokens & {
        "door", "exit", "leave", "outside", "entrance", "enter",
        "porta", "sair", "saída", "saida", "entrada",
    })
    if not exit_intent:
        return None

    if decision.skill == "navigate_to" and decision.args.target_position is not None:
        px, py, pz = decision.args.target_position
        near = [door for door in doors if math.dist(door.position, (px, py, pz)) <= 180.0]
        return min(near, key=lambda actor: actor.distance) if near else None

    explicit_door = bool(tokens & {"door", "porta"})
    if explicit_door and decision.skill in {"move", "turn", "camera_center", "explore_area"} and len(doors) == 1:
        return doors[0]
    return None


def _traversal_intent_direction(decision: Decision) -> str | None:
    if decision.skill == "traverse":
        return decision.args.direction
    tokens = set(re.findall(r"[a-zà-ÿ]+", f"{decision.goal} {decision.summary}".lower()))
    if tokens & {"descend", "descending", "down", "downstairs",
                 "descer", "descendo", "baixo", "embaixo"}:
        return "down"
    if tokens & {"ascend", "ascending", "climb", "upstairs",
                 "subir", "subindo", "cima"}:
        return "up"
    return None


def _as_door_interaction(decision: Decision, door) -> Decision:
    args = decision.args.model_copy(update={
        "target_actor_id": door.actor_id,
        "target_actor_params": door.params,
        "target_position": None,
        "stop_distance": None,
        "duration_ms": max(5000, decision.args.duration_ms),
    })
    return Decision.model_validate({
        **decision.model_dump(),
        "skill": "interact_with_actor",
        "summary": f"Use dedicated door controller: {decision.summary}"[:280],
        "args": args.model_dump(),
    })
