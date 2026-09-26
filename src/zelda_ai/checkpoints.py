"""Deterministic observable quest checkpoints.

The plan does not encode physical routes or hidden transition destinations. It
only keeps the high-level vanilla opening sequence stable and derives completion
from native save/progress state so it survives backend restarts.
"""
from __future__ import annotations

from .models import Decision, GameState


PLAN_ID = "vanilla_child_opening_v1"


def _has_equipment_value(game: GameState, equipment_type: str, value: int) -> bool:
    return any(row.equipment_type == equipment_type and row.value == value
               for row in game.progress.equipment)


def _equipped_value(game: GameState, equipment_type: str, value: int) -> bool:
    return any(row.equipment_type == equipment_type and row.value == value and row.equipped
               for row in game.progress.equipment)


def _flag(game: GameState, name: str) -> bool:
    return bool(game.progress.story_flags.get(name, False))


def _inside_deku_tree(game: GameState) -> bool:
    label = game.scene_name.casefold()
    return "deku tree" in label and "kokiri forest" not in label


def opening_checkpoint_plan(game: GameState) -> dict:
    """Return the observable child-opening plan and its single active checkpoint."""
    if game.source == "soh" and "story_progress_v1" not in game.capabilities:
        return {
            "plan_id": PLAN_ID,
            "available": False,
            "reason": "story_progress_bridge_upgrade_required",
            "active": False,
            "completed_count": 0,
            "total": 8,
            "completed": [],
            "current": None,
            "upcoming": [],
            "do_not_repeat": [],
        }
    has_sword = _has_equipment_value(game, "sword", 1)
    has_shield = _has_equipment_value(game, "shield", 1)
    has_emerald = _flag(game, "obtained_kokiri_emerald")
    sword_equipped = _equipped_value(game, "sword", 1)
    shield_equipped = _equipped_value(game, "shield", 1)
    greeted_saria = _flag(game, "greeted_by_saria")
    showed_mido = _flag(game, "showed_mido_sword_shield")
    met_deku_tree = _flag(game, "met_deku_tree")
    tree_open = _flag(game, "deku_tree_opened_mouth")

    steps = [
        {
            "id": "leave_links_house",
            "title": "Leave Link's House",
            "instruction": "Leave the house through the observed scene-exit surface. Do not wait for a door actor.",
            "done": game.scene_name != "Link's House",
            "completion": "scene changes away from Link's House",
        },
        {
            "id": "greet_saria_once",
            "title": "Complete Saria's opening greeting once",
            "instruction": "Finish the initial Saria greeting/dialogue once. After the native greeted flag is set, do not talk to Saria again for this opening plan.",
            "done": greeted_saria,
            "completion": "native story flag greeted_by_saria",
        },
        {
            "id": "obtain_kokiri_sword",
            "title": "Obtain the Kokiri Sword",
            "instruction": "Explore Kokiri Forest specifically for the Kokiri Sword. Do not pursue Mido yet and do not return to Saria. If straight NavMesh is blocked by water or disconnected platforms, treat that as a route-discovery problem instead of repeating navigate_to.",
            "done": has_sword,
            "completion": "Kokiri Sword appears in owned equipment",
        },
        {
            "id": "obtain_deku_shield",
            "title": "Obtain the Deku Shield",
            "instruction": "Acquire the Deku Shield. Gather rupees as needed and use the Kokiri Shop when discovered. Do not approach Mido until both required items are owned.",
            "done": has_shield,
            "completion": "Deku Shield appears in owned equipment",
        },
        {
            "id": "equip_sword_and_shield",
            "title": "Equip Kokiri Sword and Deku Shield",
            "instruction": "Equip both the Kokiri Sword and Deku Shield using the equipment controller. Do not seek Mido until both are equipped.",
            "done": (sword_equipped and shield_equipped) or showed_mido,
            "completion": "both equipment entries are equipped",
        },
        {
            "id": "pass_mido",
            "title": "Pass Mido's equipment gate",
            "instruction": "Now find Mido on the route toward the Deku Tree and interact once with the required sword and shield equipped. Do not return to Saria.",
            "done": showed_mido or tree_open or met_deku_tree,
            "completion": "native story flag showed_mido_sword_shield or later tree progress",
        },
        {
            "id": "meet_deku_tree",
            "title": "Meet the Great Deku Tree",
            "instruction": "Continue to the Great Deku Tree and complete the initial interaction that opens the dungeon entrance.",
            "done": met_deku_tree or tree_open or _inside_deku_tree(game),
            "completion": "native met/open-tree flag or dungeon scene observed",
        },
        {
            "id": "enter_deku_tree",
            "title": "Enter the Deku Tree",
            "instruction": "Enter the newly available Deku Tree interior and continue the main quest.",
            "done": _inside_deku_tree(game) or has_emerald,
            "completion": "Inside the Deku Tree scene observed",
        },
    ]

    completed = [{"id": row["id"], "title": row["title"]} for row in steps if row["done"]]
    current_index = next((i for i, row in enumerate(steps) if not row["done"]), len(steps))
    current = None if current_index >= len(steps) else {
        key: value for key, value in steps[current_index].items() if key != "done"
    }
    upcoming = [
        {"id": row["id"], "title": row["title"]}
        for row in steps[current_index + 1:current_index + 4]
    ] if current else []

    blocked = []
    if greeted_saria:
        blocked.append("Do not talk to Saria again during this opening plan unless a future checkpoint explicitly requires it.")
    if not (has_sword and has_shield and sword_equipped and shield_equipped):
        blocked.append("Do not make Mido the active goal before Kokiri Sword and Deku Shield are owned and equipped.")
    if showed_mido:
        blocked.append("Do not repeat the Mido equipment-gate conversation after it is cleared.")

    return {
        "plan_id": PLAN_ID,
        "available": True,
        "reason": "",
        "active": current is not None,
        "completed_count": len(completed),
        "total": len(steps),
        "completed": completed,
        "current": current,
        "upcoming": upcoming,
        "do_not_repeat": blocked,
    }


def checkpoint_blocks_decision(game: GameState, decision: Decision, checkpoint: dict) -> dict | None:
    """Fail closed on obvious repeated NPC interactions that contradict completed checkpoints."""
    if not checkpoint.get("available", True) or not checkpoint.get("active", False):
        return None
    if decision.skill not in {"talk_to_actor", "interact_with_actor", "approach_actor", "follow_actor"}:
        return None
    if decision.args.target_actor_id is None:
        return None

    actor = next((row for row in game.room_actors
        if row.actor_id == decision.args.target_actor_id and
        (decision.args.target_actor_params is None or row.params == decision.args.target_actor_params)), None)
    if actor is None:
        return None

    label = f"{actor.name} {actor.description}".casefold()
    current_id = (checkpoint.get("current") or {}).get("id")
    if _flag(game, "greeted_by_saria") and "saria" in label and current_id != "greet_saria_once":
        return {
            "status": "failed",
            "reason": "checkpoint_repeat_blocked",
            "blocked": "saria_already_greeted",
            "checkpoint": current_id,
            "skill": decision.skill,
        }
    if "mido" in label and current_id != "pass_mido":
        return {
            "status": "failed",
            "reason": "checkpoint_repeat_blocked",
            "blocked": ("mido_gate_already_cleared" if _flag(game, "showed_mido_sword_shield")
                        else "mido_not_current_checkpoint"),
            "checkpoint": current_id,
            "skill": decision.skill,
        }
    return None
