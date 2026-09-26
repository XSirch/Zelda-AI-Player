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
    inside_tree = _inside_deku_tree(game)
    terminal_opening = has_emerald
    passed_mido = showed_mido or tree_open or met_deku_tree or inside_tree or terminal_opening
    greeted_done = greeted_saria or passed_mido
    sword_done = has_sword or passed_mido
    shield_done = has_shield or passed_mido
    equipment_done = (sword_equipped and shield_equipped) or passed_mido
    tree_met_done = met_deku_tree or tree_open or inside_tree or terminal_opening
    tree_entered_done = inside_tree or terminal_opening

    steps = [
        {
            "id": "leave_links_house",
            "title": "Sair da Casa do Link",
            "instruction": "Saia da casa pela superfície de transição observada. Não espere existir um ator de porta.",
            "done": game.scene_name != "Link's House" or passed_mido,
            "completion": "A cena muda para fora da Casa do Link",
        },
        {
            "id": "greet_saria_once",
            "title": "Concluir uma vez a saudação inicial da Saria",
            "instruction": "Conclua uma vez a saudação/diálogo inicial da Saria. Depois que o progresso nativo registrar isso, não fale com Saria novamente neste plano inicial.",
            "done": greeted_done,
            "completion": "O save registra que Saria já cumprimentou Link",
        },
        {
            "id": "obtain_kokiri_sword",
            "title": "Obter a Kokiri Sword",
            "instruction": "Explore Kokiri Forest especificamente para obter a Kokiri Sword. Não persiga Mido ainda e não volte para Saria. Se o NavMesh direto estiver bloqueado por água ou plataformas desconectadas, trate isso como descoberta de rota em vez de repetir navigate_to.",
            "done": sword_done,
            "completion": "Kokiri Sword aparece entre os equipamentos possuídos",
        },
        {
            "id": "obtain_deku_shield",
            "title": "Obter o Deku Shield",
            "instruction": "Adquira o Deku Shield. Junte rupias conforme necessário e use a Kokiri Shop quando ela for descoberta. Não se aproxime de Mido até possuir os dois equipamentos exigidos.",
            "done": shield_done,
            "completion": "Deku Shield aparece entre os equipamentos possuídos",
        },
        {
            "id": "equip_sword_and_shield",
            "title": "Equipar Kokiri Sword e Deku Shield",
            "instruction": "Equipe Kokiri Sword e Deku Shield usando o controlador de equipamento. Não procure Mido até os dois estarem equipados.",
            "done": equipment_done,
            "completion": "Os dois equipamentos estão equipados",
        },
        {
            "id": "pass_mido",
            "title": "Passar pelo bloqueio de equipamento do Mido",
            "instruction": "Agora encontre Mido no caminho para a Great Deku Tree e interaja uma vez com espada e escudo exigidos equipados. Não volte para Saria.",
            "done": passed_mido,
            "completion": "O save registra que Mido verificou espada/escudo ou progresso posterior da árvore",
        },
        {
            "id": "meet_deku_tree",
            "title": "Encontrar a Great Deku Tree",
            "instruction": "Continue até a Great Deku Tree e conclua a interação inicial que libera a entrada da dungeon.",
            "done": tree_met_done,
            "completion": "O save registra o encontro/abertura da árvore ou a cena interna é observada",
        },
        {
            "id": "enter_deku_tree",
            "title": "Entrar na Deku Tree",
            "instruction": "Entre no interior agora disponível da Deku Tree e continue a missão principal.",
            "done": tree_entered_done,
            "completion": "A cena interna da Deku Tree é observada ou o Kokiri Emerald já foi obtido",
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
        blocked.append("Não fale com Saria novamente durante este plano inicial, salvo se um checkpoint futuro exigir explicitamente.")
    if not (has_sword and has_shield and sword_equipped and shield_equipped):
        blocked.append("Não torne Mido o objetivo ativo antes de possuir e equipar Kokiri Sword e Deku Shield.")
    if showed_mido:
        blocked.append("Não repita a conversa do bloqueio de equipamento do Mido depois que ela for concluída.")

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
    if decision.args.target_actor_id is None and decision.args.target_actor_uid is None:
        return None

    actor = next((row for row in game.room_actors
        if ((decision.args.target_actor_uid is not None and
             row.actor_uid == decision.args.target_actor_uid) or
            (decision.args.target_actor_uid is None and
             decision.args.target_actor_id is not None and
             row.actor_id == decision.args.target_actor_id and
             (decision.args.target_actor_params is None or
              row.params == decision.args.target_actor_params)))), None)
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
