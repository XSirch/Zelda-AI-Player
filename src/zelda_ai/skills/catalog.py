"""Local catalog controller helpers."""
from __future__ import annotations

from ..models import Decision

BUTTONS = {"A": 0x8000, "B": 0x4000, "Z": 0x2000, "START": 0x1000, "R": 0x0010,
    "C_UP": 0x0008, "C_LEFT": 0x0002, "C_DOWN": 0x0004, "C_RIGHT": 0x0001}


DIALOGUE_SKILLS = {"advance_dialogue", "choose_dialogue"}


MENU_SKILLS = {"pause_toggle", "menu_move", "menu_confirm", "menu_cancel", "menu_assign",
    "continue_gameover", "equip_item", "equip_gear"}


NAVIGATION_SKILLS = {"move", "turn", "interact", "wait", "camera_center", "roll", "backflip", "sidestep",
    "navigate_to", "approach_actor", "interact_with_actor", "traverse", "traverse_to", "traverse_exit"}


SONG_IDS = {"minuet": 0, "bolero": 1, "serenade": 2, "requiem": 3, "nocturne": 4, "prelude": 5,
    "sarias": 6, "eponas": 7, "lullaby": 8, "suns": 9, "time": 10, "storms": 11}


SONG_NOTES = {
    "minuet": ["A", "C_UP", "C_LEFT", "C_RIGHT", "C_LEFT", "C_RIGHT"],
    "bolero": ["C_DOWN", "A", "C_DOWN", "A", "C_RIGHT", "C_DOWN", "C_RIGHT", "C_DOWN"],
    "serenade": ["A", "C_DOWN", "C_RIGHT", "C_RIGHT", "C_LEFT"],
    "requiem": ["A", "C_DOWN", "A", "C_RIGHT", "C_DOWN", "A"],
    "nocturne": ["C_LEFT", "C_RIGHT", "C_RIGHT", "A", "C_LEFT", "C_RIGHT", "C_DOWN"],
    "prelude": ["C_UP", "C_RIGHT", "C_UP", "C_RIGHT", "C_LEFT", "C_UP"],
    "sarias": ["C_DOWN", "C_RIGHT", "C_LEFT", "C_DOWN", "C_RIGHT", "C_LEFT"],
    "eponas": ["C_UP", "C_LEFT", "C_RIGHT", "C_UP", "C_LEFT", "C_RIGHT"],
    "lullaby": ["C_LEFT", "C_UP", "C_RIGHT", "C_LEFT", "C_UP", "C_RIGHT"],
    "suns": ["C_RIGHT", "C_DOWN", "C_UP", "C_RIGHT", "C_DOWN", "C_UP"],
    "time": ["C_RIGHT", "A", "C_DOWN", "C_RIGHT", "A", "C_DOWN"],
    "storms": ["A", "C_DOWN", "C_UP", "A", "C_DOWN", "C_UP"],
}


SKILL_CATALOG = [
    {"id": "move", "name": "Translação curta", "status": "implemented", "version": "1.0"},
    {"id": "turn", "name": "Giro local com feedback de yaw", "status": "implemented", "version": "1.0"},
    {"id": "interact", "name": "Interagir / confirmar (A)", "status": "implemented", "version": "1.0"},
    {"id": "advance_dialogue", "name": "Avançar página de diálogo", "status": "implemented", "version": "1.0"},
    {"id": "choose_dialogue", "name": "Selecionar opção de diálogo", "status": "implemented", "version": "1.0"},
    {"id": "camera_center", "name": "Recentrar câmera", "status": "implemented", "version": "1.0"},
    {"id": "roll", "name": "Rolamento para frente", "status": "implemented", "version": "1.0"},
    {"id": "backflip", "name": "Backflip Link-relative com confirmação nativa", "status": "implemented", "version": "3.0"},
    {"id": "sidestep", "name": "Esquiva lateral Link-relative com confirmação nativa", "status": "implemented", "version": "3.0"},
    {"id": "jump_attack", "name": "Jump attack (Z+A)", "status": "implemented", "version": "1.0"},
    {"id": "attack", "name": "Ataque básico (B)", "status": "implemented", "version": "1.0"},
    {"id": "defend", "name": "Defesa (Z+R)", "status": "implemented", "version": "1.0"},
    {"id": "target", "name": "Z-target", "status": "implemented", "version": "1.0"},
    {"id": "use_item", "name": "Usar item já equipado em C", "status": "implemented", "version": "1.0"},
    {"id": "wait", "name": "Esperar com controle neutro", "status": "implemented", "version": "1.0"},
    {"id": "pause_toggle", "name": "Abrir/fechar pause menu", "status": "implemented", "version": "1.0"},
    {"id": "menu_move", "name": "Mover cursor do pause menu", "status": "implemented", "version": "1.0"},
    {"id": "menu_confirm", "name": "Confirmar seleção no menu", "status": "implemented", "version": "1.0"},
    {"id": "menu_cancel", "name": "Cancelar/voltar no menu", "status": "implemented", "version": "1.0"},
    {"id": "menu_assign", "name": "Atribuir item selecionado a C", "status": "implemented", "version": "1.0"},
    {"id": "continue_gameover", "name": "Confirmar continue após game over", "status": "implemented", "version": "1.0"},
    {"id": "play_song", "name": "Tocar música conhecida na ocarina", "status": "implemented", "version": "1.0"},
    {"id": "navigate_to", "name": "NavMesh local + A* até posição observada", "status": "implemented", "version": "3.0"},
    {"id": "approach_actor", "name": "Aproximar-se de ator via NavMesh local", "status": "implemented", "version": "3.0"},
    {"id": "follow_actor", "name": "Seguir ator móvel via NavMesh local", "status": "implemented", "version": "3.0"},
    {"id": "talk_to_actor", "name": "Aproximar e iniciar conversa com ator", "status": "implemented", "version": "2.0"},
    {"id": "interact_with_actor", "name": "Aproximar/interagir com ator e verificar efeito", "status": "implemented", "version": "2.0"},
    {"id": "equip_item", "name": "Equipar item possuído em C via pause menu", "status": "implemented", "version": "2.0"},
    {"id": "equip_gear", "name": "Equipar espada/escudo/túnica/botas via pause menu", "status": "implemented", "version": "2.0"},
    {"id": "aim_at", "name": "Mira fechada e disparo com item C equipado", "status": "implemented", "version": "2.0"},
    {"id": "face_target", "name": "Orientar Link para ator/posição", "status": "implemented", "version": "2.0"},
    {"id": "shield_face", "name": "Orientar e sustentar escudo para alvo", "status": "implemented", "version": "2.0"},
    {"id": "fight_enemy", "name": "Combate aprendido por tipo de inimigo", "status": "implemented", "version": "4.0"},
    {"id": "manipulate_object", "name": "Agarrar/empurrar/puxar objeto observado", "status": "implemented", "version": "2.0"},
    {"id": "explore_area", "name": "Exploração local com NavMesh e descoberta", "status": "implemented", "version": "3.0"},
    {"id": "traverse", "name": "Subir/descer escadas, ladders e superfícies escaláveis", "status": "implemented", "version": "3.0"},
    {"id": "traverse_to", "name": "Navegar até affordance vertical e atravessar", "status": "implemented", "version": "1.0"},
    {"id": "traverse_exit", "name": "Entrar em surface de saída de cena observada", "status": "implemented", "version": "1.0"},
    {"id": "gameover_recovery", "name": "Save/continue e respawn automáticos", "status": "implemented", "version": "2.0"},
    {"id": "vision_fallback", "name": "Visão sob demanda após falhas estruturadas", "status": "planned", "version": None},
]


def controller_input(decision: Decision) -> tuple[int, int, int]:
    skill, args = decision.skill, decision.args
    amount = round(80 * args.strength)
    if skill == "move":
        x, y = {"forward": (0, amount), "back": (0, -amount), "left": (-amount, 0), "right": (amount, 0)}[args.direction]
        return 0, x, y
    if skill == "turn":
        steer = max(28, round(72 * args.strength))
        return 0, -steer if args.direction == "left" else steer, 12
    if skill == "sidestep":
        return BUTTONS["Z"] | BUTTONS["A"], -max(50, amount) if args.direction == "left" else max(50, amount), 0
    if skill == "backflip":
        return BUTTONS["Z"] | BUTTONS["A"], 0, -max(50, amount)
    if skill == "roll":
        return BUTTONS["A"], 0, max(45, amount)
    if skill == "jump_attack":
        return BUTTONS["Z"] | BUTTONS["A"], 0, 0
    if skill == "menu_move":
        return 0, {"left": -60, "right": 60}.get(args.direction, 0), {"up": 60, "down": -60}.get(args.direction, 0)
    button = {
        "interact": BUTTONS["A"],
        "advance_dialogue": BUTTONS["A"],
        "attack": BUTTONS["B"],
        "defend": BUTTONS["Z"] | BUTTONS["R"],
        "target": BUTTONS["Z"],
        "camera_center": BUTTONS["Z"],
        "wait": 0,
        "choose_dialogue": 0,
        "pause_toggle": BUTTONS["START"],
        "menu_confirm": BUTTONS["A"],
        "menu_cancel": BUTTONS["B"],
        "continue_gameover": BUTTONS["A"],
        "play_song": 0,
        "menu_move": 0,
    }.get(skill)
    if skill in {"use_item", "menu_assign"}:
        button = BUTTONS[f"C_{args.slot.upper()}"]
    return button or 0, 0, 0
