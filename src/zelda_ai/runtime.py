"""One model decision at a time; local skill execution never calls a model per frame."""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import time
from collections import deque

from pydantic import ValidationError

from .bridge import Bridge
from .models import Decision, GameState, ModelInfo, RunConfig, SwitchConfig
from .providers.base import ProviderFailure
from .providers.openrouter import reserve_cost
from .store import Store

CONTRACT_VERSION = "state-v2/skills-v2/trajectory-v1/prompt-v5"
BUTTONS = {"A": 0x8000, "B": 0x4000, "Z": 0x2000, "START": 0x1000, "R": 0x0010,
    "C_UP": 0x0008, "C_LEFT": 0x0002, "C_DOWN": 0x0004, "C_RIGHT": 0x0001}
DIALOGUE_SKILLS = {"advance_dialogue", "choose_dialogue"}
MENU_SKILLS = {"pause_toggle", "menu_move", "menu_confirm", "menu_cancel", "menu_assign",
    "continue_gameover", "equip_item", "equip_gear"}
NAVIGATION_SKILLS = {"move", "turn", "interact", "wait", "camera_center", "roll", "backflip", "sidestep",
    "navigate_to", "approach_actor", "interact_with_actor"}
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
    {"id": "backflip", "name": "Backflip com Z-target", "status": "implemented", "version": "1.0"},
    {"id": "sidestep", "name": "Esquiva lateral com Z-target", "status": "implemented", "version": "1.0"},
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
    {"id": "navigate_to", "name": "Servo local até posição observada", "status": "implemented", "version": "2.0"},
    {"id": "approach_actor", "name": "Aproximar-se de ator desenhado", "status": "implemented", "version": "2.0"},
    {"id": "follow_actor", "name": "Seguir ator móvel mantendo distância", "status": "implemented", "version": "2.0"},
    {"id": "talk_to_actor", "name": "Aproximar e iniciar conversa com ator", "status": "implemented", "version": "2.0"},
    {"id": "interact_with_actor", "name": "Aproximar/interagir com ator e verificar efeito", "status": "implemented", "version": "2.0"},
    {"id": "equip_item", "name": "Equipar item possuído em C via pause menu", "status": "implemented", "version": "2.0"},
    {"id": "equip_gear", "name": "Equipar espada/escudo/túnica/botas via pause menu", "status": "implemented", "version": "2.0"},
    {"id": "aim_at", "name": "Mira fechada e disparo com item C equipado", "status": "implemented", "version": "2.0"},
    {"id": "fight_enemy", "name": "Combate genérico contra inimigo observado", "status": "implemented", "version": "2.0"},
    {"id": "manipulate_object", "name": "Agarrar/empurrar/puxar objeto observado", "status": "implemented", "version": "2.0"},
    {"id": "explore_area", "name": "Exploração local com colisão e descoberta", "status": "implemented", "version": "2.0"},
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
        return BUTTONS["Z"], -amount if args.direction == "left" else amount, 0
    if skill == "backflip":
        return BUTTONS["Z"], 0, -max(50, amount)
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


async def _choose_dialogue(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    if not current or not bridge.connected:
        raise RuntimeError("bridge_disconnected")
    if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    target = decision.args.choice_index
    dialogue = current.dialogue
    if not dialogue.active or dialogue.choice_count < 2 or target is None or target >= dialogue.choice_count:
        return {"status": "failed", "reason": "dialogue_choice_not_available", "skill": decision.skill}

    acknowledged = False
    first_command = None
    for _ in range(5):
        current = bridge.state
        if not current or (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
            return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
        if not current.dialogue.active or current.dialogue.choice_count < 2:
            return {"status": "interrupted", "reason": "dialogue_changed", "skill": decision.skill}
        if current.dialogue.choice_index == target:
            break
        stick_y = -60 if current.dialogue.choice_index < target else 60
        command_id = bridge.send(stick_y=stick_y, lease_ms=120)
        if first_command is None:
            first_command = command_id
        await asyncio.sleep(0.14)
        bridge.release()
        await asyncio.sleep(0.14)
        sample = bridge.state
        if sample and first_command is not None:
            acknowledged |= sample.last_command_seq >= first_command
    current = bridge.state
    if not current or current.dialogue.choice_index != target:
        return {"status": "failed", "reason": "choice_cursor_did_not_reach_target",
            "acknowledged": acknowledged, "skill": decision.skill}

    command_id = bridge.send(buttons=BUTTONS["A"], lease_ms=120)
    if first_command is None:
        first_command = command_id
    await asyncio.sleep(0.14)
    bridge.release()
    await asyncio.sleep(0.2)
    after = bridge.state
    if after and first_command is not None:
        acknowledged |= after.last_command_seq >= first_command
    return {"status": "completed" if acknowledged else "failed",
        "reason": "choice_confirmed" if acknowledged else "input_not_acknowledged",
        "choice_index": target, "acknowledged": acknowledged, "skill": decision.skill}


async def _play_song(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    song = decision.args.song
    if not current or not bridge.connected:
        raise RuntimeError("bridge_disconnected")
    if song not in SONG_NOTES:
        return {"status": "failed", "reason": "unknown_song", "skill": decision.skill}
    if current.ocarina_mode == 0:
        return {"status": "failed", "reason": "ocarina_not_active", "skill": decision.skill}
    before_event_ids = {event.id for event in current.events}
    acknowledged = False
    first_command = None
    for note in SONG_NOTES[song]:
        current = bridge.state
        if not current or (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
            return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
        if current.ocarina_mode == 0:
            return {"status": "interrupted", "reason": "ocarina_closed", "skill": decision.skill}
        command_id = bridge.send(buttons=BUTTONS[note], lease_ms=90)
        if first_command is None:
            first_command = command_id
        await asyncio.sleep(0.11)
        bridge.release()
        await asyncio.sleep(0.07)
        sample = bridge.state
        if sample and first_command is not None:
            acknowledged |= sample.last_command_seq >= first_command
    await asyncio.sleep(0.35)
    after = bridge.state
    if after and first_command is not None:
        acknowledged |= after.last_command_seq >= first_command
    expected = SONG_IDS[song]
    confirmed = bool(after and any(
        event.id not in before_event_ids and event.kind == "ocarina_song_action" and event.detail == str(expected)
        for event in after.events))
    return {"status": "completed" if acknowledged else "failed",
        "reason": "song_confirmed" if confirmed else ("notes_sent" if acknowledged else "input_not_acknowledged"),
        "song": song, "song_id": expected, "confirmed": confirmed,
        "acknowledged": acknowledged, "skill": decision.skill}



def _matching_actor(game: GameState, actor_id: int, params: int | None):
    candidates = list(game.nearby_actors)
    if game.target_actor is not None:
        candidates.append(game.target_actor)
    matches = [actor for actor in candidates
        if actor.actor_id == actor_id and (params is None or actor.params == params)]
    return min(matches, key=lambda actor: actor.distance) if matches else None


def _steer_to(game: GameState, target: tuple[float, float, float], strength: float) -> tuple[int, int, float]:
    if not game.player:
        return 0, 0, float("inf")
    px, _, pz = game.player.position
    dx, dz = target[0] - px, target[2] - pz
    distance = math.hypot(dx, dz)
    if distance < 1e-6:
        return 0, 0, 0.0

    if game.camera_eye is not None and game.camera_at is not None:
        fx = game.camera_at[0] - game.camera_eye[0]
        fz = game.camera_at[2] - game.camera_eye[2]
        flen = math.hypot(fx, fz)
    else:
        flen = 0.0
    if flen < 1e-4:
        angle = game.player.yaw * math.pi / 32768.0
        fx, fz = math.sin(angle), math.cos(angle)
    else:
        fx, fz = fx / flen, fz / flen

    # Camera-relative right vector. N64 stick X is right, Y is forward.
    rx, rz = fz, -fx
    ux, uz = dx / distance, dz / distance
    local_x = ux * rx + uz * rz
    local_y = ux * fx + uz * fz
    scale = max(28, min(80, round(80 * max(0.35, strength))))
    return (
        max(-80, min(80, round(local_x * scale))),
        max(-80, min(80, round(local_y * scale))),
        distance,
    )


async def _pulse(bridge: Bridge, *, buttons: int = 0, stick_x: int = 0, stick_y: int = 0,
                 hold_ms: int = 100, settle_s: float = 0.12) -> bool:
    command_id = bridge.send(buttons=buttons, stick_x=stick_x, stick_y=stick_y,
        lease_ms=max(50, min(300, hold_ms)))
    await asyncio.sleep(max(0.05, hold_ms / 1000))
    bridge.release()
    await asyncio.sleep(settle_s)
    current = bridge.state
    return bool(current and current.last_command_seq >= command_id)


async def _navigate_local(bridge: Bridge, decision: Decision, observation: GameState,
                          *, actor_mode: bool, talk: bool = False, interact: bool = False) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}

    deadline = time.monotonic() + min(8.0, max(0.25, decision.args.duration_ms / 1000))
    start_position = before.player.position
    stop_distance = decision.args.stop_distance
    interaction_mode = talk or interact
    if stop_distance is None:
        stop_distance = 90.0 if interaction_mode else (70.0 if actor_mode else 35.0)

    first_command = None
    acknowledged = False
    best_distance = float("inf")
    stagnant_samples = 0
    recentered = False
    detour_attempts = 0
    last_actor = None
    last_progress_seq = -1

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "completed" if interaction_mode else "interrupted",
                    "reason": "world_transition_after_interaction" if interaction_mode else "world_changed",
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
                return {"status": "completed" if interaction_mode else "interrupted",
                    "reason": "interaction_started" if interaction_mode else "cutscene_started", "skill": decision.skill}

            if actor_mode:
                actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params)
                if actor is None:
                    return {"status": "failed", "reason": "target_actor_not_observed", "skill": decision.skill}
                last_actor = actor
                target = actor.position
            else:
                target = decision.args.target_position

            stick_x, stick_y, target_distance = _steer_to(current, target, decision.args.strength)
            if target_distance <= stop_distance:
                if not interaction_mode:
                    return {"status": "completed", "reason": "target_reached",
                        "target_distance": target_distance,
                        "distance": math.dist(start_position, current.player.position),
                        "acknowledged": acknowledged, "skill": decision.skill}

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

            if current.seq != last_progress_seq:
                last_progress_seq = current.seq
                if target_distance + 2.0 < best_distance:
                    best_distance = target_distance
                    stagnant_samples = 0
                else:
                    stagnant_samples += 1

            if (current.player.bg_check_flags & 0x008) and stagnant_samples >= 4 and detour_attempts < 2:
                # Local collision recovery only: sidestep around the contact, then resume target steering.
                side = 60 if detour_attempts == 0 else -60
                acknowledged |= await _pulse(bridge, stick_x=side, stick_y=20, hold_ms=260, settle_s=0.06)
                detour_attempts += 1
                stagnant_samples = 0
                continue
            if stagnant_samples >= 5 and not recentered:
                bridge.release()
                acknowledged |= await _pulse(bridge, buttons=BUTTONS["Z"], hold_ms=80, settle_s=0.15)
                recentered = True
                stagnant_samples = 0
                continue
            if stagnant_samples >= 8:
                return {"status": "failed", "reason": "navigation_no_progress",
                    "target_distance": target_distance,
                    "distance": math.dist(start_position, current.player.position),
                    "acknowledged": acknowledged, "skill": decision.skill}

            command_id = bridge.send(stick_x=stick_x, stick_y=stick_y, lease_ms=250)
            if first_command is None:
                first_command = command_id
            acknowledged |= current.last_command_seq >= first_command
            await asyncio.sleep(0.10)
    finally:
        bridge.release()

    after = bridge.state
    if after and first_command is not None:
        acknowledged |= after.last_command_seq >= first_command
    return {"status": "failed", "reason": "navigation_timeout",
        "target_distance": (last_actor.distance if last_actor else None),
        "distance": math.dist(start_position, after.player.position) if after and after.player else None,
        "acknowledged": acknowledged, "skill": decision.skill}


def _aim_error(game: GameState, target: tuple[float, float, float]) -> tuple[float, float] | None:
    if game.camera_eye is None or game.camera_at is None:
        return None
    eye = game.camera_eye
    at = game.camera_at
    current = (at[0] - eye[0], at[1] - eye[1], at[2] - eye[2])
    desired = (target[0] - eye[0], target[1] - eye[1], target[2] - eye[2])
    cur_h = math.hypot(current[0], current[2])
    dst_h = math.hypot(desired[0], desired[2])
    if cur_h < 1e-4 or dst_h < 1e-4:
        return None
    current_yaw = math.atan2(current[0], current[2])
    desired_yaw = math.atan2(desired[0], desired[2])
    yaw_error = (desired_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi
    current_pitch = math.atan2(current[1], cur_h)
    desired_pitch = math.atan2(desired[1], dst_h)
    return yaw_error, desired_pitch - current_pitch


def _aim_stick(error: float, sign: int) -> int:
    degrees = math.degrees(error)
    if abs(degrees) < 1.2:
        return 0
    magnitude = max(12, min(55, round(abs(degrees) * 2.2)))
    return magnitude * (1 if degrees > 0 else -1) * sign


async def _aim_at(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    if not current or not current.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    if current.paused or current.dialogue.active or current.cutscene_active or current.game_over_state != 0:
        return {"status": "stale", "reason": "gameplay_state_blocks_aim", "skill": decision.skill}

    slot = decision.args.slot
    if slot is None:
        return {"status": "failed", "reason": "aim_slot_missing", "skill": decision.skill}
    button_index = {"left": 1, "down": 2, "right": 3}[slot]
    if len(current.equipped) <= button_index or current.equipped[button_index] in {0xFE, 0xFF}:
        return {"status": "failed", "reason": "aim_slot_empty", "slot": slot, "skill": decision.skill}
    item_id = current.equipped[button_index]
    button = BUTTONS[f"C_{slot.upper()}"]

    deadline = time.monotonic() + min(8.0, max(0.6, decision.args.duration_ms / 1000))
    first_command = None
    acknowledged = False
    sign_x = sign_y = 1
    flipped_x = flipped_y = False
    previous_abs = None
    previous_stick = (0, 0)
    stable = 0
    last_seq = -1
    before_events = {event.id for event in current.events}

    # Give the equipped aiming item a short moment to enter first-person/aim state.
    first_command = bridge.send(buttons=button, lease_ms=300)
    await asyncio.sleep(0.22)

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
            if current.paused or current.dialogue.active or current.cutscene_active or current.game_over_state != 0:
                return {"status": "interrupted", "reason": "gameplay_state_changed", "skill": decision.skill}

            if current.last_command_seq >= first_command:
                acknowledged = True
            if current.seq == last_seq:
                await asyncio.sleep(0.04)
                continue
            last_seq = current.seq

            actor = None
            if decision.args.target_actor_id is not None:
                actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params)
                if actor is None:
                    return {"status": "failed", "reason": "aim_target_actor_not_observed",
                        "item_id": item_id, "skill": decision.skill}
                target = actor.focus_position or actor.position
            else:
                target = decision.args.target_position
            error = _aim_error(current, target)
            if error is None:
                return {"status": "failed", "reason": "camera_state_unavailable",
                    "item_id": item_id, "skill": decision.skill}
            yaw_error, pitch_error = error
            abs_error = (abs(yaw_error), abs(pitch_error))

            if previous_abs is not None:
                if previous_stick[0] != 0 and abs_error[0] > previous_abs[0] + math.radians(0.5) and not flipped_x:
                    sign_x *= -1
                    flipped_x = True
                if previous_stick[1] != 0 and abs_error[1] > previous_abs[1] + math.radians(0.5) and not flipped_y:
                    sign_y *= -1
                    flipped_y = True

            if abs_error[0] <= math.radians(2.2) and abs_error[1] <= math.radians(2.2):
                stable += 1
                if stable >= 2:
                    bridge.release()  # Releasing the held C-button fires bow/slingshot/hookshot-style items.
                    await asyncio.sleep(0.35)
                    after = bridge.state
                    defeated = None
                    if after:
                        acknowledged |= after.last_command_seq >= first_command
                        defeated = next((event for event in after.events
                            if event.id not in before_events and event.kind in {"enemy_defeated", "boss_defeated"}
                            and (decision.args.target_actor_id is None or
                                 event.detail == str(decision.args.target_actor_id))), None)
                    return {"status": "completed",
                        "reason": defeated.kind if defeated else "aligned_shot_released",
                        "item_id": item_id, "slot": slot,
                        "yaw_error_deg": round(math.degrees(yaw_error), 2),
                        "pitch_error_deg": round(math.degrees(pitch_error), 2),
                        "target_defeated": defeated is not None,
                        "acknowledged": acknowledged, "skill": decision.skill}
            else:
                stable = 0

            stick_x = _aim_stick(yaw_error, sign_x)
            stick_y = _aim_stick(pitch_error, sign_y)
            previous_abs = abs_error
            previous_stick = (stick_x, stick_y)
            command_id = bridge.send(buttons=button, stick_x=stick_x, stick_y=stick_y, lease_ms=260)
            if first_command is None:
                first_command = command_id
            await asyncio.sleep(0.10)
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "failed", "reason": "aim_timeout", "item_id": item_id, "slot": slot,
        "acknowledged": bool(after and after.last_command_seq >= first_command) or acknowledged,
        "skill": decision.skill}


async def _fight_enemy(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    actor = _matching_actor(before, decision.args.target_actor_id, decision.args.target_actor_params)
    if actor is None:
        return {"status": "failed", "reason": "target_actor_not_observed", "skill": decision.skill}
    if actor.category not in {5, 9}:  # ACTORCAT_ENEMY / ACTORCAT_BOSS in the pinned SoH revision.
        return {"status": "failed", "reason": "target_is_not_enemy_category",
            "actor_category": actor.category, "skill": decision.skill}

    start_position = before.player.position
    start_health = before.player.health
    deadline = time.monotonic() + min(10.0, max(0.5, decision.args.duration_ms / 1000))
    before_events = {event.id for event in before.events}
    acknowledged = False
    first_command = None
    missing_samples = 0
    cycle = 0

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
            if not current.in_game or not current.player:
                return {"status": "interrupted", "reason": "game_not_ready", "skill": decision.skill}
            if current.dialogue.active or current.cutscene_active or current.paused:
                return {"status": "interrupted", "reason": "gameplay_state_changed", "skill": decision.skill}

            if first_command is not None:
                acknowledged |= current.last_command_seq >= first_command

            defeated = next((event for event in current.events
                if event.id not in before_events and event.kind in {"enemy_defeated", "boss_defeated"}
                and event.detail == str(decision.args.target_actor_id)), None)
            if defeated is not None:
                return {"status": "completed", "reason": defeated.kind,
                    "health_lost": max(0, start_health - current.player.health),
                    "distance": math.dist(start_position, current.player.position),
                    "acknowledged": acknowledged, "skill": decision.skill}

            actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params)
            if actor is None:
                missing_samples += 1
                if missing_samples >= 5:
                    return {"status": "failed", "reason": "enemy_lost_without_defeat_event",
                        "health_lost": max(0, start_health - current.player.health),
                        "acknowledged": acknowledged, "skill": decision.skill}
                await asyncio.sleep(0.1)
                continue
            missing_samples = 0

            if actor.distance > 260:
                stick_x, stick_y, _ = _steer_to(current, actor.position, min(1.0, max(0.65, decision.args.strength)))
                command_id = bridge.send(buttons=BUTTONS["Z"], stick_x=stick_x, stick_y=stick_y, lease_ms=220)
            elif actor.distance > 120:
                stick_x, stick_y, _ = _steer_to(current, actor.position, max(0.45, decision.args.strength))
                command_id = bridge.send(buttons=BUTTONS["Z"], stick_x=stick_x, stick_y=max(20, stick_y), lease_ms=180)
            else:
                # Keep this deliberately generic. Boss-specific openings remain a planner concern.
                pattern = cycle % 5
                if pattern in {0, 1, 3}:
                    buttons, stick_x, stick_y = BUTTONS["Z"] | BUTTONS["B"], 0, 12
                elif pattern == 2:
                    buttons, stick_x, stick_y = BUTTONS["Z"] | BUTTONS["R"], 0, 0
                else:
                    buttons, stick_x, stick_y = BUTTONS["Z"] | BUTTONS["A"], 0, 0
                command_id = bridge.send(buttons=buttons, stick_x=stick_x, stick_y=stick_y, lease_ms=140)
                cycle += 1

            if first_command is None:
                first_command = command_id
            await asyncio.sleep(0.14)
    finally:
        bridge.release()

    after = bridge.state
    if after and first_command is not None:
        acknowledged |= after.last_command_seq >= first_command
    return {"status": "failed", "reason": "combat_timeout",
        "health_lost": max(0, start_health - after.player.health) if after and after.player else None,
        "distance": math.dist(start_position, after.player.position) if after and after.player else None,
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
                acknowledged |= current.last_command_seq >= first_command
            actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params)
            if actor is None:
                missing += 1
                if missing >= 6:
                    return {"status": "failed", "reason": "follow_target_lost",
                        "last_target_distance": last_distance,
                        "acknowledged": acknowledged, "skill": decision.skill}
                await asyncio.sleep(0.10)
                continue
            missing = 0
            last_distance = actor.distance

            if actor.distance > desired + 35:
                stick_x, stick_y, _ = _steer_to(current, actor.position, max(0.55, decision.args.strength))
                command_id = bridge.send(stick_x=stick_x, stick_y=stick_y, lease_ms=240)
            elif actor.distance < max(45.0, desired - 50):
                stick_x, stick_y, _ = _steer_to(current, actor.position, max(0.4, decision.args.strength))
                command_id = bridge.send(stick_x=-stick_x, stick_y=-max(20, stick_y), lease_ms=180)
            else:
                command_id = bridge.send(lease_ms=160)

            if first_command is None:
                first_command = command_id
            await asyncio.sleep(0.10)
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "completed", "reason": "follow_window_complete",
        "last_target_distance": last_distance,
        "distance": math.dist(start_position, after.player.position) if after and after.player else None,
        "acknowledged": bool(after and first_command is not None and
            after.last_command_seq >= first_command) or acknowledged,
        "skill": decision.skill}


async def _manipulate_object(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    actor = _matching_actor(before, decision.args.target_actor_id, decision.args.target_actor_params)
    if actor is None:
        return {"status": "failed", "reason": "target_actor_not_observed", "skill": decision.skill}

    approach = await _navigate_local(bridge, decision, observation, actor_mode=True)
    if approach.get("status") != "completed":
        return {"status": approach.get("status", "failed"),
            "reason": f"approach_failed:{approach.get('reason', 'unknown')}",
            "skill": decision.skill}

    current = bridge.state
    if not current or not current.player:
        return {"status": "failed", "reason": "state_lost_after_approach", "skill": decision.skill}
    actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params)
    if actor is None:
        return {"status": "failed", "reason": "target_actor_lost_after_approach", "skill": decision.skill}

    start_actor_position = actor.position
    start_player_position = current.player.position
    before_events = {event.id for event in current.events}
    start_epoch = current.scene_epoch
    start_context = current.context_action.label
    acknowledged = await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=180, settle_s=0.10)
    direction_y = 62 if decision.args.direction == "forward" else -62
    deadline = time.monotonic() + min(4.0, max(0.6, decision.args.duration_ms / 1000))

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not current.player:
                return {"status": "interrupted", "reason": "game_state_lost", "skill": decision.skill}
            if current.scene_epoch != start_epoch:
                return {"status": "completed", "reason": "world_transition_after_manipulation",
                    "acknowledged": acknowledged, "skill": decision.skill}
            if current.dialogue.active or current.cutscene_active:
                return {"status": "completed", "reason": "interaction_started",
                    "acknowledged": acknowledged, "skill": decision.skill}

            outcome = next((event for event in current.events
                if event.id not in before_events and event.kind in {
                    "item_received", "scene_flag_set", "scene_flag_unset"}), None)
            if outcome is not None:
                return {"status": "completed", "reason": outcome.kind, "detail": outcome.detail,
                    "acknowledged": acknowledged, "skill": decision.skill}

            actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params)
            if actor is not None:
                displacement = math.dist(start_actor_position, actor.position)
                if displacement >= 8.0:
                    return {"status": "completed", "reason": "object_displaced",
                        "object_distance": displacement,
                        "player_distance": math.dist(start_player_position, current.player.position),
                        "acknowledged": acknowledged, "skill": decision.skill}
            if current.context_action.label in {"throw", "drop"} and current.context_action.label != start_context:
                return {"status": "completed", "reason": "object_grabbed",
                    "context_action": current.context_action.label,
                    "acknowledged": acknowledged, "skill": decision.skill}

            command_id = bridge.send(buttons=BUTTONS["A"], stick_y=direction_y, lease_ms=260)
            acknowledged |= current.last_command_seq >= command_id
            await asyncio.sleep(0.12)
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "failed", "reason": "object_manipulation_unconfirmed",
        "player_distance": math.dist(start_player_position, after.player.position)
            if after and after.player else None,
        "acknowledged": acknowledged, "skill": decision.skill}


async def _explore_area(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not before.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    if before.paused or before.dialogue.active or before.cutscene_active or before.game_over_state != 0:
        return {"status": "stale", "reason": "gameplay_state_blocks_exploration", "skill": decision.skill}

    baseline_actors = {(actor.actor_id, actor.params) for actor in before.nearby_actors}
    baseline_context = before.context_action.label
    start_health = before.player.health
    start_position = before.player.position
    deadline = time.monotonic() + min(12.0, max(1.0, decision.args.duration_ms / 1000))
    last_position = start_position
    last_seq = before.seq
    stagnant = 0
    turn_ticks = 0
    turn_side = 1
    first_command = None
    acknowledged = False

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
                acknowledged |= current.last_command_seq >= first_command
            new_actors = [actor for actor in current.nearby_actors
                if (actor.actor_id, actor.params) not in baseline_actors]
            if new_actors:
                actor = min(new_actors, key=lambda row: row.distance)
                return {"status": "completed", "reason": "actor_discovered",
                    "actor": actor.model_dump(), "distance": math.dist(start_position, current.player.position),
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
                last_position = current.player.position
                last_seq = current.seq
                if moved < 1.2:
                    stagnant += 1
                else:
                    stagnant = max(0, stagnant - 1)

            if (current.player.bg_check_flags & 0x008) or stagnant >= 3:
                turn_ticks = max(turn_ticks, 5)
                stagnant = 0
                turn_side *= -1

            if turn_ticks > 0:
                buttons, stick_x, stick_y = 0, 58 * turn_side, 10
                turn_ticks -= 1
            else:
                buttons, stick_x, stick_y = 0, 0, 58

            command_id = bridge.send(buttons=buttons, stick_x=stick_x, stick_y=stick_y, lease_ms=240)
            if first_command is None:
                first_command = command_id
            await asyncio.sleep(0.10)
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "completed", "reason": "exploration_window_complete",
        "distance": math.dist(start_position, after.player.position) if after and after.player else None,
        "acknowledged": bool(after and first_command is not None and
            after.last_command_seq >= first_command) or acknowledged,
        "skill": decision.skill}


def _inventory_slot(game: GameState, item_id: int) -> int | None:
    for item in game.inventory_named:
        if item.item_id == item_id:
            return item.slot
    for slot, value in enumerate(game.inventory):
        if value == item_id:
            return slot
    return None


def _menu_grid_directions(current_slot: int, target_slot: int, width: int = 6) -> list[str]:
    current_row, current_col = divmod(current_slot, width)
    target_row, target_col = divmod(target_slot, width)
    preferred = []
    if current_col != target_col:
        preferred.append("right" if target_col > current_col else "left")
    if current_row != target_row:
        preferred.append("down" if target_row > current_row else "up")
    for direction in ("right", "down", "left", "up"):
        if direction not in preferred:
            preferred.append(direction)
    return preferred


async def _wait_pause_ready(bridge: Bridge, timeout_s: float = 2.5) -> GameState | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = bridge.state
        if current and current.pause_menu.active and current.pause_menu.ready:
            return current
        await asyncio.sleep(0.08)
    return None


async def _equip_item(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    item_id = decision.args.item_id
    c_slot = decision.args.slot
    if not current or not current.player or item_id is None or c_slot is None:
        return {"status": "failed", "reason": "equip_arguments_or_state_missing", "skill": decision.skill}
    if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    if current.game_over_state != 0:
        return {"status": "stale", "reason": "game_over_active", "skill": decision.skill}
    if current.dialogue.active or current.cutscene_active:
        return {"status": "stale", "reason": "gameplay_state_blocks_pause", "skill": decision.skill}

    target_slot = _inventory_slot(current, item_id)
    if target_slot is None or target_slot >= 24:
        return {"status": "failed", "reason": "item_not_owned_or_not_assignable",
            "item_id": item_id, "skill": decision.skill}
    button_index = {"left": 1, "down": 2, "right": 3}[c_slot]
    if len(current.equipped) > button_index and current.equipped[button_index] == item_id:
        return {"status": "completed", "reason": "already_equipped", "item_id": item_id,
            "slot": c_slot, "skill": decision.skill}

    acknowledged = False
    opened_here = not current.pause_menu.active
    if opened_here:
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["START"], hold_ms=90, settle_s=0.12)
    current = await _wait_pause_ready(bridge)
    if current is None:
        return {"status": "failed", "reason": "pause_menu_not_ready",
            "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}

    # R is the stable right-page control in the pinned SoH revision. Cycle until the item page (0).
    for _ in range(5):
        if current.pause_menu.page_index == 0:
            break
        old_page = current.pause_menu.page_index
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["R"], hold_ms=80, settle_s=0.16)
        current = await _wait_pause_ready(bridge, 1.5)
        if current is None:
            return {"status": "failed", "reason": "pause_page_transition_timeout",
                "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}
        if current.pause_menu.page_index == old_page:
            continue
    if current.pause_menu.page_index != 0:
        return {"status": "failed", "reason": "item_page_unreachable",
            "item_id": item_id, "page": current.pause_menu.page_index,
            "acknowledged": acknowledged, "skill": decision.skill}

    # If the cursor sits on a page-switch sentinel, move it back toward the item grid.
    if current.pause_menu.cursor_special_pos == 10:
        acknowledged |= await _pulse(bridge, stick_x=60, hold_ms=100, settle_s=0.12)
        current = await _wait_pause_ready(bridge, 1.0) or current
    elif current.pause_menu.cursor_special_pos == 11:
        acknowledged |= await _pulse(bridge, stick_x=-60, hold_ms=100, settle_s=0.12)
        current = await _wait_pause_ready(bridge, 1.0) or current

    visited = {}
    for _ in range(32):
        current = bridge.state or current
        slots = current.pause_menu.cursor_slot
        if not current.pause_menu.active or not current.pause_menu.ready or not slots:
            current = await _wait_pause_ready(bridge, 1.0)
            if current is None:
                break
            slots = current.pause_menu.cursor_slot
        cursor = slots[0] if slots else -1
        if cursor == target_slot:
            break
        if cursor < 0 or cursor >= 24:
            return {"status": "failed", "reason": "invalid_item_cursor",
                "cursor": cursor, "item_id": item_id, "skill": decision.skill}

        moved = False
        for direction in _menu_grid_directions(cursor, target_slot):
            key = (cursor, direction)
            if visited.get(key, 0) >= 2:
                continue
            visited[key] = visited.get(key, 0) + 1
            x = {"left": -60, "right": 60}.get(direction, 0)
            y = {"up": 60, "down": -60}.get(direction, 0)
            acknowledged |= await _pulse(bridge, stick_x=x, stick_y=y, hold_ms=100, settle_s=0.12)
            sample = await _wait_pause_ready(bridge, 0.9)
            if sample is None:
                continue
            new_cursor = sample.pause_menu.cursor_slot[0] if sample.pause_menu.cursor_slot else cursor
            current = sample
            if new_cursor != cursor:
                moved = True
                break
        if not moved:
            return {"status": "failed", "reason": "item_cursor_stuck",
                "cursor": cursor, "target_slot": target_slot, "item_id": item_id,
                "acknowledged": acknowledged, "skill": decision.skill}

    current = bridge.state or current
    cursor = current.pause_menu.cursor_slot[0] if current.pause_menu.cursor_slot else -1
    if cursor != target_slot:
        return {"status": "failed", "reason": "item_cursor_target_not_reached",
            "cursor": cursor, "target_slot": target_slot, "item_id": item_id,
            "acknowledged": acknowledged, "skill": decision.skill}

    acknowledged |= await _pulse(bridge, buttons=BUTTONS[f"C_{c_slot.upper()}"],
        hold_ms=90, settle_s=0.15)

    equip_deadline = time.monotonic() + 2.5
    equipped = False
    while time.monotonic() < equip_deadline:
        sample = bridge.state
        if sample and len(sample.equipped) > button_index and sample.equipped[button_index] == item_id:
            equipped = True
            current = sample
            break
        await asyncio.sleep(0.08)
    if not equipped:
        return {"status": "failed", "reason": "equip_not_confirmed",
            "item_id": item_id, "slot": c_slot, "acknowledged": acknowledged,
            "skill": decision.skill}

    # Wait out the pause equip animation before closing. Failure to close is reported but does not
    # erase the verified equipment success.
    ready = await _wait_pause_ready(bridge, 2.5)
    if ready is not None:
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["START"], hold_ms=90, settle_s=0.15)
        close_deadline = time.monotonic() + 1.5
        while time.monotonic() < close_deadline:
            sample = bridge.state
            if sample and not sample.pause_menu.active:
                return {"status": "completed", "reason": "item_equipped",
                    "item_id": item_id, "slot": c_slot, "pause_closed": True,
                    "acknowledged": acknowledged, "skill": decision.skill}
            await asyncio.sleep(0.08)
    return {"status": "completed", "reason": "item_equipped_pause_left_open",
        "item_id": item_id, "slot": c_slot, "pause_closed": False,
        "acknowledged": acknowledged, "skill": decision.skill}



def _gear_observation(game: GameState, item_id: int):
    return next((row for row in game.progress.equipment if row.item_id == item_id), None)


def _equipment_point(item_id: int) -> int | None:
    # Vanilla equipment item IDs are four contiguous rows of three:
    # swords 0x3B..0x3D, shields 0x3E..0x40, tunics 0x41..0x43, boots 0x44..0x46.
    if not 0x3B <= item_id <= 0x46:
        return None
    offset = item_id - 0x3B
    return (offset // 3) * 4 + (offset % 3) + 1


async def _equip_gear(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    item_id = decision.args.item_id
    if not current or not current.player or item_id is None:
        return {"status": "failed", "reason": "equip_gear_arguments_or_state_missing", "skill": decision.skill}
    if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    if current.game_over_state != 0:
        return {"status": "stale", "reason": "game_over_active", "skill": decision.skill}
    if current.dialogue.active or current.cutscene_active:
        return {"status": "stale", "reason": "gameplay_state_blocks_pause", "skill": decision.skill}

    gear = _gear_observation(current, item_id)
    target_point = _equipment_point(item_id)
    if gear is None or target_point is None:
        return {"status": "failed", "reason": "gear_not_owned_or_not_equipable",
            "item_id": item_id, "skill": decision.skill}
    if gear.equipped:
        return {"status": "completed", "reason": "already_equipped",
            "item_id": item_id, "equipment_type": gear.equipment_type, "skill": decision.skill}

    acknowledged = False
    if not current.pause_menu.active:
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["START"], hold_ms=90, settle_s=0.12)
    current = await _wait_pause_ready(bridge)
    if current is None:
        return {"status": "failed", "reason": "pause_menu_not_ready",
            "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}

    # R cycles pages in the pinned SoH build. Stop on PAUSE_EQUIP == 3.
    for _ in range(5):
        if current.pause_menu.page_index == 3:
            break
        old_page = current.pause_menu.page_index
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["R"], hold_ms=80, settle_s=0.16)
        current = await _wait_pause_ready(bridge, 1.5)
        if current is None:
            return {"status": "failed", "reason": "equipment_page_transition_timeout",
                "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}
        if current.pause_menu.page_index == old_page:
            continue
    if current.pause_menu.page_index != 3:
        return {"status": "failed", "reason": "equipment_page_unreachable",
            "item_id": item_id, "page": current.pause_menu.page_index,
            "acknowledged": acknowledged, "skill": decision.skill}

    if current.pause_menu.cursor_special_pos == 10:
        acknowledged |= await _pulse(bridge, stick_x=60, hold_ms=100, settle_s=0.12)
        current = await _wait_pause_ready(bridge, 1.0) or current
    elif current.pause_menu.cursor_special_pos == 11:
        acknowledged |= await _pulse(bridge, stick_x=-60, hold_ms=100, settle_s=0.12)
        current = await _wait_pause_ready(bridge, 1.0) or current

    visited = {}
    for _ in range(28):
        current = bridge.state or current
        if not current.pause_menu.active or not current.pause_menu.ready:
            current = await _wait_pause_ready(bridge, 1.0)
            if current is None:
                break
        points = current.pause_menu.cursor_point
        cursor = points[3] if len(points) > 3 else -1
        if cursor == target_point:
            break
        if cursor < 0 or cursor >= 16:
            return {"status": "failed", "reason": "invalid_equipment_cursor",
                "cursor": cursor, "target_point": target_point, "item_id": item_id,
                "skill": decision.skill}

        moved = False
        for direction in _menu_grid_directions(cursor, target_point, width=4):
            key = (cursor, direction)
            if visited.get(key, 0) >= 2:
                continue
            visited[key] = visited.get(key, 0) + 1
            x = {"left": -60, "right": 60}.get(direction, 0)
            y = {"up": 60, "down": -60}.get(direction, 0)
            acknowledged |= await _pulse(bridge, stick_x=x, stick_y=y, hold_ms=100, settle_s=0.12)
            sample = await _wait_pause_ready(bridge, 0.9)
            if sample is None:
                continue
            new_cursor = sample.pause_menu.cursor_point[3] if len(sample.pause_menu.cursor_point) > 3 else cursor
            current = sample
            if new_cursor != cursor:
                moved = True
                break
        if not moved:
            return {"status": "failed", "reason": "equipment_cursor_stuck",
                "cursor": cursor, "target_point": target_point, "item_id": item_id,
                "acknowledged": acknowledged, "skill": decision.skill}

    current = bridge.state or current
    cursor = current.pause_menu.cursor_point[3] if len(current.pause_menu.cursor_point) > 3 else -1
    if cursor != target_point:
        return {"status": "failed", "reason": "equipment_cursor_target_not_reached",
            "cursor": cursor, "target_point": target_point, "item_id": item_id,
            "acknowledged": acknowledged, "skill": decision.skill}

    acknowledged |= await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=90, settle_s=0.12)
    deadline = time.monotonic() + 2.0
    verified = None
    while time.monotonic() < deadline:
        sample = bridge.state
        if sample:
            candidate = _gear_observation(sample, item_id)
            if candidate is not None and candidate.equipped:
                verified = candidate
                current = sample
                break
        await asyncio.sleep(0.08)
    if verified is None:
        return {"status": "failed", "reason": "gear_equip_not_confirmed",
            "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}

    ready = await _wait_pause_ready(bridge, 2.0)
    if ready is not None:
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["START"], hold_ms=90, settle_s=0.15)
        close_deadline = time.monotonic() + 1.5
        while time.monotonic() < close_deadline:
            sample = bridge.state
            if sample and not sample.pause_menu.active:
                return {"status": "completed", "reason": "gear_equipped",
                    "item_id": item_id, "equipment_type": verified.equipment_type,
                    "pause_closed": True, "acknowledged": acknowledged, "skill": decision.skill}
            await asyncio.sleep(0.08)
    return {"status": "completed", "reason": "gear_equipped_pause_left_open",
        "item_id": item_id, "equipment_type": verified.equipment_type,
        "pause_closed": False, "acknowledged": acknowledged, "skill": decision.skill}


async def execute_skill(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not bridge.connected:
        raise RuntimeError("bridge_disconnected")
    if (before.instance_id, before.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    if not before.in_game:
        return {"status": "stale", "reason": "game_not_ready", "skill": decision.skill}
    if decision.skill == "equip_item":
        return await _equip_item(bridge, decision, observation)
    if decision.skill == "equip_gear":
        return await _equip_gear(bridge, decision, observation)
    if decision.skill == "navigate_to":
        return await _navigate_local(bridge, decision, observation, actor_mode=False)
    if decision.skill == "approach_actor":
        return await _navigate_local(bridge, decision, observation, actor_mode=True)
    if decision.skill == "follow_actor":
        return await _follow_actor(bridge, decision, observation)
    if decision.skill == "talk_to_actor":
        return await _navigate_local(bridge, decision, observation, actor_mode=True, talk=True)
    if decision.skill == "interact_with_actor":
        return await _navigate_local(bridge, decision, observation, actor_mode=True, interact=True)
    if decision.skill == "aim_at":
        return await _aim_at(bridge, decision, observation)
    if decision.skill == "fight_enemy":
        return await _fight_enemy(bridge, decision, observation)
    if decision.skill == "explore_area":
        return await _explore_area(bridge, decision, observation)
    if decision.skill == "manipulate_object":
        return await _manipulate_object(bridge, decision, observation)
    if before.paused and decision.skill not in MENU_SKILLS:
        return {"status": "stale", "reason": "pause_menu_active", "skill": decision.skill}
    if not before.paused and decision.skill in {"menu_move", "menu_confirm", "menu_cancel", "menu_assign"}:
        return {"status": "stale", "reason": "pause_menu_not_active", "skill": decision.skill}
    if before.dialogue.active and decision.skill not in DIALOGUE_SKILLS and decision.skill != "wait":
        return {"status": "stale", "reason": "dialogue_requires_handling", "skill": decision.skill}
    if before.cutscene_active and not before.dialogue.active and decision.skill not in {"wait", "continue_gameover"}:
        return {"status": "stale", "reason": "cutscene_active", "skill": decision.skill}
    if decision.skill == "choose_dialogue":
        return await _choose_dialogue(bridge, decision, observation)
    if decision.skill == "play_song":
        return await _play_song(bridge, decision, observation)

    buttons, x, y = controller_input(decision)
    start_yaw = before.player.yaw if before.player else None
    target_turn_units = None
    if decision.skill == "turn" and start_yaw is not None:
        target_turn_units = max(3600, min(18200,
            int(decision.args.duration_ms * 9.1 * max(0.35, decision.args.strength))))
    effective_duration_ms = (min(decision.args.duration_ms, 180)
        if decision.skill in MENU_SKILLS | {"advance_dialogue"}
        else min(decision.args.duration_ms, 2000))
    deadline = time.monotonic() + effective_duration_ms / 1000
    status, reason, acknowledged = "completed", "duration_elapsed", False
    first_command = None
    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not bridge.connected or not current:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (before.instance_id, before.scene_epoch):
                status, reason = "interrupted", "world_changed"
                break
            if not current.in_game:
                status, reason = "interrupted", "game_not_ready"
                break
            if current.paused and decision.skill not in MENU_SKILLS:
                status, reason = "interrupted", "pause_menu_opened"
                break
            if not current.paused and decision.skill in {"menu_move", "menu_confirm", "menu_cancel", "menu_assign"}:
                status, reason = "interrupted", "pause_menu_closed"
                break
            if (not before.dialogue.active and current.dialogue.active and
                    decision.skill not in DIALOGUE_SKILLS):
                status, reason = "interrupted", "dialogue_opened"
                break
            if (not before.cutscene_active and current.cutscene_active and
                    not current.dialogue.active and decision.skill != "wait"):
                status, reason = "interrupted", "cutscene_started"
                break
            if target_turn_units is not None and current.player:
                delta = ((current.player.yaw - start_yaw + 32768) % 65536) - 32768
                if abs(delta) >= target_turn_units:
                    reason = "heading_reached"
                    break
            command_id = bridge.send(buttons=buttons, stick_x=x, stick_y=y, lease_ms=300)
            if first_command is None:
                first_command = command_id
            acknowledged |= current.last_command_seq >= first_command
            await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))
    finally:
        bridge.release()

    await asyncio.sleep(0.22)
    after = bridge.state
    distance, damage, yaw_delta = None, None, None
    if after:
        acknowledged |= first_command is not None and after.last_command_seq >= first_command
    if before.player and after and after.player and before.scene_epoch == after.scene_epoch:
        distance = math.dist(before.player.position, after.player.position)
        damage = max(0, before.player.health - after.player.health)
        yaw_delta = ((after.player.yaw - before.player.yaw + 32768) % 65536) - 32768
    if status == "completed" and not acknowledged and decision.skill != "wait":
        status, reason = "failed", "input_not_acknowledged"
    elif status == "completed" and decision.skill == "move" and distance is not None and distance < 1:
        status, reason = "failed", "no_displacement_observed"
    elif status == "completed" and decision.skill == "turn" and yaw_delta is not None and abs(yaw_delta) < 900:
        status, reason = "failed", "no_heading_change_observed"
    return {"status": status, "reason": reason, "distance": distance, "yaw_delta": yaw_delta,
        "health_lost": damage, "acknowledged": acknowledged, "skill": decision.skill}


class Runtime:
    def __init__(self, bridge: Bridge, store: Store, providers: dict):
        self.bridge, self.store, self.providers = bridge, store, providers
        self.run_id: str | None = None
        self.segment_id: str | None = None
        self.namespace = ""
        self.config: RunConfig | None = None
        self.selected_model: ModelInfo | None = None
        self.state = "idle"
        self.reason = ""
        self.task: asyncio.Task | None = None
        self.lock = asyncio.Lock()
        self.started = 0.0
        self.last_decision: dict | None = None
        self.last_result: dict | None = None
        self.recent = deque(maxlen=12)
        self.hints = deque(maxlen=3)
        self.dialogue_transcript = deque(maxlen=20)
        self.seen_events: set[str] = set()
        self.bridge.on_state = self.on_state
        self.subscribers: set[asyncio.Queue] = set()
        self.last_publish = 0.0
        self.pending_switch: tuple[RunConfig, ModelInfo] | None = None
        self.game_instance: str | None = None
        self.metrics_cache: dict | None = None
        self.metrics_at = 0.0
        self.trajectory_trace = deque(maxlen=96)
        self.trace_origin: GameState | None = None
        self.trajectory_tainted = False
        self.replaying_trajectory = False
        self.replay_attempts: set[tuple[str, int]] = set()
        self.stuck_score = 0
        self.stuck_notified_at = 0

    def publish(self, force=False):
        if not force and time.monotonic() - self.last_publish < 0.2:
            return
        self.last_publish = time.monotonic()
        for queue in list(self.subscribers):
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(True)

    def log(self, kind: str, data: dict | None = None):
        self.recent.append({"kind": kind, "data": data or {}, "at": time.time()})
        if self.run_id:
            self.store.event(self.run_id, kind, data)
        self.publish(True)

    def _update_stuck(self, decision: Decision, result: dict):
        status = result.get("status")
        reason = result.get("reason")
        if status == "interrupted" and reason in {"world_changed", "dialogue_opened", "cutscene_started",
                                                  "pause_menu_opened", "pause_menu_closed"}:
            self.stuck_score = 0
        elif status in {"failed", "stale"}:
            self.stuck_score = min(20, self.stuck_score + 2)
        elif decision.skill == "move" and (result.get("distance") or 0) >= 20:
            self.stuck_score = max(0, self.stuck_score - 3)
        elif decision.skill == "turn" and abs(result.get("yaw_delta") or 0) >= 1800:
            self.stuck_score = max(0, self.stuck_score - 2)
        else:
            self.stuck_score = max(0, self.stuck_score - 1)
        if self.stuck_score >= 6 and self.stuck_score - self.stuck_notified_at >= 2:
            self.stuck_notified_at = self.stuck_score
            self.replay_attempts.clear()
            self.trajectory_tainted = True
            self.log("stuck_detected", {"score": self.stuck_score, "last_skill": decision.skill,
                "instruction": "Replan; do not repeat the same failed local action."})

    def _reset_trajectory_trace(self, state: GameState | None = None, *, tainted: bool = False):
        self.trajectory_trace.clear()
        self.trace_origin = state.model_copy(deep=True) if state and state.player else None
        self.trajectory_tainted = tainted

    def _record_trajectory_action(self, decision: Decision, state: GameState):
        if self.replaying_trajectory or not self.config or self.config.memory_mode != "adaptive":
            return
        if decision.skill not in NAVIGATION_SKILLS:
            self._reset_trajectory_trace(state)
            return
        if (not self.trace_origin or
                (self.trace_origin.scene, self.trace_origin.room) != (state.scene, state.room)):
            self._reset_trajectory_trace(state)
        if len(self.trajectory_trace) >= 96:
            self.trajectory_tainted = True
            return
        self.trajectory_trace.append({"skill": decision.skill, "args": decision.args.model_dump()})

    def _discard_failed_trajectory_action(self, decision: Decision):
        if self.replaying_trajectory or not self.trajectory_trace:
            return
        expected = {"skill": decision.skill, "args": decision.args.model_dump()}
        if self.trajectory_trace[-1] == expected:
            self.trajectory_trace.pop()

    def _learn_transition(self, state: GameState, old: GameState | None):
        if not old or not self.config or self.config.memory_mode != "adaptive":
            return
        if old.instance_id != state.instance_id or not old.in_game or not state.in_game:
            return
        if min(old.scene, state.scene, old.room, state.room) < 0:
            return
        if (old.scene, old.room) == (state.scene, state.room):
            return
        edge_id = self.store.learn_world_edge(self.namespace,
            {"scene": old.scene, "scene_name": old.scene_name, "room": old.room,
             "position": old.player.position if old.player else None},
            {"scene": state.scene, "scene_name": state.scene_name, "room": state.room,
             "position": state.player.position if state.player else None,
             "entrance_index": state.entrance_index})
        if edge_id:
            self.log("world_edge_observed", {"edge_id": edge_id,
                "from": [old.scene, old.room], "from_name": old.scene_name,
                "to": [state.scene, state.room], "to_name": state.scene_name,
                "entrance_index": state.entrance_index})
        if (not self.replaying_trajectory and not self.trajectory_tainted and self.trace_origin
                and self.trajectory_trace):
            origin = self.trace_origin
            route_id = self.store.learn_trajectory(self.namespace,
                {"scene": origin.scene, "room": origin.room,
                 "position": origin.player.position if origin.player else None,
                 "yaw": origin.player.yaw if origin.player else None},
                {"scene": state.scene, "room": state.room},
                list(self.trajectory_trace))
            if route_id:
                self.log("trajectory_learned", {"trajectory_id": route_id,
                    "from": [origin.scene, origin.room], "to": [state.scene, state.room],
                    "steps": len(self.trajectory_trace)})
        self._reset_trajectory_trace(state)

    async def _try_replay_trajectory(self, game: GameState) -> bool:
        if (not self.config or self.config.memory_mode != "adaptive" or self.replaying_trajectory
                or not game.player or game.paused or game.dialogue.active or game.game_over_state != 0):
            return False
        route = self.store.best_trajectory(self.namespace, game.scene, game.room,
            game.player.position, game.player.yaw)
        if not route:
            return False
        attempt_key = (route["id"], game.scene_epoch)
        if attempt_key in self.replay_attempts:
            return False
        self.replay_attempts.add(attempt_key)
        self.replaying_trajectory = True
        self.log("trajectory_replay_started", {"trajectory_id": route["id"],
            "from": [route["from_scene"], route["from_room"]],
            "to": [route["to_scene"], route["to_room"]],
            "steps": len(route["actions"])})
        success = False
        try:
            for raw in route["actions"][:96]:
                current = self.bridge.state
                if not self.bridge.connected or not current or not current.in_game or current.paused:
                    break
                if (current.scene, current.room) == (route["to_scene"], route["to_room"]):
                    success = True
                    break
                if (current.scene, current.room) != (route["from_scene"], route["from_room"]):
                    break
                decision = Decision(goal="Replay learned trajectory",
                    summary="Replaying a previously successful local route.",
                    skill=raw["skill"], args=raw["args"], memory_note=None)
                self.last_decision = decision.model_dump()
                self.last_result = await execute_skill(self.bridge, decision, current)
                self.log("trajectory_replay_step", {"trajectory_id": route["id"],
                    "skill": decision.skill, "result": self.last_result})
                if self.last_result["status"] != "completed":
                    break
                await asyncio.sleep(0.05)
            current = self.bridge.state
            success = bool(current and (current.scene, current.room) ==
                (route["to_scene"], route["to_room"]))
            self.store.trajectory_outcome(route["id"], success)
            self.log("trajectory_replay_succeeded" if success else "trajectory_replay_failed",
                {"trajectory_id": route["id"], "scene": current.scene if current else None,
                 "room": current.room if current else None})
            return success
        finally:
            self.replaying_trajectory = False

    def on_state(self, state: GameState, old: GameState | None):
        self._learn_transition(state, old)
        if self.run_id and self.state in {"running", "paused"}:
            if old and old.instance_id != state.instance_id:
                self.log("game_instance_changed")
                self.bridge.release()
            if old and old.instance_id == state.instance_id and (
                    old.scene != state.scene or old.room != state.room or
                    old.scene_epoch != state.scene_epoch):
                self.bridge.release()
                self.replay_attempts.clear()
                self.stuck_score = 0
                self.stuck_notified_at = 0
                self.last_result = {"status": "interrupted", "reason": "world_transition",
                    "from": [old.scene, old.room], "to": [state.scene, state.room]}
                self.log("world_transition", {"from_scene": old.scene, "from_room": old.room,
                    "to_scene": state.scene, "to_room": state.room,
                    "entrance_index": state.entrance_index, "scene_epoch": state.scene_epoch})
            if old:
                if not old.dialogue.active and state.dialogue.active:
                    self.bridge.release()
                    entry = {"text_id": state.dialogue.text_id, "text": state.dialogue.text,
                        "choices": state.dialogue.choices,
                        "speaker": state.dialogue.speaker.model_dump() if state.dialogue.speaker else None}
                    if state.dialogue.text:
                        self.dialogue_transcript.append(entry)
                    self.log("dialogue_detected", entry)
                elif old.dialogue.active and not state.dialogue.active:
                    self.log("dialogue_closed", {"text_id": old.dialogue.text_id})
                elif state.dialogue.active and (
                        old.dialogue.text_id != state.dialogue.text_id or
                        old.dialogue.text != state.dialogue.text or
                        old.dialogue.choice_count != state.dialogue.choice_count):
                    entry = {"text_id": state.dialogue.text_id, "text": state.dialogue.text,
                        "choices": state.dialogue.choices,
                        "speaker": state.dialogue.speaker.model_dump() if state.dialogue.speaker else None}
                    if state.dialogue.text and (not self.dialogue_transcript or
                            self.dialogue_transcript[-1].get("text") != state.dialogue.text):
                        self.dialogue_transcript.append(entry)
                    self.log("dialogue_changed", entry)
            for event in state.events:
                key = f"{state.instance_id}:{event.id}"
                if key not in self.seen_events:
                    self.seen_events.add(key)
                    self.log(event.kind, {"detail": event.detail, "scene": state.scene, "room": state.room})
                    if event.kind == "game_completed" and self.state == "running":
                        self.bridge.release()
                        self.state, self.reason = "completed", "game_completed"
                        self.store.update_run(self.run_id, status="completed", reason="game_completed")
                        self.store.end_segments(self.run_id)
                        self.log("run_completed", {"reason": event.detail, "scene": state.scene,
                            "scene_name": state.scene_name})
            if old and old.player and state.player and old.player.health > 0 and state.player.health == 0:
                self.log("player_died", {"scene": state.scene, "last_skill": self.last_decision})
                self.store.remember(self.namespace, state.scene,
                    f"Death observed after skill: {(self.last_decision or {}).get('skill', 'unknown')}. "
                    "Causality is unconfirmed; reconsider the tactic.")
        self.publish()

    async def _handle_dialogue(self, game: GameState) -> bool:
        if not game.dialogue.active:
            return False
        if game.dialogue.text and (not self.dialogue_transcript or
                self.dialogue_transcript[-1].get("text") != game.dialogue.text):
            self.dialogue_transcript.append({"text_id": game.dialogue.text_id,
                "text": game.dialogue.text, "choices": game.dialogue.choices,
                "speaker": game.dialogue.speaker.model_dump() if game.dialogue.speaker else None})
        if game.dialogue.choice_count > 0 and game.dialogue.can_advance:
            return False  # A model decision is required for a semantic choice.
        self.bridge.release()
        if not game.dialogue.can_advance:
            await asyncio.sleep(0.08)
            return True
        acknowledged = await _pulse(self.bridge, buttons=BUTTONS["A"], hold_ms=90, settle_s=0.12)
        self.log("dialogue_auto_advance", {"text_id": game.dialogue.text_id,
            "state": game.dialogue.state, "acknowledged": acknowledged})
        return True

    async def _handle_gameover(self, game: GameState) -> bool:
        if game.game_over_state == 0:
            return False
        self.bridge.release()
        pause_state = game.pause_menu.state
        if pause_state in {0xE, 0x10}:
            if game.pause_menu.prompt_choice != 0:
                await _pulse(self.bridge, stick_x=-60, hold_ms=90, settle_s=0.12)
            acknowledged = await _pulse(self.bridge, buttons=BUTTONS["A"], hold_ms=90, settle_s=0.18)
            self.log("gameover_auto_confirm", {"pause_state": pause_state,
                "prompt_choice": game.pause_menu.prompt_choice, "acknowledged": acknowledged})
        else:
            await asyncio.sleep(0.15)
        return True

    async def validate_model(self, config: RunConfig) -> ModelInfo:
        if config.provider not in self.providers:
            raise ValueError("Provider not available")
        provider = self.providers[config.provider]
        status = await provider.status()
        if not status.get("connected"):
            raise ValueError(status.get("message", "Provider not connected"))
        catalog = await provider.models()
        info = next((m for m in catalog if m.id == config.model), None)
        if not info:
            raise ValueError("Model not present in the provider catalog")
        if not info.structured_output:
            raise ValueError("This milestone requires a model with structured-output support")
        if config.effort is not None and config.effort not in info.efforts:
            raise ValueError("Effort not supported by this model's catalog")
        return info

    def new_namespace(self, config: RunConfig) -> str:
        identity = hashlib.sha256(f"{config.provider}|{config.model}|{config.effort}|{CONTRACT_VERSION}".encode()).hexdigest()[:24]
        return f"adaptive:{identity}" if config.memory_mode == "adaptive" else f"run:{self.run_id}:{identity}"

    async def start(self, config: RunConfig):
        async with self.lock:
            if self.state in {"running", "paused"}:
                raise ValueError("Stop the current run before starting another")
            game = self.bridge.state
            if not self.bridge.connected or not game or not game.in_game or not game.player:
                raise ValueError("Connect the bridge and load a playable save in SoH first")
            if config.provider == "demo" and game.source != "simulator":
                raise ValueError("The deterministic demo is not a game-playing model")
            info = await self.validate_model(config)
            # Discovery/auth can be slow. Recheck game availability before authorizing any inference.
            if not self.bridge.connected or not self.bridge.state or not self.bridge.state.in_game:
                raise ValueError("Bridge disconnected while validating the provider")
            if self.bridge.state.instance_id != game.instance_id:
                raise ValueError("Game instance changed while validating the provider")
            game = self.bridge.state
            self.game_instance = game.instance_id
            self.pending_switch = None
            self.metrics_cache = None
            self.config, self.selected_model = config, info
            fingerprint = hashlib.sha256(json.dumps({"contract": CONTRACT_VERSION,
                "revision": game.upstream_revision, "initial": game.model_dump(exclude={"seq", "events", "last_command_seq"}),
                "checkpoint_label": config.checkpoint_label}, sort_keys=True).encode()).hexdigest()
            self.run_id = self.store.new_run(config.model_dump(), game.source, fingerprint)
            self.namespace = self.new_namespace(config)
            self.segment_id = self.store.segment(self.run_id, config.model_dump(), self.namespace)
            self.seen_events = {f"{game.instance_id}:{e.id}" for e in game.events}
            self.last_decision = self.last_result = None
            self.replay_attempts.clear()
            self.replaying_trajectory = False
            self.stuck_score = 0
            self.stuck_notified_at = 0
            self._reset_trajectory_trace(game)
            self.recent.clear()
            self.hints.clear()
            self.dialogue_transcript.clear()
            self.started = time.monotonic()
            self.state, self.reason = "running", ""
            self.log("run_started", {"contract": CONTRACT_VERSION, "checkpoint_certified": False})
            self.task = asyncio.create_task(self.loop())

    async def halt(self, state: str, reason: str):
        # Release immediately, before waiting for provider cancellation.
        self.bridge.release()
        self.state, self.reason = state, reason
        if self.run_id:
            self.store.update_run(self.run_id, status=state, reason=reason)
            if state == "stopped":
                self.pending_switch = None
                self.store.end_segments(self.run_id)
        task, self.task = self.task, None
        if task and task is not asyncio.current_task():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self.publish(True)

    async def control(self, action: str):
        async with self.lock:
            if not self.run_id or self.state not in {"running", "paused"}:
                raise ValueError("No active run")
            if action in {"pause", "stop", "take_control"}:
                if action == "take_control":
                    self.store.update_run(self.run_id, assisted=True)
                    self._reset_trajectory_trace(self.bridge.state, tainted=True)
                    self.log("take_control")
                await self.halt("stopped" if action == "stop" else "paused", action)
            elif action == "resume":
                if self.state != "paused":
                    raise ValueError("The run is not paused")
                if not self.bridge.connected:
                    raise ValueError("Reconnect the bridge before resuming")
                self.state, self.reason = "running", ""
                self.store.update_run(self.run_id, status="running", reason="")
                self.task = asyncio.create_task(self.loop())
            else:
                raise ValueError("Unknown control action")
            self.publish(True)

    async def switch(self, change: SwitchConfig):
        async with self.lock:
            if self.state not in {"running", "paused"} or not self.config or not self.run_id:
                raise ValueError("No active run")
            config = RunConfig.model_validate({**self.config.model_dump(), **change.model_dump()})
            if config.provider == "demo" and self.bridge.state and self.bridge.state.source != "simulator":
                raise ValueError("Demo provider cannot control SoH")
            info = await self.validate_model(config)
            if self.state not in {"running", "paused"}:
                raise ValueError("Run ended while validating the new model")
            self.pending_switch = (config, info)
            self.log("model_switch_queued", change.model_dump())
            if self.state == "paused":
                self.apply_switch()
            self.publish(True)

    def apply_switch(self):
        if self.pending_switch is None:
            return
        self.config, self.selected_model = self.pending_switch
        self.pending_switch = None
        self.namespace = self.new_namespace(self.config)
        self.segment_id = self.store.segment(self.run_id, self.config.model_dump(), self.namespace)
        self.replay_attempts.clear()
        self._reset_trajectory_trace(self.bridge.state)
        self.store.update_run(self.run_id, mixed=True)
        self.log("model_changed", {"provider": self.config.provider,
            "model": self.config.model, "effort": self.config.effort})

    def hint(self, text: str):
        if not self.run_id or self.state not in {"running", "paused"}:
            raise ValueError("No active run")
        self.hints.append(text)
        self.store.update_run(self.run_id, assisted=True)
        self._reset_trajectory_trace(self.bridge.state, tainted=True)
        self.log("human_hint", {"text": text})

    def budget_check(self, prompt: str):
        assert self.run_id and self.config and self.selected_model
        metrics = self.store.metrics(self.run_id)
        if metrics["unknown_usage_calls"]:
            raise ProviderFailure("unreconciled_token_usage; audit this run before continuing")
        if metrics["calls"] >= self.config.max_calls:
            raise ProviderFailure("call_budget_reached")
        if metrics["total_tokens"] >= self.config.max_tokens:
            raise ProviderFailure("token_budget_reached")
        if time.monotonic() - self.started >= self.config.max_runtime_s:
            raise ProviderFailure("runtime_budget_reached")
        if self.config.provider == "openrouter":
            if metrics["unknown_cost_calls"]:
                raise ProviderFailure("unreconciled_openrouter_cost; stop this run and audit provider billing")
            reservation = reserve_cost(self.selected_model, prompt, self.config.max_output_tokens)
            if metrics["known_cost_usd"] + reservation > self.config.max_cost_usd:
                raise ProviderFailure("cost_budget_reservation_exceeded")

    async def loop(self):
        try:
            while self.state == "running":
                self.apply_switch()
                assert self.config and self.run_id and self.segment_id
                game = self.bridge.state
                if not self.bridge.connected or not game:
                    raise ProviderFailure("bridge_disconnected")
                if game.instance_id != self.game_instance:
                    raise ProviderFailure("game_instance_changed; start a new run")
                if not game.in_game:
                    if time.monotonic() - self.started >= self.config.max_runtime_s:
                        raise ProviderFailure("runtime_budget_reached")
                    await asyncio.sleep(0.25)
                    continue
                if await self._handle_gameover(game):
                    continue
                # Do not spend inference calls while an uninterruptible animation/cutscene owns Link.
                if game.cutscene_active and not game.dialogue.active and game.game_over_state == 0:
                    self.bridge.release()
                    await asyncio.sleep(0.15)
                    continue
                if await self._handle_dialogue(game):
                    continue
                if await self._try_replay_trajectory(game):
                    await asyncio.sleep(0.1)
                    continue
                game = self.bridge.state or game
                observation = {"contract": CONTRACT_VERSION, "objective": self.config.goal,
                    "state": game.model_dump(exclude={"events", "upstream_revision", "last_command_seq"}),
                    "last_decision": self.last_decision, "last_result": self.last_result,
                    "events": list(self.recent)[-5:], "dialogue_transcript": list(self.dialogue_transcript),
                    "memory": [r["note"] for r in self.store.recall(self.namespace, game.scene, limit=6)],
                    "recent_global_memory": [r["note"] for r in self.store.recall(self.namespace, limit=8)],
                    "known_world_edges": self.store.world_neighbors(self.namespace, game.scene, game.room),
                    "stuck_score": self.stuck_score,
                    "human_hints": list(self.hints)}
                prompt = json.dumps(observation, separators=(",", ":"), ensure_ascii=False)
                self.budget_check(prompt)
                call_id = self.store.begin_call(self.run_id, self.segment_id, observation)
                started = time.monotonic()
                result = None
                try:
                    result = await self.providers[self.config.provider].decide(self.config, prompt)
                    decision = Decision.model_validate_json(result.text)
                    self.store.finish_call(call_id, status="completed", decision=decision.model_dump(),
                        usage=result.usage.model_dump(), latency_ms=(time.monotonic() - started) * 1000)
                except asyncio.CancelledError:
                    self.store.finish_call(call_id, status="cancelled", error="Cancelled; usage may be incomplete.",
                        latency_ms=(time.monotonic() - started) * 1000)
                    raise
                except (ProviderFailure, ValidationError, asyncio.TimeoutError) as exc:
                    usage = result.usage.model_dump() if result else getattr(exc, "usage", None)
                    if hasattr(usage, "model_dump"):
                        usage = usage.model_dump()
                    self.store.finish_call(call_id, status="invalid_output" if result else "failed", usage=usage or {},
                        error="Invalid structured decision" if isinstance(exc, ValidationError) else str(exc),
                        latency_ms=(time.monotonic() - started) * 1000)
                    raise ProviderFailure("invalid_structured_decision" if isinstance(exc, ValidationError) else str(exc)) from None
                except Exception as exc:
                    self.store.finish_call(call_id, status="failed", usage={},
                        error=f"Unexpected provider failure ({type(exc).__name__}); usage unknown.",
                        latency_ms=(time.monotonic() - started) * 1000)
                    raise ProviderFailure(f"provider_error:{type(exc).__name__}") from None
                if result.usage.actual_model and result.usage.actual_model != self.config.model:
                    self.store.update_run(self.run_id, mixed=True)
                    self.log("provider_rerouted", {"requested": self.config.model, "actual": result.usage.actual_model})
                self.last_decision = decision.model_dump()
                if not game.dialogue.active and self.dialogue_transcript:
                    self.dialogue_transcript.clear()
                if decision.memory_note:
                    self.store.remember(self.namespace, game.scene, decision.memory_note)
                self.log("decision", {"summary": decision.summary, "skill": decision.skill})
                self._record_trajectory_action(decision, game)
                self.last_result = await execute_skill(self.bridge, decision, game)
                if self.last_result["status"] not in {"completed"}:
                    self._discard_failed_trajectory_action(decision)
                event_kind = {
                    "failed": "skill_failed",
                    "interrupted": "skill_interrupted",
                    "stale": "skill_stale",
                }.get(self.last_result["status"], "skill_result")
                self.log(event_kind, self.last_result)
                self._update_stuck(decision, self.last_result)
                await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # A failed run must be visible and must never silently fall back to the demo provider.
            message = str(exc)[:300] or type(exc).__name__
            self.log("run_paused", {"reason": message})
            await self.halt("paused", message)
        finally:
            self.bridge.release()
            self.publish(True)

    def snapshot(self) -> dict:
        if self.run_id and (self.metrics_cache is None or time.monotonic() - self.metrics_at >= 1):
            self.metrics_cache = self.store.metrics(self.run_id)
            self.metrics_at = time.monotonic()
        return {"status": self.state, "reason": self.reason, "run_id": self.run_id,
            "config": self.config.model_dump() if self.config else None,
            "metrics": self.metrics_cache if self.run_id else None,
            "pending_switch": self.pending_switch[0].model_dump() if self.pending_switch else None,
            "elapsed_s": round(time.monotonic() - self.started) if self.run_id else 0,
            "last_decision": self.last_decision, "last_result": self.last_result,
            "events": list(self.recent), "dialogue_transcript": list(self.dialogue_transcript),
            "bridge": self.bridge.status(), "memory": self.store.recall(self.namespace, limit=30) if self.namespace else [],
            "trajectories": self.store.list_trajectories(self.namespace, limit=30) if self.namespace else [],
            "world_edges": self.store.list_world_edges(self.namespace, limit=100) if self.namespace else []}
