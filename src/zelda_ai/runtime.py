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

CONTRACT_VERSION = "state-v2/skills-v1/trajectory-v1/prompt-v4"
BUTTONS = {"A": 0x8000, "B": 0x4000, "Z": 0x2000, "START": 0x1000, "R": 0x0010,
    "C_UP": 0x0008, "C_LEFT": 0x0002, "C_DOWN": 0x0004, "C_RIGHT": 0x0001}
DIALOGUE_SKILLS = {"advance_dialogue", "choose_dialogue"}
MENU_SKILLS = {"pause_toggle", "menu_move", "menu_confirm", "menu_cancel", "menu_assign", "continue_gameover"}
NAVIGATION_SKILLS = {"move", "turn", "interact", "wait", "camera_center", "roll", "backflip", "sidestep"}
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
    {"id": "navigate", "name": "Navegação espacial até ator/saída/posição", "status": "planned", "version": None},
    {"id": "talk_to", "name": "Aproximar e conversar com ator", "status": "planned", "version": None},
    {"id": "equip_item", "name": "Equipar item por ID usando o pause menu", "status": "planned", "version": None},
    {"id": "aim_at", "name": "Mira calibrada para arco/estilingue/Hookshot", "status": "planned", "version": None},
    {"id": "fight_enemy", "name": "Combate composto com feedback", "status": "planned", "version": None},
    {"id": "manipulate_object", "name": "Empurrar/puxar/carregar/lançar objetos", "status": "planned", "version": None},
    {"id": "explore_area", "name": "Exploração e grafo de saídas", "status": "planned", "version": None},
    {"id": "death_recovery", "name": "Game over e recuperação autônoma", "status": "planned", "version": None},
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


async def execute_skill(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not bridge.connected:
        raise RuntimeError("bridge_disconnected")
    if (before.instance_id, before.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    if not before.in_game:
        return {"status": "stale", "reason": "game_not_ready", "skill": decision.skill}
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
    deadline = time.monotonic() + decision.args.duration_ms / 1000
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
                or not game.player):
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
                if self.last_result["status"] == "failed":
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
                self.last_result = {"status": "interrupted", "reason": "world_transition",
                    "from": [old.scene, old.room], "to": [state.scene, state.room]}
                self.log("world_transition", {"from_scene": old.scene, "from_room": old.room,
                    "to_scene": state.scene, "to_room": state.room,
                    "entrance_index": state.entrance_index, "scene_epoch": state.scene_epoch})
            if old:
                if not old.dialogue.active and state.dialogue.active:
                    self.bridge.release()
                    self.log("dialogue_detected", {"text_id": state.dialogue.text_id,
                        "text": state.dialogue.text, "choices": state.dialogue.choices,
                        "speaker": state.dialogue.speaker.model_dump() if state.dialogue.speaker else None})
                elif old.dialogue.active and not state.dialogue.active:
                    self.log("dialogue_closed", {"text_id": old.dialogue.text_id})
                elif state.dialogue.active and (
                        old.dialogue.text_id != state.dialogue.text_id or
                        old.dialogue.text != state.dialogue.text or
                        old.dialogue.choice_count != state.dialogue.choice_count):
                    self.log("dialogue_changed", {"text_id": state.dialogue.text_id,
                        "text": state.dialogue.text, "choices": state.dialogue.choices})
            for event in state.events:
                key = f"{state.instance_id}:{event.id}"
                if key not in self.seen_events:
                    self.seen_events.add(key)
                    self.log(event.kind, {"detail": event.detail, "scene": state.scene, "room": state.room})
            if old and old.player and state.player and old.player.health > 0 and state.player.health == 0:
                self.log("player_died", {"scene": state.scene, "last_skill": self.last_decision})
                self.store.remember(self.namespace, state.scene,
                    f"Death observed after skill: {(self.last_decision or {}).get('skill', 'unknown')}. "
                    "Causality is unconfirmed; reconsider the tactic.")
        self.publish()

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
            self._reset_trajectory_trace(game)
            self.recent.clear()
            self.hints.clear()
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
                # Do not spend inference calls while an uninterruptible animation/cutscene owns Link.
                if game.cutscene_active and not game.dialogue.active:
                    self.bridge.release()
                    await asyncio.sleep(0.15)
                    continue
                # Wait locally for the current textbox to reach an actionable state.
                if game.dialogue.active and not game.dialogue.can_advance:
                    self.bridge.release()
                    await asyncio.sleep(0.08)
                    continue
                if await self._try_replay_trajectory(game):
                    await asyncio.sleep(0.1)
                    continue
                game = self.bridge.state or game
                observation = {"contract": CONTRACT_VERSION, "objective": self.config.goal,
                    "state": game.model_dump(exclude={"events", "upstream_revision", "last_command_seq"}),
                    "last_result": self.last_result, "events": list(self.recent)[-5:],
                    "memory": [r["note"] for r in self.store.recall(self.namespace, game.scene)],
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
            "events": list(self.recent), "bridge": self.bridge.status(),
            "memory": self.store.recall(self.namespace, limit=30) if self.namespace else [],
            "trajectories": self.store.list_trajectories(self.namespace, limit=30) if self.namespace else []}
