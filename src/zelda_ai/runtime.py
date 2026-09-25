"""Run lifecycle, budgets, provider decisions and observed learning."""
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
from .budgets import budget_reason, runtime_exhausted
from .control.authority import owned
from .control.feedback import feedback
from .control.diagnostics import run_input_diagnostic
from .models import Decision, GameState, ModelInfo, RunConfig, SwitchConfig
from .combat_learning import compact_profile, enemy_key
from .providers.base import ProviderFailure
from .providers.openrouter import reserve_cost
from .store import Store
from .skills.catalog import BUTTONS, NAVIGATION_SKILLS
from .skills.common import _matching_actor, _pulse
from .skills.navigation import _backtrack_recovery
from .skills.executor import execute_skill


# Compatibility exports; implementations live in skills/.
from .skills.catalog import controller_input as controller_input
from .skills.catalog import BUTTONS as BUTTONS
from .skills.catalog import DIALOGUE_SKILLS as DIALOGUE_SKILLS
from .skills.catalog import MENU_SKILLS as MENU_SKILLS
from .skills.catalog import NAVIGATION_SKILLS as NAVIGATION_SKILLS
from .skills.catalog import SONG_IDS as SONG_IDS
from .skills.catalog import SONG_NOTES as SONG_NOTES
from .skills.catalog import SKILL_CATALOG as SKILL_CATALOG
from .skills.common import _matching_actor as _matching_actor
from .skills.common import _is_door_actor as _is_door_actor
from .skills.common import _probe_stick as _probe_stick
from .skills.common import _steer_to as _steer_to
from .skills.common import _pulse as _pulse
from .skills.common import _target_position as _target_position
from .skills.common import _yaw_to_target as _yaw_to_target
from .skills.common import _yaw_error_units as _yaw_error_units
from .skills.common import _aim_error as _aim_error
from .skills.common import _aim_stick as _aim_stick
from .skills.intents import _door_intent_actor as _door_intent_actor
from .skills.intents import _traversal_intent_direction as _traversal_intent_direction
from .skills.intents import _as_door_interaction as _as_door_interaction
from .skills.menus import _choose_dialogue as _choose_dialogue
from .skills.menus import _play_song as _play_song
from .skills.menus import _inventory_slot as _inventory_slot
from .skills.menus import _menu_grid_directions as _menu_grid_directions
from .skills.menus import _wait_pause_ready as _wait_pause_ready
from .skills.menus import _equip_item as _equip_item
from .skills.menus import _gear_observation as _gear_observation
from .skills.menus import _equipment_point as _equipment_point
from .skills.menus import _equip_gear as _equip_gear
from .skills.navigation import _recovery_inputs as _recovery_inputs
from .skills.navigation import _safe_escape_target as _safe_escape_target
from .skills.navigation import _backtrack_recovery as _backtrack_recovery
from .skills.navigation import _navigate_local as _navigate_local
from .skills.navigation import _face_target as _face_target
from .skills.navigation import _follow_actor as _follow_actor
from .skills.navigation import _explore_area as _explore_area
from .skills.interactions import _wait_interaction_evidence as _wait_interaction_evidence
from .skills.interactions import _interact_with_door as _interact_with_door
from .skills.interactions import _manipulate_object as _manipulate_object
from .skills.aim import _aim_at as _aim_at
from .skills.combat import _fight_enemy as _fight_enemy
from .skills.executor import execute_skill as execute_skill
from .skills.executor import _execute_skill as _execute_skill
from .skills.traversal import _best_traversal_probe as _best_traversal_probe
from .skills.traversal import _best_climb_surface_probe as _best_climb_surface_probe
from .skills.traversal import _traverse_local as _traverse_local

CONTRACT_VERSION = "state-v8/skills-v9/trajectory-v3/prompt-v12"

def _sanitize_transition_actor(actor: dict | None) -> dict | None:
    if actor is None:
        return None
    row = dict(actor)
    category = str(row.get("category_name") or "").lower()
    metadata = f"{row.get('name') or ''} {row.get('description') or ''}".lower()
    transition_words = ("warp", "scene change", "scene_change", "scene exit",
                        "loading zone", "entrance", "exit", "portal", "teleport")
    if category == "door":
        row["name"] = "Door"
        row["description"] = ""
    elif any(word in metadata for word in transition_words):
        row["name"] = "Transition Object"
        row["description"] = ""
    return row


def _strip_transition_ids(value):
    if isinstance(value, dict):
        row = {key: _strip_transition_ids(item) for key, item in value.items()
               if key not in {"exit_index", "entrance_index", "floor_exit_index"}}
        if "actor_id" in row and ("name" in row or "category_name" in row):
            row = _sanitize_transition_actor(row) or {}
        return row
    if isinstance(value, list):
        return [_strip_transition_ids(item) for item in value]
    return value


def _model_state_payload(game: GameState) -> dict:
    """Model-facing state: transition surfaces are physical but destination-opaque."""
    payload = game.model_dump(exclude={"events", "upstream_revision", "last_command_seq",
        "input_receipts", "last_received_seq", "last_applied_command_seq", "owner_epoch",
        "capture_tick", "input_tick", "event_floor", "event_seq", "full_seq", "navmesh",
        "scene_exits", "entrance_index"})
    payload["scene_exits"] = [{"position": list(row.position)} for row in game.scene_exits]
    payload = _strip_transition_ids(payload)
    if game.room_actors:
        payload.pop("nearby_actors", None)
    return payload


def _model_world_edges(rows: list[dict]) -> list[dict]:
    """Only empirically observed topology is model-visible; native entrance IDs remain local."""
    allowed = ("from_scene", "from_scene_name", "from_room", "from_position",
               "to_scene", "to_scene_name", "to_room", "to_position", "traversals")
    return [{key: row.get(key) for key in allowed} for row in rows]


def _model_last_decision(row: dict | None) -> dict | None:
    if not row:
        return None
    # Do not echo model-authored goal/summary/memory_note back into inference.
    # They may contain an unsupported transition guess. The executable action
    # itself is enough context for interpreting last_result.
    return _strip_transition_ids({
        "skill": row.get("skill"),
        "args": row.get("args") or {},
    })


def _strip_model_authored_prose(value):
    if isinstance(value, dict):
        return {key: _strip_model_authored_prose(item) for key, item in value.items()
                if key not in {"goal", "summary", "memory_note"}}
    if isinstance(value, list):
        return [_strip_model_authored_prose(item) for item in value]
    return value


def _model_recent_events(rows: list[dict]) -> list[dict]:
    visible = []
    for raw in rows:
        row = _strip_model_authored_prose(_strip_transition_ids(raw))
        if row.get("kind") == "decision":
            data = row.get("data") or {}
            row["data"] = {"skill": data.get("skill")}
        visible.append(row)
    return visible


def _model_memory_note_persistent(_: Decision) -> bool:
    """Free-form model notes are non-authoritative; persistent learning is runtime-owned."""
    return False



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
        self.lifecycle = 0
        self.retired_tasks: set[asyncio.Task] = set()
        self.closed = False
        self.switch_request = 0
        self.started = 0.0
        self.last_decision: dict | None = None
        self.last_result: dict | None = None
        self.recent = deque(maxlen=12)
        self.hints = deque(maxlen=3)
        self.dialogue_transcript = deque(maxlen=12)
        self.seen_events: set[str] = set()
        self.seen_event_order: deque[str] = deque()
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
        self.unstick_attempted_at_score = 0
        self.unstick_attempts = 0
        self.last_diagnostic: dict | None = None
        self.diagnostic_active = False
        self.combat_profiles_cache: list[dict] | None = None
        self.combat_profiles_at = 0.0
        self.combat_learning_tainted = False

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
        elif status in {"failed", "stale"} and decision.skill in NAVIGATION_SKILLS and reason not in {
                "target_on_different_floor", "no_traversal_affordance_observed", "world_changed_during_inference",
                "navigation_no_path", "navigation_path_blocked"}:
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
                "instruction": "Replan. Create clearance by moving back before rotating; do not repeat the same forward/lateral probe."})

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
        if self.replaying_trajectory:
            return
        self.trajectory_tainted = True
        # A failed command may have moved Link; removing it creates a fictional trajectory.


    def _transition_origin_position(self, old: GameState):
        decision = self.last_decision or {}
        args = decision.get("args") or {}

        evidence = self.bridge.previous_state
        if (not evidence or evidence.instance_id != old.instance_id or
                (evidence.scene, evidence.room) != (old.scene, old.room)):
            evidence = old

        if decision.get("skill") == "traverse_exit" and old.scene_exits and evidence.player:
            target = args.get("target_position")
            if isinstance(target, list) and len(target) == 3:
                requested = tuple(float(v) for v in target)
                nearest = min(old.scene_exits, key=lambda row: math.dist(row.position, requested))
                if (math.dist(nearest.position, requested) <= 140.0 and
                        evidence.player.floor_exit_index == nearest.exit_index):
                    # This is the only case where an exit-surface coordinate is
                    # promoted: Link's actual pre-transition floor poly identifies
                    # the same exit the model selected.
                    return nearest.position

        # A different exit, door, script, or accidental transition occurred.
        # Preserve the latest actual player position and do not assign the
        # destination to the planned surface.
        return evidence.player.position if evidence.player else (
            old.player.position if old.player else None)

    def _learn_transition(self, state: GameState, old: GameState | None):
        if self.state != "running" or self.trajectory_tainted:
            return
        if not old or not self.config or self.config.memory_mode != "adaptive":
            return
        if old.instance_id != state.instance_id or not old.in_game or not state.in_game:
            return
        if min(old.scene, state.scene, old.room, state.room) < 0:
            return
        if (old.scene, old.room) == (state.scene, state.room):
            return
        transition_origin = self._transition_origin_position(old)
        edge_id = self.store.learn_world_edge(self.namespace,
            {"scene": old.scene, "scene_name": old.scene_name, "room": old.room,
             "position": transition_origin},
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
                self.unstick_attempted_at_score = 0
                self.unstick_attempts = 0
                self.last_result = {"status": "interrupted", "reason": "world_transition",
                    "from": [old.scene, old.room], "to": [state.scene, state.room]}
                self.log("world_transition", {"from_scene": old.scene, "from_room": old.room,
                    "to_scene": state.scene, "to_room": state.room,
                    "entrance_index": state.entrance_index, "scene_epoch": state.scene_epoch})
            if old:
                if not old.dialogue.active and state.dialogue.active:
                    self.bridge.release()
                    entry = {"text_id": state.dialogue.text_id, "text": state.dialogue.text[:1200],
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
                    entry = {"text_id": state.dialogue.text_id, "text": state.dialogue.text[:1200],
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
                    self.seen_event_order.append(key)
                    while len(self.seen_event_order) > 4096:
                        self.seen_events.discard(self.seen_event_order.popleft())
                    self.log(event.kind, {"detail": event.detail, "scene": state.scene, "room": state.room})
                    if event.kind == "bridge_event_gap":
                        self.trajectory_tainted = True
                    if event.kind == "game_completed" and self.state == "running":
                        self.lifecycle += 1
                        self.bridge.revoke()
                        self.state, self.reason = "completed", "game_completed"
                        self.store.update_run(self.run_id, status="completed", reason="game_completed")
                        self.store.end_segments(self.run_id)
                        self.log("run_completed", {"reason": event.detail, "scene": state.scene,
                            "scene_name": state.scene_name})
                        if self.task and self.task is not asyncio.current_task():
                            self.task.cancel()
            if old and old.player and state.player and old.player.health > 0 and state.player.health == 0:
                self.log("player_died", {"scene": state.scene,
                    "last_skill": (self.last_decision or {}).get("skill")})
                self.store.remember(self.namespace, state.scene,
                    f"Death observed after skill: {(self.last_decision or {}).get('skill', 'unknown')}. "
                    "Causality is unconfirmed; reconsider the tactic.")
        self.publish()

    async def _handle_dialogue(self, game: GameState) -> bool:
        with owned(self.bridge, 'dialogue'):
            if not game.dialogue.active:
                return False
            if game.dialogue.text and (not self.dialogue_transcript or
                    self.dialogue_transcript[-1].get("text") != game.dialogue.text):
                self.dialogue_transcript.append({"text_id": game.dialogue.text_id,
                    "text": game.dialogue.text[:1200], "choices": game.dialogue.choices,
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
        with owned(self.bridge, 'gameover'):
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

    async def _auto_unstick(self, game: GameState) -> bool:
        with owned(self.bridge, 'recovery'):
            if (self.stuck_score < 6 or self.stuck_score <= self.unstick_attempted_at_score or
                    not game.player or game.paused or game.dialogue.active or game.cutscene_active or
                    game.game_over_state != 0):
                return False
            # If the game already exposes an actionable A prompt, let the planner interact instead of backing away.
            if game.context_action.label != "none":
                return False

            self.unstick_attempted_at_score = self.stuck_score
            attempt = self.unstick_attempts
            self.unstick_attempts += 1
            self.trajectory_tainted = True
            recovery = await _backtrack_recovery(self.bridge, game, attempt)
            moved = recovery["distance"]
            turned = abs(recovery["yaw_delta"])
            recovered = recovery["world_changed"] or moved >= 8.0 or turned >= 1200
            if recovered:
                self.stuck_score = max(2, self.stuck_score - 2)
            self.last_result = {"status": "completed" if recovered else "failed",
                "reason": "auto_unstick_reverse_escape" if recovered else "auto_unstick_no_progress",
                "distance": moved, "yaw_delta": recovery["yaw_delta"],
                "world_changed": recovery["world_changed"], "recovery_attempt": attempt + 1,
                "acknowledged": recovery["acknowledged"]}
            self.log("auto_unstick", self.last_result)
            return True

    async def _defensive_guard(self):
        with owned(self.bridge, "inference_guard"):
            try:
                while self.state == "running":
                    game = self.bridge.state
                    if not game:
                        self.bridge.release()
                        await asyncio.sleep(.1)
                        continue
                    ready = (game.in_game and game.player and not game.paused and not game.dialogue.active
                        and not game.cutscene_active and not game.game_over_state)
                    threat = ready and any(a.category in {5, 9} and a.distance <= 180 for a in game.room_actors)
                    shield = any(g.equipment_type == "shield" and g.equipped for g in game.progress.equipment)
                    if threat and shield:
                        self.bridge.send(buttons=BUTTONS["Z"] | BUTTONS["R"], lease_ms=300)
                    else:
                        self.bridge.release()
                    await feedback(self.bridge, game, .1)
            except RuntimeError:
                # Feedback loss stops guard renewal; native watchdog releases even if Python stalls.
                self.bridge.release()
            finally:
                self.bridge.release()

    async def _decide_with_guard(self, config: RunConfig, prompt: str, *, call_id: str | None = None):
        guard = asyncio.create_task(self._defensive_guard())
        inference = asyncio.create_task(self.providers[config.provider].decide(config, prompt))
        try:
            # Shield lets the runtime stop immediately even if a provider ignores cancellation.
            return await asyncio.shield(inference)
        finally:
            guard.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await guard
            if not inference.done():
                inference.cancel()
                self._retire_task(inference, call_id=call_id)
            elif inference.cancelled():
                pass
            else:
                # Retrieve possible exceptions when caller cancellation races with completion.
                inference.exception()
    
    def _enemy_learning_context(self, game: GameState, limit: int = 8) -> list[dict]:
        if not self.config or self.config.memory_mode != "adaptive" or not self.namespace or not game.player:
            return []
        actors = []
        seen: set[str] = set()
        ordered = list(game.room_actors)
        if game.target_actor is not None:
            ordered.insert(0, game.target_actor)
        for actor in ordered:
            if actor.category not in {5, 9}:
                continue
            key = enemy_key(game, actor)
            if key in seen:
                continue
            seen.add(key)
            profile = self.store.combat_profile(self.namespace, key, actor_id=actor.actor_id,
                category=actor.category, enemy_name=actor.description or actor.name, create=False)
            if profile:
                actors.append(compact_profile(profile))
            else:
                actors.append({"enemy_key": key, "actor_id": actor.actor_id,
                    "enemy_name": actor.description or actor.name or f"Actor {actor.actor_id}",
                    "category": actor.category, "encounters": 0, "wins": 0, "losses": 0,
                    "incomplete": 0, "damage_taken": 0, "best_by_state": {}})
            if len(actors) >= limit:
                break
        return actors

    def _combat_profile_for_decision(self, game: GameState, decision: Decision) -> tuple[dict | None, dict | None]:
        if decision.skill != "fight_enemy" or decision.args.target_actor_id is None or not game.player:
            return None, None
        actor = _matching_actor(game, decision.args.target_actor_id,
            decision.args.target_actor_params, decision.args.target_actor_uid)
        if actor is None or actor.category not in {5, 9}:
            return None, None
        key = enemy_key(game, actor)
        enemy = {"enemy_key": key, "actor_id": actor.actor_id, "actor_uid": actor.actor_uid, "params": actor.params,
            "category": actor.category, "enemy_name": actor.description or actor.name or f"Actor {actor.actor_id}",
            "scene": game.scene, "room": game.room}
        if not self.config or self.config.memory_mode != "adaptive" or not self.namespace:
            return None, enemy
        profile = self.store.combat_profile(self.namespace, key, actor_id=actor.actor_id,
            category=actor.category, enemy_name=enemy["enemy_name"], create=True)
        return profile, enemy

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
            if self.state in {"starting", "running", "paused"}:
                raise ValueError("Stop the current run before starting another")
            if self.diagnostic_active:
                raise ValueError("Release or wait for the input diagnostic before starting a run")
            game = self.bridge.state
            if not self.bridge.connected or not game or not game.in_game or not game.player:
                raise ValueError("Connect the bridge and load a playable save in SoH first")
            if config.provider == "demo" and game.source != "simulator":
                raise ValueError("The deterministic demo is not a game-playing model")
            self.lifecycle += 1
            generation = self.lifecycle
            instance_id = game.instance_id
            self.run_id = self.segment_id = None
            self.namespace = ""
            self.metrics_cache = None
            self.state, self.reason = "starting", "validating_provider"
            self.publish(True)
        try:
            info = await self.validate_model(config)
        except BaseException:
            async with self.lock:
                if generation == self.lifecycle and self.state == "starting":
                    self.state, self.reason = "idle", "provider_validation_failed"
                    self.publish(True)
            raise
        async with self.lock:
            if generation != self.lifecycle or self.state != "starting":
                raise ValueError("Start cancelled while validating the provider")
            if (not self.bridge.connected or not self.bridge.state or not self.bridge.state.in_game
                    or self.bridge.state.instance_id != instance_id or not self.bridge.state.player):
                self.state, self.reason = "idle", "game_changed_during_validation"
                raise ValueError("Game changed while validating the provider")
            game = self.bridge.state
            self.game_instance = game.instance_id
            self.pending_switch = None
            self.metrics_cache = None
            self.config, self.selected_model = config, info
            self.combat_profiles_cache = None
            self.combat_learning_tainted = False
            fingerprint = hashlib.sha256(json.dumps({"contract": CONTRACT_VERSION,
                "revision": game.upstream_revision, "initial": game.model_dump(exclude={"seq", "events", "last_command_seq"}),
                "checkpoint_label": config.checkpoint_label}, sort_keys=True).encode()).hexdigest()
            self.run_id = self.store.new_run(config.model_dump(), game.source, fingerprint)
            self.namespace = self.new_namespace(config)
            self.segment_id = self.store.segment(self.run_id, config.model_dump(), self.namespace)
            self.seen_events = {f"{game.instance_id}:{e.id}" for e in game.events}
            self.seen_event_order = deque(self.seen_events)
            self.last_decision = self.last_result = None
            self.replay_attempts.clear()
            self.replaying_trajectory = False
            self.stuck_score = 0
            self.stuck_notified_at = 0
            self.unstick_attempted_at_score = 0
            self.unstick_attempts = 0
            self._reset_trajectory_trace(game)
            self.recent.clear()
            self.hints.clear()
            self.dialogue_transcript.clear()
            self.started = time.monotonic()
            self.state, self.reason = "running", ""
            self.log("run_started", {"contract": CONTRACT_VERSION, "checkpoint_certified": False})
            self.bridge.enable_control()
            self.task = asyncio.create_task(self.loop(self.lifecycle))

    async def halt(self, state: str, reason: str):
        # Revoke first. Never wait for network/provider teardown to stop Link.
        self.lifecycle += 1
        self.bridge.revoke()
        self.state, self.reason = state, reason
        self.pending_switch = None
        if self.run_id:
            self.store.update_run(self.run_id, status=state, reason=reason)
            if state in {"stopped", "completed"}:
                self.store.end_segments(self.run_id)
        task, self.task = self.task, None
        if task and task is not asyncio.current_task():
            task.cancel()
            self._retire_task(task)
        self.publish(True)

    async def control(self, action: str):
        if action not in {"pause", "stop", "take_control", "resume"}:
            raise ValueError("Unknown control action")
        # Only short state transitions hold this lock; discovery runs outside it.
        async with self.lock:
            if self.state == "starting" and action != "resume":
                await self.halt("stopped", action)
                return
            if not self.run_id or self.state not in {"running", "paused"}:
                raise ValueError("No active run")
            if action in {"pause", "stop", "take_control"}:
                # Human input during a pause can alter subsequent combat context. Do not
                # promote later combat as autonomous learning in this run.
                self._reset_trajectory_trace(self.bridge.state, tainted=True)
                if action in {"pause", "take_control"}:
                    self.combat_learning_tainted = True
                if action == "take_control":
                    self.store.update_run(self.run_id, assisted=True)
                    self.log("take_control")
                await self.halt("stopped" if action == "stop" else "paused", action)
            else:
                if self.state != "paused":
                    raise ValueError("The run is not paused")
                if not self.bridge.connected:
                    raise ValueError("Reconnect the bridge before resuming")
                if self.config and runtime_exhausted(self.config, time.monotonic() - self.started):
                    raise ValueError("runtime_budget_reached; start a new run or configure duration 0")
                self.lifecycle += 1
                self.bridge.enable_control()
                self.state, self.reason = "running", ""
                self.store.update_run(self.run_id, status="running", reason="")
                self._reset_trajectory_trace(self.bridge.state)
                self.task = asyncio.create_task(self.loop(self.lifecycle))
            self.publish(True)

    async def input_diagnostic(self, action: str) -> dict:
        """Run a local controller diagnostic without provider calls, run accounting or learned memory."""
        async with self.lock:
            if self.state in {"starting", "running", "paused"}:
                raise ValueError("Stop the agent before running input diagnostics")
            if self.diagnostic_active:
                raise ValueError("Another input diagnostic is already running")
            game = self.bridge.state
            if not self.bridge.connected or not game or not game.in_game or not game.player:
                raise ValueError("Load a playable SoH save before running input diagnostics")
            if not self.bridge.realtime:
                raise ValueError("Input diagnostics require BRIDGE V2")
            self.diagnostic_active = True
            self.bridge.enable_control()
            self.publish(True)
        try:
            result = await run_input_diagnostic(self.bridge, action)
            self.last_diagnostic = result
            return result
        finally:
            # Never hold the runtime lock while motor tests run; emergency handoff stays available.
            self.bridge.revoke()
            async with self.lock:
                self.diagnostic_active = False
            self.publish(True)

    def release_diagnostic_control(self) -> dict:
        """Immediate human handoff. Safe to call while the diagnostic coroutine is still running."""
        if self.diagnostic_active:
            self.bridge.revoke()
        self.publish(True)
        return {"released": True, "diagnostic_active": self.diagnostic_active}

    async def switch(self, change: SwitchConfig):
        async with self.lock:
            if self.state not in {"running", "paused"} or not self.config or not self.run_id:
                raise ValueError("No active run")
            generation, run_id = self.lifecycle, self.run_id
            self.switch_request += 1
            request = self.switch_request
            config = RunConfig.model_validate({**self.config.model_dump(), **change.model_dump()})
            if config.provider == "demo" and self.bridge.state and self.bridge.state.source != "simulator":
                raise ValueError("Demo provider cannot control SoH")
        info = await self.validate_model(config)
        async with self.lock:
            if (generation != self.lifecycle or run_id != self.run_id or request != self.switch_request
                    or self.state not in {"running", "paused"}):
                raise ValueError("Run changed while validating the new model; change discarded")
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
        self.combat_learning_tainted = True
        self.namespace = self.new_namespace(self.config)
        self.combat_profiles_cache = None
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
        self.combat_learning_tainted = True
        self.store.update_run(self.run_id, assisted=True)
        self._reset_trajectory_trace(self.bridge.state, tainted=True)
        self.log("human_hint", {"text": text})

    def budget_check(self, prompt: str):
        assert self.run_id and self.config and self.selected_model
        reason = budget_reason(self.config, self.store.metrics(self.run_id),
            time.monotonic() - self.started,
            reserve=lambda: reserve_cost(self.selected_model, prompt, self.config.max_output_tokens))
        if reason:
            raise ProviderFailure(reason)

    def _retire_task(self, task: asyncio.Task, *, call_id: str | None = None):
        self.retired_tasks.add(task)
        def finished(done: asyncio.Task):
            self.retired_tasks.discard(done)
            try:
                result = done.result()
            except asyncio.CancelledError:
                return
            except Exception as exc:
                result = None
                usage = getattr(exc, "usage", None)
            else:
                usage = getattr(result, "usage", None)
            # Reconcile a late paid response, never execute its decision or relabel it successful.
            if call_id and usage is not None and not self.closed:
                try:
                    self.store.finish_call(call_id, usage=usage.model_dump())
                    self.metrics_cache = None
                except Exception:
                    # Keep the persisted cancellation/unknown accounting; no automatic retry.
                    pass
        task.add_done_callback(finished)

    async def loop(self, generation: int | None = None):
        generation = self.lifecycle if generation is None else generation
        if generation != self.lifecycle or self.state != "running":
            return
        assert self.config
        remaining = (max(0.0, self.config.max_runtime_s - (time.monotonic() - self.started))
                     if self.config.max_runtime_s > 0 else None)
        with self.bridge.authority.run_scope():
            try:
                async with asyncio.timeout(remaining):
                    await self._run_loop(generation)
            except TimeoutError:
                if generation == self.lifecycle:
                    self.log("run_paused", {"reason": "runtime_budget_reached"})
                    await self.halt("paused", "runtime_budget_reached")

    async def _run_loop(self, generation: int):
        try:
            while self.state == "running" and generation == self.lifecycle:
                self.apply_switch()
                assert self.config and self.run_id and self.segment_id
                if runtime_exhausted(self.config, time.monotonic() - self.started):
                    raise ProviderFailure("runtime_budget_reached")
                game = self.bridge.state
                if not self.bridge.connected or not game:
                    raise ProviderFailure("bridge_disconnected")
                if game.instance_id != self.game_instance:
                    raise ProviderFailure("game_instance_changed; start a new run")
                if not game.in_game:
                    if runtime_exhausted(self.config, time.monotonic() - self.started):
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
                game = self.bridge.state or game
                if await self._auto_unstick(game):
                    await asyncio.sleep(0.08)
                    game = self.bridge.state or game
                state_payload = _model_state_payload(game)
                observation = {"contract": CONTRACT_VERSION, "objective": self.config.goal,
                    "state": state_payload,
                    "last_decision": _model_last_decision(self.last_decision),
                    "last_result": _strip_transition_ids(self.last_result),
                    "events": _model_recent_events(list(self.recent)[-5:]),
                    "dialogue_transcript": _strip_transition_ids(list(self.dialogue_transcript)),
                    "memory": [r["note"] for r in self.store.recall(self.namespace, game.scene, limit=6)],
                    "recent_global_memory": [r["note"] for r in self.store.recall(self.namespace, limit=8)],
                    "known_world_edges": _model_world_edges(
                        self.store.world_neighbors(self.namespace, game.scene, game.room)),
                    "enemy_learning": self._enemy_learning_context(game),
                    "navigation_mesh": {
                        "available": game.navmesh.available,
                        "cells": len(game.navmesh.cells),
                        "step": game.navmesh.step if game.navmesh.available else None,
                        "radius": game.navmesh.step * game.navmesh.half_extent if game.navmesh.available else None,
                    },
                    "stuck_score": self.stuck_score,
                    "human_hints": list(self.hints)}
                prompt = json.dumps(observation, separators=(",", ":"), ensure_ascii=False)
                self.budget_check(prompt)
                call_id = self.store.begin_call(self.run_id, self.segment_id, observation)
                started = time.monotonic()
                result = None
                try:
                    result = await self._decide_with_guard(self.config, prompt, call_id=call_id)
                    if generation != self.lifecycle or self.state != "running":
                        self.store.finish_call(call_id, status="cancelled", usage=result.usage.model_dump())
                        raise asyncio.CancelledError
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
                self.log("decision", {"summary": decision.summary, "skill": decision.skill})
                self._record_trajectory_action(decision, game)
                combat_profile, combat_enemy = self._combat_profile_for_decision(game, decision)
                self.last_result = await execute_skill(self.bridge, decision, game,
                    combat_profile=combat_profile)
                if decision.memory_note:
                    if _model_memory_note_persistent(decision):
                        self.store.remember(self.namespace, game.scene, decision.memory_note)
                    else:
                        self.log("memory_note_skipped_unverified", {
                            "skill": decision.skill,
                            "reason": "freeform_model_memory_is_not_persistent",
                        })
                if (combat_enemy and self.config.memory_mode == "adaptive" and self.namespace
                        and not self.combat_learning_tainted
                        and isinstance(self.last_result.get("learning_trace"), list)
                        and self.last_result["learning_trace"]):
                    updated_profile = self.store.record_combat_encounter(
                        self.namespace, self.run_id, combat_enemy, self.last_result)
                    if updated_profile:
                        self.combat_profiles_cache = None
                        self.log("combat_profile_updated", compact_profile(updated_profile) or {})
                elif (combat_enemy and self.combat_learning_tainted
                        and isinstance(self.last_result.get("learning_trace"), list)
                        and self.last_result["learning_trace"]):
                    self.log("combat_learning_skipped_assisted", {
                        "enemy_key": combat_enemy["enemy_key"], "reason": "human_intervention_in_run"})
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
            if generation != self.lifecycle:
                return
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
        if self.namespace and (self.combat_profiles_cache is None or
                time.monotonic() - self.combat_profiles_at >= 1):
            self.combat_profiles_cache = [compact_profile(row) for row in
                self.store.list_combat_profiles(self.namespace, limit=40)]
            self.combat_profiles_at = time.monotonic()
        return {"control_generation": self.lifecycle, "status": self.state, "reason": self.reason, "run_id": self.run_id,
            "config": self.config.model_dump() if self.config else None,
            "metrics": self.metrics_cache if self.run_id else None,
            "pending_switch": self.pending_switch[0].model_dump() if self.pending_switch else None,
            "elapsed_s": round(time.monotonic() - self.started) if self.run_id else 0,
            "last_decision": self.last_decision, "last_result": self.last_result,
            "diagnostic": self.last_diagnostic, "diagnostic_active": self.diagnostic_active,
            "combat_learning_tainted": self.combat_learning_tainted,
            "events": list(self.recent), "dialogue_transcript": list(self.dialogue_transcript),
            "bridge": self.bridge.status(), "memory": self.store.recall(self.namespace, limit=30) if self.namespace else [],
            "trajectories": self.store.list_trajectories(self.namespace, limit=30) if self.namespace else [],
            "world_edges": self.store.list_world_edges(self.namespace, limit=100) if self.namespace else [],
            "combat_profiles": self.combat_profiles_cache or []}
