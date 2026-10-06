"""Bounded local task execution under an immutable strategic objective.

This supervisor does not know an OoT route or change ObjectiveCompletion. Its
postconditions use observed effects; a portal is never complete by proximity.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from .lifetime import ObservationLifetime
from .locomotion import camera_modal_active, locomotor_mode


@dataclass
class TaskAttempt:
    identity: str
    kind: str
    phase: str
    context: tuple
    target: tuple | None
    started: float
    last_progress: float
    best_distance: float
    result: str | None = None


def physical_context(game):
    return (game.instance_id, game.scene, game.room, game.mirrored_world,
            game.player.age if game.player else None)


class ExecutionSupervisor:
    version = "local-executor-v1"

    def __init__(self, *, room_budget_s=120.0, input_budget_s=3.0, modal_budget_s=300.0):
        self.room_budget_s = room_budget_s
        self.input_budget_s = input_budget_s
        self.modal_budget_s = modal_budget_s
        self.modal_signature = None
        self.modal_effect_at = None
        self.task: TaskAttempt | None = None
        self.plan_revision = 0
        self.state = "idle"
        self.reason = ""
        self.context = None
        self.progress_at = time.monotonic()
        self.campaign_progress_at = self.progress_at
        self.last_observed_at = self.progress_at
        self.input_at = self.progress_at
        self.last_input_seq = 0
        self.completed = 0
        self.failed = 0
        self.incident: dict | None = None
        self.trace = deque(maxlen=128)
        self.diagnostics_error = ""
        self.lifetime = ObservationLifetime()

    def observe(self, game, guidance, *, macro_progress=False, durable_progress=False, now=None):
        now = time.monotonic() if now is None else now
        elapsed = max(0.0, now - self.last_observed_at)
        self.last_observed_at = now
        context = physical_context(game)
        if self.lifetime.observe(game):
            self.task = None
            self.context = None
        if self.context != context:
            if self.task:
                normal_crossing = (self.context is not None and self.context[0] == context[0]
                    and self.context[1:3] != context[1:3] and self.context[3:] == context[3:]
                    and not game.game_over_state and game.player and game.player.health > 0)
                self.task.result = "success" if normal_crossing and self.task.kind == "traverse_portal" else "wrong_context"
                self.completed += int(self.task.result == "success")
            self.context = context
            self.task = None
            self.progress_at = self.input_at = now
            self.last_input_seq = game.last_applied_command_seq
            self.state, self.reason = "executing", ""
            self.modal_signature = None
            self.modal_effect_at = None
        if macro_progress or durable_progress:
            self.progress_at = now
        if durable_progress:
            self.campaign_progress_at = now
            # A new prerequisite/state can justify a previously blocked attempt.
            if self.state == "blocked":
                self.state, self.reason = "executing", ""
        consumed = game.last_applied_command_seq
        if consumed > self.last_input_seq:
            self.last_input_seq, self.input_at = consumed, now
        if self.state == "blocked":
            return
        modal = (game.dialogue.active or game.pause_menu.active or game.cutscene_active or game.paused
                 or camera_modal_active(game.player))
        if modal:
            signature = (game.dialogue.active, game.dialogue.text_id, game.dialogue.text,
                         game.dialogue.choice_index, game.pause_menu.active,
                         game.pause_menu.page_index, game.pause_menu.cursor_point,
                         game.cutscene_active, game.paused, camera_modal_active(game.player))
            if signature != self.modal_signature:
                self.modal_signature, self.modal_effect_at = signature, now
            if now - self.modal_effect_at >= self.modal_budget_s:
                self.block("modal_without_observed_effect", now)
                return
            self.state = "modal"
            self.progress_at += elapsed
            self.campaign_progress_at += elapsed
            self.input_at += elapsed
            if self.task:
                self.task.last_progress = now
            return
        self.modal_signature = None
        self.modal_effect_at = None
        if game.game_over_state or not game.player or game.player.health <= 0:
            self.task = None
            self.block("game_over_requires_recovery", now)
            return
        self.state = "executing"
        if now - self.campaign_progress_at >= 600.0:
            self.block("campaign_loop_without_progress", now)
            return
        if now - self.progress_at >= self.room_budget_s:
            self.block("no_durable_or_macro_progress", now)
            return
        if game.protocol == 3 and now - self.input_at >= self.input_budget_s:
            self.block("input_not_consumed", now)
            return
        target = guidance.get("target")
        if not guidance.get("active") or target is None:
            return
        kind = "traverse_portal" if guidance.get("exit_active") else "traverse_surface" if guidance.get("traversal_route_active") else "move_to_observed_region"
        identity = str(guidance.get("escape_key") or guidance.get("route_waypoint_id") or target)
        distance = math.dist(game.player.position, target)
        if not self.task or self.task.identity != identity:
            self.plan_revision += 1
            self.task = TaskAttempt(identity, kind, "prepare", context, tuple(target), now, now, distance)
        task = self.task
        task.phase = "verify" if distance <= 45.0 else "execute"
        if distance < task.best_distance - 8.0:
            task.best_distance, task.last_progress = distance, now
        # Only movement-to-region can finish geometrically. A portal requires
        # context change, a surface traversal requires actual locomotor effects.
        if task.kind == "move_to_observed_region" and distance <= 35.0:
            if task.result != "success":
                self.completed += 1
            task.result = "success"
            task.phase = "succeeded"

    def block(self, reason, now):
        self.state, self.reason = "blocked", reason
        self.failed += 1
        if self.task:
            self.task.result, self.task.phase = "timeout", "failed"
        if self.incident is None:
            self.incident = {"reason": reason, "monotonic_s": now, "executor": self.snapshot(), "trace": list(self.trace)}

    def record(self, game, guidance, setpoint, bridge):
        # Whitelist diagnostics: no transport token, provider auth, ROM or save.
        self.trace.append({
            "seq": game.seq, "full_seq": game.full_seq, "capture_tick": game.capture_tick,
            "scene_epoch": game.scene_epoch, "context_epoch": game.context_epoch,
            "instance_id": game.instance_id, "scene": game.scene, "room": game.room,
            "position": game.player.position if game.player else None,
            "locomotor_mode": locomotor_mode(game),
            "camera_input_yaw": game.camera_input_yaw,
            "dialogue_active": game.dialogue.active, "pause_active": game.pause_menu.active,
            "target": guidance.get("target"), "source": guidance.get("source"),
            "blocked": guidance.get("blocked"), "detour": guidance.get("detour"),
            "requested": asdict(setpoint), "last_consumed": game.last_applied_command_seq,
            "input_tick": game.input_tick,
            "receipts": [row.model_dump() for row in list(bridge.receipts.values())[-4:]],
        })

    def snapshot(self):
        task = asdict(self.task) if self.task else None
        return {"version": self.version, "state": self.state, "reason": self.reason,
                "plan_revision": self.plan_revision, "task": task, "completed": self.completed,
                "failed": self.failed, "diagnostics_error": self.diagnostics_error or None}

    def save_incident(self, directory: Path):
        if self.incident is None:
            return None
        data = json.dumps(self.incident, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"incident-{int(time.time())}-{uuid4().hex[:8]}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        self.incident = None
        return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}
