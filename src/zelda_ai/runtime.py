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

CONTRACT_VERSION = "state-v1/skills-v0.2/prompt-v2"
BUTTONS = {"A": 0x8000, "B": 0x4000, "Z": 0x2000, "R": 0x0010,
    "C_LEFT": 0x0002, "C_DOWN": 0x0004, "C_RIGHT": 0x0001}
SKILL_CATALOG = [
    {"id": "move", "name": "Translação curta", "status": "implemented", "version": "0.2"},\n    {"id": "turn", "name": "Giro local com feedback de yaw", "status": "implemented", "version": "0.2"},
    {"id": "interact", "name": "Interagir / confirmar (A)", "status": "implemented", "version": "0.1"},
    {"id": "attack", "name": "Ataque básico (B)", "status": "implemented", "version": "0.1"},
    {"id": "defend", "name": "Defesa (Z + R)", "status": "implemented", "version": "0.1"},
    {"id": "target", "name": "Z-target", "status": "implemented", "version": "0.1"},
    {"id": "use_item", "name": "Item já equipado em C", "status": "implemented", "version": "0.1"},
    {"id": "wait", "name": "Esperar com controle neutro", "status": "implemented", "version": "0.1"},
    {"id": "aim", "name": "Mira calibrada / arco / Hookshot", "status": "planned", "version": None},
    {"id": "navigate", "name": "Navegação espacial e colisão", "status": "planned", "version": None},
    {"id": "combat_learning", "name": "Aprendizado motor e novas skills", "status": "planned", "version": None},
]


def controller_input(decision: Decision) -> tuple[int, int, int]:
    skill, args = decision.skill, decision.args
    if skill == "move":
        amount = round(80 * args.strength)
        x, y = {"forward": (0, amount), "back": (0, -amount), "left": (-amount, 0), "right": (amount, 0)}[args.direction]
        return 0, x, y
    button = {"interact": BUTTONS["A"], "attack": BUTTONS["B"], "defend": BUTTONS["Z"] | BUTTONS["R"],
        "target": BUTTONS["Z"], "wait": 0}.get(skill)
    if skill == "use_item":
        button = BUTTONS[f"C_{args.slot.upper()}"]
    return button or 0, 0, 0


async def execute_skill(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    before = bridge.state
    if not before or not bridge.connected:
        raise RuntimeError("bridge_disconnected")
    if (before.instance_id, before.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "scene_changed_during_inference"}
    if before.paused or not before.in_game:
        return {"status": "stale", "reason": "game_not_ready"}
    buttons, x, y = controller_input(decision)
    deadline = time.monotonic() + decision.args.duration_ms / 1000
    status, reason, acknowledged = "completed", "duration_elapsed", False
    first_command = None
    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not bridge.connected or not current:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (before.instance_id, before.scene_epoch):
                reason = "scene_changed"
                break
            if current.paused or not current.in_game:
                reason = "game_not_ready"
                break
            command_id = bridge.send(buttons=buttons, stick_x=x, stick_y=y, lease_ms=300)
            if first_command is None:
                first_command = command_id
            acknowledged |= current.last_command_seq >= first_command
            await asyncio.sleep(min(0.1, max(0, deadline - time.monotonic())))
    finally:
        bridge.release()
    # Wait for a post-action sample; never report a hit or path success from elapsed time alone.
    await asyncio.sleep(0.22)
    after = bridge.state
    distance, damage, yaw_delta = None, None, None
    if after:
        acknowledged |= first_command is not None and after.last_command_seq >= first_command
    if before.player and after and after.player and before.scene_epoch == after.scene_epoch:
        distance = math.dist(before.player.position, after.player.position)
        damage = max(0, before.player.health - after.player.health)\n        yaw_delta = ((after.player.yaw - before.player.yaw + 32768) % 65536) - 32768
    if not acknowledged:
        status, reason = "failed", "input_not_acknowledged"
    elif decision.skill == "move" and distance is not None and distance < 1:
        status, reason = "failed", "no_displacement_observed"
    return {"status": status, "reason": reason, "distance": distance, "health_lost": damage,
        "acknowledged": acknowledged, "skill": decision.skill}


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

    def on_state(self, state: GameState, old: GameState | None):
        if self.run_id and self.state in {"running", "paused"}:
            if old and old.instance_id != state.instance_id:
                self.log("game_instance_changed")
                self.bridge.release()
            for event in state.events:
                key = f"{state.instance_id}:{event.id}"
                if key not in self.seen_events:
                    self.seen_events.add(key)
                    self.log(event.kind, {"detail": event.detail, "scene": state.scene})
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
        self.store.update_run(self.run_id, mixed=True)
        self.log("model_changed", {"provider": self.config.provider,
            "model": self.config.model, "effort": self.config.effort})

    def hint(self, text: str):
        if not self.run_id or self.state not in {"running", "paused"}:
            raise ValueError("No active run")
        self.hints.append(text)
        self.store.update_run(self.run_id, assisted=True)
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
                if game.paused or not game.in_game:
                    if time.monotonic() - self.started >= self.config.max_runtime_s:
                        raise ProviderFailure("runtime_budget_reached")
                    await asyncio.sleep(0.25)
                    continue
                observation = {"contract": CONTRACT_VERSION, "objective": self.config.goal,
                    "state": game.model_dump(exclude={"events", "upstream_revision", "last_command_seq"}),
                    "last_result": self.last_result, "events": list(self.recent)[-5:],
                    "memory": [r["note"] for r in self.store.recall(self.namespace, game.scene)],
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
                self.last_result = await execute_skill(self.bridge, decision, game)
                self.log("skill_failed" if self.last_result["status"] == "failed" else "skill_result", self.last_result)
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
            "memory": self.store.recall(self.namespace, limit=30) if self.namespace else []}
