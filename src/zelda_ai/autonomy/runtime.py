from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import time
from collections import deque
from pathlib import Path

from pydantic import ValidationError

from ..budgets import budget_reason, runtime_exhausted
from ..models import ModelInfo, RunConfig, SwitchConfig
from ..providers.base import ProviderFailure
from ..providers.openrouter import reserve_cost
from .controller import ContinuousController
from .models import AgentIntent
from .prompt import AUTONOMY_SYSTEM_PROMPT, build_cognition_observation


CONTRACT_VERSION = "autonomy-v3/raw-controller-v1/ppo-rnd-v1"


class AutonomyRuntime:
    """Continuous autonomous runtime.

    Motor control, ML training and LLM cognition are independent tasks.  Slow
    provider inference therefore never creates the old move -> think -> move gap.
    """

    def __init__(self, bridge, store, providers: dict, model_dir: Path):
        self.bridge = bridge
        self.store = store
        self.providers = providers
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.state = "idle"
        self.reason = ""
        self.run_id: str | None = None
        self.segment_id: str | None = None
        self.config: RunConfig | None = None
        self.selected_model: ModelInfo | None = None
        self.namespace = "autonomy-v3:shared-world"
        self.game_instance: str | None = None
        self.started = 0.0
        self.lifecycle = 0
        self.lock = asyncio.Lock()
        self.tasks: set[asyncio.Task] = set()
        self.retired_tasks: set[asyncio.Task] = set()
        self.controller_task: asyncio.Task | None = None
        self.cognition_task: asyncio.Task | None = None
        self.watchdog_task: asyncio.Task | None = None
        self.usage_task: asyncio.Task | None = None
        self.settling_controller: asyncio.Task | None = None
        self.subscribers: set[asyncio.Queue] = set()
        self.persist_tasks: set[asyncio.Task] = set()
        self.persistence_lock = asyncio.Lock()
        self.persistence_error = ""
        self.closed = False
        self.last_publish = 0.0

        self.controller: ContinuousController | None = None
        self.cognition_state = "idle"
        self.cognition_error = ""
        self.thought = "Ready to start autonomous play."
        self.thinking_since: float | None = None
        self.last_cognition_at = 0.0
        self.pending_switch: tuple[RunConfig, ModelInfo] | None = None
        self.switch_request = 0

        self.recent = deque(maxlen=24)
        self.dialogue_transcript = deque(maxlen=12)
        self.seen_events: set[str] = set()
        self.seen_event_order: deque[str] = deque()
        self.cognition_trigger = asyncio.Event()
        self.cognition_signature = None
        self.metrics_cache: dict | None = None
        self.metrics_at = 0.0
        self.provider_usage: dict = {"available": False, "windows": []}
        self.provider_usage_at = 0.0

        self.bridge.on_state = self.on_state

    @staticmethod
    def _empty_metrics() -> dict:
        return {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_input_tokens": 0,
            "reasoning_output_tokens": 0,
            "total_tokens": 0,
            "unknown_usage_calls": 0,
            "unknown_cost_calls": 0,
            "known_cost_usd": 0.0,
            "cost_usd": None,
            "mean_latency_ms": None,
            "deaths": 0,
            "boss_events": 0,
            "game_completions": 0,
            "interventions": 0,
            "vision_calls": 0,
        }

    async def _db_write(self, func, *args, **kwargs):
        async with self.persistence_lock:
            return await asyncio.to_thread(func, *args, **kwargs)

    def _persist(self, func, *args, **kwargs):
        if self.closed:
            return
        async def worker():
            try:
                await self._db_write(func, *args, **kwargs)
            except Exception as exc:
                self.persistence_error = f"{type(exc).__name__}: {str(exc)[:180]}"
        task = asyncio.create_task(worker())
        self.persist_tasks.add(task)
        task.add_done_callback(self.persist_tasks.discard)

    async def _drain_persistence(self):
        while self.persist_tasks:
            tasks = list(self.persist_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _refresh_metrics(self, run_id: str | None = None):
        target_run = run_id or self.run_id
        if not target_run:
            if run_id is None:
                self.metrics_cache = None
            return
        metrics = await asyncio.to_thread(self.store.metrics, target_run)
        if target_run == self.run_id:
            self.metrics_cache = metrics
            self.metrics_at = time.monotonic()

    async def _finish_call(self, run_id: str, call_id: str, **values):
        await self._db_write(self.store.finish_call, call_id, **values)
        await self._refresh_metrics(run_id)

    async def close(self):
        self.closed = True
        await self._drain_persistence()
        retired = [task for task in self.retired_tasks if not task.done()]
        if retired:
            _, pending = await asyncio.wait(retired, timeout=6.0)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    def publish(self, force: bool = False):
        now = time.monotonic()
        if not force and now - self.last_publish < 0.05:
            return
        self.last_publish = now
        for queue in list(self.subscribers):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(True)

    def log(self, kind: str, data: dict | None = None):
        row = {"kind": kind, "data": data or {}, "at": time.time()}
        self.recent.append(row)
        if self.run_id:
            self._persist(self.store.event, self.run_id, kind, data or {})
        self.publish(True)

    @staticmethod
    def _signature(game):
        target = game.target_actor.actor_uid if game.target_actor else None
        return (
            game.scene,
            game.room,
            game.scene_epoch,
            game.dialogue.active,
            game.dialogue.text_id,
            game.dialogue.choice_index,
            game.pause_menu.active,
            game.game_over_state,
            game.context_action.code,
            target,
            tuple(game.inventory),
            tuple(game.progress.quest_items),
            tuple(game.progress.owned_equipment),
            game.progress.small_keys,
        )

    def on_state(self, state, old):
        signature = self._signature(state)
        if signature != self.cognition_signature:
            self.cognition_signature = signature
            self.cognition_trigger.set()

        if not self.run_id or self.state != "running":
            self.publish()
            return

        if old and old.instance_id != state.instance_id:
            # Revoke synchronously; lifecycle/task cleanup can safely happen on
            # the event loop without allowing one more controller lease.
            self.bridge.revoke()
            self.log("game_instance_changed")
            asyncio.create_task(self.halt("paused", "game_instance_changed"))
            return

        if old and old.instance_id == state.instance_id and (
            old.scene,
            old.room,
            old.scene_epoch,
        ) != (
            state.scene,
            state.room,
            state.scene_epoch,
        ):
            if old.player and state.player:
                self._persist(
                    self.store.learn_world_edge,
                    self.namespace,
                    {
                        "scene": old.scene,
                        "scene_name": old.scene_name,
                        "room": old.room,
                        "position": old.player.position,
                    },
                    {
                        "scene": state.scene,
                        "scene_name": state.scene_name,
                        "room": state.room,
                        "position": state.player.position,
                        "entrance_index": state.entrance_index,
                    },
                )
            self.log(
                "world_transition",
                {
                    "from_scene": old.scene,
                    "from_room": old.room,
                    "to_scene": state.scene,
                    "to_room": state.room,
                },
            )

        if old:
            if not old.dialogue.active and state.dialogue.active:
                entry = {
                    "text_id": state.dialogue.text_id,
                    "text": state.dialogue.text[:1400],
                    "choices": state.dialogue.choices,
                }
                self.dialogue_transcript.append(entry)
                self.log("dialogue_started", entry)
                speaker = state.dialogue.speaker
                speaker_label = (
                    (speaker.description or speaker.name or f"actor {speaker.actor_id}")
                    if speaker else "unknown speaker"
                )
                dialogue_evidence = state.dialogue.text.strip()[:220] or (
                    f"text_id={state.dialogue.text_id}"
                )
                self._persist(
                    self.store.remember,
                    self.namespace,
                    state.scene,
                    f"Observed dialogue with {speaker_label} in "
                    f"{state.scene_name or state.scene}/room {state.room}: "
                    f"{dialogue_evidence}",
                )
            elif state.dialogue.active and (
                old.dialogue.text_id != state.dialogue.text_id
                or old.dialogue.text != state.dialogue.text
            ):
                entry = {
                    "text_id": state.dialogue.text_id,
                    "text": state.dialogue.text[:1400],
                    "choices": state.dialogue.choices,
                }
                if not self.dialogue_transcript or self.dialogue_transcript[-1] != entry:
                    self.dialogue_transcript.append(entry)
                self.log("dialogue_changed", entry)
            old_items = {row.item_id: row.name for row in old.inventory_named}
            new_items = {row.item_id: row.name for row in state.inventory_named}
            acquired_items = [
                name for item_id, name in new_items.items() if item_id not in old_items
            ]
            acquired_quest = [
                name for name in state.progress.quest_items
                if name not in old.progress.quest_items
            ]
            acquired_equipment = [
                name for name in state.progress.owned_equipment
                if name not in old.progress.owned_equipment
            ]
            if acquired_items or acquired_quest or acquired_equipment:
                evidence = []
                if acquired_items:
                    evidence.append("inventory=" + ", ".join(acquired_items[:8]))
                if acquired_quest:
                    evidence.append("quest=" + ", ".join(acquired_quest[:8]))
                if acquired_equipment:
                    evidence.append("equipment=" + ", ".join(acquired_equipment[:8]))
                note = "Observed durable progress: " + "; ".join(evidence)
                self._persist(self.store.remember, self.namespace, state.scene, note)
                self.log("durable_progress_observed", {"evidence": evidence})

            if old.player and state.player and old.player.health > 0 and state.player.health == 0:
                self.log("player_died", {"scene": state.scene, "room": state.room})
                self._persist(
                    self.store.remember,
                    self.namespace,
                    state.scene,
                    f"Death observed in {state.scene_name or state.scene}/room {state.room}.",
                )

        for event in state.events:
            key = f"{state.instance_id}:{event.id}"
            if key in self.seen_events:
                continue
            self.seen_events.add(key)
            self.seen_event_order.append(key)
            while len(self.seen_event_order) > 4096:
                self.seen_events.discard(self.seen_event_order.popleft())
            self.log(event.kind, {"detail": event.detail, "scene": state.scene, "room": state.room})
            if event.kind == "game_completed" and self.state == "running":
                asyncio.create_task(self.halt("completed", "game_completed"))

        self.publish()

    async def validate_model(self, config: RunConfig) -> ModelInfo:
        provider = self.providers.get(config.provider)
        if provider is None:
            raise ValueError("Provider not available")
        status = await provider.status()
        if not status.get("connected"):
            raise ValueError(status.get("message", "Provider not connected"))
        catalog = await provider.models()
        info = next((row for row in catalog if row.id == config.model), None)
        if info is None:
            raise ValueError("Model not present in provider catalog")
        if not info.structured_output:
            raise ValueError("Autonomy cognition requires structured output support")
        if config.effort is not None and config.effort not in info.efforts:
            raise ValueError("Configured reasoning effort is not supported by this model")
        return info

    def _budget_reason(self, prompt: str) -> str | None:
        if not self.config or not self.run_id:
            return None
        metrics = self.metrics_cache or self._empty_metrics()
        reserve = None
        if self.config.provider == "openrouter" and self.selected_model is not None:
            reserve = lambda: reserve_cost(
                self.selected_model,
                prompt,
                self.config.max_output_tokens,
                system_prompt=AUTONOMY_SYSTEM_PROMPT,
                output_schema=AgentIntent.model_json_schema(),
            )
        return budget_reason(
            self.config,
            metrics,
            time.monotonic() - self.started,
            reserve=reserve,
        )

    async def _settle_previous_controller(self):
        task = self.settling_controller
        if task is None:
            return
        if not task.done():
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await asyncio.shield(task)
        if self.settling_controller is task:
            self.settling_controller = None

    async def start(self, config: RunConfig):
        await self._settle_previous_controller()
        await self._drain_persistence()
        async with self.lock:
            if self.state in {"running", "starting", "paused"}:
                raise ValueError("Stop the current run before starting another")
            game = self.bridge.state
            if not self.bridge.connected or not game or not game.in_game or not game.player:
                raise ValueError("Connect Ship of Harkinian and load a playable save first")

            self.lifecycle += 1
            self.state = "starting"
            self.reason = ""
            self.config = config
            self.selected_model = None
            self.metrics_cache = self._empty_metrics()
            self.metrics_at = time.monotonic()
            self.provider_usage = {"available": False, "windows": []}
            self.provider_usage_at = 0.0
            self.game_instance = game.instance_id
            self.started = time.monotonic()
            self.cognition_state = "connecting"
            self.cognition_error = ""
            self.thought = "Starting ML actor immediately; cognition is connecting in parallel."
            self.thinking_since = None
            self.last_cognition_at = 0.0
            self.recent.clear()
            self.dialogue_transcript.clear()
            self.seen_events = {f"{game.instance_id}:{event.id}" for event in game.events}
            self.seen_event_order = deque(self.seen_events)
            self.cognition_signature = self._signature(game)
            self.cognition_trigger.clear()

            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "contract": CONTRACT_VERSION,
                        "revision": game.upstream_revision,
                        "instance": game.instance_id,
                        "initial_scene": [game.scene, game.room],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            self.run_id = self.store.new_run(config.model_dump(), game.source, fingerprint)
            self.segment_id = self.store.segment(
                self.run_id, config.model_dump(), self.namespace
            )
            self.controller = ContinuousController(
                self.bridge,
                self.model_dir / "raw-controller-ppo-rnd-v1.pt",
                on_achievement=lambda achievement: self.log(
                    "ml_achievement", achievement
                ),
            )
            self.bridge.enable_control()
            self.state = "running"
            self.log(
                "run_started",
                {
                    "contract": CONTRACT_VERSION,
                    "ml": "ppo+rnd",
                    "control": "raw_stick+raw_buttons",
                },
            )
            self._spawn_tasks()
            self.publish(True)

    def _spawn(self, coroutine):
        task = asyncio.create_task(coroutine)
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)
        return task

    def _spawn_tasks(self):
        generation = self.lifecycle
        assert self.controller is not None
        self.controller_task = self._spawn(
            self.controller.run(
                lambda: generation == self.lifecycle and self.state == "running",
                self.publish,
            )
        )
        self.cognition_task = self._spawn(self._cognition_loop(generation))
        self.watchdog_task = self._spawn(self._runtime_watchdog(generation))
        self.usage_task = self._spawn(self._provider_usage_loop(generation))

    async def _refresh_provider_usage(self):
        if not self.config:
            return
        provider = self.providers.get(self.config.provider)
        if provider is None:
            return
        try:
            quota_reader = getattr(provider, "quota", None)
            if quota_reader is not None:
                usage = await quota_reader()
            elif self.config.provider == "openrouter":
                status = await provider.status()
                usage = {
                    "available": bool(status.get("connected")),
                    "connected": bool(status.get("connected")),
                    "limit": status.get("limit"),
                    "limit_remaining": status.get("limit_remaining"),
                    "windows": [],
                }
            else:
                usage = {"available": False, "windows": []}
            self.provider_usage = usage if isinstance(usage, dict) else {
                "available": False, "windows": []
            }
            self.provider_usage_at = time.time()
        except (ProviderFailure, asyncio.TimeoutError, ValueError) as exc:
            # Keep the last valid quota snapshot; freshness/error is explicit.
            self.provider_usage = {
                **self.provider_usage,
                "error": str(exc)[:180],
            }
            self.provider_usage_at = time.time()
        self.publish(True)

    async def _provider_usage_loop(self, generation: int):
        while generation == self.lifecycle and self.state == "running":
            await self._refresh_provider_usage()
            for _ in range(40):
                if generation != self.lifecycle or self.state != "running":
                    return
                await asyncio.sleep(0.5)

    async def _runtime_watchdog(self, generation: int):
        while generation == self.lifecycle and self.state == "running":
            if self.config and runtime_exhausted(
                self.config, time.monotonic() - self.started
            ):
                await self.halt("paused", "runtime_budget_reached")
                return
            await asyncio.sleep(0.25)

    async def _cognition_loop(self, generation: int):
        while generation == self.lifecycle and self.state == "running":
            assert self.config and self.run_id and self.segment_id and self.controller
            self.apply_switch()

            game = self.bridge.state
            if not self.bridge.connected or game is None or not game.in_game:
                self.cognition_state = "waiting_game"
                self.publish(True)
                await asyncio.sleep(0.5)
                continue
            if game.instance_id != self.game_instance:
                await self.halt("paused", "game_instance_changed")
                return

            if self.selected_model is None:
                self.cognition_state = "connecting"
                self.publish(True)
                try:
                    self.selected_model = await self.validate_model(self.config)
                    self.cognition_error = ""
                except (ValueError, ProviderFailure, asyncio.TimeoutError) as exc:
                    self.cognition_state = "provider_unavailable"
                    self.cognition_error = str(exc)[:240]
                    self.thought = (
                        "ML motor policy is still playing; high-level cognition is unavailable: "
                        + self.cognition_error
                    )
                    self.publish(True)
                    await asyncio.sleep(3.0)
                    continue

            # High-level cognition is event-driven but rate-limited. The ML actor
            # continues controlling Link during this debounce.
            since_last = time.monotonic() - self.last_cognition_at
            if since_last < 3.0:
                await asyncio.sleep(3.0 - since_last)
                if generation != self.lifecycle or self.state != "running":
                    return

            telemetry = self.controller.telemetry()
            learning = telemetry.get("learning", {})
            compact_learning = {
                key: learning.get(key)
                for key in (
                    "updates",
                    "samples_trained",
                    "last_reward",
                    "total_reward",
                    "reward_breakdown",
                    "objective_score",
                    "achievements",
                )
                if key in learning
            }
            recent_for_model = [
                row for row in self.recent
                if row.get("kind") not in {"intent_updated", "provider_rerouted"}
            ][-8:]
            world_edges, memory_rows = await asyncio.gather(
                asyncio.to_thread(
                    self.store.world_neighbors,
                    self.namespace,
                    game.scene,
                    game.room,
                ),
                asyncio.to_thread(self.store.recall, self.namespace, None, 12),
            )
            if generation != self.lifecycle or self.state != "running":
                return
            observation = build_cognition_observation(
                game=game,
                objective=self.config.goal,
                current_intent=self.controller.intent,
                motor={
                    "summary": telemetry.get("motor"),
                    "setpoint": telemetry.get("setpoint"),
                    "actions_sampled": telemetry.get("actions_sampled"),
                },
                ml_learning=compact_learning,
                world_edges=world_edges,
                memory=[row["note"] for row in memory_rows],
                recent_events=recent_for_model,
                dialogue_transcript=list(self.dialogue_transcript),
            )
            prompt = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
            reason = self._budget_reason(prompt)
            if reason:
                await self.halt("paused", reason)
                return

            call_run_id = self.run_id
            call_segment_id = self.segment_id
            begin_task = asyncio.create_task(self._db_write(
                self.store.begin_call,
                call_run_id,
                call_segment_id,
                observation,
            ))
            try:
                call_id = await asyncio.shield(begin_task)
            except asyncio.CancelledError:
                async def cleanup_pending_call():
                    try:
                        pending_id = await begin_task
                        await self._finish_call(
                            call_run_id,
                            pending_id,
                            status="cancelled",
                            error="Run stopped while the call record was being created.",
                        )
                    except Exception:
                        pass
                cleanup = asyncio.create_task(cleanup_pending_call())
                self.retired_tasks.add(cleanup)
                cleanup.add_done_callback(self.retired_tasks.discard)
                raise
            if generation != self.lifecycle or self.state != "running":
                await self._finish_call(
                    call_run_id,
                    call_id,
                    status="cancelled",
                    error="Run changed before provider inference started.",
                )
                return
            self.cognition_state = "thinking"
            self.thinking_since = time.monotonic()
            self.publish(True)
            result = None
            started = time.monotonic()
            try:
                provider = self.providers[self.config.provider]
                think = getattr(provider, "think", None)
                if think is None:
                    raise ProviderFailure(
                        "Provider does not implement Autonomy V3 intent inference"
                    )
                result = await think(self.config, prompt)
                if generation != self.lifecycle or self.state != "running":
                    await self._finish_call(
                        call_run_id,
                        call_id,
                        status="cancelled",
                        usage=result.usage.model_dump(),
                        latency_ms=(time.monotonic() - started) * 1000,
                    )
                    return
                intent = AgentIntent.model_validate_json(result.text)
                await self._finish_call(
                    call_run_id,
                    call_id,
                    status="completed",
                    decision=intent.model_dump(),
                    usage=result.usage.model_dump(),
                    latency_ms=(time.monotonic() - started) * 1000,
                )
            except asyncio.CancelledError:
                if self.run_id:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(self._finish_call(
                            call_run_id,
                            call_id,
                            status="cancelled",
                            error="Cognition cancelled; motor control was independently revoked or continued.",
                            latency_ms=(time.monotonic() - started) * 1000,
                        ))
                raise
            except (ProviderFailure, ValidationError, asyncio.TimeoutError) as exc:
                usage = result.usage.model_dump() if result else getattr(exc, "usage", {})
                if hasattr(usage, "model_dump"):
                    usage = usage.model_dump()
                await self._finish_call(
                    call_run_id,
                    call_id,
                    status="failed",
                    usage=usage or {},
                    error=str(exc)[:300],
                    latency_ms=(time.monotonic() - started) * 1000,
                )
                self.cognition_state = "error"
                self.cognition_error = str(exc)[:240]
                self.thought = (
                    "Cognition failed: "
                    + self.cognition_error
                    + " The ML motor actor keeps playing with the last intent; "
                    "stop and start the run to explicitly retry the provider."
                )
                self.publish(True)
                return

            if result.usage.actual_model and result.usage.actual_model != self.config.model:
                await self._db_write(self.store.update_run, self.run_id, mixed=True)
                self.log(
                    "provider_rerouted",
                    {
                        "requested": self.config.model,
                        "actual": result.usage.actual_model,
                    },
                )

            self.last_cognition_at = time.monotonic()
            self.controller.set_intent(intent)
            self.thought = intent.summary
            self.cognition_state = "acting"
            self.cognition_error = ""
            self.thinking_since = None
            self.log(
                "intent_updated",
                {
                    "objective": intent.objective,
                    "mode": intent.mode,
                    "summary": intent.summary,
                },
            )

            timeout = max(5.0, min(30.0, intent.horizon_ms / 1000.0))
            self.cognition_trigger.clear()
            try:
                await asyncio.wait_for(self.cognition_trigger.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass

    async def halt(self, state: str, reason: str):
        async with self.lock:
            if self.state not in {"running", "starting", "paused"} and state != "completed":
                return
            self.lifecycle += 1
            self.bridge.revoke()
            if self.controller:
                self.controller.neutralize(reason)
            self.state = state
            self.reason = reason
            self.cognition_state = "idle" if state in {"stopped", "completed"} else "paused"
            self.pending_switch = None
            await self._drain_persistence()
            if self.run_id:
                await self._db_write(self.store.update_run, self.run_id, status=state, reason=reason)
                if state in {"stopped", "completed"}:
                    await self._db_write(self.store.end_segments, self.run_id)
            current = asyncio.current_task()

            # Controller exits from its active() predicate and is intentionally
            # not cancelled: an in-flight PPO worker must finish its atomic
            # checkpoint after controller input has already been revoked.
            controller_task = self.controller_task
            if controller_task and controller_task is not current and not controller_task.done():
                self.settling_controller = controller_task
                self.retired_tasks.add(controller_task)
                controller_task.add_done_callback(self.retired_tasks.discard)

            for task in (self.cognition_task, self.watchdog_task, self.usage_task):
                if task and task is not current and not task.done():
                    task.cancel()
                    self.retired_tasks.add(task)
                    task.add_done_callback(self.retired_tasks.discard)

            self.controller_task = None
            self.cognition_task = None
            self.watchdog_task = None
            self.usage_task = None
            self.tasks.clear()
            self.log("run_" + state, {"reason": reason})
            self.publish(True)

    async def control(self, action: str):
        if action not in {"stop", "pause", "resume", "take_control"}:
            raise ValueError("Unknown control action")
        if action == "stop":
            await self.halt("stopped", "stop")
            return
        if action in {"pause", "take_control"}:
            if action == "take_control" and self.run_id:
                await self._db_write(self.store.update_run, self.run_id, assisted=True)
            await self.halt("paused", action)
            return

        await self._settle_previous_controller()
        async with self.lock:
            if self.state != "paused" or not self.run_id:
                raise ValueError("No paused run to resume")
            if not self.bridge.connected or not self.bridge.state or not self.bridge.state.in_game:
                raise ValueError("Reconnect the game before resuming")
            self.lifecycle += 1
            self.state = "running"
            self.reason = ""
            self.bridge.enable_control()
            if self.controller is None:
                self.controller = ContinuousController(
                    self.bridge,
                    self.model_dir / "raw-controller-ppo-rnd-v1.pt",
                )
            self.controller.reset_episode_state()
            await self._db_write(self.store.update_run, self.run_id, status="running", reason="")
            self._spawn_tasks()
            self.publish(True)

    async def switch(self, change: SwitchConfig):
        async with self.lock:
            if self.state not in {"running", "paused"} or not self.config or not self.run_id:
                raise ValueError("No active run")
            self.switch_request += 1
            request = self.switch_request
            config = RunConfig.model_validate(
                {**self.config.model_dump(), **change.model_dump()}
            )
        info = await self.validate_model(config)
        async with self.lock:
            if request != self.switch_request or self.state not in {"running", "paused"}:
                raise ValueError("Run changed while validating the model")
            self.pending_switch = (config, info)
            if self.state == "paused":
                self.apply_switch()
            self.publish(True)

    def apply_switch(self):
        if self.pending_switch is None or not self.run_id:
            return
        self.config, self.selected_model = self.pending_switch
        self.pending_switch = None
        self.provider_usage = {"available": False, "windows": []}
        self.provider_usage_at = 0.0
        self.segment_id = self.store.segment(
            self.run_id, self.config.model_dump(), self.namespace
        )
        self.store.update_run(self.run_id, mixed=True)
        self.log(
            "model_changed",
            {
                "provider": self.config.provider,
                "model": self.config.model,
                "effort": self.config.effort,
            },
        )

    def snapshot(self) -> dict:
        bridge_status = self.bridge.telemetry()
        controller = self.controller.telemetry() if self.controller else {
            "intent": AgentIntent.bootstrap().model_dump(),
            "motor": "Controller not started.",
            "setpoint": {
                "buttons": 0,
                "button_names": [],
                "stick_x": 0,
                "stick_y": 0,
                "reason": "idle",
            },
            "learning": {},
        }
        realtime = bridge_status.get("realtime") or {}
        return {
            "status": self.state,
            "reason": self.reason,
            "run_id": self.run_id,
            "elapsed_s": round(time.monotonic() - self.started, 1) if self.run_id else 0.0,
            "connection": {
                "game": bool(bridge_status.get("connected")),
                "source": bridge_status.get("source"),
                "realtime": bool(realtime.get("enabled")),
                "state_hz": realtime.get("state_hz"),
                "ai": bool(
                    self.state == "running"
                    and self.selected_model is not None
                    and self.cognition_state not in {"provider_unavailable", "error"}
                ),
                "ai_state": self.cognition_state,
                "ai_error": self.cognition_error,
            },
            "thought": {
                "summary": self.thought,
                "state": self.cognition_state,
                "thinking_ms": round((time.monotonic() - self.thinking_since) * 1000)
                    if self.thinking_since else 0,
                "intent": controller.get("intent"),
                "motor": controller.get("motor"),
            },
            "input": controller.get("setpoint"),
            "learning": controller.get("learning"),
            "metrics": self.metrics_cache if self.run_id else None,
            "usage": {
                "provider": self.config.provider if self.config else None,
                "model": self.config.model if self.config else None,
                "input_tokens": (self.metrics_cache or {}).get("input_tokens", 0),
                "output_tokens": (self.metrics_cache or {}).get("output_tokens", 0),
                "cached_input_tokens": (self.metrics_cache or {}).get("cached_input_tokens", 0),
                "reasoning_output_tokens": (self.metrics_cache or {}).get("reasoning_output_tokens", 0),
                "total_tokens": (self.metrics_cache or {}).get("total_tokens", 0),
                "cost_usd": (
                    None
                    if self.config and self.config.provider == "codex"
                    else (self.metrics_cache or {}).get("cost_usd")
                ),
                "known_cost_usd": (self.metrics_cache or {}).get("known_cost_usd", 0),
                "quota": self.provider_usage,
                "quota_updated_at": self.provider_usage_at or None,
            },
            # Compact compatibility block; the realtime web panel intentionally
            # does not stream the full GameState at 20 Hz.
            "bridge": {
                "connected": bool(bridge_status.get("connected")),
                "last_seen_age_s": bridge_status.get("last_seen_age_s"),
                "realtime": realtime,
            },
        }
