from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

from pydantic import ValidationError

from ..budgets import budget_reason, runtime_exhausted
from ..models import ModelInfo, RunConfig, SwitchConfig
from ..providers.base import ProviderFailure
from ..providers.openrouter import reserve_cost
from .champions import ChampionStore
from .controller import ContinuousController
from .features import target_point
from .models import AgentIntent
from .prompt import AUTONOMY_SYSTEM_PROMPT, build_cognition_observation


CONTRACT_VERSION = "autonomy-v3/goal-conditioned-controller-v6/residual-stick-v1/route-memory-v1/ppo-rnd-v3/reward-v7"
COGNITION_EVENT_DEBOUNCE_S = 1.5
COGNITION_MIN_INTERVAL_S = 8.0
COGNITION_STUCK_AFTER_S = 90.0
COGNITION_STUCK_COOLDOWN_S = 180.0
COGNITION_BLOCKED_AFTER_S = 20.0
COGNITION_BLOCKED_COOLDOWN_S = 60.0
COGNITION_IDLE_POLL_S = 5.0


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
        # V3 changes the policy action semantics from absolute sampled stick to
        # residual stick mixed with structured guidance. Keep the old V2 file
        # untouched for inspection; incompatible weights must not be reinterpreted.
        self.training_checkpoint = self.model_dir / "raw-controller-ppo-rnd-v3.pt"
        self.champions = ChampionStore(self.model_dir / "champions")
        self.champion_catalog = self.champions.catalog()
        self.active_champion: dict | None = None
        self.last_champion_error = ""
        self.completion_champion_saved_run_id: str | None = None
        self.completion_capture_ready = asyncio.Event()
        self.completion_capture_ready.set()

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
        self.last_cognition_reasons: list[str] = []
        self.pending_switch: tuple[RunConfig, ModelInfo] | None = None
        self.switch_request = 0

        self.recent = deque(maxlen=24)
        self.dialogue_transcript = deque(maxlen=12)
        self.seen_events: set[str] = set()
        self.seen_event_order: deque[str] = deque()
        self.cognition_trigger = asyncio.Event()
        self.cognition_reasons: set[str] = set()
        self.cognition_seen_dialogue_triggers: set[tuple] = set()
        self.cognition_seen_target_reached: set[tuple] = set()
        self.last_stuck_replan_at = 0.0
        self.guidance_blocked_since: float | None = None
        self.last_blocked_replan_at = 0.0
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
            "elapsed_s": 0.0,
            "usage_by_model": [],
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
    def _durable_progress_gained(old, new) -> bool:
        old_items = {row.item_id for row in old.inventory_named}
        new_items = {row.item_id for row in new.inventory_named}
        if new_items - old_items:
            return True
        if set(new.progress.quest_items) - set(old.progress.quest_items):
            return True
        if set(new.progress.owned_equipment) - set(old.progress.owned_equipment):
            return True
        if set(new.progress.dungeon_items) - set(old.progress.dungeon_items):
            return True
        if new.progress.heart_pieces > old.progress.heart_pieces:
            return True
        if new.progress.skull_tokens > old.progress.skull_tokens:
            return True
        if new.progress.small_keys > old.progress.small_keys:
            return True
        if new.progress.magic_acquired and not old.progress.magic_acquired:
            return True
        if new.progress.double_magic and not old.progress.double_magic:
            return True
        if new.progress.double_defense and not old.progress.double_defense:
            return True
        for name, level in new.progress.upgrade_levels.items():
            if level > old.progress.upgrade_levels.get(name, 0):
                return True
        for name, enabled in new.progress.story_flags.items():
            if enabled and not old.progress.story_flags.get(name, False):
                return True
        return False

    def _request_dialogue_cognition(self, reason: str, key: tuple):
        if key in self.cognition_seen_dialogue_triggers:
            return
        self.cognition_seen_dialogue_triggers.add(key)
        self._request_cognition(reason)

    def _check_intent_target_reached(self, state):
        if not self.controller or not state.player:
            return
        intent = self.controller.intent
        if intent.mode not in {"navigate", "explore", "observe"}:
            return
        point = target_point(state, intent)
        if point is None:
            return
        distance = math.dist(state.player.position, point)
        threshold = 95.0 if (
            intent.target_actor_uid is not None
            or intent.target_actor_id is not None
        ) else 80.0
        if distance > threshold:
            return
        key = (
            state.scene,
            state.room,
            intent.mode,
            intent.target_actor_uid,
            intent.target_actor_id,
            intent.target_actor_params,
            tuple(intent.target_position) if intent.target_position is not None else None,
            intent.target_item_id,
        )
        if key in self.cognition_seen_target_reached:
            return
        self.cognition_seen_target_reached.add(key)
        self.log(
            "intent_target_reached",
            {
                "distance": round(distance, 2),
                "target": list(point),
                "mode": intent.mode,
            },
        )
        self._request_cognition("intent_target_reached")

    def _request_cognition(self, reason: str):
        if self.state != "running":
            return
        self.cognition_reasons.add(reason)
        self.cognition_trigger.set()

    def _take_cognition_reasons(self) -> list[str]:
        reasons = sorted(self.cognition_reasons)
        self.cognition_reasons.clear()
        self.cognition_trigger.clear()
        return reasons

    def on_state(self, state, old):
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

        self._check_intent_target_reached(state)

        if old and old.instance_id == state.instance_id and (
            old.scene,
            old.room,
        ) != (
            state.scene,
            state.room,
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
            self._request_cognition("world_transition")

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
                if state.dialogue.choice_count > 0:
                    self._request_dialogue_cognition(
                        "dialogue_choice",
                        (
                            state.scene,
                            state.room,
                            state.dialogue.text_id,
                            tuple(state.dialogue.choices),
                            "choice",
                        ),
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
                if state.dialogue.choice_count > 0:
                    self._request_dialogue_cognition(
                        "dialogue_choice",
                        (
                            state.scene,
                            state.room,
                            state.dialogue.text_id,
                            tuple(state.dialogue.choices),
                            "choice",
                        ),
                    )

            if old.dialogue.active and not state.dialogue.active:
                # Linear text/signposts normally do not need LLM calls. If the
                # current intent explicitly waited on that text, one resolution
                # trigger prevents the planner from remaining stuck on "observe".
                waiting_on_dialogue = bool(
                    self.controller
                    and self.controller.intent.mode in {"dialogue", "observe"}
                )
                if (
                    old.dialogue.speaker is not None
                    or old.dialogue.choice_count > 0
                    or waiting_on_dialogue
                ):
                    self._request_dialogue_cognition(
                        "dialogue_resolved",
                        (
                            state.scene,
                            state.room,
                            old.dialogue.text_id,
                            "resolved",
                        ),
                    )

            if self._durable_progress_gained(old, state):
                self._request_cognition("durable_progress")

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
                self._request_cognition("player_died")
                self._persist(
                    self.store.remember,
                    self.namespace,
                    state.scene,
                    f"Death observed in {state.scene_name or state.scene}/room {state.room}.",
                )

            if old.game_over_state == 0 and state.game_over_state != 0:
                self._request_cognition("game_over")

        for event in state.events:
            key = f"{state.instance_id}:{event.id}"
            if key in self.seen_events:
                continue
            self.seen_events.add(key)
            self.seen_event_order.append(key)
            while len(self.seen_event_order) > 4096:
                self.seen_events.discard(self.seen_event_order.popleft())
            self.log(event.kind, {"detail": event.detail, "scene": state.scene, "room": state.room})
            if event.kind == "boss_defeated":
                self._request_cognition("boss_defeated")
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

    def _make_controller(self, config: RunConfig) -> ContinuousController:
        checkpoint = self.training_checkpoint
        route_graph_path = self.model_dir / "route-graph-v1.json"
        training_enabled = True
        self.active_champion = None
        if config.run_mode == "evaluation":
            champion, checkpoint = self.champions.resolve(config.champion_id)
            if champion.get("contract") != CONTRACT_VERSION:
                raise ValueError(
                    "Champion contract is incompatible with the current motor architecture"
                )
            frozen_routes = self.champions.resolve_route_graph(champion)
            if frozen_routes is None:
                raise ValueError("Champion route graph metadata is missing")
            route_graph_path = frozen_routes
            self.active_champion = champion
            training_enabled = False

        try:
            controller = ContinuousController(
                self.bridge,
                checkpoint,
                on_achievement=lambda achievement: self.log(
                    "ml_achievement", achievement
                ),
                training_enabled=training_enabled,
                route_graph_path=route_graph_path,
            )
            if config.run_mode == "evaluation" and controller.route_graph.load_error:
                raise ValueError(
                    "Champion route graph is invalid: "
                    + controller.route_graph.load_error
                )
            return controller
        except (RuntimeError, ValueError) as exc:
            if config.run_mode == "evaluation":
                raise ValueError(str(exc)) from None
            raise

    def refresh_champions(self) -> dict:
        self.champion_catalog = self.champions.catalog()
        return self.champion_catalog

    def champion_status(self) -> dict:
        return {
            **self.champion_catalog,
            "capture_pending": not self.completion_capture_ready.is_set(),
            "error": self.last_champion_error or None,
        }

    async def _capture_completion_champion(
        self,
        controller: ContinuousController,
        run_id: str,
        elapsed_s: float,
        config: RunConfig,
    ):
        if self.completion_champion_saved_run_id == run_id:
            return
        try:
            await asyncio.to_thread(controller.policy.save)
            await asyncio.to_thread(controller.route_graph.save, force=True)
            telemetry = controller.telemetry()
            learning = telemetry.get("learning") or {}
            summary = await asyncio.to_thread(self.store.benchmark, run_id)
            metrics = (summary or {}).get("metrics")
            if not isinstance(metrics, dict):
                metrics = await asyncio.to_thread(self.store.metrics, run_id)
            persisted_elapsed_s = (
                (summary or {}).get("elapsed_s")
                if isinstance(summary, dict)
                else metrics.get("elapsed_s")
            )
            if not isinstance(persisted_elapsed_s, (int, float)):
                persisted_elapsed_s = elapsed_s
            metadata = {
                "run_id": run_id,
                "elapsed_s": round(float(persisted_elapsed_s), 3),
                "contract": CONTRACT_VERSION,
                "updates": int(learning.get("updates") or 0),
                "samples_trained": int(learning.get("samples_trained") or 0),
                "run_updates": int(learning.get("run_updates") or 0),
                "run_samples_trained": int(
                    learning.get("run_samples_trained") or 0
                ),
                "objective_score": int(learning.get("objective_score") or 0),
                "total_reward": float(learning.get("total_reward") or 0.0),
                "provider": config.provider,
                "model": config.model,
                "effort": config.effort,
                "calls": int(metrics.get("calls") or 0),
                "input_tokens": int(metrics.get("input_tokens") or 0),
                "output_tokens": int(metrics.get("output_tokens") or 0),
                "cached_input_tokens": int(
                    metrics.get("cached_input_tokens") or 0
                ),
                "reasoning_output_tokens": int(
                    metrics.get("reasoning_output_tokens") or 0
                ),
                "total_tokens": int(metrics.get("total_tokens") or 0),
                "cost_usd": metrics.get("cost_usd"),
                "known_cost_usd": float(metrics.get("known_cost_usd") or 0.0),
                "unknown_usage_calls": int(
                    metrics.get("unknown_usage_calls") or 0
                ),
                "unknown_cost_calls": int(
                    metrics.get("unknown_cost_calls") or 0
                ),
                "usage_by_model": metrics.get("usage_by_model") or [],
                "route_memory": learning.get("route_memory") or {},
            }
            champion = await asyncio.to_thread(
                self.champions.capture,
                self.training_checkpoint,
                metadata,
                controller.route_graph.path,
            )
            self.completion_champion_saved_run_id = run_id
            self.last_champion_error = ""
            self.refresh_champions()
            self.log(
                "champion_saved",
                {
                    "champion_id": champion["id"],
                    "elapsed_s": champion["elapsed_s"],
                    "sha256": champion["sha256"],
                    "route_graph_sha256": champion.get("route_graph_sha256"),
                },
            )
        except Exception as exc:
            self.last_champion_error = f"{type(exc).__name__}: {str(exc)[:180]}"
            self.log(
                "champion_save_failed",
                {"error": self.last_champion_error},
            )
        self.publish(True)

    async def start(self, config: RunConfig):
        # A completed training run is not startable until its immutable champion
        # snapshot has either been captured or explicitly failed. This prevents
        # a new run from racing the final checkpoint save/copy.
        await self.completion_capture_ready.wait()
        await self._settle_previous_controller()
        await self._drain_persistence()
        async with self.lock:
            if self.state in {"running", "starting", "paused"}:
                raise ValueError("Stop the current run before starting another")
            game = self.bridge.state
            if not self.bridge.connected or not game or not game.in_game or not game.player:
                raise ValueError("Connect Ship of Harkinian and load a playable save first")

            if config.run_mode == "evaluation":
                champion, _ = self.champions.resolve(config.champion_id)
                config = config.model_copy(
                    update={"champion_id": champion["id"]}
                )
                self.refresh_champions()

            # Load/validate the policy before opening a new run record. A corrupt
            # immutable champion therefore fails cleanly while the runtime is idle.
            controller = self._make_controller(config)

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
            self.last_cognition_reasons = []
            self.last_stuck_replan_at = 0.0
            self.guidance_blocked_since = None
            self.last_blocked_replan_at = 0.0
            self.recent.clear()
            self.dialogue_transcript.clear()
            self.seen_events = {f"{game.instance_id}:{event.id}" for event in game.events}
            self.seen_event_order = deque(self.seen_events)
            self.cognition_reasons = {"run_started"}
            self.cognition_seen_dialogue_triggers.clear()
            self.cognition_seen_target_reached.clear()
            self.cognition_trigger.set()

            fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "contract": CONTRACT_VERSION,
                        "revision": game.upstream_revision,
                        "instance": game.instance_id,
                        "initial_scene": [game.scene, game.room],
                        "run_mode": config.run_mode,
                        "champion_id": config.champion_id,
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            self.run_id = self.store.new_run(config.model_dump(), game.source, fingerprint)
            self.segment_id = self.store.segment(
                self.run_id, config.model_dump(), self.namespace
            )
            self.controller = controller
            self.bridge.enable_control()
            self.state = "running"
            self.log(
                "run_started",
                {
                    "contract": CONTRACT_VERSION,
                    "ml": "ppo+rnd",
                    "control": "raw_stick+raw_buttons",
                    "run_mode": config.run_mode,
                    "champion_id": config.champion_id,
                    "training_enabled": config.run_mode == "train",
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

    @staticmethod
    def _stuck_signal(learning: dict) -> tuple[float, str]:
        stalled_s = float(
            learning.get("seconds_since_useful_progress") or 0.0
        )
        exploration = learning.get("exploration") or {}
        local_dwell_s = float(
            exploration.get("local_dwell_seconds") or 0.0
        )
        if local_dwell_s >= stalled_s:
            return local_dwell_s, "local_area_stuck"
        return stalled_s, "motor_stuck"

    async def _wait_for_cognition_need(self, generation: int) -> bool:
        """Wait for a strategic event or sustained motor stagnation.

        There is deliberately no periodic LLM refresh. A valid high-level intent
        remains active until game evidence says it should change.
        """
        while generation == self.lifecycle and self.state == "running":
            if self.cognition_trigger.is_set():
                since_last = time.monotonic() - self.last_cognition_at
                delay = max(
                    COGNITION_EVENT_DEBOUNCE_S,
                    COGNITION_MIN_INTERVAL_S - since_last,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                return generation == self.lifecycle and self.state == "running"

            try:
                await asyncio.wait_for(
                    self.cognition_trigger.wait(),
                    timeout=COGNITION_IDLE_POLL_S,
                )
                continue
            except asyncio.TimeoutError:
                pass

            if not self.controller:
                continue
            telemetry = self.controller.telemetry()
            learning = telemetry.get("learning", {})
            guidance = telemetry.get("guidance") or {}
            stuck_age_s, stuck_reason = self._stuck_signal(learning)
            now = time.monotonic()

            hard_blocked = bool(
                guidance.get("blocked")
                and not guidance.get("detour")
                and guidance.get("active")
            )
            if hard_blocked:
                if self.guidance_blocked_since is None:
                    self.guidance_blocked_since = now
                blocked_age = now - self.guidance_blocked_since
                if (
                    blocked_age >= COGNITION_BLOCKED_AFTER_S
                    and now - self.last_blocked_replan_at >= COGNITION_BLOCKED_COOLDOWN_S
                    and now - self.last_cognition_at >= COGNITION_BLOCKED_AFTER_S
                ):
                    self.last_blocked_replan_at = now
                    self._request_cognition("guidance_blocked")
                    continue
            else:
                self.guidance_blocked_since = None

            if (
                stuck_age_s >= COGNITION_STUCK_AFTER_S
                and now - self.last_stuck_replan_at >= COGNITION_STUCK_COOLDOWN_S
                and now - self.last_cognition_at >= COGNITION_STUCK_AFTER_S
            ):
                self.last_stuck_replan_at = now
                self._request_cognition(stuck_reason)
        return False

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

            if self.last_cognition_at > 0:
                if not await self._wait_for_cognition_need(generation):
                    return

            trigger_reasons = self._take_cognition_reasons()
            if not trigger_reasons:
                trigger_reasons = ["run_started"] if self.last_cognition_at == 0 else ["strategic_event"]
            self.last_cognition_reasons = trigger_reasons

            telemetry = self.controller.telemetry()
            learning = telemetry.get("learning", {})
            exploration = learning.get("exploration") or {}
            compact_learning = {
                "run_updates": learning.get("run_updates", 0),
                "run_samples_trained": learning.get("run_samples_trained", 0),
                "objective_score": learning.get("objective_score", 0),
                "useful_progress_rate": learning.get("useful_progress_rate", 0.0),
                "seconds_since_useful_progress": learning.get(
                    "seconds_since_useful_progress", 0.0
                ),
                "local_dwell_seconds": exploration.get(
                    "local_dwell_seconds", 0.0
                ),
                "unique_macro_regions": exploration.get(
                    "unique_macro_regions", 0
                ),
                "local_frontier_radius": exploration.get(
                    "local_frontier_radius", 0.0
                ),
                "reward_breakdown": learning.get("reward_breakdown", {}),
                "route_memory": learning.get("route_memory", {}),
                "recent_achievements": (learning.get("achievements") or [])[-4:],
            }
            recent_for_model = [
                row for row in self.recent
                if row.get("kind") not in {"intent_updated", "provider_rerouted"}
            ][-5:]
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
                    "seconds_since_useful_progress": learning.get(
                        "seconds_since_useful_progress", 0.0
                    ),
                    "guidance": telemetry.get("guidance"),
                },
                ml_learning=compact_learning,
                world_edges=world_edges,
                memory=[row["note"] for row in memory_rows],
                recent_events=recent_for_model,
                dialogue_transcript=list(self.dialogue_transcript),
                trigger_reasons=trigger_reasons,
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

            # No horizon-based refresh here. The next iteration blocks until a
            # strategic trigger or sustained stuck condition requests replanning.

    async def halt(self, state: str, reason: str):
        completion_controller: ContinuousController | None = None
        completion_run_id: str | None = None
        completion_elapsed_s = 0.0
        completion_config: RunConfig | None = None
        cancelled_tasks: list[asyncio.Task] = []
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
                await self._refresh_metrics(self.run_id)
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
                    cancelled_tasks.append(task)

            if (
                state == "completed"
                and self.config is not None
                and self.config.run_mode == "train"
                and self.controller is not None
                and self.run_id is not None
                and self.completion_champion_saved_run_id != self.run_id
                and self.completion_capture_ready.is_set()
            ):
                self.completion_capture_ready.clear()
                completion_controller = self.controller
                completion_run_id = self.run_id
                completion_elapsed_s = max(
                    0.0, time.monotonic() - self.started
                )
                completion_config = self.config.model_copy(deep=True)

            self.controller_task = None
            self.cognition_task = None
            self.watchdog_task = None
            self.usage_task = None
            self.tasks.clear()
            self.log("run_" + state, {"reason": reason})
            self.publish(True)

        if state in {"stopped", "completed"}:
            if cancelled_tasks:
                await asyncio.gather(*cancelled_tasks, return_exceptions=True)
            await self._drain_persistence()
            if self.run_id:
                summary = await self._db_write(self.store.finalize_run, self.run_id)
                if summary and isinstance(summary.get("metrics"), dict):
                    self.metrics_cache = summary["metrics"]
                    self.metrics_at = time.monotonic()
                else:
                    await self._refresh_metrics(self.run_id)

        if (
            completion_controller is not None
            and completion_run_id is not None
            and completion_config is not None
        ):
            try:
                # Wait until queued/in-flight PPO work and the partial final rollout
                # finish. The champion therefore represents the policy that actually
                # completed the run, including its final training update.
                await self._settle_previous_controller()
                await self._capture_completion_champion(
                    completion_controller,
                    completion_run_id,
                    completion_elapsed_s,
                    completion_config,
                )
            finally:
                # Capture failure is visible in champion metadata/status, but must
                # never leave the runtime permanently unable to start another run.
                self.completion_capture_ready.set()
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
                self.controller = self._make_controller(self.config)
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
            "run_mode": self.config.run_mode if self.config else "train",
            "active_champion": self.active_champion,
            "champions": self.champion_status(),
            "elapsed_s": (
                round(float((self.metrics_cache or {}).get("elapsed_s") or 0.0), 1)
                if self.run_id and self.state in {"stopped", "completed"}
                else round(time.monotonic() - self.started, 1) if self.run_id else 0.0
            ),
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
                "guidance": controller.get("guidance"),
                "trigger": ", ".join(self.last_cognition_reasons) if self.last_cognition_reasons else None,
            },
            "input": controller.get("setpoint"),
            "learning": controller.get("learning"),
            "metrics": self.metrics_cache if self.run_id else None,
            "usage": {
                "provider": self.config.provider if self.config else None,
                "model": self.config.model if self.config else None,
                "calls": (self.metrics_cache or {}).get("calls", 0),
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
