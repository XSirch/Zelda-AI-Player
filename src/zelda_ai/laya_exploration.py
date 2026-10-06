"""Continuous frozen goal pursuit; no prescribed Zelda route or provider call."""
from __future__ import annotations

import asyncio
import json
import math
import time

from .autonomy.camera_task import ObservedCameraReturnTask
from .autonomy.context_tasks import LinearDialogueTask, NativeModalWaitTask, ObservedContextTask
from .autonomy.controller import ContinuousController
from .autonomy.exploration_plan import CurrentTasksCooling, ObservedExplorationPlan
from .autonomy.ground_descent_task import GroundDescentApproachTask
from .autonomy.ladder_task import LadderDescentTask
from .autonomy.laya_ladder_policy import encode_ladder
from .autonomy.locomotion import camera_modal_active
from .autonomy.models import AgentIntent, ObjectiveCompletion
from .autonomy.objectives import ObjectiveTracker
from .g1 import ARTIFACTS, context, write_json
from .laya_context import execute_context_task, execute_modal_wait
from .laya_motor import execute_frozen_task
from .laya_planning_wait import wait_for_replan
from .laya_traversal import fresh_collision_snapshot


async def exploration_episode(bridge, frozen, walking, ladder, directory, *, seconds, on_started=None,
                              contextual_interactions=False):
    if not 10 <= seconds <= 300 or ladder is None:
        raise ValueError("Continuous exploration needs 10..300 seconds and both explicit frozen candidates")
    directory.mkdir(parents=True)
    controller = await asyncio.to_thread(ContinuousController, bridge, frozen / ARTIFACTS[0],
        training_enabled=False, route_graph_path=frozen / ARTIFACTS[1], room_map_path=frozen / ARTIFACTS[2])
    objective = AgentIntent(objective="Obtain the Kokiri Sword", summary="Obtain the Kokiri Sword",
        mode="explore", completion=ObjectiveCompletion(kind="equipment", name="Kokiri Sword"))
    locked_objective = objective.model_copy(deep=True)
    controller.set_intent(objective)
    initial = bridge.state.model_copy(deep=True)
    tracker, planner = ObjectiveTracker(), ObservedExplorationPlan(contextual_interactions=contextual_interactions)
    tracker.assign(locked_objective, initial)
    baseline_completed = tracker.evaluate(initial).completed
    controller.executor.room_budget_s = 120  # Macro clock is never restarted by a new local task.
    started, deadline = time.monotonic(), time.monotonic() + seconds
    stages, reason, cancelled, first_task = [], None, False, None
    planning_waits = []
    selection_state = initial
    landings = 0
    episode_valid = True
    initial_events = {event.id for event in initial.events}

    def context_guard_active(game):
        guard = getattr(controller, "dialogue_reentry_guard", None)
        return bool(guard and time.monotonic() < guard["until"]
            and (game.scene, game.room) == (guard["scene"], guard["room"])
            and game.player and guard.get("anchor")
            and math.dist(game.player.position, guard["anchor"]) < 120)

    def observe(game):
        nonlocal episode_valid
        # This flag is sticky: a later truncated event list cannot make a new
        # save/native lifetime count as completion of the original attempt.
        episode_valid = episode_valid and game.instance_id == initial.instance_id and not any(
            event.kind in {"save_loaded", "player_died", "game_over"} and event.id not in initial_events
            for event in game.events)
        if episode_valid:
            planner.observe(game)
            task = getattr(controller, "local_task", None)
            if (contextual_interactions and task and task.kind == "observed_cell" and not task.terminal
                    and not context_guard_active(game) and planner.context_available(game)):
                task.interrupt("observed_context_interaction")
                bridge.release()

    previous_observer = getattr(bridge, "on_state", None)

    def on_state(game, old):
        try:
            if previous_observer:
                previous_observer(game, old)
        finally:
            observe(game)
            if old and camera_modal_active(old.player) and not camera_modal_active(game.player):
                controller.note_camera_return_closed(old, game)
            if old and old.dialogue.active and not game.dialogue.active:
                controller.note_dialogue_closed(old, game)
                bridge.release()  # Closing input is revoked before a repeated motor tick.

    if contextual_interactions:
        bridge.on_state = on_state

    async def execute(task, policy, *, encoder=None):
        path = directory / f"stage-{len(stages)+1:03d}"
        options = {"encoder": encoder} if encoder else {}
        was_cancelled = False
        try:
            if isinstance(task, NativeModalWaitTask):
                walking.invalidate()
                ladder.invalidate()
                result = await execute_modal_wait(controller, bridge, task, path,
                    on_started=on_started, observe=observe)
            elif isinstance(task, ObservedContextTask):
                walking.invalidate()
                ladder.invalidate()
                result = await execute_context_task(controller, bridge, task, path,
                    on_started=on_started, observe=observe)
            else:
                result = await execute_frozen_task(controller, bridge, task, policy, path,
                    on_started=on_started, observe=observe, **options)
        except asyncio.CancelledError:
            result = json.loads(await asyncio.to_thread((path / "motor.json").read_text, encoding="utf-8"))
            was_cancelled = True
        stages.append({"name": path.name, "after_verified_landing": landings > 0,
            **{key: result[key] for key in (
            "success", "task", "consumed_actions", "raw_button_actions", "objective_unchanged",
            "run_updates", "motor_failure")},
            "causal_button_actions": result.get("causal_button_actions", 0),
            "unowned_button_actions": result.get("unowned_button_actions", result["raw_button_actions"])})
        if was_cancelled:
            raise asyncio.CancelledError
        return result

    try:
        if baseline_completed:
            reason = "objective_already_satisfied_at_start"
        else:
            while time.monotonic() < deadline - 1 and len(stages) < 64:
                if controller.executor.state == "blocked":
                    reason = f"supervisor:{controller.executor.reason}"
                    break
                await fresh_collision_snapshot(bridge)
                observe(bridge.state)
                if not episode_valid:
                    reason = "episode_identity_or_load_changed"
                    break
                if tracker.evaluate(bridge.state).completed:
                    reason = "objective_telemetry_completed"
                    break
                selection_state = bridge.state.model_copy(deep=True)
                try:
                    task = planner.choose(selection_state, budget_s=min(300, deadline-time.monotonic()-.5),
                        suppress_context=context_guard_active(selection_state))
                except CurrentTasksCooling as cooling:
                    if len(planning_waits) >= 64:
                        reason = "planning_wait_count_budget"
                        break
                    walking.invalidate()
                    ladder.invalidate()
                    result = await wait_for_replan(controller,bridge,
                        directory/f"wait-{len(planning_waits)+1:03d}",retry_at=cooling.retry_at,
                        deadline=deadline,observe=observe)
                    planning_waits.append({k:result[k] for k in ("reason","game_progress",
                        "elapsed_seconds","consumed_actions","raw_button_actions")})
                    if result["reason"] == "bridge_unavailable" or result["reason"].startswith("supervisor:"):
                        reason = result["reason"]
                        break
                    continue
                first_task = first_task or task
                result = await execute(task, walking)
                observe(bridge.state)
                if not episode_valid:
                    reason = "episode_identity_or_load_changed"
                    break
                if (isinstance(task, GroundDescentApproachTask)
                        and task.failure == "unsupported_locomotor_mode" and bridge.state.player
                        and bridge.state.player.climbing_ladder and not result["motor_failure"]):
                    await bridge.next_state(bridge.state.seq, timeout=1.5)
                    attached = LadderDescentTask.continue_from(bridge.state, task)
                    result = await execute(attached, ladder, encoder=encode_ladder)
                    landing = bool(result["success"] and attached.landing_verified(bridge.state))
                    landings += int(landing)
                    planner.outcome(task, success=landing)
                else:
                    planner.outcome(task, success=result["success"])
                if controller.intent != locked_objective:
                    reason = "objective_lock_changed"
                    break
                if result["motor_failure"]:
                    reason = result["motor_failure"]
                    break
                if isinstance(task, LinearDialogueTask) and not result["success"]:
                    reason = task.failure or "linear_dialogue_incomplete"
                    break
                if isinstance(task, NativeModalWaitTask) and not result["success"]:
                    reason = task.failure or "native_modal_handoff_incomplete"
                    break
                if isinstance(task, ObservedCameraReturnTask) and not result["success"]:
                    reason = task.failure or "camera_return_incomplete"
                    break
            reason = reason or ("task_count_budget" if len(stages) >= 64 else "session_budget_exhausted")
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        reason = f"{type(exc).__name__}:{str(exc)[:180]}"
    except asyncio.CancelledError:
        reason, cancelled = "exploration_cancelled", True
    finally:
        bridge.release()
        if contextual_interactions:
            bridge.on_state = previous_observer
        walking.invalidate()
        ladder.invalidate()
    observe(bridge.state)
    status = tracker.evaluate(bridge.state)
    unchanged = controller.intent == locked_objective and all(row["objective_unchanged"] for row in stages)
    updates = controller.policy.updates - controller.starting_updates
    consumed = sum(row["consumed_actions"] for row in stages)
    wait_consumed = sum(row["consumed_actions"] for row in planning_waits)
    buttons = sum(row["raw_button_actions"] for row in stages)
    causal_buttons = sum(row["causal_button_actions"] for row in stages)
    unowned_buttons = sum(row["unowned_button_actions"] for row in stages)
    acquired = status.completed and not baseline_completed and episode_valid
    runtime_clean = not cancelled and not any(stage["motor_failure"] for stage in stages)
    report = {"source": "soh", "success": bool(acquired and unchanged and updates == 0
        and buttons == causal_buttons and unowned_buttons == 0 and runtime_clean),
        "runtime_clean": runtime_clean,
        "reason": reason, "objective_completed": status.completed, "objective_acquired_in_run": acquired,
        "completion_evidence": status.evidence, "stages": stages, "planning": planner.snapshot(),
        "elapsed_seconds": time.monotonic()-started, "session_budget_seconds": seconds,
        "attached_ladder": any(s["task"]["kind"] == "ladder_down"
            and s["task"]["version"] == LadderDescentTask.VERSION for s in stages),
        "landing_verified": landings > 0, "verified_landings": landings,
        "post_descent_walks_successful": sum(s["success"] for s in stages
            if s["task"]["kind"] == "observed_cell" and s["after_verified_landing"]),
        "consumed_actions": consumed+wait_consumed, "raw_button_actions": buttons, "objective_unchanged": unchanged,
        "movement_and_interaction_consumed_actions": consumed,
        "planning_wait_consumed_actions": wait_consumed,"planning_waits":planning_waits,
        "causal_button_actions": causal_buttons, "unowned_button_actions": unowned_buttons,
        "contextual_interactions": contextual_interactions,
        "interaction_memory": controller.interaction_memory.snapshot() if contextual_interactions else None,
        "run_updates": updates, "provider_calls": 0, "reference_blend": 0, "demonstration_labels": 0,
        "objective": locked_objective.model_dump(), "final_objective": controller.intent.model_dump(),
        "episode_identity_unchanged": episode_valid,
        "task": first_task.snapshot() if first_task else None,
        "initial_context": context(initial), "final_context": context(bridge.state)}
    await asyncio.to_thread(write_json, directory / "planning-state.json", {
        "schema": "observed-exploration-planning-state-v1", "source": "soh", "reason": reason,
        "observation": selection_state.model_dump(), "planning": planner.snapshot(include_details=True),
        "objective": locked_objective.model_dump(), "provider_calls": 0, "training_updates": 0})
    await asyncio.to_thread(write_json, directory / "exploration.json", report)
    if cancelled:
        raise asyncio.CancelledError
    return report
