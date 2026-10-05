"""Observed portal/descent composition under one immutable strategic goal."""
from __future__ import annotations

import asyncio
import json
import math
import random
import time

from .autonomy.controller import ContinuousController
from .autonomy.ground_descent_task import GroundDescentApproachTask
from .autonomy.ladder_task import LadderDescentTask
from .autonomy.laya_ladder_policy import encode_ladder
from .autonomy.local_tasks import LocalTask
from .autonomy.models import AgentIntent, ObjectiveCompletion
from .g1 import ARTIFACTS, context, write_json
from .laya_curriculum import observed_target
from .laya_motor import execute_frozen_task


def select_descent(game):
    rejected = []
    for row in sorted((r for r in game.traversal_affordances if r.kind == "ledge_down"),
                      key=lambda r: (r.distance, math.dist(game.player.position, r.target_position))):
        try:
            return GroundDescentApproachTask.create(game, row), rejected
        except ValueError as exc:
            rejected.append({"reason": str(exc)[:160]})
    raise ValueError(f"No currently supported observed descent: {rejected}")


async def fresh_collision_snapshot(bridge, *, timeout=1.5):
    """A fast attached-state packet cannot certify a new ground walking mesh."""
    initial, start = bridge.state, time.monotonic()
    initial_context = context(initial)
    while bridge.state.full_seq <= initial.full_seq:
        await bridge.next_state(bridge.state.seq, timeout=max(.001, timeout-(time.monotonic()-start)))
        if context(bridge.state) != initial_context or bridge.state.scene_epoch != initial.scene_epoch:
            raise RuntimeError("Physical context changed during collision handoff")
        if time.monotonic()-start >= timeout:
            raise TimeoutError("Fresh collision handoff timed out")
    return bridge.state


async def traversal_episode(bridge, frozen, walking, ladder, directory, *, seconds, select_portal,
                            on_started=None, post_descent_walks=0, seed=0):
    directory.mkdir(parents=True)
    controller = await asyncio.to_thread(ContinuousController, bridge, frozen / ARTIFACTS[0],
        training_enabled=False, route_graph_path=frozen / ARTIFACTS[1], room_map_path=frozen / ARTIFACTS[2])
    objective = AgentIntent(objective="Obtain the Kokiri Sword", summary="Obtain the Kokiri Sword",
                            mode="explore", completion=ObjectiveCompletion(kind="equipment", name="Kokiri Sword"))
    controller.set_intent(objective)
    # Local tasks keep their own deadlines. The supervisor's macro-progress
    # clock spans this whole plan and is neither reset nor shortened per skill.
    controller.executor.room_budget_s = seconds + 20 + 12 * post_descent_walks + 2
    stages, rejected = [], []
    portal = None
    landing, attached, failure, cancelled = False, False, None, False

    async def execute(name, task, policy, **kwargs):
        cancelled_stage = False
        try:
            result = await execute_frozen_task(controller, bridge, task, policy, directory / name,
                                               on_started=on_started, **kwargs)
        except asyncio.CancelledError:
            # The shared motor preserves partial receipts before propagating
            # cancellation. Keep that attempted stage in the chain denominator.
            raw = await asyncio.to_thread((directory / name / "motor.json").read_text, encoding="utf-8")
            result, cancelled_stage = json.loads(raw), True
        stages.append({"name": name, **{k: result[k] for k in (
            "success", "task", "consumed_actions", "raw_button_actions", "objective_unchanged",
            "run_updates", "motor_failure")}})
        if cancelled_stage:
            raise asyncio.CancelledError
        return result

    try:
        await bridge.next_state(bridge.state.seq, timeout=1.5)
        portal, rows = select_portal(bridge.state, budget_s=seconds)
        rejected.extend(rows)
        result = await execute("portal", portal, walking)
        if not (result["success"] and portal.crossing_context and portal.verification_frames >= 3):
            raise RuntimeError(result["motor_failure"] or portal.failure or "portal_not_verified")
        await bridge.next_state(bridge.state.seq, timeout=1.5)
        descent, rows = select_descent(bridge.state)
        rejected.extend(rows)
        result = await execute("ground-approach", descent, walking)
        if (descent.failure == "unsupported_locomotor_mode" and bridge.state.player
                and bridge.state.player.climbing_ladder and not result["motor_failure"]):
            # Reobserve attachment after release. Preserve the original landing,
            # physical context and deadline; never carry a walking lease across.
            await bridge.next_state(bridge.state.seq, timeout=1.5)
            descent = LadderDescentTask.continue_from(bridge.state, descent)
            attached = True
            result = await execute("attached-ladder", descent, ladder, encoder=encode_ladder)
        landing = bool(result["success"] and descent.landing_verified(bridge.state))
        if not landing:
            raise RuntimeError(result["motor_failure"] or descent.failure or "descent_not_verified")
        rng = random.Random(seed)
        for index in range(post_descent_walks):
            await fresh_collision_snapshot(bridge)
            task = LocalTask.observed_cell(bridge.state, observed_target(bridge.state, rng), budget_s=12)
            result = await execute(f"post-descent-walk-{index+1}", task, walking)
            if not result["success"]:
                raise RuntimeError(result["motor_failure"] or task.failure or "post_descent_walk_not_verified")
    except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
        failure = f"{type(exc).__name__}:{str(exc)[:180]}"
    except asyncio.CancelledError:
        failure, cancelled = "chain_cancelled", True
    finally:
        bridge.release()
        walking.invalidate()
        ladder.invalidate()
    unchanged = controller.intent == objective and all(s["objective_unchanged"] for s in stages)
    updates = controller.policy.updates - controller.starting_updates
    consumed, buttons = sum(s["consumed_actions"] for s in stages), sum(s["raw_button_actions"] for s in stages)
    report = {"source": "soh", "success": bool(failure is None and landing and unchanged and updates == 0
                and buttons == 0 and stages and stages[0]["success"]),
        "reason": failure, "stages": stages, "attached_ladder": attached, "landing_verified": landing,
        "post_descent_walks_planned": post_descent_walks,
        "post_descent_walks_successful": sum(s["success"] for s in stages if s["name"].startswith("post-descent-walk-")),
        "consumed_actions": consumed, "raw_button_actions": buttons, "objective_unchanged": unchanged,
        "run_updates": updates, "provider_calls": 0, "reference_blend": 0, "demonstration_labels": 0,
        "objective": objective.model_dump(), "final_objective": controller.intent.model_dump(),
        "task": portal.snapshot() if portal else None,
        "initial_context": portal.context if portal else None, "final_context": context(bridge.state),
        "rejected_proposals": rejected}
    await asyncio.to_thread(write_json, directory / "chain.json", report)
    if cancelled:
        raise asyncio.CancelledError
    return report
