import asyncio
import json
import time
from types import SimpleNamespace

import pytest

from zelda_ai.autonomy.models import AgentIntent, ObjectiveCompletion
from zelda_ai.laya_motor import execute_frozen_task


def harness(state, *, cancel=False):
    goal = AgentIntent(objective="Obtain the Kokiri Sword", summary="Keep the sword goal", mode="explore",
        completion=ObjectiveCompletion(kind="equipment", name="Kokiri Sword"))
    bridge = SimpleNamespace(state=state, connected=True, receipts={}, command_seq=10, releases=0)
    def release():
        bridge.releases += 1
    bridge.release = release
    task = SimpleNamespace(deadline=time.monotonic()+20, phase="execute", failure=None)
    task.snapshot = lambda: {"phase":task.phase,"failure":task.failure}
    policy = SimpleNamespace(invalidations=0)
    def invalidate():
        policy.invalidations += 1
    policy.invalidate = invalidate
    controller = SimpleNamespace(training_enabled=False, intent=goal, policy=SimpleNamespace(updates=0),
        executor=SimpleNamespace(room_budget_s=30), pending={"command_seqs":[9]}, local_task=None,
        last_setpoint=SimpleNamespace(reason="local_task",stick_x=0,stick_y=60,buttons=0))
    def start(owned, *, stick_policy):
        assert controller.pending is None
        assert controller.intent is goal
        assert stick_policy is policy
        controller.local_task = owned
    controller.start_local_task = start
    started = asyncio.Event()
    async def run(active, publish):
        assert active()
        bridge.command_seq += 1
        bridge.receipts[11] = SimpleNamespace(first_tick=50, model_dump=lambda: {"pressed":0,"first_tick":50})
        publish()
        started.set()
        if cancel:
            await asyncio.Event().wait()
        task.phase, controller.local_task = "succeeded", None
    controller.run = run
    return controller, bridge, task, policy, started


def test_frozen_stage_retains_goal_but_does_not_reuse_old_consumption(state, tmp_path):
    async def scenario():
        controller, bridge, task, policy, _ = harness(state)
        bridge.receipts[9] = SimpleNamespace(first_tick=40, model_dump=lambda: {"pressed":64,"first_tick":40})
        result = await execute_frozen_task(controller, bridge, task, policy, tmp_path, encoder=lambda g,t:[1]*9)
        assert result["success"] and result["consumed_actions"] == 1
        assert result["raw_button_actions"] == 0 and result["objective_unchanged"]
        assert bridge.releases == 1 and policy.invalidations == 2
    asyncio.run(scenario())


def test_cancelled_stage_preserves_partial_receipts_then_propagates_cancel(state, tmp_path):
    async def scenario():
        controller, bridge, task, policy, started = harness(state, cancel=True)
        running = asyncio.create_task(execute_frozen_task(controller, bridge, task, policy, tmp_path,
            encoder=lambda g,t:[1]*9))
        await started.wait()
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        result = json.loads((tmp_path / "motor.json").read_text(encoding="utf-8"))
        assert not result["success"] and result["motor_failure"] == "stage_cancelled"
        assert result["consumed_actions"] == 1 and result["objective_unchanged"]
        assert bridge.releases == 1 and policy.invalidations == 2
    asyncio.run(scenario())


def test_actual_motor_observations_can_feed_ephemeral_planner_without_another_sender(state, tmp_path):
    async def scenario():
        controller, bridge, task, policy, _ = harness(state)
        observations = []
        result = await execute_frozen_task(controller, bridge, task, policy, tmp_path,
            encoder=lambda g,t:[1]*9, observe=lambda game: observations.append(game.seq))
        assert result["success"] and observations == [state.seq]
        assert result["consumed_actions"] == 1 and bridge.releases == 1
    asyncio.run(scenario())


def test_dialogue_guard_keeps_consumed_candidate_motion_in_audit(state, tmp_path):
    async def scenario():
        controller, bridge, task, policy, _ = harness(state)
        controller.last_setpoint.reason = "dialogue_disengage"
        result = await execute_frozen_task(controller, bridge, task, policy, tmp_path,
            encoder=lambda g,t:[1]*9)
        assert result["success"] and result["consumed_actions"] == 1
        assert result["requested"][11]["source"] == "dialogue_disengage"
        assert result["raw_button_actions"] == 0 and result["run_updates"] == 0
    asyncio.run(scenario())
