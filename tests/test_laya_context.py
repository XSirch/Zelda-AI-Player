import asyncio
import json
from types import SimpleNamespace

import pytest

from zelda_ai.autonomy.context_tasks import NativeModalWaitTask, ObservedContextTask
from zelda_ai.autonomy.controller import Setpoint
from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.laya_context import execute_context_task, execute_modal_wait
from zelda_ai.models import InputReceipt


@pytest.mark.parametrize("source,expected", [("interaction_probe", True), ("ml_policy", False)])
def test_causal_stage_requires_owned_single_button_receipt_even_if_task_reports_success(state, tmp_path, source, expected):
    state.player.bg_check_flags = 1
    state.player.floor_height = state.player.position[1]
    state.context_action.code, state.context_action.label = 1, "check"
    task = ObservedContextTask.create(state)
    receipt = InputReceipt(seq=2, owner_epoch=1, status="completed", first_tick=2, last_tick=3,
                           pressed=0x8000, released=0x8000)
    bridge = SimpleNamespace(state=state, connected=True, command_seq=1, receipts={}, release=lambda: None,
                             command_consumed=lambda seq: seq == 2)
    controller = SimpleNamespace(training_enabled=False, policy=SimpleNamespace(updates=0),
        intent=AgentIntent.bootstrap(), pending=None, pending_interaction_probe=None,
        interaction_memory=SimpleNamespace(snapshot=lambda: {}))

    def start(owned, *, stick_policy):
        controller.local_task = owned
        assert stick_policy(state, owned) == (0, 0)

    async def run(active, publish):
        # Software integration fixture, never native evidence.
        bridge.command_seq, bridge.receipts = 2, {2: receipt}
        controller.pending = {"trainable": False}
        controller.pending_interaction_probe = {"kind": "context", "key": task.context_action_key,
            "button": "A", "command_seqs": [2], "at": task.started}
        controller.last_setpoint = Setpoint(buttons=0x8000, reason=source)
        publish()
        task.phase, task.consumed, task.observed_effect = "succeeded", True, "dialogue_started"
        controller.local_task = None

    controller.start_local_task, controller.run = start, run
    result = asyncio.run(execute_context_task(controller, bridge, task, tmp_path / "motor"))
    assert result["success"] is expected
    assert result["causal_button_actions"] == int(expected)
    assert result["unowned_button_actions"] == int(not expected)
    assert result["run_updates"] == 0 and result["candidate_button_actions"] == 0
    assert json.loads((tmp_path / "motor/motor.json").read_text(encoding="utf-8"))["input_source"] == "causal_context_controller"


def test_modal_wait_polls_same_bridge_releases_ownership_and_does_not_claim_neural_actions(state, tmp_path):
    state.cutscene_active = True
    task = NativeModalWaitTask.create(state)
    releases, observations = [], []
    bridge = SimpleNamespace(state=state, connected=True, command_seq=1, receipts={},
        release=lambda: releases.append("release"))
    controller = SimpleNamespace(training_enabled=False, policy=SimpleNamespace(updates=0),
        intent=AgentIntent.bootstrap(), pending={"old": True}, pending_interaction_probe={"old": True},
        local_task=object(), local_stick_policy=object(),
        executor=SimpleNamespace(state="executing", observe=lambda game, evidence: observations.append(game.seq)))
    calls = 0

    async def next_state(seq, *, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TimeoutError("temporary silence")
        state.seq += 1
        state.cutscene_active = False
        return state

    bridge.next_state = next_state
    result = asyncio.run(execute_modal_wait(controller, bridge, task, tmp_path / "wait"))
    assert result["success"] and calls == 4 and len(releases) == 2
    assert result["consumed_actions"] == result["raw_button_actions"] == 0
    assert result["input_source"] == "neutral_native_modal_wait" and result["run_updates"] == 0
    assert result["objective_unchanged"] and observations[-1] == state.seq
    assert controller.pending is controller.pending_interaction_probe is controller.local_task is None
