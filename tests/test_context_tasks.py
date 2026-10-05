import hashlib
import time
from types import SimpleNamespace

import pytest

from zelda_ai.autonomy.context_tasks import LinearDialogueTask, ObservedContextTask
from zelda_ai.autonomy.controller import ContinuousController
from zelda_ai.autonomy.interaction_memory import ObservedInteractionMemory, context_key
from zelda_ai.autonomy.routes import LearnedRouteGraph
from zelda_ai.bridge import Bridge


def context(state):
    state.player.bg_check_flags = 1
    state.player.floor_height = state.player.position[1]
    state.player.speed_xz = 0
    state.context_action.code, state.context_action.label = 15, "speak"


def test_context_waits_for_stopped_fresh_frames_and_cannot_succeed_without_button_consumption(state):
    context(state)
    task = ObservedContextTask.create(state, now=0)
    for i in range(1, 4):
        state.seq += 1
        task.observe(state, consumed=False, now=i/10)
    assert task.phase == "execute" and task.allows_context_buttons(state)
    assert task.reference_stick(state) == (0, 0)
    state.dialogue.active = True
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=False, now=.5+i/10)
    assert not task.terminal and task.verification_frames == 0
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=1+i/10)
    assert task.phase == "succeeded" and task.observed_effect == "dialogue_started"
    assert task.verification_frames == 3


def test_neutral_wrong_mask_and_old_receipt_cannot_pay_a_contextual_button(state):
    context(state)
    task = ObservedContextTask.create(state, now=0)
    task.register_button(0, 0x8000)
    task.register_button(4, 0x1000)
    task.register_button(5, 0xC000)
    task.register_button(6, 0x8000)
    assert task.button_commands == {6: 0x8000}
    bridge = SimpleNamespace(receipts={5: SimpleNamespace(pressed=0x8000), 6: SimpleNamespace(pressed=0)},
                             command_consumed=lambda seq: True)
    assert not task.button_consumed(bridge)
    bridge.receipts[6].pressed = 0x4000
    assert not task.button_consumed(bridge)
    bridge.receipts[6].pressed = 0x8000
    assert task.button_consumed(bridge)


def test_context_changed_or_native_reload_cannot_count_as_causal_effect(state):
    from zelda_ai.models import GameEvent
    context(state)
    task = ObservedContextTask.create(state, now=0)
    state.seq += 1
    state.room += 1
    state.events = [GameEvent(id="new-load", kind="save_loaded")]
    task.observe(state, consumed=True, now=.1)
    assert task.phase == "interrupted" and task.observed_effect is None


def test_linear_dialogue_verifies_close_and_refuses_semantic_choice(state):
    context(state)
    state.dialogue.active, state.dialogue.can_advance = True, True
    task = LinearDialogueTask.create(state, now=0)
    state.seq += 1
    state.dialogue.choice_count = 2
    task.observe(state, consumed=False, now=.1)
    assert task.failure == "semantic_choice_requires_cognition" and task.terminal
    state.dialogue.choice_count = 0
    task = LinearDialogueTask.create(state, now=1)
    state.dialogue.active = False
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=1.2+i/10)
    assert task.phase == "succeeded" and task.observed_effect == "dialogue_closed"
    assert task.verification_frames == 3 and not task.allows_context_buttons(state)


@pytest.mark.parametrize("label", ["attack", "putaway", "none"])
def test_general_button_labels_do_not_create_world_interaction_tasks(state, label):
    context(state)
    state.context_action.label = label
    with pytest.raises(ValueError):
        ObservedContextTask.create(state)


def test_readonly_control_memory_reuses_observed_success_and_evicts_bad_legacy_binding_without_mutation(tmp_path):
    path = tmp_path / "graph.json"
    seed = LearnedRouteGraph(path)
    seed.record_interaction_success("opaque-context", "A")
    seed.save(force=True)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    graph = LearnedRouteGraph(path, writable=False)
    memory = ObservedInteractionMemory(graph)
    for _ in range(3):
        memory.record_interaction_failure("opaque-context", "A")
    assert memory.interaction_button("opaque-context") is None
    assert graph.interaction_button("opaque-context") == "A"
    memory.record_interaction_success("opaque-context", "R")
    assert memory.interaction_button("opaque-context") == "R"
    assert graph.interaction_button("opaque-context") == "A" and not graph.dirty
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_typed_context_can_use_empirical_button_without_changing_goal_or_training(state, tmp_path):
    context(state)
    bridge = Bridge("x" * 32)
    bridge.state, bridge.last_seen = state, time.monotonic()
    seed = ContinuousController(bridge, tmp_path / "policy.pt", training_enabled=True)
    seed.route_graph.record_interaction_success(context_key(state), "R")
    seed.route_graph.save(force=True)
    seed.policy.save()
    controller = ContinuousController(bridge, tmp_path / "policy.pt", training_enabled=False)
    objective = controller.intent.model_copy(deep=True)
    task = ObservedContextTask.create(state)
    task.phase, task.stopped_frames = "execute", 3
    controller.start_local_task(task, stick_policy=lambda game, owned: (0, 0))
    setpoint = controller._ml_step(state)
    assert setpoint.reason == "interaction_memory" and setpoint.buttons == 0x10
    assert setpoint.stick_x == setpoint.stick_y == 0
    assert controller.pending["trainable"] is False and controller.intent == objective
    assert controller.policy.updates == controller.starting_updates
