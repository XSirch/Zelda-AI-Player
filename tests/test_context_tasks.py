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


def test_fresh_native_chest_effect_survives_animation_but_old_events_and_item_received_do_not_pay(state):
    from zelda_ai.models import ActorObservation, GameEvent
    context(state)
    state.context_action.label = "open"
    state.context_actor = ActorObservation(actor_uid="container", actor_id=999, category=10,
        category_name="chest", params=0, position=(0, 0, 30), distance=30, drawn=True)
    state.events = [GameEvent(id="old-chest", kind="chest_opened")]
    task = ObservedContextTask.create(state, now=0)
    task.phase = "execute"
    state.cutscene_active = True
    state.events.append(GameEvent(id="only-item", kind="item_received"))
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert not task.terminal and task.observed_effect is None
    state.events.append(GameEvent(id="new-chest", kind="chest_opened", detail=f"{state.scene}:7"))
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.2+i/10)
    assert task.phase == "succeeded" and task.observed_effect == "chest_opened"
    assert task.effect_event_id == "new-chest" and task.verification_frames == 3


def test_chest_event_before_consumption_cannot_be_reused_afterward(state):
    from zelda_ai.models import ActorObservation, GameEvent
    context(state)
    state.context_action.label = "open"
    state.context_actor = ActorObservation(actor_uid="container", actor_id=999, category=10,
        category_name="chest", params=0, position=(0, 0, 30), distance=30, drawn=True)
    task = ObservedContextTask.create(state, now=0)
    task.phase = "execute"
    state.events = [GameEvent(id="too-early", kind="chest_opened", detail=f"{state.scene}:1")]
    state.seq += 1
    task.observe(state, consumed=False, now=.1)
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.2+i/10)
    assert task.observed_effect is None and task.effect_event_id is None


@pytest.mark.parametrize("handoff", ["ground", "dialogue"])
def test_native_animation_wait_requires_three_fresh_handoff_frames_and_no_button(state, handoff):
    from zelda_ai.autonomy.context_tasks import NativeModalWaitTask
    context(state)
    state.cutscene_active = True
    task = NativeModalWaitTask.create(state, now=0)
    state.seq += 1
    task.observe(state, now=.1)
    assert not task.terminal
    if handoff == "ground":
        state.cutscene_active = False
    else:
        state.dialogue.active = True
    for i in range(3):
        state.seq += 1
        task.observe(state, now=.2+i/10)
        task.observe(state, now=.2+i/10)  # Duplicate telemetry cannot pay.
    assert task.phase == "succeeded" and task.verification_frames == 3
    assert not task.consumed


def test_native_animation_wait_stays_bounded_and_rejects_reload(state):
    from zelda_ai.autonomy.context_tasks import NativeModalWaitTask
    from zelda_ai.models import GameEvent
    context(state)
    state.cutscene_active = True
    task = NativeModalWaitTask.create(state, now=0, budget_s=1)
    state.seq += 1
    task.observe(state, now=1.1)
    assert task.failure == "native_modal_wait_timeout"
    task = NativeModalWaitTask.create(state, now=2)
    state.seq += 1
    state.events = [GameEvent(id="new-load", kind="save_loaded")]
    task.observe(state, now=2.1)
    assert task.failure == "context_episode_changed"


def test_causal_chest_binding_requires_a_new_effect_and_matching_consumed_physical_button(state):
    from zelda_ai.models import GameEvent
    state.protocol = 3
    memory = []
    consumed = False
    receipt = SimpleNamespace(pressed=0x10)
    controller = SimpleNamespace(
        bridge=SimpleNamespace(command_consumed=lambda seq: consumed, receipts={8: receipt}),
        pending_interaction_probe={"kind": "context", "instance_id": state.instance_id,
            "at": time.monotonic(), "command_seqs": [8], "scene": state.scene, "room": state.room,
            "dialogue_active": False, "key": "opaque", "button": "R", "known": False,
            "event_ids": (), "container_context": True, "context_actor_uid": "container"},
        interaction_memory=SimpleNamespace(record_interaction_success=lambda *args: memory.append(args)),
        interaction_probe_successes=0, last_interaction_source="none")
    state.events = [GameEvent(id="before-consumption", kind="chest_opened", detail=f"{state.scene}:1")]
    ContinuousController._observe_interaction_outcome(controller, state, None)
    consumed = True
    ContinuousController._observe_interaction_outcome(controller, state, None)
    assert not memory
    state.events.append(GameEvent(id="new-effect", kind="chest_opened", detail=f"{state.scene}:2"))
    receipt.pressed = 0x8000
    ContinuousController._observe_interaction_outcome(controller, state, None)
    assert not memory
    receipt.pressed = 0x10
    ContinuousController._observe_interaction_outcome(controller, state, None)
    assert memory == [("opaque", "R")] and controller.pending_interaction_probe is None


def test_cutscene_arbiter_observes_owned_container_effect_with_neutral_input(state, tmp_path):
    import asyncio

    from zelda_ai.models import ActorObservation, GameEvent
    context(state)
    state.context_action.label = "open"
    state.context_actor = ActorObservation(actor_uid="container", actor_id=999, category=10,
        category_name="chest", params=0, position=(0, 0, 30), distance=30, drawn=True)
    task = ObservedContextTask.create(state)
    task.phase = "execute"
    task.register_button(6, 0x10)
    bridge = Bridge("x" * 32)
    bridge.state, bridge.last_seen = state, time.monotonic()
    bridge.command_consumed = lambda seq: seq == 6
    bridge.receipts[6] = SimpleNamespace(pressed=0x10, model_dump=lambda: {"seq": 6, "pressed": 0x10})
    sent, snapshots = [], []

    def send(**packet):
        sent.append(packet)
        bridge.command_seq += 1
        return bridge.command_seq

    bridge.send, bridge.release = send, lambda: None
    controller = ContinuousController(bridge, tmp_path / "policy.pt", training_enabled=False)
    controller.start_local_task(task, stick_policy=lambda game, owned: (0, 0))
    state.cutscene_active = True
    state.seq += 1
    state.events = [GameEvent(id="new-chest", kind="chest_opened", detail=f"{state.scene}:1")]
    asyncio.run(controller.run(lambda: not snapshots, lambda: snapshots.append(task.snapshot())))
    assert snapshots[0]["observed_effect"] == "chest_opened" and snapshots[0]["verification_frames"] == 1
    assert sent == [{"buttons": 0, "stick_x": 0, "stick_y": 0, "lease_ms": 150}]


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
