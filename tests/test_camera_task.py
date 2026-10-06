"""Software camera ownership/causality contracts, not gameplay evidence."""
import time
from types import SimpleNamespace

import pytest

from zelda_ai.autonomy.camera_task import ObservedCameraReturnTask
from zelda_ai.autonomy.controller import ContinuousController, Setpoint
from zelda_ai.autonomy.interaction_memory import context_key
from zelda_ai.autonomy.locomotion import FIRST_PERSON, camera_modal_active, grounded, locomotor_mode
from zelda_ai.bridge import Bridge
from zelda_ai.models import GameEvent


def setup(state):
    state.player.bg_check_flags, state.player.floor_height, state.player.speed_xz = 1, 0, 0
    state.player.state_flags_1 = FIRST_PERSON
    state.context_action.code, state.context_action.label = 4, "return"


def advance(state, task, *, consumed=False, now=.1):
    state.seq += 1
    task.observe(state, consumed=consumed, now=now)


def test_camera_return_requires_native_mode_change_not_hud_change_or_camera_rotation(state):
    setup(state)
    task = ObservedCameraReturnTask.create(state, now=0)
    assert grounded(state.player) and locomotor_mode(state) == "first_person"
    assert task.context_action_key.startswith("first_person:")
    for i in range(3):
        advance(state, task, now=.1+i/10)
    assert task.phase == "execute" and task.allows_context_buttons(state)
    assert task.guidance(state)["camera_return_active"] and task.reference_stick(state) == (0, 0)
    state.camera_input_yaw, state.player.yaw = 16000, 16000
    advance(state, task, consumed=True, now=.5)
    assert not task.terminal and task.observed_effect is None
    state.context_action.label = "attack"
    advance(state, task, consumed=True, now=.6)
    assert task.phase == "interrupted" and task.observed_effect is None


def test_consumed_exit_needs_three_fresh_stopped_frames_and_resets_on_reentry(state):
    setup(state)
    task = ObservedCameraReturnTask.create(state, now=0)
    state.player.state_flags_1 = 0
    advance(state, task, consumed=True)
    assert task.phase == "verify" and task.verification_frames == 1
    task.observe(state, consumed=True, now=.2)
    assert task.verification_frames == 1
    state.player.speed_xz = 2
    advance(state, task, consumed=True, now=.3)
    assert task.verification_frames == 0
    state.player.speed_xz, state.player.state_flags_1 = 0, FIRST_PERSON
    advance(state, task, consumed=True, now=.4)
    assert task.verification_frames == 0 and not task.terminal
    state.player.state_flags_1 = 0
    for i in range(3):
        advance(state, task, consumed=True, now=.5+i/10)
    assert task.phase == "succeeded" and task.observed_effect == "first_person_closed"
    assert task.verification_frames == 3 and not task.allows_context_buttons(state)


def test_exit_before_owned_button_consumption_cannot_be_credited_later(state):
    setup(state)
    task = ObservedCameraReturnTask.create(state, now=0)
    state.player.state_flags_1 = 0
    advance(state, task)
    assert task.phase == "interrupted" and task.failure == "camera_exit_without_button_consumption"
    for i in range(3):
        advance(state, task, consumed=True, now=.2+i/10)
    assert task.phase != "succeeded"


@pytest.mark.parametrize("loss", ["reload", "scene", "modal", "water", "death"])
def test_camera_return_releases_on_ownership_or_mode_loss(state, loss):
    setup(state)
    task = ObservedCameraReturnTask.create(state, now=0)
    if loss == "reload":
        state.events = [GameEvent(id="new", kind="save_loaded")]
    elif loss == "scene":
        state.scene += 1
    elif loss == "modal":
        state.dialogue.active = True
    elif loss == "water":
        state.player.state_flags_1 |= 1 << 27
    else:
        state.player.health = 0
    advance(state, task, consumed=True)
    assert task.phase == "interrupted" and task.reference_stick(state) == (0, 0)


def test_camera_return_attempt_stays_bounded_without_an_effect(state):
    setup(state)
    task = ObservedCameraReturnTask.create(state, now=0, budget_s=2)
    advance(state, task, consumed=True, now=2.1)
    assert task.phase == "failed" and task.failure == "camera_return_no_observed_effect"


@pytest.mark.parametrize("missing", ["flag", "prompt", "ground", "pause", "cutscene"])
def test_selection_requires_the_specific_observed_camera_context(state, missing):
    setup(state)
    if missing == "flag":
        state.player.state_flags_1 = 0
    elif missing == "prompt":
        state.context_action.label = "speak"
    elif missing == "ground":
        state.player.bg_check_flags = 0
    elif missing == "pause":
        state.pause_menu.active = True
    else:
        state.cutscene_active = True
    with pytest.raises(ValueError):
        ObservedCameraReturnTask.create(state, now=0)


def test_native_camera_and_normal_context_bindings_are_isolated(state):
    setup(state)
    camera_key = context_key(state)
    state.player.state_flags_1 = 0
    assert context_key(state) != camera_key
    assert not context_key(state).startswith("first_person:")


def test_rotating_or_top_down_camera_geometry_does_not_claim_first_person_ownership(state):
    setup(state)
    state.player.state_flags_1 = 0
    for eye,at,yaw in [((0,600,0),(0,0,0),12000),((120,70,0),(0,0,0),-12000)]:
        state.camera_eye,state.camera_at,state.camera_input_yaw = eye,at,yaw
        assert not camera_modal_active(state.player)
        assert grounded(state.player) and locomotor_mode(state)=="ground"


def probe_controller(state, *, delivered=True, mask=0x10):
    setup(state)
    state.protocol = 3
    memory = []
    controller = SimpleNamespace(pending_interaction_probe={"kind":"context", "camera_return_active":True,
        "instance_id":state.instance_id, "event_ids":(), "scene":state.scene,"room":state.room,
        "command_seqs":[6],"key":context_key(state),"button":"R","known":False,"at":time.monotonic()},
        bridge=SimpleNamespace(command_consumed=lambda seq:delivered and seq==6,
            receipts={6:SimpleNamespace(pressed=mask)}),
        interaction_memory=SimpleNamespace(record_interaction_success=lambda k,b:memory.append((k,b))),
        interaction_probe_successes=0,last_interaction_source="none")
    return controller,memory


def test_binding_is_learned_only_after_exact_consumption_and_three_fresh_camera_exit_frames(state):
    controller,memory = probe_controller(state)
    key = controller.pending_interaction_probe['key']
    state.player.state_flags_1 = 0
    for i in range(3):
        state.seq += 1
        ContinuousController._observe_interaction_outcome(controller,state,None)
        if i<2:
            assert not memory
            ContinuousController._observe_interaction_outcome(controller,state,None)
            assert not memory
    assert memory == [(key,"R")] and controller.pending_interaction_probe is None


def test_camera_binding_cannot_reuse_a_mode_change_observed_before_consumption(state):
    controller,memory = probe_controller(state,delivered=False)
    state.player.state_flags_1 = 0
    state.seq += 1
    ContinuousController._observe_interaction_outcome(controller,state,None)
    assert controller.pending_interaction_probe is None
    controller.bridge.command_consumed = lambda seq:True
    for i in range(3):
        state.seq += 1
        ContinuousController._observe_interaction_outcome(controller,state,None)
    assert not memory


@pytest.mark.parametrize("missing", ["consumption", "matching_mask", "native_flag_change", "same_scene", "modal_clear"])
def test_unproven_camera_exit_never_teaches_a_button(state, missing):
    controller,memory = probe_controller(state,delivered=missing!='consumption',mask=0x8000 if missing=='matching_mask' else 0x10)
    if missing != "native_flag_change":
        state.player.state_flags_1 = 0
    if missing == "same_scene":
        state.scene += 1
    elif missing == "modal_clear":
        state.dialogue.active = True
    for i in range(3):
        state.seq += 1
        ContinuousController._observe_interaction_outcome(controller,state,None)
    assert not memory


def test_closing_button_is_released_before_repeat_and_guard_keeps_escape_stick(state):
    setup(state)
    old = state.model_copy(deep=True)
    released = []
    controller = SimpleNamespace(dialogue_reentry_guard=None,pending_interaction_probe={"button":"R"},
        pending={"trainable":True},last_setpoint=Setpoint(buttons=0x10),
        bridge=SimpleNamespace(release=lambda:released.append(True)),dialogue_reentry_suppressed=0)
    state.seq += 1
    state.player.state_flags_1 = 0
    ContinuousController.note_camera_return_closed(controller,old,state)
    assert released == [True] and controller.last_setpoint.buttons == 0
    assert controller.pending["trainable"] is False and controller.pending_interaction_probe is not None
    setpoint,sample,overridden = ContinuousController._dialogue_reentry_override(controller,state,
        Setpoint(buttons=0x8000,stick_x=21,stick_y=45),{"buttons":[1]*9})
    assert overridden and setpoint.buttons == 0 and (setpoint.stick_x,setpoint.stick_y)==(21,45)
    assert sample["buttons"] == [0.]*9


def test_main_controller_automatically_owns_observed_camera_return_without_frontier_or_ppo_training(state,tmp_path):
    setup(state)
    bridge = Bridge("x"*32)
    bridge.state,bridge.release = state,lambda:None
    controller = ContinuousController(bridge,tmp_path/'policy.pt',training_enabled=False)
    controller._ml_step(state)
    assert isinstance(controller.local_task,ObservedCameraReturnTask)
    assert controller.route_graph.active_frontier is None
    assert controller.last_setpoint.stick_x == controller.last_setpoint.stick_y == 0
    assert controller.pending["trainable"] is False
