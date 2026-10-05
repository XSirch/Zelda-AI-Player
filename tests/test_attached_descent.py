import pytest

from zelda_ai.autonomy.attached_descent import PROBES, AttachedDescentTask
from zelda_ai.autonomy.descent_task import ObservedDescentTask
from zelda_ai.models import NavigationMeshSnapshot, TraversalAffordanceObservation


def handoff(state):
    state.player.position = (0, 100, 0)
    state.player.floor_height = 100
    state.player.bg_check_flags = 1
    state.player.speed_xz = 0
    state.camera_input_yaw = 0
    state.navmesh = NavigationMeshSnapshot(origin=(0, 100, 0), step=70, half_extent=3,
        cells=[(0, 0, 100., 0), (1, 0, 0., 0)])
    row = TraversalAffordanceObservation(kind="ledge_down", direction="down",
        target_position=(70, 0, 0), approach_position=state.player.position, distance=0, height_delta=-100)
    state.traversal_affordances = [row]
    parent = ObservedDescentTask.create(state, row, now=0)
    state.player.climbing_ladder = True
    state.seq += 1
    parent.observe(state, consumed=True, now=.1)
    assert parent.failure == "unsupported_locomotor_mode"
    return AttachedDescentTask.continue_from(state, parent, now=.2), parent


def test_ladder_handoff_preserves_the_observed_floor_and_original_budget(state):
    task, parent = handoff(state)
    assert task.target == parent.target
    assert task.deadline <= parent.deadline
    assert task.reference_stick(state) == (0, 0)
    assert not task.grounded(state)  # Top ladder animation still has native ground bit.


@pytest.mark.parametrize("bad", ("context", "deadline", "missing_floor", "wrong_attachment"))
def test_ladder_handoff_does_not_invent_a_new_floor_or_context(state, bad):
    _, parent = handoff(state)
    if bad == "context":
        state.scene_epoch += 1
    elif bad == "missing_floor":
        state.navmesh.cells = [(0, 0, 100., 0)]
    elif bad == "wrong_attachment":
        state.player.hanging_ledge = True
    with pytest.raises(ValueError):
        AttachedDescentTask.continue_from(state, parent, now=21 if bad == "deadline" else .3)


@pytest.mark.parametrize("consumed", (True, False))
def test_mode_vector_is_learned_only_after_consumed_downward_motion(state, consumed):
    task, _ = handoff(state)
    task.raw_probe = PROBES[1]
    task.probe_at, task.probe_seq = .2, state.seq
    task.probe_position = state.player.position
    state.seq += 1
    state.player.position = (0, 85, 0)
    task.observe(state, consumed=consumed, now=.6)
    assert task.learned_stick == (PROBES[1] if consumed else None)
    assert task.calibration[-1]["consumed"] is consumed


def test_camera_frame_change_releases_the_empirical_raw_vector(state):
    task, _ = handoff(state)
    task.learned_stick = PROBES[1]
    task.input_frame = (0, state.player.yaw)
    state.seq += 1
    state.camera_input_yaw = 16384
    task.observe(state, consumed=True, now=.4)
    assert task.learned_stick is None
    assert task.reference_stick(state) == (0, 0)


def test_gradual_camera_orbit_invalidates_the_original_calibration_frame(state):
    task, _ = handoff(state)
    task.learned_stick = PROBES[1]
    task.input_frame = (0, state.player.yaw)
    for index, yaw in enumerate((1000, 2000, 3000)):
        state.seq += 1
        state.camera_input_yaw = yaw
        task.observe(state, consumed=True, now=.3 + .1 * index)
    assert task.learned_stick is None
    assert task.reference_stick(state) == (0, 0)


def test_ladder_vector_never_continues_as_walking_after_attachment_ends(state):
    task, _ = handoff(state)
    task.learned_stick = PROBES[1]
    state.player.climbing_ladder = False
    state.seq += 1
    task.observe(state, consumed=True, now=.4)
    assert task.failure == "attachment_left_before_landing"
    assert task.reference_stick(state) == (0, 0)


def test_ladder_success_requires_actual_released_ground_landing(state):
    task, _ = handoff(state)
    state.player.position = task.target
    state.player.floor_height = 0
    for index in range(4):
        state.seq += 1
        task.observe(state, consumed=True, now=.3 + .1 * index)
    assert not task.landing_verified(state)
    state.player.climbing_ladder = False
    for index in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.8 + .1 * index)
    assert task.landing_verified(state)
