import pytest

from zelda_ai.autonomy.ground_descent_task import GroundDescentApproachTask
from zelda_ai.autonomy.imitation import encode_surface
from zelda_ai.autonomy.ladder_task import LadderDescentTask
from zelda_ai.models import NavigationMeshSnapshot, TraversalAffordanceObservation


def descent(state):
    state.player.position = (0, 100, 0)
    state.player.floor_height, state.player.bg_check_flags = 100, 1
    state.player.speed_xz = 0
    state.camera_input_yaw = 0
    state.navmesh = NavigationMeshSnapshot(origin=(0, 100, 0), step=70, half_extent=3,
        cells=[(0, 0, 100., 0), (1, 0, 0., 0)])
    row = TraversalAffordanceObservation(kind="ledge_down", direction="down",
        target_position=(70, 0, 0), approach_position=state.player.position, distance=0, height_delta=-100)
    state.traversal_affordances = [row]
    return row


def test_dry_descent_uses_point_encoding_without_changing_actual_landing(state):
    row = descent(state)
    task = GroundDescentApproachTask.create(state, row, now=0)
    features = encode_surface(state, task)
    assert features[4] == 0 and features[8] == 1
    assert task.steering_point(state) == (70, 100, 0)
    assert task.target == (70, 0, 0)
    state.player.position = (70, 100, 0)
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1 + .1 * i)
    assert task.phase != "succeeded"  # Reaching the upper projection cannot prove a descent.


def test_actual_attachment_interrupts_walk_and_preserves_landing_budget(state):
    task = GroundDescentApproachTask.create(state, descent(state), now=0, budget_s=5)
    state.player.climbing_ladder = True
    state.seq += 1
    task.observe(state, consumed=True, now=1)
    assert task.failure == "unsupported_locomotor_mode"
    attached = LadderDescentTask.continue_from(state, task, now=1.1)
    assert attached.target == task.target and attached.deadline <= task.deadline
    assert not attached.consumed  # The prior walking command is not attached-policy evidence.


def test_lost_current_landing_evidence_stops_approach(state):
    task = GroundDescentApproachTask.create(state, descent(state), now=0)
    state.navmesh.cells = [(0, 0, 100., 0)]
    state.traversal_affordances = []
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert task.phase == "interrupted" and task.failure == "current_landing_support_lost"


def test_lower_landing_requires_three_stopped_dry_frames(state):
    task = GroundDescentApproachTask.create(state, descent(state), now=0)
    state.player.position = task.target
    state.player.floor_height = 0
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1 + .1 * i)
        assert task.landing_verified(state) is (i == 2)


def test_upper_approach_does_not_cut_a_corner_at_35_units(state):
    row = descent(state)
    row.distance, row.approach_position, row.target_position = 70, (70, 100, 70), (140, 0, 70)
    state.navmesh.cells = [(0, 0, 100., 1), (0, 1, 100., 4), (1, 1, 100., 0), (2, 1, 0., 0)]
    task = GroundDescentApproachTask.create(state, row, now=0)
    assert task.phase == "prepare" and task.steering_point(state) == (0, 100, 70)
    state.player.position = (0, 100, 40)
    state.seq += 1
    task.observe(state, consumed=True, now=1)
    assert task.steering_point(state) == (0, 100, 70)
    state.player.position = (0, 100, 60)
    state.seq += 1
    task.observe(state, consumed=True, now=1.5)
    assert task.steering_point(state) == (70, 100, 70) and task.phase == "prepare"


def test_invented_or_unsupported_descent_is_rejected(state):
    row = descent(state)
    state.traversal_affordances = []
    with pytest.raises(ValueError, match="currently observed"):
        GroundDescentApproachTask.create(state, row)
