import pytest

from zelda_ai.autonomy.descent_task import ObservedDescentTask
from zelda_ai.models import NavigationMeshSnapshot, TraversalAffordanceObservation


def ledge(state):
    state.player.position = (0, 100, 0)
    state.player.floor_height = 100
    state.player.bg_check_flags = 1
    state.navmesh = NavigationMeshSnapshot(origin=(0, 100, 0), step=70, half_extent=3,
        cells=[(0, 0, 100., 4), (1, 0, 100., 0), (3, 0, 0., 0)])
    row = TraversalAffordanceObservation(kind="ledge_down", direction="down",
        approach_position=(140, -80, 0), target_position=(210, 0, 0),
        distance=140, height_delta=-100)
    state.traversal_affordances = [row]
    return row


def test_descent_approaches_only_via_current_upper_floor_cells(state):
    row = ledge(state)
    task = ObservedDescentTask.create(state, row, now=0)
    assert task.approach == (70, 100, 0)
    assert task.approach != row.approach_position
    assert task.target == row.target_position
    assert task.kind == "ledge_down"
    assert task.phase == "prepare"


@pytest.mark.parametrize("failure", ("unobserved", "inconsistent_height", "too_high", "no_approach"))
def test_descent_never_creates_unobserved_floor_or_directed_edges(state, failure):
    row = ledge(state)
    if failure == "unobserved":
        state.traversal_affordances = []
    elif failure == "inconsistent_height":
        row.height_delta = -20
    elif failure == "too_high":
        row.target_position = (210, -50, 0)
    else:
        state.navmesh.cells[0] = (0, 0, 100., 0)
    with pytest.raises(ValueError):
        ObservedDescentTask.create(state, row, now=0)


def test_descent_success_requires_consumption_and_three_stopped_ground_frames(state):
    task = ObservedDescentTask.create(state, ledge(state), now=0)
    state.player.position = task.target
    state.player.floor_height = 0
    state.player.bg_check_flags = 0
    for i in range(4):
        state.seq += 1
        task.observe(state, consumed=True, now=.1 + i * .1)
    assert task.phase != "succeeded"  # Passing the landing while still airborne.
    state.player.bg_check_flags = 1
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.6 + i * .1)
    assert task.landing_verified(state)


@pytest.mark.parametrize("mode", ("hanging_ledge", "climbing_ladder", "climbing_ledge"))
def test_descent_does_not_reuse_camera_walking_control_while_attached(state, mode):
    task = ObservedDescentTask.create(state, ledge(state), now=0)
    setattr(state.player, mode, True)
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert task.failure == "unsupported_locomotor_mode"
    assert task.reference_stick(state) == (0, 0)


def test_direct_native_landing_sample_is_not_discarded_for_a_different_coarse_height(state):
    row = ledge(state)
    state.navmesh.cells = [(0, 0, 100., 4), (1, 0, 100., 0), (3, 0, -13., 0)]
    task = ObservedDescentTask.create(state, row, now=0)
    assert task.target == (210, 0, 0)  # Native floor sample, not the adjacent coarse cell.


def test_descent_cannot_start_from_an_airborne_pose(state):
    row = ledge(state)
    state.player.bg_check_flags = 0
    with pytest.raises(ValueError):
        ObservedDescentTask.create(state, row, now=0)
