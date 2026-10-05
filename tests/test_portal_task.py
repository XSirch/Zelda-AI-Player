import pytest

from zelda_ai.autonomy.imitation import encode_surface
from zelda_ai.autonomy.local_tasks import LocalTask
from zelda_ai.autonomy.portal_task import ObservedPortalTask
from zelda_ai.models import GameEvent, NavigationMeshSnapshot, SceneExitObservation


def observed(state, *, direct=False):
    state.player.position = (0, 0, 0)
    state.player.floor_height = 0
    state.player.bg_check_flags = 1
    state.player.speed_xz = 0
    state.camera_input_yaw = 0
    state.navmesh = NavigationMeshSnapshot(origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 1), (0, 1, 0., 0)])
    row = SceneExitObservation(exit_index=1, entrance_index=4, position=(0, 0, 160),
                               samples=2, direct_reachable=direct)
    state.scene_exits = [row]
    return row


def test_partial_portal_approach_advances_only_through_new_collision_observations(state):
    row = observed(state)
    task = ObservedPortalTask.create(state, row, now=0)
    assert task.steering_point(state) == (0, 0, 70)
    assert encode_surface(state, task) == encode_surface(state, LocalTask.observed_cell(state, (0, 0, 70)))
    state.seq += 1
    state.player.position = (0, 0, 60)
    state.navmesh = NavigationMeshSnapshot(origin=(0, 0, 60), step=70, half_extent=1,
        cells=[(0, 0, 0., 1), (0, 1, 0., 0)])
    task.observe(state, consumed=True, now=1)
    assert task.steering_point(state) == (0, 0, 130)
    assert task.phase == "execute" and task.corridor_replans == 2
    assert task.target == row.position


def test_portal_proximity_does_not_finish_or_brake_and_transition_must_settle(state):
    row = observed(state, direct=True)
    task = ObservedPortalTask.create(state, row, now=0)
    state.player.position = (0, 0, 140)
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1 + i * .1)
    assert task.phase == "execute" and task.reference_stick(state) != (0, 0)
    state.scene += 1
    state.scene_epoch += 1
    state.player.position = (100, 0, 20)
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=False, now=1 + i * .1)
        assert task.reference_stick(state) == (0, 0)
        assert (task.phase == "succeeded") is (i == 2)
    assert task.crossing_context[1] == state.scene


@pytest.mark.parametrize("invalid", ("same_epoch", "no_input", "far_from_exit", "reload", "other_instance", "death"))
def test_unrelated_context_change_never_counts_as_portal_traversal(state, invalid):
    row = observed(state, direct=True)
    task = ObservedPortalTask.create(state, row, now=0)
    state.player.position = (0, 0, 140) if invalid != "far_from_exit" else (0, 0, 0)
    state.seq += 1
    task.observe(state, consumed=invalid != "no_input", now=.1)
    state.scene += 1
    if invalid != "same_epoch":
        state.scene_epoch += 1
    if invalid == "reload":
        state.events = [GameEvent(id="new-load", kind="save_loaded", detail="1")]
    if invalid == "other_instance":
        state.instance_id = "another-process"
    if invalid == "death":
        state.events = [GameEvent(id="new-death", kind="player_died", detail="")]
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=False, now=1 + i * .1)
    assert task.phase == "interrupted" and task.phase != "succeeded"


def test_exhausted_partial_approach_never_becomes_a_collision_blind_exit_target(state):
    task = ObservedPortalTask.create(state, observed(state), now=0)
    state.player.position = (0, 0, 70)
    state.seq += 1
    state.navmesh = NavigationMeshSnapshot(origin=(0, 0, 70), step=70, half_extent=1,
                                          cells=[(0, 0, 0., 0)])
    task.observe(state, consumed=True, now=1)
    assert task.phase == "failed" and task.failure == "observed_approach_exhausted"
    assert task.reference_stick(state) == (0, 0)


def test_observed_connected_detour_can_temporarily_move_away_from_exit(state):
    row = observed(state)
    row.position = (0, 0, -210)
    # The only observed path goes around an obstacle. Every reachable endpoint
    # initially lies farther from the exit; continuing straight is unproven.
    state.navmesh = NavigationMeshSnapshot(origin=(0, 0, 0), step=70, half_extent=2,
        cells=[(0, 0, 0., 1), (0, 1, 0., 4), (1, 1, 0., 4), (2, 1, 0., 16), (2, 0, 0., 0)])
    task = ObservedPortalTask.create(state, row, now=0)
    assert task.steering_point(state) == (0, 0, 70)
    assert task.target == (0, 0, -210)
    assert task.corridor[-1] == (140, 0, 0)
    assert task.phase == "execute"


def test_portal_loading_releases_control_but_does_not_reset_its_budget(state):
    task = ObservedPortalTask.create(state, observed(state, direct=True), now=0, budget_s=3)
    state.in_game, state.player = False, None
    state.seq += 1
    task.observe(state, consumed=True, now=1)
    assert task.phase == "transition" and task.reference_stick(state) == (0, 0)
    assert task.guidance(state)["active"] is False
    task.observe(state, consumed=False, now=3)
    assert task.phase == "failed" and task.failure == "attempt_timeout"


def test_portal_creation_requires_current_native_exit_and_grounded_dry_control(state):
    row = observed(state, direct=True)
    state.scene_exits = []
    with pytest.raises(ValueError, match="currently observed"):
        ObservedPortalTask.create(state, row)
    state.scene_exits = [row]
    state.player.state_flags_1 = 1 << 27
    with pytest.raises(ValueError, match="dry ground"):
        ObservedPortalTask.create(state, row)
