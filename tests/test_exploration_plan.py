import pytest

from zelda_ai.autonomy.exploration_plan import ObservedExplorationPlan
from zelda_ai.models import (
    GameEvent,
    NavigationMeshSnapshot,
    SceneExitObservation,
    TraversalAffordanceObservation,
)


def ground(state):
    state.player.bg_check_flags = 1
    state.player.floor_height = 0
    state.player.speed_xz = 0
    state.camera_input_yaw = 0
    state.navmesh = NavigationMeshSnapshot(step=70, half_extent=4,
        cells=[(0, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 68), (3, 0, 0., 64)])


def test_frontier_prefers_unoccupied_current_region_and_never_marks_proposed_target_visited(state):
    ground(state)
    plan = ObservedExplorationPlan()
    task = plan.choose(state, budget_s=12, now=0)
    assert task.target == (210, 0, 0) and task.kind == "observed_cell"
    assert task.corridor == ((70, 0, 0), (140, 0, 0), (210, 0, 0))
    assert len(plan.visits) == 1 and plan.region(task.context, task.target) not in plan.visits
    state.player.position = (210, 0, 0)
    state.seq += 1
    state.navmesh.origin = state.player.position
    next_task = plan.choose(state, budget_s=12, now=1)
    assert next_task.target == (420, 0, 0) and len(plan.visits) == 2


def test_native_portal_competes_without_a_destination_lookup(state):
    ground(state)
    state.scene_exits = [SceneExitObservation(exit_index=1, entrance_index=123,
        position=(0, 0, 70), direct_reachable=True, samples=1)]
    task = ObservedExplorationPlan().choose(state, budget_s=20, now=0)
    assert task.kind == "observed_portal" and task.exit_index == 1


def test_new_room_holds_back_entry_exit_and_prefers_observed_unvisited_descent(state):
    ground(state)
    plan = ObservedExplorationPlan()
    plan.observe(state, now=0)
    state.scene += 1
    state.seq += 1
    state.scene_exits = [SceneExitObservation(exit_index=1, entrance_index=123,
        position=(0, 0, 70), direct_reachable=True, samples=1)]
    state.traversal_affordances = [TraversalAffordanceObservation(kind="ledge_down", direction="down",
        approach_position=(0, 0, 0), target_position=(70, -100, 0), distance=0, height_delta=-100)]
    task = plan.choose(state, budget_s=20, now=1)
    assert task.kind == "observed_descent_approach" and task.target == (70, -100, 0)
    assert plan.actual_transitions == 1


def test_vertical_choice_prefers_current_upper_approach_before_a_farther_frontier(state):
    ground(state)
    state.player.position = (0, 100, 0)
    state.player.floor_height = 100
    state.navmesh.origin = state.player.position
    state.navmesh.cells = [(0, 0, 100., 4), (1, 0, 100., 4), (2, 0, 100., 0)]
    near = TraversalAffordanceObservation(kind="ledge_down", direction="down", distance=0,
        approach_position=(0, 100, 0), target_position=(0, -80, 70), height_delta=-180)
    farther = TraversalAffordanceObservation(kind="ledge_down", direction="down", distance=70,
        approach_position=(70, 100, 0), target_position=(140, -80, 0), height_delta=-180)
    state.traversal_affordances = [farther, near]
    task = ObservedExplorationPlan().choose(state, budget_s=20, now=0)
    assert task.target == near.target_position


def test_failed_target_is_cooled_and_success_heals_negative_working_memory(state):
    ground(state)
    plan = ObservedExplorationPlan()
    task = plan.choose(state, budget_s=12, now=0)
    task.phase, task.failure = "failed", "no_geometric_progress"
    plan.outcome(task, success=False, now=1)
    replacement = plan.choose(state, budget_s=12, now=2)
    assert replacement.target != task.target
    assert plan.failures[plan.task_key(task)] == 1
    plan.outcome(task, success=True, now=25)
    assert plan.failures[plan.task_key(task)] == 0


def test_failed_composed_descent_is_remembered_after_expected_approach_interrupt(state):
    ground(state)
    state.traversal_affordances = [TraversalAffordanceObservation(kind="ledge_down", direction="down",
        approach_position=(0, 0, 0), target_position=(70, -100, 0), distance=0, height_delta=-100)]
    plan = ObservedExplorationPlan()
    task = plan.choose(state, budget_s=20, now=0)
    task.interrupt("unsupported_locomotor_mode")
    # The approach yielded to the attached profile; that complete local
    # attempt then failed. Its expected handoff must not erase failure cost.
    plan.outcome(task, success=False, now=1)
    assert plan.failures[plan.task_key(task)] == 1


def test_modal_interrupt_does_not_blame_observed_geometry(state):
    ground(state)
    plan = ObservedExplorationPlan()
    task = plan.choose(state, budget_s=12, now=0)
    task.interrupt("modal_owns_control")
    plan.outcome(task, success=False, now=1)
    assert plan.failures[plan.task_key(task)] == 0


def test_unlinked_cells_and_rearward_paths_do_not_replace_observed_forward_frontier(state):
    ground(state)
    state.navmesh.cells = [(0, 0, 0., 17), (0, 1, 0., 1), (0, 2, 0., 0),
                           (0, -1, 0., 16), (0, -2, 0., 0), (4, 0, 0., 0)]
    task = ObservedExplorationPlan().choose(state, budget_s=12, now=0)
    assert task.target == (0, 0, 140)


def test_fine_movement_and_camera_churn_do_not_create_new_macro_footprints(state):
    ground(state)
    plan = ObservedExplorationPlan()
    plan.observe(state, now=0)
    for seq in range(11, 40):
        state.seq, state.camera_input_yaw = seq, seq*10
        state.player.position = (seq, 0, 0)
        plan.observe(state, now=seq/10)
    assert len(plan.visits) == 1 and next(iter(plan.visits.values())) == 1


def test_reload_does_not_count_as_a_physically_traversed_portal(state):
    ground(state)
    plan = ObservedExplorationPlan()
    plan.observe(state, now=0)
    state.seq += 1
    state.scene += 1
    state.events = [GameEvent(id="new-load", kind="save_loaded")]
    plan.observe(state, now=1)
    assert plan.actual_transitions == 0


@pytest.mark.parametrize("mode", ["modal", "water", "ladder"])
def test_unsupported_mode_does_not_choose_ground_frontiers(state, mode):
    ground(state)
    plan = ObservedExplorationPlan()
    plan.choose(state, budget_s=12, now=0)
    state.seq += 1
    if mode == "modal":
        state.dialogue.active = True
    elif mode == "water":
        state.player.state_flags_1 |= 1 << 27
    else:
        state.player.climbing_ladder = True
    with pytest.raises(ValueError, match="modal_requires|unsupported_planning_mode"):
        plan.choose(state, budget_s=12, now=1)
    assert plan.selection["observation_seq"] == state.seq
    assert plan.selection["eligible_floor_candidates"] == 0
