import pytest

from zelda_ai.autonomy.controller import ContinuousController
from zelda_ai.autonomy.local_tasks import LocalTask
from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.models import TraversalAffordanceObservation


def stair(state):
    row = TraversalAffordanceObservation(kind="stairs_or_slope_up", direction="up",
        approach_position=state.player.position, target_position=(0, 14, 70),
        distance=0, height_delta=14)
    state.traversal_affordances = [row]
    return row


def test_traversal_requires_current_observation(state):
    row = stair(state)
    state.traversal_affordances = []
    with pytest.raises(ValueError, match="observed"):
        LocalTask.traversal(state, row, now=0)


def test_raised_sample_over_flat_walkable_cells_is_not_a_valid_walking_surface(state):
    from zelda_ai.models import NavigationMeshSnapshot
    row = stair(state)
    state.navmesh = NavigationMeshSnapshot(origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 1), (0, 1, 0., 0)])
    with pytest.raises(ValueError, match="supported"):
        LocalTask.traversal(state, row)


def test_coarse_mesh_does_not_discard_observed_descent_between_grid_samples(state):
    from zelda_ai.models import NavigationMeshSnapshot
    state.player.position = (0, 14, 0)
    row = TraversalAffordanceObservation(kind="stairs_or_slope_down", direction="down",
        approach_position=state.player.position, target_position=(64, 0, -27),
        distance=0, height_delta=-14)
    state.traversal_affordances = [row]
    state.navmesh = NavigationMeshSnapshot(origin=(0, 14, 0), step=70, half_extent=1,
        cells=[(0, 0, 14., 1), (0, -1, 0., 0)])
    assert LocalTask.traversal(state, row).kind == "stairs_or_slope_down"


def test_surface_failure_cools_down_without_creating_edges_and_success_heals(state, tmp_path):
    from zelda_ai.autonomy.routes import LearnedRouteGraph
    graph = LearnedRouteGraph(tmp_path / "route.json")
    row = stair(state)
    graph.record_surface_outcome(state, row.target_position, success=False)
    assert graph.traversal_waypoint(state) is None
    assert sum(graph.frontier_failures.values()) == 1
    assert graph.edges == {}
    graph.save(force=True)
    read_only = LearnedRouteGraph(tmp_path / "route.json", writable=False)
    before = (tmp_path / "route.json").read_bytes()
    read_only.record_surface_outcome(state, row.target_position, success=True)
    read_only.save(force=True)
    assert (tmp_path / "route.json").read_bytes() == before
    graph.record_surface_outcome(state, row.target_position, success=True)
    assert graph.frontier_failures == {}
    assert graph.traversal_waypoint(state) is not None


def test_success_requires_height_and_consumed_input_and_fresh_verification(state):
    task = LocalTask.traversal(state, stair(state), now=0)
    state.player.position = (0, 0, 70)
    state.seq += 1
    task.observe(state, consumed=True, now=1)
    assert task.phase == "execute"
    state.player.position = (0, 14, 70)
    task.observe(state, consumed=False, now=1.1)
    assert task.phase != "succeeded"
    for index in range(3):
        state.seq += 1
        task.observe(state, consumed=False, now=1.2 + index * .1)
    assert task.phase == "succeeded"


def test_near_target_without_any_consumed_input_cannot_succeed(state):
    task = LocalTask.traversal(state, stair(state), now=0)
    state.player.position = (0, 14, 70)
    for index in range(4):
        state.seq += 1
        task.observe(state, consumed=False, now=.1 + index * .1)
    assert task.phase != "succeeded"


def test_context_reset_and_modal_interrupt_do_not_move_underneath(state):
    task = LocalTask.traversal(state, stair(state), now=0)
    state.dialogue.active = True
    task.observe(state, consumed=True, now=1)
    assert task.phase == "interrupted"
    assert task.failure == "modal_owns_control"
    assert task.reference_stick(state) == (0, 0)
    state.dialogue.active = False
    task = LocalTask.traversal(state, stair(state), now=0)
    state.scene_epoch += 1
    task.observe(state, consumed=True, now=1)
    assert task.failure == "context_changed"


def test_stall_is_bounded_and_does_not_count_button_churn(state):
    task = LocalTask.traversal(state, stair(state), now=0)
    state.seq += 1
    task.observe(state, consumed=True, now=3)
    assert task.phase == "failed"
    assert task.failure == "no_geometric_progress"


def test_walking_task_rejects_ladder_mode_instead_of_using_camera_projection(state):
    task = LocalTask.traversal(state, stair(state), now=0)
    state.player.climbing_ladder = True
    task.observe(state, consumed=True, now=.1)
    assert task.failure == "unsupported_locomotor_mode"
    assert task.reference_stick(state) == (0, 0)


def test_active_physical_task_preserves_objective_and_excludes_ppo_and_context_buttons(state, tmp_path):
    class Bridge:
        def command_consumed(self, seq):
            return False
    bridge = Bridge()
    bridge.state = state
    controller = ContinuousController(bridge, tmp_path / "policy.pt")
    intent = AgentIntent.bootstrap()
    controller.set_intent(intent)
    task = LocalTask.traversal(state, stair(state))
    controller.start_local_task(task)
    controller.route_graph.next_waypoint = lambda *a, **kw: pytest.fail("planner stole task control")
    controller.route_graph.exploration_waypoint = lambda *a, **kw: pytest.fail("exploration stole task control")
    controller._interaction_override = lambda *a: pytest.fail("context button stole surface control")
    expected = task.reference_stick(state)
    actual = controller._ml_step(state)
    assert (actual.stick_x, actual.stick_y) == expected
    assert actual.buttons == 0
    assert controller.pending["trainable"] is False
    assert controller.intent == intent
    # Repeating the actor step must not append an overridden action to PPO.
    state.seq += 1
    controller._ml_step(state)
    assert controller.rollout == []


def test_external_stick_candidate_has_full_authority_and_no_reference_blend(state, tmp_path):
    class Bridge:
        def command_consumed(self, seq):
            return True
    bridge = Bridge()
    bridge.state = state
    controller = ContinuousController(bridge, tmp_path / "policy.pt")
    controller.start_local_task(LocalTask.traversal(state, stair(state)), stick_policy=lambda *a: (23, -41))
    actual = controller._ml_step(state)
    assert (actual.stick_x, actual.stick_y) == (23, -41)
    state.camera_input_yaw = 16384
    controller._refresh_camera_relative_setpoint(state)
    assert controller.last_setpoint == actual
    assert controller.pending["trainable"] is False
