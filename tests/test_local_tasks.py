from dataclasses import replace

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
    state.player.floor_height = 14
    state.player.bg_check_flags = 1
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


@pytest.mark.parametrize("reason", ["local_task", "dialogue_disengage"])
def test_nonblocking_local_policy_response_is_applied_between_ml_samples(state, tmp_path, reason):
    class Bridge:
        def command_consumed(self, seq):
            return True
    class Policy:
        refresh_at_motor_cadence = True
        response = (0, 0)
        def __call__(self, game, task):
            return self.response
    bridge = Bridge()
    bridge.state = state
    controller = ContinuousController(bridge, tmp_path / "policy.pt")
    policy = Policy()
    controller.start_local_task(LocalTask.traversal(state, stair(state)), stick_policy=policy)
    controller._ml_step(state)
    assert controller.last_setpoint.stick_x == 0
    controller.last_setpoint = replace(controller.last_setpoint, reason=reason)
    policy.response = (-40, 20)  # Worker completes between residual-policy samples.
    controller._refresh_camera_relative_setpoint(state)
    assert (controller.last_setpoint.stick_x, controller.last_setpoint.stick_y) == (-40, 20)
    assert controller.last_setpoint.buttons == 0
    assert controller.last_setpoint.reason == reason
    assert controller.pending["trainable"] is False
    assert controller.pending["stick"] == [-.5, .25]
    controller.local_task.phase = "verify"
    controller._refresh_camera_relative_setpoint(state)
    assert (controller.last_setpoint.stick_x, controller.last_setpoint.stick_y) == (0, 0)


def test_observed_cell_uses_current_body_contact_when_thin_probes_miss_the_wall(state):
    from zelda_ai.autonomy.features import PROBE_NAMES
    from zelda_ai.models import NavigationMeshSnapshot, NavigationProbe

    state.player.yaw = -32768
    state.player.wall_yaw = 0
    state.player.bg_check_flags = 1 | 8
    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0.0, 16 | 4 | 64), (0, -1, 0.0, 0),
               (1, 0, 0.0, 0), (-1, 0, 0.0, 0)],
    )
    state.navigation_probes = [
        NavigationProbe(direction=name, distance=70, floor_found=True, floor_y=0, delta_y=0)
        for name in PROBE_NAMES
    ]
    task = LocalTask.observed_cell(state, (0, 0, -70), now=0)
    task.observe(state, consumed=True, now=.1)
    assert task.detour is not None
    assert abs(task.detour[0]) == pytest.approx(70)
    assert abs(task.detour[2]) < 1e-4  # A currently probed tangent, not into the wall.


def test_ground_detour_cannot_turn_an_unsupported_raised_probe_into_a_walking_target(state):
    from zelda_ai.models import NavigationMeshSnapshot, NavigationProbe

    state.navmesh = NavigationMeshSnapshot(origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 1), (0, 1, 0., 0)])
    state.player.yaw = 0
    state.navigation_probes = [
        NavigationProbe(direction="forward", distance=70, floor_found=True, floor_y=0, delta_y=0, wall_hit=True),
        NavigationProbe(direction="forward_left", distance=70, floor_found=True, floor_y=54, delta_y=54),
    ]
    task = LocalTask.observed_cell(state, (0, 0, 70), now=0)
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert task.detour is None


@pytest.mark.parametrize("linked", [False, True])
def test_ground_detour_requires_directed_links_and_can_choose_a_supported_alternative(state, linked):
    from zelda_ai.models import NavigationMeshSnapshot, NavigationProbe

    state.navmesh = NavigationMeshSnapshot(origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 1 | (4 if linked else 0)), (0, 1, 0., 0), (1, 0, 0., 0)])
    state.player.yaw = 0
    state.navigation_probes = [
        NavigationProbe(direction="forward", distance=70, floor_found=True, floor_y=0, delta_y=0, wall_hit=True),
        NavigationProbe(direction="forward_left", distance=70, floor_found=True, floor_y=54, delta_y=54),
        NavigationProbe(direction="left", distance=70, floor_found=True, floor_y=0, delta_y=0),
    ]
    task = LocalTask.observed_cell(state, (0, 0, 70), now=0)
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert task.detour == ((70., 0., 0.) if linked else None)


@pytest.mark.parametrize("contact,target", ((1, (0, 0, -70)), (1 | 8, (0, 0, 70))))
def test_body_contact_does_not_redirect_stale_wall_data_or_motion_away_from_wall(state, contact, target):
    from zelda_ai.autonomy.features import PROBE_NAMES
    from zelda_ai.models import NavigationMeshSnapshot, NavigationProbe

    state.player.yaw = -32768
    state.player.wall_yaw = 0
    state.player.wall_flags = 1  # A retained surface flag alone is not current contact.
    state.player.bg_check_flags = contact
    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0.0, 17), (0, -1, 0.0, 0), (0, 1, 0.0, 0)],
    )
    state.navigation_probes = [
        NavigationProbe(direction=name, distance=70, floor_found=True, floor_y=0, delta_y=0)
        for name in PROBE_NAMES
    ]
    task = LocalTask.observed_cell(state, target, now=0)
    task.observe(state, consumed=True, now=.1)
    assert task.detour is None


def test_observed_corridor_keeps_its_unreached_world_waypoint_when_mesh_recenters(state):
    from zelda_ai.models import NavigationMeshSnapshot

    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 16), (0, -1, 0., 64), (-1, -1, 0., 0)],
    )
    task = LocalTask.observed_cell(state, (-70, 0, -70), now=0)
    task.observe(state, consumed=True, now=.1)
    assert task.steering_point(state) == (0, 0, -70)
    state.player.position = (0, 0, -30)
    state.seq += 1
    # Native mesh origin follows Link; these newly sampled cells do not mean
    # that the previous observed world waypoint has been reached.
    state.navmesh.origin = state.player.position
    task.observe(state, consumed=True, now=1)
    assert task.steering_point(state) == (0, 0, -70)
    assert task.target == (-70, 0, -70)


def test_observed_corridor_progress_can_temporarily_increase_final_target_distance(state):
    from zelda_ai.models import NavigationMeshSnapshot

    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 16), (0, -1, 0., 64),
               (-1, -1, 0., 1), (-1, 0, 0., 0)],
    )
    task = LocalTask.observed_cell(state, (-70, 0, 0), now=0)
    task.observe(state, consumed=True, now=.1)
    assert task.steering_point(state) == (0, 0, -70)
    state.player.position = (0, 0, -42)
    state.seq += 1
    task.observe(state, consumed=True, now=2.6)
    assert task.phase == "execute"
    assert task.failure is None
    assert task.target == (-70, 0, 0)


def test_observed_corridor_checks_collision_toward_next_waypoint_not_final_target(state):
    from zelda_ai.autonomy.features import PROBE_NAMES
    from zelda_ai.models import NavigationMeshSnapshot, NavigationProbe

    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 16), (0, -1, 0., 64),
               (-1, -1, 0., 1), (-1, 0, 0., 0)],
    )
    state.navigation_probes = [
        NavigationProbe(direction=name, distance=70, floor_found=True, floor_y=0,
                        delta_y=0, wall_hit=name == "right")
        for name in PROBE_NAMES
    ]
    task = LocalTask.observed_cell(state, (-70, 0, 0), now=0)
    task.observe(state, consumed=True, now=.1)
    assert task.detour is None
    assert task.steering_point(state) == (0, 0, -70)


@pytest.mark.parametrize("fresh,consumed", ((False, True), (True, False)))
def test_observed_corridor_requires_fresh_consumed_evidence_for_progress(state, fresh, consumed):
    from zelda_ai.models import NavigationMeshSnapshot

    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 16), (0, -1, 0., 64),
               (-1, -1, 0., 1), (-1, 0, 0., 0)],
    )
    task = LocalTask.observed_cell(state, (-70, 0, 0), now=0)
    task.observe(state, consumed=True, now=.1)
    state.player.position = (0, 0, -42)
    state.seq += int(fresh)
    task.observe(state, consumed=consumed, now=2.6)
    assert task.failure == "no_geometric_progress"


def test_observed_corridor_revisits_do_not_renew_best_waypoint_progress(state):
    from zelda_ai.models import NavigationMeshSnapshot

    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 16), (0, -1, 0., 64),
               (-1, -1, 0., 1), (-1, 0, 0., 0)],
    )
    task = LocalTask.observed_cell(state, (-70, 0, 0), now=0)
    for seq, z, now in ((11, -30, .5), (12, -5, 1), (13, -30, 3.1)):
        state.seq, state.player.position = seq, (0, 0, z)
        task.observe(state, consumed=True, now=now)
    assert task.failure == "no_geometric_progress"
    assert task.progress_at == .5


def test_observed_probe_detour_is_not_released_before_rounding_its_endpoint(state):
    from zelda_ai.autonomy.features import PROBE_NAMES
    from zelda_ai.models import NavigationMeshSnapshot, NavigationProbe

    state.player.yaw = -32768
    state.player.wall_yaw = 0
    state.player.bg_check_flags = 1 | 8
    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 16 | 4 | 64), (0, -1, 0., 0),
               (1, 0, 0., 0), (-1, 0, 0., 0)],
    )
    state.navigation_probes = [
        NavigationProbe(direction=name, distance=70, floor_found=True, floor_y=0, delta_y=0)
        for name in PROBE_NAMES
    ]
    task = LocalTask.observed_cell(state, (0, 0, -70), now=0)
    task.observe(state, consumed=True, now=.1)
    point = task.detour
    state.player.position = (point[0] * 5 / 7, point[1], point[2] * 5 / 7)
    state.seq += 1
    state.player.bg_check_flags = 1
    task.observe(state, consumed=True, now=.5)
    assert task.detour == point  # Twenty units short of an intermediate corner.


def test_observed_held_detour_reacts_to_new_body_contact_instead_of_pushing_into_wall(state):
    from zelda_ai.autonomy.features import PROBE_NAMES
    from zelda_ai.models import NavigationMeshSnapshot, NavigationProbe

    state.player.yaw = -32768
    state.player.wall_yaw = 0
    state.player.bg_check_flags = 1 | 8
    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 16 | 4 | 64), (0, -1, 0., 0),
               (1, 0, 0., 0), (-1, 0, 0., 0)],
    )
    state.navigation_probes = [
        NavigationProbe(direction=name, distance=70, floor_found=True, floor_y=0, delta_y=0)
        for name in PROBE_NAMES
    ]
    task = LocalTask.observed_cell(state, (0, 0, -70), now=0)
    task.observe(state, consumed=True, now=.1)
    previous = task.detour
    state.player.position = (previous[0] / 3, 0, 0)
    state.player.wall_yaw = -16384 if previous[0] > 0 else 16384
    state.seq += 1
    task.observe(state, consumed=True, now=.5)
    assert task.detour != previous
    assert task.steering_point(state) == task.target
    assert task.progress_at == .5  # Prior physical gain, not the replacement.


def test_observed_detour_handoff_respects_native_body_clearance_at_a_wall(state):
    from zelda_ai.models import NavigationMeshSnapshot

    state.navmesh = NavigationMeshSnapshot(
        origin=(0, 0, 0), step=70, half_extent=1,
        cells=[(0, 0, 0., 1), (0, 1, 0., 0)],
    )
    task = LocalTask.observed_cell(state, (0, 0, 70), now=0)
    task.detour = (70, 0, 0)
    task._track_waypoint(state)
    state.player.position = (53, 0, 0)
    state.player.bg_check_flags = 1 | 8
    state.player.wall_yaw = -16384
    state.seq += 1
    task.observe(state, consumed=True, now=.5)
    assert task.detour is None
