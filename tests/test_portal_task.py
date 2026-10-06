import pytest

from zelda_ai.autonomy.imitation import encode_surface
from zelda_ai.autonomy.local_tasks import LocalTask
from zelda_ai.autonomy.portal_task import ObservedPortalTask
from zelda_ai.models import GameEvent, NavigationMeshSnapshot, SceneExitObservation


def test_cutscene_arbiter_holds_portal_without_running_its_ground_observer(state, tmp_path):
    import asyncio
    import time

    from zelda_ai.autonomy.controller import ContinuousController
    from zelda_ai.bridge import Bridge

    row = observed(state, direct=True)
    task = ObservedPortalTask.create(state, row)
    bridge = Bridge("x" * 32)
    bridge.state, bridge.last_seen = state, time.monotonic()
    sent, snapshots = [], []

    def send(**packet):
        sent.append(packet)
        bridge.command_seq += 1
        return bridge.command_seq

    bridge.send, bridge.release = send, lambda: None
    controller = ContinuousController(bridge, tmp_path / "policy.pt", training_enabled=False)
    controller.start_local_task(task, stick_policy=lambda game, owned: (0, 60))
    state.cutscene_active = True
    state.seq += 1
    asyncio.run(controller.run(lambda: not snapshots, lambda: snapshots.append(task.snapshot())))
    assert snapshots[0]["failure"] is None and snapshots[0]["phase"] == "execute"
    assert sent == [{"buttons": 0, "stick_x": 0, "stick_y": 0, "lease_ms": 150}]


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


def refinement_boundary(state):
    row = observed(state)
    state.protocol = 3
    state.capabilities = ['navmesh_refinement']
    state.full_seq = state.seq
    task = ObservedPortalTask.create(state,row,now=0)
    state.player.position = (0,0,70)
    state.seq += 1
    state.full_seq = state.seq
    state.navmesh = NavigationMeshSnapshot(origin=(0,0,70),step=70,half_extent=1,
        cells=[(0,0,0.,0)])
    task.observe(state,consumed=True,now=1)
    return task


def test_exhausted_current_coarse_portal_requests_neutral_refinement_before_failing(state):
    task = refinement_boundary(state)
    assert task.phase == 'collision_refinement' and task.failure is None
    assert task.reference_stick(state) == (0,0)
    assert task.guidance(state)['active'] is False
    assert task.deadline == 30 and task.progress_at == 1


def test_fresh_owned_fine_collision_can_resume_same_bounded_portal_without_synthesized_path(state):
    task = refinement_boundary(state)
    assert task.phase == 'collision_refinement'
    state.seq += 1
    state.full_seq = state.seq
    state.navmesh = NavigationMeshSnapshot(origin=(0,0,70),step=35,half_extent=2,
        cells=[(0,0,0.,1),(0,1,0.,1),(0,2,0.,0)],refinement_request_id=task.origin_seq)
    task.observe(state,consumed=False,now=1.2)
    assert task.phase == 'execute' and task.failure is None
    assert task.steering_point(state) == (0,0,105)
    assert task.target == (0,0,160) and task.deadline == 30
    assert task.progress_at == 1


@pytest.mark.parametrize('change',['old_snapshot','wrong_request','coarse'])
def test_unowned_or_old_collision_cannot_resume_a_waiting_portal(state,change):
    task = refinement_boundary(state)
    state.seq += 1
    if change != 'old_snapshot':
        state.full_seq = state.seq
    state.navmesh = NavigationMeshSnapshot(origin=(0,0,70),step=70 if change=='coarse' else 35,
        half_extent=2,cells=[(0,0,0.,1),(0,1,0.,0)],
        refinement_request_id=task.origin_seq+1 if change=='wrong_request' else task.origin_seq)
    task.observe(state,consumed=False,now=1.2)
    assert task.phase == 'collision_refinement' and task.reference_stick(state)==(0,0)


def test_fine_query_without_real_continuation_fails_without_second_query_or_wall_steering(state):
    task = refinement_boundary(state)
    state.seq += 1
    state.full_seq = state.seq
    state.navmesh = NavigationMeshSnapshot(origin=(0,0,70),step=35,half_extent=2,
        cells=[(0,0,0.,0)],refinement_request_id=task.origin_seq)
    task.observe(state,consumed=False,now=1.2)
    assert task.phase=='failed' and task.failure=='observed_approach_exhausted'
    assert task.collision_refinement_count==1 and task.reference_stick(state)==(0,0)


@pytest.mark.parametrize('expired',['wait','geometric_progress','task_deadline'])
def test_collision_wait_preserves_original_progress_and_deadline_budgets(state,expired):
    task = refinement_boundary(state)
    if expired=='geometric_progress':
        task.progress_at = -2
    now=30 if expired=='task_deadline' else 2
    task.observe(state,consumed=False,now=now)
    assert task.phase=='failed'
    assert task.failure=={'wait':'collision_refinement_timeout','geometric_progress':'no_geometric_progress',
        'task_deadline':'attempt_timeout'}[expired]


def test_motor_refresh_does_not_call_laya_or_apply_old_stick_while_collision_query_owns_wait(state,tmp_path):
    import time

    from zelda_ai.autonomy.controller import ContinuousController, Setpoint
    from zelda_ai.bridge import Bridge

    task = refinement_boundary(state)
    class Policy:
        refresh_at_motor_cadence = True
        def __call__(self,*args):
            raise AssertionError('A waiting query cannot invoke a movement candidate')
    bridge = Bridge('x'*32)
    bridge.state,bridge.last_seen = state,time.monotonic()
    controller = ContinuousController(bridge,tmp_path/'policy.pt',training_enabled=False)
    controller.local_task,controller.local_stick_policy = task,Policy()
    controller.last_setpoint = Setpoint(stick_x=60,reason='local_task')
    controller._refresh_camera_relative_setpoint(state)
    assert controller.last_setpoint.stick_x==controller.last_setpoint.stick_y==0


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
