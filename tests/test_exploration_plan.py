import pytest

from zelda_ai.autonomy.exploration_plan import ObservedExplorationPlan
from zelda_ai.models import (
    ActorObservation,
    GameEvent,
    NavigationMeshSnapshot,
    SceneExitObservation,
    TraversalAffordanceObservation,
)


def chest(state, *, drawn=True, uid="observed-container"):
    return ActorObservation(actor_uid=uid, actor_id=999, category=10, category_name="chest",
        params=0, position=(180, 0, 0), distance=180, drawn=drawn)


def test_current_drawn_container_precedes_frontier_and_failure_cools_actor_identity(state):
    ground(state)
    state.nearby_actors = [chest(state)]
    plan = ObservedExplorationPlan(contextual_interactions=True)
    task = plan.choose(state, budget_s=20, now=0)
    assert task.kind == "observed_container_approach"
    assert task.actor_uid == "observed-container" and task.target == (180, 0, 0)
    assert plan.region(task.context, task.target) not in plan.visits
    task.phase, task.failure = "failed", "container_no_observed_prompt"
    plan.outcome(task, success=False, now=1)
    assert plan.choose(state, budget_s=12, now=2).kind == "observed_cell"


def test_drawn_collectible_with_current_floor_approach_is_attempted_before_random_frontier(state):
    ground(state)
    actor = ActorObservation(actor_uid="current-collectible", actor_id=21, category=6,
        category_name="misc", name="En_Item00", description="Collectibles", params=-1,
        position=(180, 1, 0), distance=180, drawn=True)
    state.nearby_actors = [actor]
    plan = ObservedExplorationPlan(contextual_interactions=True)
    task = plan.choose(state, budget_s=20, now=0)
    assert task.kind == "observed_collectible_approach"
    assert task.actor_uid == actor.actor_uid


def test_repeated_optional_collectible_failure_yields_to_current_exit_and_cools_uid(state):
    ground(state)
    actor = ActorObservation(actor_uid="current-collectible", actor_id=21, category=6,
        category_name="misc", name="En_Item00", params=-1,
        position=(180, 1, 0), distance=180, drawn=True)
    state.nearby_actors = [actor]
    state.scene_exits = [SceneExitObservation(exit_index=1, entrance_index=123,
        position=(70, 0, 0), direct_reachable=True, samples=1)]
    plan = ObservedExplorationPlan(contextual_interactions=True)
    for start in [0, 22]:
        task = plan.choose(state, budget_s=20, now=start)
        assert task.actor_uid == actor.actor_uid
        task.phase, task.failure = "failed", "collectible_no_observed_gain"
        plan.outcome(task, success=False, now=start+1)
        assert plan.choose(state, budget_s=20, now=start+2).kind == "observed_portal"
    assert plan.choose(state, budget_s=20, now=44).kind == "observed_portal"


def test_current_visible_open_lid_is_skipped_without_fabricating_an_open_event(state):
    ground(state)
    state.nearby_actors = [chest(state).model_copy(update={"container_lid_pose": "open"})]
    plan = ObservedExplorationPlan(contextual_interactions=True)
    assert plan.choose(state, budget_s=20, now=0).kind == "observed_cell"
    assert not plan.opened_containers and not plan.failures
    assert plan.selection["visibly_open_containers_skipped"] == 1


@pytest.mark.parametrize("pose", ["closed", "unknown"])
def test_unknown_or_closed_lid_retains_observed_container_attempt(state, pose):
    ground(state)
    state.nearby_actors = [chest(state).model_copy(update={"container_lid_pose": pose})]
    plan = ObservedExplorationPlan(contextual_interactions=True)
    assert plan.choose(state, budget_s=20, now=0).kind == "observed_container_approach"


def test_repeated_container_failures_yield_to_an_observed_native_exit(state):
    ground(state)
    state.nearby_actors = [chest(state)]
    state.scene_exits = [SceneExitObservation(exit_index=1, entrance_index=123,
        position=(70, 0, 0), direct_reachable=True, samples=1)]
    plan = ObservedExplorationPlan(contextual_interactions=True)
    for start, end in [(0, 1), (22, 23)]:
        task = plan.choose(state, budget_s=20, now=start)
        assert task.kind == 'observed_container_approach'
        task.phase, task.failure, task.consumed = 'failed', 'container_no_observed_prompt', True
        plan.outcome(task, success=False, now=end)
    assert plan.choose(state, budget_s=20, now=44).kind == 'observed_portal'


@pytest.mark.parametrize('alternative', ['no_exit', 'new_actor'])
def test_container_failures_preserve_unknown_interactions_and_other_actor_identity(state, alternative):
    ground(state)
    state.nearby_actors = [chest(state)]
    plan = ObservedExplorationPlan(contextual_interactions=True)
    for start, end in [(0, 1), (22, 23)]:
        task = plan.choose(state, budget_s=20, now=start)
        task.phase, task.failure, task.consumed = 'failed', 'container_no_observed_prompt', True
        plan.outcome(task, success=False, now=end)
    if alternative == 'new_actor':
        state.nearby_actors.append(chest(state, uid='newly_observed'))
        state.scene_exits = [SceneExitObservation(exit_index=1, entrance_index=123,
            position=(70, 0, 0), direct_reachable=True, samples=1)]
    chosen = plan.choose(state, budget_s=20, now=44)
    assert chosen.kind == 'observed_container_approach'
    assert chosen.actor_uid == ('newly_observed' if alternative == 'new_actor' else 'observed-container')


@pytest.mark.parametrize("reason", ["not_drawn", "room_only", "unlinked", "raised"])
def test_container_selection_requires_current_drawn_and_reachable_floor(state, reason):
    ground(state)
    actor = chest(state, drawn=reason != "not_drawn")
    if reason == "room_only":
        state.room_actors = [actor]
    else:
        state.nearby_actors = [actor]
    if reason == "unlinked":
        state.navmesh.cells = [(0, 0, 0., 0), (3, 0, 0., 0)]
    if reason == "raised":
        actor.position = (180, 60, 0)
    plan = ObservedExplorationPlan(contextual_interactions=True)
    try:
        task = plan.choose(state, budget_s=12, now=0)
    except ValueError:
        return
    assert task.kind != "observed_container_approach"


def ground(state):
    state.player.bg_check_flags = 1
    state.player.floor_height = 0
    state.player.speed_xz = 0
    state.camera_input_yaw = 0
    state.navmesh = NavigationMeshSnapshot(step=70, half_extent=4,
        cells=[(0, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 68), (3, 0, 0., 64)])


def stationary_departure(state):
    from zelda_ai.autonomy.local_tasks import LocalTask
    ground(state)
    state.navmesh.cells = [(0, 0, 0., 65), (0, 1, 0., 1), (0, 2, 0., 1),
                          (0, 3, 0., 0), (-1, 0, 0., 0)]
    task = LocalTask.observed_cell(state, (0, 0, 210), now=0)
    plan = ObservedExplorationPlan()
    plan.observe(state, now=0)
    plan.visits[plan.region(task.context, (-70, 0, 0))] = 1
    state.seq += 1
    task.observe(state, consumed=True, now=3)
    assert task.failure == 'no_geometric_progress'
    plan.observe(state, now=3)
    return plan, task


def test_stationary_failed_departure_is_not_retried_under_another_target(state):
    plan, task = stationary_departure(state)
    plan.outcome(task, success=False, now=3)
    next_task = plan.choose(state, budget_s=12, now=4)
    assert next_task.steering_point(state) != (0, 0, 70)
    assert next_task.target == (-70, 0, 0)
    assert plan.selection['cooled_departure_candidates'] > 0


@pytest.mark.parametrize('missing', ['consumption', 'fresh_state', 'stationary_origin',
                                   'held_first_point', 'first_segment', 'physical_failure'])
def test_departure_memory_requires_a_consumed_stationary_first_attempt(state, missing):
    plan, task = stationary_departure(state)
    if missing == 'consumption':
        task.consumed = False
    elif missing == 'fresh_state':
        plan.last_seq = task.origin_seq
    elif missing == 'stationary_origin':
        plan.last_position = (0, 0, 40)
    elif missing == 'held_first_point':
        task.progress_point = (0, 0, 140)
    elif missing == 'first_segment':
        task.waypoint_index = 1
    else:
        task.phase, task.failure = 'interrupted', 'modal_owns_control'
    plan.outcome(task, success=False, now=3)
    assert not plan.failed_departures
    assert plan.choose(state, budget_s=12, now=4).steering_point(state) == (0, 0, 70)


def test_departure_cooldown_expires_without_creating_traversed_routes(state):
    plan, task = stationary_departure(state)
    plan.outcome(task, success=False, now=3)
    assert plan.choose(state, budget_s=12, now=4).target == (-70, 0, 0)
    assert plan.choose(state, budget_s=12, now=24).steering_point(state) == (0, 0, 70)
    assert len(plan.visits) == 2  # Only the observed origin and seeded earlier west visit.
    assert plan.departure_deferrals > 0


@pytest.mark.parametrize('changed', ['instance', 'scene', 'room', 'age', 'mirror', 'floor', 'origin'])
def test_failed_departure_is_local_to_the_observed_physical_context(state, changed):
    from dataclasses import replace
    plan, task = stationary_departure(state)
    plan.outcome(task, success=False, now=3)
    context = list(task.context)
    if changed in {'instance', 'scene', 'room', 'age', 'mirror'}:
        index = {'instance': 0, 'scene': 1, 'room': 2, 'mirror': 3, 'age': 4}[changed]
        context[index] = 'other' if index in {0, 4} else not context[index] if index == 3 else context[index]+1
        candidate = replace(task, context=tuple(context))
    elif changed == 'floor':
        candidate = replace(task, origin=(0, 10, 0), corridor=((0, 10, 70),))
    else:
        candidate = replace(task, origin=(40, 0, 0), corridor=((0, 0, 70),))
    assert not plan.departure_deferred(candidate, now=4)


def test_observed_portal_is_not_suppressed_by_failed_horizontal_departure(state):
    plan, task = stationary_departure(state)
    plan.outcome(task, success=False, now=3)
    state.scene_exits = [SceneExitObservation(exit_index=1, entrance_index=123,
        position=(0, 0, 70), direct_reachable=True, samples=1)]
    next_task = plan.choose(state, budget_s=20, now=4)
    assert next_task.kind == 'observed_portal'
    assert not plan.departure_deferred(next_task, now=4)


def test_verified_physical_reach_heals_a_failed_departure(state):
    from zelda_ai.autonomy.local_tasks import LocalTask
    plan, task = stationary_departure(state)
    plan.outcome(task, success=False, now=3)
    successful = LocalTask.observed_cell(state, (0, 0, 70), now=4)
    state.player.position = (0, 0, 70)
    for now in [5, 5.05, 5.1]:
        state.seq += 1
        successful.observe(state, consumed=True, now=now)
        plan.observe(state, now=now)
    assert successful.phase == 'succeeded'
    plan.outcome(successful, success=True, now=5.1)
    assert not plan.failed_departures


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


def upward_surface(state, *, height=18, linked=True):
    ground(state)
    state.navmesh.cells = [(0, 0, 0., 5 if linked else 4), (0, 1, float(height), 0),
                           (1, 0, 0., 4), (2, 0, 0., 4), (3, 0, 0., 0)]
    state.traversal_affordances = [TraversalAffordanceObservation(
        kind="stairs_or_slope_up", direction="up", distance=0,
        approach_position=(0, 0, 0), target_position=(0, height, 70), height_delta=height)]


@pytest.mark.parametrize("height", [18, 38])
def test_current_linked_upward_floor_precedes_horizontal_frontier_with_real_height_postcondition(state, height):
    upward_surface(state, height=height)
    plan = ObservedExplorationPlan()
    task = plan.choose(state, budget_s=20, now=0)
    assert task.target == (0, height, 70)
    assert task.kind == "observed_cell" and task.corridor == ((0, height, 70),)
    assert plan.selection["eligible_upward_candidates"] == 1
    # A small rise may share the macro-region with its base. That must not
    # erase the observed traversal or reset the macro dwell clock.
    if height == 18:
        assert plan.region(task.context, task.target) in plan.visits
    state.player.position = (0, 0, 70)
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1+i/10)
    assert not task.terminal
    state.player.position, state.player.floor_height = (0, height, 70), height
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.4+i/10)
    assert task.phase == "succeeded" and task.verification_frames == 3


@pytest.mark.parametrize("missing", ["directed_link", "matching_floor", "affordance"])
def test_upward_priority_needs_current_floor_and_directed_path_not_just_a_surface_label(state, missing):
    upward_surface(state, height=38, linked=missing != "directed_link")
    if missing == "matching_floor":
        state.navmesh.cells[1] = (0, 1, 0., 0)
    elif missing == "affordance":
        state.traversal_affordances = []
    plan = ObservedExplorationPlan()
    assert plan.choose(state, budget_s=20, now=0).target == (210, 0, 0)
    assert plan.selection["eligible_upward_candidates"] == 0


def test_actually_occupied_upper_floor_loses_vertical_priority_without_new_macro_expansion(state):
    upward_surface(state)
    plan = ObservedExplorationPlan()
    plan.observe(state, now=0)
    state.seq += 1
    state.player.position, state.player.floor_height = (0, 18, 70), 18
    plan.observe(state, now=1)
    state.seq += 1
    state.player.position, state.player.floor_height = (0, 0, 0), 0
    assert plan.choose(state, budget_s=20, now=2).target == (210, 0, 0)
    assert plan.selection["eligible_upward_candidates"] == 0
    assert len(plan.visits) == 1 and plan.actual_transitions == 0


def test_upward_proposal_or_airborne_pass_does_not_mark_the_landing_occupied(state):
    upward_surface(state)
    plan = ObservedExplorationPlan()
    first = plan.choose(state, budget_s=20, now=0)
    assert not plan.upward_floor_visited(first.context, first.target)
    state.seq += 1
    state.player.position, state.player.floor_height = (0, 18, 70), 18
    state.player.bg_check_flags = 0
    plan.observe(state, now=1)
    assert not plan.upward_floor_visited(first.context, first.target)
    state.seq += 1
    state.player.position, state.player.floor_height, state.player.bg_check_flags = (0, 0, 0), 0, 1
    assert plan.choose(state, budget_s=20, now=2).target == first.target


def test_failed_supported_upward_attempt_cools_down_before_horizontal_recovery(state):
    upward_surface(state)
    plan = ObservedExplorationPlan()
    first = plan.choose(state, budget_s=20, now=0)
    first.phase, first.failure = "failed", "no_geometric_progress"
    plan.outcome(first, success=False, now=1)
    next_task = plan.choose(state, budget_s=20, now=2)
    assert next_task.target == (210, 0, 0)
    assert plan.selection["eligible_upward_candidates"] == 0
    assert not plan.upward_floor_visited(first.context, first.target)


def test_contextual_mode_prioritizes_current_prompt_and_linear_text_without_ground_frontier(state):
    ground(state)
    state.context_action.code, state.context_action.label = 1, "check"
    plan = ObservedExplorationPlan(contextual_interactions=True)
    task = plan.choose(state, budget_s=20, now=0)
    assert task.kind == "observed_context_interaction"
    assert len(plan.visits) == 1
    plan.outcome(task, success=True, now=1)
    assert not plan.context_available(state, now=2)
    assert plan.choose(state, budget_s=12, now=2).kind == "observed_cell"
    state.seq += 1
    state.dialogue.active = True
    state.navmesh.cells = []
    assert plan.choose(state, budget_s=20, now=3).kind == "observed_linear_dialogue"


def test_verified_visible_read_defers_same_actor_but_not_another_or_durable_progress(state):
    from zelda_ai.autonomy.context_tasks import LinearDialogueTask
    from zelda_ai.models import DialogueState
    ground(state)
    actor = ActorObservation(actor_uid="reader", actor_id=999, category=4, category_name="npc",
        params=0, position=(100, 0, 0), distance=100, drawn=True)
    state.context_actor = actor
    state.context_action.code, state.context_action.label = 7, "speak"
    plan = ObservedExplorationPlan(contextual_interactions=True)
    plan.observe(state, now=0)
    state.dialogue = DialogueState(active=True, text_id=42, text="observed page",
        message_mode=53, text_visible=True, visible_bytes=13, can_advance=True, speaker=actor)
    task = LinearDialogueTask.create(state, now=0)
    state.dialogue = DialogueState()
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1+i/10)
    assert task.phase == "succeeded"
    plan.outcome(task, success=True, now=1)
    assert plan.choose(state, budget_s=12, now=30).kind == "observed_cell"
    assert plan.choose(state, budget_s=12, now=121).kind == "observed_context_interaction"
    state.context_actor = actor.model_copy(update={"actor_uid": "other"})
    assert plan.choose(state, budget_s=12, now=31).kind == "observed_context_interaction"
    state.context_actor = actor
    state.seq += 1
    state.progress.owned_equipment = ["new observed equipment"]
    assert plan.choose(state, budget_s=12, now=32).kind == "observed_context_interaction"


@pytest.mark.parametrize("missing_proof", ["visibility", "speaker", "complete_page", "consumption"])
def test_incomplete_dialogue_evidence_does_not_defer_actor(state, missing_proof):
    from zelda_ai.autonomy.context_tasks import LinearDialogueTask
    from zelda_ai.models import DialogueState
    ground(state)
    actor = ActorObservation(actor_uid="reader", actor_id=999, category=4, category_name="npc",
        params=0, position=(100, 0, 0), distance=100, drawn=True)
    state.context_actor = actor
    state.context_action.code, state.context_action.label = 7, "speak"
    plan = ObservedExplorationPlan(contextual_interactions=True)
    plan.observe(state, now=0)
    state.dialogue = DialogueState(active=True, text_id=42, text="observed page", message_mode=53,
        text_visible=None if missing_proof == "visibility" else True, visible_bytes=13,
        can_advance=missing_proof != "complete_page", speaker=None if missing_proof == "speaker" else actor)
    task = LinearDialogueTask.create(state, now=0)
    state.dialogue = DialogueState()
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=missing_proof != "consumption", now=.1+i/10)
    plan.outcome(task, success=task.phase == "succeeded", now=1)
    assert not plan.read_actor_contexts
    assert plan.choose(state, budget_s=12, now=30).kind == "observed_context_interaction"


def test_dialogue_escape_guard_can_suppress_context_while_preserving_walking(state):
    ground(state)
    state.context_action.code, state.context_action.label = 15, "speak"
    plan = ObservedExplorationPlan(contextual_interactions=True)
    assert plan.choose(state, budget_s=12, now=0, suppress_context=True).kind == "observed_cell"


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


def test_only_linked_short_fine_step_can_escape_without_new_macro_progress(state):
    ground(state)
    state.player.position = (-83, 0, -53)
    state.navmesh = NavigationMeshSnapshot(origin=state.player.position, step=35, half_extent=8,
        cells=[(-1, 0, 0., 4), (0, 0, 0., 64), (2, 0, 0., 4), (3, 0, 0., 64)])
    plan = ObservedExplorationPlan()
    task = plan.choose(state, budget_s=12, now=0)
    assert task.kind == "observed_cell" and task.target == (-118, 0, -53)
    assert task.corridor == ((-118, 0, -53),)
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1+i/10)
    assert task.phase != "verify" and not task.terminal
    # Move within the same macro-region: this is an escape step, not expansion.
    state.seq += 1
    state.player.position = (-84, 0, -53)
    plan.observe(state, now=1)
    assert len(plan.visits) == 1
    state.seq += 1
    state.player.position = (-118, 0, -53)
    plan.observe(state, now=2)
    assert len(plan.visits) == 1
    task.phase, task.failure = "failed", "no_geometric_progress"
    plan.outcome(task, success=False, now=2)
    with pytest.raises(ValueError, match="no_eligible_current_collision_task"):
        plan.choose(state, budget_s=12, now=3)


def test_short_step_completes_only_after_consumed_geometric_gain_and_stopped_frames(state):
    ground(state)
    state.navmesh = NavigationMeshSnapshot(step=35, half_extent=8,
        cells=[(0, 0, 0., 4), (1, 0, 0., 64)])
    task = ObservedExplorationPlan().choose(state, budget_s=12, now=0)
    state.player.position = (26, 0, 0)
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=False, now=.1+i/10)
    assert not task.terminal and task.verification_frames == 0
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.4+i/10)
    assert task.phase == "succeeded" and task.verification_frames == 3


def test_short_step_does_not_replace_available_longer_frontier(state):
    ground(state)
    state.navmesh = NavigationMeshSnapshot(step=35, half_extent=8,
        cells=[(0, 0, 0., 68), (-1, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 64)])
    task = ObservedExplorationPlan().choose(state, budget_s=12, now=0)
    assert task.target == (70, 0, 0)


def test_unlinked_short_cell_cannot_supply_an_escape(state):
    ground(state)
    state.navmesh = NavigationMeshSnapshot(step=35, half_extent=8,
        cells=[(0, 0, 0., 0), (-1, 0, 0., 4)])
    with pytest.raises(ValueError, match="no_eligible_current_collision_task"):
        ObservedExplorationPlan().choose(state, budget_s=12, now=0)


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
