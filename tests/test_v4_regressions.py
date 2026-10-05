import math
import time

import pytest

from zelda_ai.autonomy.controller import ContinuousController, Setpoint
from zelda_ai.autonomy.execution import ExecutionSupervisor
from zelda_ai.autonomy.features import camera_relative_stick, goal_guidance
from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.autonomy.navigation import observed_local_path
from zelda_ai.autonomy.room_map import RoomMapMemory
from zelda_ai.autonomy.routes import LearnedRouteGraph
from zelda_ai.bridge import Bridge
from zelda_ai.models import GameEvent, NavigationMeshSnapshot, SceneExitObservation


def controller_for(tmp_path, state):
    bridge = Bridge("x" * 32)
    bridge.state = state
    bridge.last_seen = time.monotonic()
    return ContinuousController(bridge, tmp_path / "policy.pt")


def test_intervention_closes_the_previous_ppo_fragment(tmp_path, state):
    controller = controller_for(tmp_path, state)
    controller._ml_step(state)
    controller._ml_step(state)
    assert controller.rollout
    # A camera/modal override replaced the action after it was sampled.
    controller.pending["trainable"] = False
    controller._ml_step(state)
    assert not controller.rollout, "GAE must not span a reference-controller intervention"
    assert not controller.training_queue.empty()


def test_repeated_camera_cuts_cannot_starve_world_space_movement(tmp_path, state):
    controller = controller_for(tmp_path, state)
    controller.last_guidance = {"active": True, "world_yaw": 0.0, "strength": 1.0}
    controller.pending = {"policy_stick": [0.0, 0.0], "guidance_strength": 1.0}
    controller.last_setpoint = Setpoint(stick_y=80, reason="ml_policy")
    neutral_ticks = 0
    for index in range(40):
        state.camera_input_yaw = 0 if index % 2 == 0 else 0x4000
        controller._refresh_camera_relative_setpoint(state)
        neutral_ticks += int(controller.last_setpoint.stick_x == controller.last_setpoint.stick_y == 0)
    assert neutral_ticks <= 4, "Camera chatter must use bounded hysteresis, then reproject"


def test_observed_exit_cooldown_also_excludes_its_remembered_alias(tmp_path, state):
    controller = controller_for(tmp_path, state)
    state.scene_exits = [SceneExitObservation(exit_index=1, entrance_index=2,
        position=(0.0, 0.0, 200.0), samples=4, direct_reachable=True)]
    controller.room_map.observe(state)
    first = controller._escape_waypoint(state)
    controller.escape_retry_after[first["escape_key"]] = time.monotonic() + 20.0
    assert controller._escape_waypoint(state) is None, "Memory must not bypass the same portal cooldown"


@pytest.mark.parametrize("interruption", ["death", "restart"])
def test_respawn_and_new_instance_do_not_create_successful_portals(tmp_path, state, interruption):
    memory = RoomMapMemory(tmp_path / "rooms.json")
    memory.observe(state)
    if interruption == "death":
        dead = state.model_copy(deep=True)
        dead.game_over_state = 1
        dead.player.health = 0
        memory.observe(dead)
    destination = state.model_copy(deep=True)
    destination.scene += 1
    destination.scene_epoch += 1
    destination.seq += 10
    if interruption == "restart":
        destination.instance_id = "new-process"
    memory.observe(destination)
    assert memory.stats()["transitions"] == 0, "Respawn/restart is not a traversed directed edge"


def test_reference_button_suppression_is_excluded_even_when_stick_is_weakened(tmp_path, state):
    controller = controller_for(tmp_path, state)
    assert not controller._ppo_transition_trainable(interaction_override=False,
        guidance={"forced_escape": True}, sample={"guidance_strength": 0.3})


def test_local_path_uses_only_observed_directed_links_and_can_detour(state):
    # A U-shaped observed corridor: the first step must move away from target.
    state.navmesh = NavigationMeshSnapshot(step=80.0, half_extent=2, cells=[
        (0, 0, 0.0, 1 << 0), (0, 1, 0.0, 1 << 2),
        (1, 1, 0.0, 1 << 2), (2, 1, 0.0, 1 << 4), (2, 0, 0.0, 0)])
    hint = observed_local_path(state, (160.0, 0.0, 0.0))
    assert hint["waypoint"] == (0.0, 0.0, 80.0)
    assert hint["path_nodes"] == 5
    assert hint["local_path"] and hint["target_gap"] == 0.0
    state.player.position = (160.0, 0.0, 0.0)
    assert observed_local_path(state, (0.0, 0.0, 0.0)) is None


def test_local_path_rejects_another_floor_and_unlinked_cells(state):
    state.navmesh = NavigationMeshSnapshot(step=80.0, half_extent=2,
        cells=[(0, 0, 0.0, 1 << 2), (1, 0, 120.0, 1 << 2), (2, 0, 120.0, 0)])
    assert observed_local_path(state, (160.0, 120.0, 0.0)) is None


def test_backface_diagnostic_does_not_invent_collision_links(state):
    from pydantic import ValidationError

    legacy = NavigationMeshSnapshot(step=80.0, half_extent=1, cells=[(0, 0, 0.0, 0), (1, 0, 0.0, 0)])
    assert legacy.backface_rejections == 0
    state.navmesh = legacy.model_copy(update={"backface_rejections": 24})
    assert observed_local_path(state, (80.0, 0.0, 0.0)) is None
    for invalid in (-1, 8193):
        with pytest.raises(ValidationError):
            NavigationMeshSnapshot(backface_rejections=invalid)


def test_lower_body_diagnostic_preserves_legacy_unknown_and_cannot_create_links(state):
    from pydantic import ValidationError

    legacy = NavigationMeshSnapshot(step=70, half_extent=1, cells=[(0, 0, 0., 0), (1, 0, 21., 0)])
    assert legacy.lower_band_rejections == 0  # Schema default, not a measured legacy count.
    state.navmesh = legacy.model_copy(update={"lower_band_rejections": 25})
    assert observed_local_path(state, (70., 21., 0.)) is None
    for invalid in (-1, 8193):
        with pytest.raises(ValidationError):
            NavigationMeshSnapshot(lower_band_rejections=invalid)


def test_navigation_refinement_diagnostics_do_not_authorize_unobserved_links(state):
    from pydantic import ValidationError

    legacy = NavigationMeshSnapshot(step=70, half_extent=1, cells=[(0, 0, 0., 0)])
    assert legacy.query_us is None and legacy.refined_component_cells == 0
    state.navmesh = NavigationMeshSnapshot(step=35, half_extent=8,
        cells=[(0, 0, 0., 0), (1, 0, 0., 0)], refined_component_cells=3, query_us=2400)
    assert observed_local_path(state, (35., 0., 0.)) is None
    for field, invalid in (("query_us", -1), ("query_us", 60_000_001),
                           ("refined_component_cells", -1), ("refined_component_cells", 5)):
        with pytest.raises(ValidationError):
            NavigationMeshSnapshot(**{field: invalid})


def test_portal_requires_context_change_and_supervisor_bounds_failed_room(state):
    supervisor = ExecutionSupervisor(room_budget_s=4.0)
    hint = {"active": True, "exit_active": True, "target": (0.0, 0.0, 0.0), "escape_key": "portal"}
    supervisor.observe(state, hint, now=10.0)
    assert supervisor.task.phase == "verify"
    assert supervisor.task.result is None
    supervisor.observe(state, hint, now=15.0)
    assert supervisor.state == "blocked"
    assert supervisor.reason == "no_durable_or_macro_progress"
    assert supervisor.incident is not None


def test_new_room_verifies_portal_and_keeps_strategic_contract(state):
    supervisor = ExecutionSupervisor()
    supervisor.observe(state, {"active": True, "exit_active": True,
        "target": (0.0, 0.0, 200.0), "escape_key": "portal"}, now=10.0)
    destination = state.model_copy(deep=True)
    destination.scene += 1
    supervisor.observe(destination, {}, now=11.0)
    assert supervisor.completed == 1
    assert supervisor.task is None


def test_ppo_rejects_old_actor_fragment_without_mutating_weights(tmp_path, state):
    controller = controller_for(tmp_path, state)
    controller.policy.updates = 2
    result = controller.policy.train_rollout([{"actor_version": 1}], bootstrap_value=0.0, bootstrap_done=False)
    assert result["skipped"] == "stale_actor_version"
    assert controller.policy.updates == 2
    assert not controller.policy.checkpoint.exists()


@pytest.mark.parametrize("effect", ["prompt_changed", "inertia"])
def test_interaction_mapping_requires_specific_causal_effect(tmp_path, state, effect):
    controller = controller_for(tmp_path, state)
    controller.pending_interaction_probe = {"kind": "context", "key": "test-context",
        "button": "R", "at": time.monotonic(), "scene": state.scene, "room": state.room,
        "dialogue_active": False, "exit_active": effect == "prompt_changed",
        "traversal_active": effect == "inertia", "player_position": (0.0, 0.0, 0.0),
        "speed_xz": 10.0, "climbing_ladder": False, "climbing_ledge": False,
        "hanging_ledge": False, "known": False}
    if effect == "prompt_changed":
        state.context_action.label = "none"
    else:
        state.player.position = (40.0, 0.0, 0.0)
        state.player.speed_xz = 10.0
    class Reward:
        breakdown = {}
    controller._observe_interaction_outcome(state, Reward())
    assert controller.route_graph.interaction_button("test-context") is None


def test_dialogue_close_does_not_learn_an_undelivered_button(tmp_path, state):
    controller = controller_for(tmp_path, state)
    state.protocol = 3
    old = state.model_copy(deep=True)
    old.dialogue.active = True
    controller.pending_interaction_probe = {"kind": "dialogue", "button": "R", "command_seqs": [12]}
    controller.note_dialogue_closed(old, state)
    assert controller.route_graph.interaction_button("dialogue:advance") is None
    assert controller.dialogue_reentry_guard is not None


def test_modal_state_cannot_revive_a_blocked_executor(state):
    supervisor = ExecutionSupervisor(room_budget_s=1.0)
    supervisor.observe(state, {}, now=10.0)
    supervisor.observe(state, {}, now=12.0)
    assert supervisor.state == "blocked"
    state.pause_menu.active = True
    supervisor.observe(state, {}, now=13.0)
    state.pause_menu.active = False
    supervisor.observe(state, {}, now=14.0)
    assert supervisor.state == "blocked"


def test_modal_loop_has_a_bounded_budget(state):
    supervisor = ExecutionSupervisor()
    state.dialogue.active = True
    supervisor.observe(state, {}, now=10.0)
    supervisor.observe(state, {}, now=311.0)
    assert supervisor.state == "blocked"
    assert supervisor.reason == "modal_without_observed_effect"


def test_age_change_is_not_a_successful_portal(state):
    supervisor = ExecutionSupervisor()
    hint = {"active": True, "exit_active": True, "target": (0.0, 0.0, 200.0), "escape_key": "portal"}
    supervisor.observe(state, hint, now=10.0)
    state.player.age = "adult"
    supervisor.observe(state, {}, now=11.0)
    assert supervisor.completed == 0


def test_proposed_local_path_is_not_a_learned_route(state):
    hint = {"waypoint": (0.0, 0.0, 80.0), "waypoint_id": "local-path",
            "path_nodes": 5, "local_path": True, "confidence": 1.0}
    intent = AgentIntent(objective="Explore", summary="Explore", mode="navigate", target_position=(160.0, 0.0, 0.0))
    guidance = goal_guidance(state, intent, route_hint=hint)
    assert guidance["local_path_active"]
    assert not guidance["route_active"]


@pytest.mark.parametrize("mirrored", [False, True])
@pytest.mark.parametrize("heading", [-2.1, -0.8, 0.4, 1.5])
def test_projection_inverts_native_soh_stick_angle(state, mirrored, heading):
    state.camera_input_yaw = 0x3000
    state.mirrored_world = mirrored
    x, y = camera_relative_stick(state, state.player, heading)
    # Pinned SoH func_80077D10: Math_Atan2S(relY, -relX), with
    # mirrored-world relX inversion, followed by Camera_GetInputDirYaw.
    rel_x = -x if mirrored else x
    observed_heading = math.atan2(-rel_x, y) + state.camera_input_yaw * math.pi / 32768.0
    error = (observed_heading - heading + math.pi) % (2 * math.pi) - math.pi
    assert error == pytest.approx(0.0, abs=1e-7)


def test_same_process_save_load_does_not_create_a_traversed_edge(tmp_path, state):
    graph = LearnedRouteGraph(tmp_path / "routes.json")
    graph.observe(state)
    state.player.position = (240.0, 0.0, 0.0)
    state.events = [GameEvent(id="load-2", kind="save_loaded")]
    graph.observe(state)
    assert not any(graph.edges.values())
    # A repeated event must not erase a subsequent legitimate traversal.
    state.player.position = (360.0, 0.0, 0.0)
    graph.observe(state)
    assert any(graph.edges.values())


@pytest.mark.parametrize("boundary", ["save_loaded", "age_change"])
def test_non_portal_world_changes_do_not_enter_room_transition_memory(tmp_path, state, boundary):
    memory = RoomMapMemory(tmp_path / "rooms.json")
    memory.observe(state)
    state.scene += 1
    if boundary == "save_loaded":
        state.events = [GameEvent(id="load-2", kind="save_loaded")]
    else:
        state.player.age = "adult"
    memory.observe(state)
    assert memory.stats()["transitions"] == 0


def test_local_expansion_does_not_reset_campaign_clock(state):
    supervisor = ExecutionSupervisor(room_budget_s=1000)
    supervisor.observe(state, {}, durable_progress=True, now=10.0)
    supervisor.observe(state, {}, macro_progress=True, now=611.0)
    assert supervisor.reason == "campaign_loop_without_progress"


def test_blocked_motor_wins_over_modal_button_probes(tmp_path, state):
    controller = controller_for(tmp_path, state)
    controller._ml_step(state)
    controller.executor.block("test_block", time.monotonic())
    state.dialogue.active = True
    state.dialogue.can_advance = True
    setpoint = controller._ml_step(state)
    assert setpoint.buttons == setpoint.stick_x == setpoint.stick_y == 0
    assert setpoint.reason == "motor_blocked"


def test_moving_mesh_origin_does_not_drag_the_committed_waypoint(tmp_path, state):
    controller = controller_for(tmp_path, state)
    cells = [(0, 0, 0.0, 1 << 0), (0, 1, 0.0, 1 << 0), (0, 2, 0.0, 0)]
    state.navmesh = NavigationMeshSnapshot(step=80.0, half_extent=2, cells=cells)
    first = controller._local_path_hint(state, None, (0.0, 0.0, 400.0))
    assert first["waypoint"] == (0.0, 0.0, 80.0)
    state.player.position = (0.0, 0.0, 20.0)
    state.navmesh.origin = (0.0, 0.0, 20.0)
    state.full_seq += 1
    second = controller._local_path_hint(state, None, (0.0, 0.0, 400.0))
    assert second["waypoint"] == first["waypoint"]
    assert second["waypoint_id"] == first["waypoint_id"]
    state.navmesh.cells = [(0, 0, 0.0, 0), (0, 1, 0.0, 0)]
    assert controller._local_path_hint(state, None, (0.0, 0.0, 400.0)) is None
    assert controller.local_path_attempt is None
