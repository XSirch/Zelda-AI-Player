from pathlib import Path

import pytest

from zelda_ai.models import GameState, NavigationProbe
from zelda_ai.navmesh import _probe_direction, plan_navmesh, primitive_move_safe, waypoint_probe_safe


def with_mesh(state, cells, *, origin=(0, 0, 0), step=70.0, half_extent=4, probes=None):
    payload = state.model_dump()
    payload["navmesh"] = {
        "origin": origin,
        "step": step,
        "half_extent": half_extent,
        "cells": cells,
    }
    if probes is not None:
        payload["navigation_probes"] = probes
    return GameState.model_validate(payload)


def test_astar_straight_path_uses_connected_lookahead(state):
    game = with_mesh(state, [
        [0, 0, 0, 0x01],
        [0, 1, 0, 0x11],
        [0, 2, 0, 0x10],
    ])
    plan = plan_navmesh(game, (0, 0, 200))
    assert plan is not None
    assert plan.path == ((0, 0), (0, 1), (0, 2))
    assert plan.waypoint == pytest.approx((0, 0, 140))
    assert plan.exact_goal_reachable


def test_astar_routes_around_blocked_direct_corridor(state):
    # Direct north cell (0,1) is absent. The only connected route goes east,
    # north twice, then west to the target-side cell.
    game = with_mesh(state, [
        [0, 0, 0, 0x04],
        [1, 0, 0, 0x41],
        [1, 1, 0, 0x11],
        [1, 2, 0, 0x50],
        [0, 2, 0, 0x04],
    ])
    plan = plan_navmesh(game, (0, 0, 140))
    assert plan is not None
    assert plan.path == ((0, 0), (1, 0), (1, 1), (1, 2), (0, 2))
    assert plan.waypoint == pytest.approx((70, 0, 0))


def test_same_cell_micro_approach_keeps_probe_guard(state):
    game = with_mesh(state, [[0, 0, 0, 0]])
    plan = plan_navmesh(game, (25, 0, 0))
    assert plan is not None
    assert plan.waypoint == pytest.approx((25, 0, 0))
    assert plan.path == ((0, 0),)


def test_primitive_move_fails_closed_without_player_telemetry(state):
    game = state.model_copy(update={"player": None, "capabilities": ["local_navmesh"]})
    assert not primitive_move_safe(game, "forward")


def test_v26_fails_closed_if_realtime_probes_disappear(state):
    game = type(state).model_validate({
        **state.model_dump(),
        "capabilities": ["local_navmesh"],
        "navigation_probes": [],
    })
    assert not waypoint_probe_safe(game, (0, 0, 70))
    assert not primitive_move_safe(game, "forward")


def test_astar_never_invents_connection_to_disconnected_target(state):
    game = with_mesh(state, [
        [0, 0, 0, 0x00],
        [0, 2, 0, 0x00],
    ])
    assert plan_navmesh(game, (0, 0, 140)) is None


def test_reciprocal_links_are_required(state):
    # A stale/partial one-way bit must fail closed instead of being traversed.
    game = with_mesh(state, [
        [0, 0, 0, 0x01],
        [0, 1, 0, 0x00],
    ])
    assert plan_navmesh(game, (0, 0, 70)) is None


def test_realtime_probe_can_veto_stale_navmesh_waypoint(state):
    safe = {
        "direction": "forward", "distance": 70, "floor_found": True,
        "floor_y": 0, "delta_y": 0, "floor_type": 0,
        "wall_hit": False, "wall_distance": None, "wall_flags": 0,
    }
    game = with_mesh(state, [[0, 0, 0, 0]], probes=[safe])
    assert waypoint_probe_safe(game, (0, 0, 70))

    blocked = {**safe, "wall_hit": True, "wall_distance": 20}
    game = with_mesh(state, [[0, 0, 0, 0]], probes=[blocked])
    assert not waypoint_probe_safe(game, (0, 0, 70))

    cliff = {**safe, "floor_found": True, "delta_y": -100}
    game = with_mesh(state, [[0, 0, 0, 0]], probes=[cliff])
    assert not waypoint_probe_safe(game, (0, 0, 70))


def test_known_exit_target_does_not_require_floor_beyond_threshold(state):
    beyond_exit = {
        "direction": "forward", "distance": 70, "floor_found": False,
        "floor_y": None, "delta_y": None, "floor_type": None,
        "wall_hit": False, "wall_distance": None, "wall_flags": 0,
    }
    game = type(state).model_validate({
        **state.model_dump(),
        "capabilities": ["local_navmesh", "probe_yaw_v2", "scene_exit_surfaces"],
        "navigation_probes": [beyond_exit],
    })
    target = (0.0, game.player.floor_height, 25.0)
    assert not waypoint_probe_safe(game, target)
    assert waypoint_probe_safe(game, target, known_floor_target=True)


def test_known_exit_target_still_rejects_intervening_wall(state):
    wall = {
        "direction": "forward", "distance": 70, "floor_found": False,
        "floor_y": None, "delta_y": None, "floor_type": None,
        "wall_hit": True, "wall_distance": 20, "wall_flags": 0,
    }
    game = type(state).model_validate({
        **state.model_dump(),
        "capabilities": ["local_navmesh", "probe_yaw_v2", "scene_exit_surfaces"],
        "navigation_probes": [wall],
    })
    target = (0.0, game.player.floor_height, 25.0)
    assert not waypoint_probe_safe(game, target, known_floor_target=True)


def test_known_exit_target_rejects_unsafe_height_change(state):
    safe_direction = {
        "direction": "forward", "distance": 70, "floor_found": False,
        "floor_y": None, "delta_y": None, "floor_type": None,
        "wall_hit": False, "wall_distance": None, "wall_flags": 0,
    }
    game = type(state).model_validate({
        **state.model_dump(),
        "capabilities": ["local_navmesh", "probe_yaw_v2", "scene_exit_surfaces"],
        "navigation_probes": [safe_direction],
    })
    target = (0.0, game.player.floor_height + 40.0, 25.0)
    assert not waypoint_probe_safe(game, target, known_floor_target=True)


def test_probe_blocks_wall_before_full_navmesh_waypoint(state):
    wall = {
        "direction": "forward", "distance": 70, "floor_found": True,
        "floor_y": 0, "delta_y": 0, "floor_type": 0,
        "wall_hit": True, "wall_distance": 60, "wall_flags": 0,
    }
    game = with_mesh(state, [[0, 0, 0, 0]], probes=[wall])
    assert not waypoint_probe_safe(game, (0, 0, 70))


def test_micro_waypoint_allows_wall_safely_beyond_body_margin(state):
    wall = {
        "direction": "forward", "distance": 70, "floor_found": True,
        "floor_y": 0, "delta_y": 0, "floor_type": 0,
        "wall_hit": True, "wall_distance": 48, "wall_flags": 0,
    }
    game = with_mesh(state, [[0, 0, 0, 0]], probes=[wall])
    assert waypoint_probe_safe(game, (0, 0, 25))


def test_long_probe_blocks_roll_before_distant_wall(state):
    safe70 = {
        "direction": "forward", "distance": 70, "floor_found": True,
        "floor_y": 0, "delta_y": 0, "floor_type": 0,
        "wall_hit": False, "wall_distance": None, "wall_flags": 0,
    }
    blocked140 = {
        **safe70,
        "distance": 140,
        "wall_hit": True,
        "wall_distance": 95,
    }
    game = with_mesh(state, [[0, 0, 0, 0]], probes=[safe70, blocked140])
    assert primitive_move_safe(game, "forward")
    assert not primitive_move_safe(
        game, "forward", probe_distance=145.0, wall_clearance=120.0)


def test_probe_direction_matches_player_stick_direction_enum(state):
    game = state.model_copy(update={"player": state.player.model_copy(update={"yaw": 0}),
        "capabilities": ["probe_yaw_v2"]})
    assert _probe_direction(game, (70, 0, 0)) == "left"
    assert _probe_direction(game, (-70, 0, 0)) == "right"


def test_primitive_left_and_right_validate_the_matching_probe(state):
    left_unsafe = {
        "direction": "left", "distance": 70, "floor_found": False,
        "floor_y": None, "delta_y": None, "floor_type": None,
        "wall_hit": False, "wall_distance": None, "wall_flags": 0,
    }
    right_safe = {
        "direction": "right", "distance": 70, "floor_found": True,
        "floor_y": 0, "delta_y": 0, "floor_type": 0,
        "wall_hit": False, "wall_distance": None, "wall_flags": 0,
    }
    game = type(state).model_validate({
        **state.model_dump(),
        "camera_input_yaw": 0,
        "capabilities": ["probe_yaw_v2"],
        "navigation_probes": [left_unsafe, right_safe],
    })
    assert not primitive_move_safe(game, "left")
    assert primitive_move_safe(game, "right")


def test_camera_relative_primitive_move_uses_world_heading_probe(state):
    right = {
        "direction": "right", "distance": 70, "floor_found": False,
        "floor_y": None, "delta_y": None, "floor_type": None,
        "wall_hit": False, "wall_distance": None, "wall_flags": 0,
    }
    game = state.model_copy(update={
        "camera_eye": (0.0, 0.0, -10.0),
        "camera_at": (0.0, 0.0, 0.0),
        "capabilities": ["probe_yaw_v2"],
        "navigation_probes": [NavigationProbe.model_validate(right)],
    })
    assert not primitive_move_safe(game, "right")


def test_navmesh_contract_rejects_duplicate_cells(state):
    payload = state.model_dump()
    payload["navmesh"] = {
        "origin": [0, 0, 0], "step": 70, "half_extent": 4,
        "cells": [[0, 0, 0, 0], [0, 0, 0, 0]],
    }
    with pytest.raises(ValueError):
        GameState.model_validate(payload)


def test_native_bridge_exposes_navmesh_only_as_slow_state():
    native = (Path(__file__).parents[1] / "native" / "ZeldaAiBridge.cpp").read_text(encoding="utf-8")
    assert 'BRIDGE_BUILD = "rt-input-v2.10"' in native
    assert '"forward", "forward_left", "left", "back_left"' in native
    assert '"back", "back_right", "right", "forward_right"' in native
    assert '"local_navmesh"' in native
    assert '"probe_yaw_v2"' in native
    assert '"scene_exit_surfaces"' in native
    assert "SurfaceType_GetSceneExitIndex" in native
    assert '"scene_exits"' in native
    assert "json NavigationMesh(Player* player)" in native
    assert "EDGE_FLOOR_SAMPLES = 4" in native
    assert "sampleIndex <= EDGE_FLOOR_SAMPLES" in native
    assert "EXIT_SCAN_STEP = 35.0f" in native
    assert "EXIT_SCAN_HALF_EXTENT = HALF_EXTENT * 2" in native
    assert "EXIT_INTERIOR_BLEND = 0.65f" in native
    assert "CollisionPoly_GetVerticesByBgId" in native
    assert "centroid" in native
    assert "targetX = x + (centroid.x - x) * EXIT_INTERIOR_BLEND" in native
    assert "DIRECT_FLOOR_SAMPLES = 4" in native
    assert '"direct_reachable"' in native
    assert '"traversal_affordances_v1"' in native
    assert '"story_progress_v1"' in native
    assert '"scene_autosave_v1"' in native
    assert "json TraversalAffordances(Player* player)" in native
    assert "DIRECTION_COUNT = 16" in native
    assert "MAX_RADIUS = 280.0f" in native
    assert "WALL_FLAG_LADDER" in native
    assert "WALL_FLAG_LADDER_TOP" in native
    assert "WALL_FLAG_CLIMBABLE" in native
    assert '"stairs_or_slope_down"' in native
    assert '"climbable_wall_up"' in native
    assert "Flags_GetInfTable(INFTABLE_GREETED_BY_SARIA)" in native
    assert "Flags_GetEventChkInf(EVENTCHKINF_SHOWED_MIDO_SWORD_SHIELD)" in native
    assert "GameInteractor::OnGameFrameUpdate" in native
    assert "TrySceneAutosave" in native
    assert "Play_PerformSave(gPlayState)" in native
    assert "scene_autosave_pending" in native
    assert "scene_autosave_completed" in native



    assert '"nearby_actors", "traversal_affordances", "scene_exits", "navmesh", "autosave"' in native
    assert 'state["navmesh"] = {{"origin", {0.0f, 0.0f, 0.0f}}' in native
