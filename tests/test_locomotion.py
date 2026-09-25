from zelda_ai.models import Decision, SkillArgs
from zelda_ai.skills.common import _dodge_direction_safe, _player_relative_stick, _probe_stick, _rotate_stick_quadrants, _world_yaw_stick
import zelda_ai.skills.interactions as interactions
from zelda_ai.runtime import _aim_error, _aim_stick, _best_climb_surface_probe, _best_traversal_probe, _door_intent_actor, _equipment_point, _inventory_slot, _matching_actor, _menu_grid_directions, _recovery_inputs, _steer_to, _traversal_intent_direction, controller_input
from zelda_ai.skills.navigation import _resolve_scene_exit
from zelda_ai.skills.traversal import _resolve_traversal_affordance, _refresh_traversal_affordance, _local_traversal_evidence, _fast_revalidation_matches_original


def decision(skill, direction, duration=700, strength=0.7, slot=None, song=None, choice_index=None,
             target_actor_id=None, target_actor_params=None, target_position=None, stop_distance=None, item_id=None):
    return Decision(goal="navigate", summary="test", skill=skill,
        args=SkillArgs(direction=direction, duration_ms=duration, strength=strength, slot=slot,
            choice_index=choice_index, song=song, target_actor_id=target_actor_id,
            target_actor_params=target_actor_params, target_position=target_position,
            stop_distance=stop_distance, item_id=item_id), memory_note=None)


def test_turn_uses_steering_plus_small_forward_bias():
    _, left_x, left_y = controller_input(decision("turn", "left"))
    _, right_x, right_y = controller_input(decision("turn", "right"))
    assert left_x < 0 < right_x
    assert left_y == right_y == 12


def test_move_forward_remains_pure_translation_command():
    buttons, x, y = controller_input(decision("move", "forward"))
    assert buttons == 0
    assert x == 0
    assert y > 0


def test_move_back_is_real_reverse_translation_command():
    buttons, x, y = controller_input(decision("move", "back"))
    assert buttons == 0
    assert x == 0
    assert y < 0


def test_wall_recovery_backs_up_before_turning_and_alternates_arc_side():
    first = _recovery_inputs(0)
    second = _recovery_inputs(1)
    assert first[0][2] < 0 and second[0][2] < 0
    assert first[1][2] < 0 and second[1][2] < 0
    assert first[1][1] == -second[1][1]
    assert first[2][1] == -second[2][1]


def test_turn_rejects_forward_and_back():
    import pytest
    from pydantic import ValidationError
    for direction in ("forward", "back"):
        with pytest.raises(ValidationError):
            decision("turn", direction)


def test_compound_local_combat_inputs():
    assert controller_input(decision("backflip", None))[0] == 0xA000
    assert controller_input(decision("backflip", None))[2] < 0
    assert controller_input(decision("roll", None))[0] == 0x8000
    assert controller_input(decision("roll", None))[2] > 0
    assert controller_input(decision("jump_attack", None))[0] == 0xA000


def test_sidestep_uses_target_plus_lateral_stick():
    left = controller_input(decision("sidestep", "left"))
    right = controller_input(decision("sidestep", "right"))
    assert left[0] == right[0] == 0xA000
    assert left[1] < 0 < right[1]


def test_advance_dialogue_is_single_a_action():
    action = decision("advance_dialogue", None, duration=120, strength=0)
    assert controller_input(action) == (0x8000, 0, 0)


def test_menu_controls_are_bounded_controller_primitives():
    assert controller_input(decision("pause_toggle", None))[0] == 0x1000
    assert controller_input(decision("menu_confirm", None))[0] == 0x8000
    assert controller_input(decision("menu_cancel", None))[0] == 0x4000
    assert controller_input(decision("menu_move", "up")) == (0, 0, 60)
    assert controller_input(decision("menu_move", "left")) == (0, -60, 0)
    assert controller_input(decision("menu_assign", None, slot="left"))[0] == 0x0002


def test_world_move_does_not_accept_menu_vertical_direction():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        decision("move", "up")


def test_play_song_requires_named_song():
    import pytest
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        decision("play_song", None)
    assert decision("play_song", None, song="lullaby").args.song == "lullaby"


def test_local_servo_steers_toward_camera_forward(state):
    game = state.model_copy(update={"camera_eye": (0.0, 0.0, -10.0),
        "camera_at": (0.0, 0.0, 0.0)})
    x, y, distance = _steer_to(game, (0.0, 0.0, 100.0), 0.8)
    assert abs(x) <= 1
    assert y > 0
    assert distance == 100.0


def test_local_servo_matches_soh_raw_stick_sign_for_positive_world_yaw(state):
    game = state.model_copy(update={"camera_input_yaw": 0, "mirrored_world": False})
    x, y, _ = _steer_to(game, (100.0, 0.0, 0.0), 0.8)
    # SoH uses Math_Atan2S(relY, -relX), so +90 world yaw requires raw X < 0.
    assert x < 0
    assert abs(y) <= 1


def test_world_yaw_stick_respects_exact_camera_yaw_and_mirroring(state):
    game = state.model_copy(update={"camera_input_yaw": 16384, "mirrored_world": False})
    assert _world_yaw_stick(game, 16384, 70) == (0, 70)
    assert _world_yaw_stick(game, 0, 70) == (70, 0)
    mirrored = game.model_copy(update={"mirrored_world": True})
    assert _world_yaw_stick(mirrored, 0, 70) == (-70, 0)


def test_door_final_micro_step_allows_door_wall_but_rejects_intervening_wall(state):
    door = {"actor_id": 9, "actor_uid": "door-1", "name": "En_Door", "description": "Door",
        "category": 10, "category_name": "door", "room": 0, "params": 2,
        "position": [0, 0, 100], "focus_position": [0, 20, 100], "distance": 100.0,
        "targeted": False, "drawn": False, "text_id": 0}
    probe = {"direction": "forward", "distance": 70, "floor_found": True,
        "floor_y": 0, "delta_y": 0, "floor_type": 0,
        "wall_hit": True, "wall_distance": 100, "wall_flags": 0}
    game = type(state).model_validate({
        **state.model_dump(), "capabilities": ["local_navmesh"],
        "room_actors": [door], "room_actor_count": 1, "navigation_probes": [probe],
    })
    actor = game.room_actors[0]
    assert interactions._door_step_safe(game, actor)
    blocked = type(state).model_validate({
        **game.model_dump(),
        "navigation_probes": [{**probe, "wall_distance": 30}],
    })
    assert not interactions._door_step_safe(blocked, blocked.room_actors[0])


def test_door_does_not_bypass_failed_navmesh_approach(state, monkeypatch):
    import asyncio

    door = {"actor_id": 9, "actor_uid": "door-1", "name": "En_Door", "description": "Door",
        "category": 10, "category_name": "door", "room": 0, "params": 2,
        "position": [160, 0, 0], "focus_position": [160, 20, 0], "distance": 160.0,
        "targeted": False, "drawn": False, "text_id": 0}
    game = type(state).model_validate({
        **state.model_dump(),
        "capabilities": ["local_navmesh"],
        "room_actors": [door],
        "room_actor_count": 1,
    })
    nav = decision("interact_with_actor", None, duration=5000, target_actor_id=9,
        target_actor_params=2).model_copy(update={
            "args": decision("interact_with_actor", None, duration=5000, target_actor_id=9,
                target_actor_params=2).args.model_copy(update={"target_actor_uid": "door-1"})
        })

    class FakeBridge:
        def __init__(self, current):
            self.state = current

    async def no_path(_bridge, _decision, _observation, *, actor_mode, talk=False, interact=False):
        return {"status": "failed", "reason": "navigation_no_path", "acknowledged": False}

    monkeypatch.setattr(interactions, "_navigate_local", no_path)
    result = asyncio.run(interactions._interact_with_door(FakeBridge(game), nav, game))
    assert result["status"] == "failed"
    assert result["reason"] == "door_approach_failed:navigation_no_path"


def test_explicit_exit_intent_promotes_unique_room_door(state):
    door = {"actor_id": 9, "name": "En_Door", "description": "Door", "category": 10,
        "category_name": "door", "room": 0, "params": 2, "position": [120, 0, 0],
        "focus_position": [0, 0, 0], "distance": 120.0, "targeted": False,
        "drawn": False, "text_id": 0}
    game = type(state).model_validate({**state.model_dump(), "room_actors": [door],
        "room_actor_count": 1})
    exit_move = decision("move", "forward").model_copy(update={
        "goal": "Exit the room through the door",
        "summary": "Leave this interior now.",
    })
    assert _door_intent_actor(game, exit_move).actor_id == 9

    ordinary_move = decision("move", "forward").model_copy(update={
        "goal": "Explore the center of the room",
        "summary": "Move toward open floor.",
    })
    assert _door_intent_actor(game, ordinary_move) is None


def test_navigate_to_door_coordinate_promotes_only_with_exit_intent(state):
    door = {"actor_id": 9, "name": "En_Door", "description": "Door", "category": 10,
        "category_name": "door", "room": 0, "params": 2, "position": [120, 0, 0],
        "focus_position": [0, 0, 0], "distance": 120.0, "targeted": False,
        "drawn": False, "text_id": 0}
    game = type(state).model_validate({**state.model_dump(), "room_actors": [door],
        "room_actor_count": 1})
    nav = decision("navigate_to", None, duration=5000, target_position=[130, 0, 5],
        stop_distance=35).model_copy(update={
            "goal": "Reach the exit door",
            "summary": "Navigate to the door.",
        })
    assert _door_intent_actor(game, nav).actor_id == 9


def test_traversal_intent_detects_down_without_center_enter_false_positive(state):
    down = decision("move", "forward").model_copy(update={
        "goal": "Descend the ladder",
        "summary": "Go down to the lower level.",
    })
    assert _traversal_intent_direction(down) == "down"

    center = decision("move", "forward").model_copy(update={
        "goal": "Explore the center of the platform",
        "summary": "Move toward center.",
    })
    assert _traversal_intent_direction(center) is None
    assert _door_intent_actor(state, center) is None


def test_ladder_top_probe_is_preferred_for_descent(state):
    game = type(state).model_validate({**state.model_dump(), "navigation_probes": [
        {"direction": "left", "distance": 70, "floor_found": False,
         "wall_hit": True, "wall_distance": 42, "wall_flags": 0x02},
        {"direction": "forward_right", "distance": 70, "floor_found": False,
         "wall_hit": True, "wall_distance": 55, "wall_flags": 0x04},
        {"direction": "forward", "distance": 70, "floor_found": True,
         "floor_y": -90, "delta_y": -90, "floor_type": 0,
         "wall_hit": False, "wall_distance": None, "wall_flags": 0},
    ]})
    probe = _best_climb_surface_probe(game, "down")
    assert probe is not None
    assert probe.direction == "forward_right"
    assert probe.wall_flags & 0x04


def test_traversal_probe_prefers_modest_safe_descent(state):
    game = type(state).model_validate({**state.model_dump(), "navigation_probes": [
        {"direction": "forward", "distance": 70, "floor_found": True,
         "floor_y": -35, "delta_y": -35, "floor_type": 0},
        {"direction": "right", "distance": 70, "floor_found": True,
         "floor_y": -200, "delta_y": -200, "floor_type": 0},
        {"direction": "back", "distance": 140, "floor_found": True,
         "floor_y": 0, "delta_y": 0, "floor_type": 0},
    ]})
    probe = _best_traversal_probe(game, "down")
    assert probe is not None
    assert probe.direction == "forward"


def test_matching_actor_uses_off_camera_room_actor(state):
    door = {"actor_id": 9, "name": "En_Door", "description": "Door", "category": 10,
        "category_name": "door", "room": 0, "params": 2, "position": [120, 0, 0],
        "focus_position": [120, 20, 0], "distance": 120.0, "targeted": False,
        "drawn": False, "text_id": 0}
    game = type(state).model_validate({**state.model_dump(), "nearby_actors": [],
        "room_actors": [door], "room_actor_count": 1})
    found = _matching_actor(game, 9, 2)
    assert found is not None
    assert found.category_name == "door"
    assert found.drawn is False


def test_inventory_slot_prefers_semantic_observation(state):
    game = type(state).model_validate({**state.model_dump(), "inventory": [255, 7, 6],
        "inventory_named": [{"slot": 1, "item_id": 7, "name": "Fairy Ocarina"}]})
    assert _inventory_slot(game, 7) == 1
    assert _inventory_slot(game, 6) == 2
    assert _inventory_slot(game, 3) is None


def test_menu_grid_prefers_axis_toward_target():
    assert _menu_grid_directions(0, 7)[:2] == ["right", "down"]
    assert _menu_grid_directions(23, 0)[:2] == ["left", "up"]


def test_equipment_grid_mapping_matches_vanilla_layout():
    assert _equipment_point(0x3B) == 1   # Kokiri Sword
    assert _equipment_point(0x3D) == 3   # BGS
    assert _equipment_point(0x3E) == 5   # Deku Shield
    assert _equipment_point(0x43) == 11  # Zora Tunic
    assert _equipment_point(0x45) == 14  # Iron Boots
    assert _equipment_point(0x46) == 15  # Hover Boots
    assert _equipment_point(7) is None


def test_aim_error_zero_when_camera_points_at_target(state):
    game = state.model_copy(update={"camera_eye": (0.0, 0.0, 0.0),
        "camera_at": (0.0, 0.0, 10.0)})
    yaw, pitch = _aim_error(game, (0.0, 0.0, 100.0))
    assert abs(yaw) < 1e-6
    assert abs(pitch) < 1e-6


def test_aim_error_and_stick_point_toward_right_and_up(state):
    game = state.model_copy(update={"camera_eye": (0.0, 0.0, 0.0),
        "camera_at": (0.0, 0.0, 10.0)})
    yaw, pitch = _aim_error(game, (10.0, 10.0, 100.0))
    assert yaw > 0 and pitch > 0
    assert _aim_stick(yaw, 1) > 0
    assert _aim_stick(pitch, 1) > 0
    assert _aim_stick(yaw, -1) < 0


def test_probe_stick_targets_link_relative_world_direction_with_rotated_camera(state):
    game = state.model_copy(update={"camera_input_yaw": 16384, "mirrored_world": False,
        "capabilities": ["probe_yaw_v2"]})
    game.player.yaw = 0
    # Link-forward world yaw 0 is camera-left when camera input yaw is +90.
    assert _probe_stick("forward", game) == (48, 0)
    # OoT enum: +90 relative yaw is LEFT, and here that equals camera-forward.
    assert _probe_stick("left", game) == (0, 48)


def test_legacy_probe_label_left_maps_to_old_negative_yaw_sample(state):
    game = state.model_copy(update={"camera_input_yaw": 0, "capabilities": []})
    game.player.yaw = 0
    # v2.5 called the -90-degree sample "left"; preserve that interpretation.
    assert _probe_stick("left", game) == (48, 0)


def test_player_relative_dodge_stick_uses_exact_camera_input_yaw(state):
    game = state.model_copy(update={"camera_input_yaw": 0, "mirrored_world": False})
    game.player.yaw = 0
    assert _player_relative_stick(game, "back", 70) == (0, -70)
    assert _player_relative_stick(game, "left", 70) == (-70, 0)
    assert _player_relative_stick(game, "right", 70) == (70, 0)


def test_player_relative_dodge_rotates_raw_stick_against_camera(state):
    game = state.model_copy(update={"camera_input_yaw": 16384, "mirrored_world": False})
    game.player.yaw = 0
    assert _player_relative_stick(game, "back", 70) == (-70, 0)


def test_player_relative_dodge_respects_mirrored_world(state):
    game = state.model_copy(update={"camera_input_yaw": 16384, "mirrored_world": True})
    game.player.yaw = 0
    assert _player_relative_stick(game, "back", 70) == (70, 0)



def test_generated_stick_classifies_as_link_backward_for_arbitrary_camera(state):
    import math
    game = state.model_copy(update={"camera_input_yaw": -12288, "mirrored_world": False})
    game.player.yaw = 8192
    x, y = _player_relative_stick(game, "back", 70)
    stick_angle = math.atan2(-x, y) * 32768 / math.pi
    world_yaw = game.camera_input_yaw + stick_angle
    relative = ((world_yaw - game.player.yaw + 32768) % 65536) - 32768
    assert abs(abs(relative) - 32768) < 800


def test_dodge_safety_uses_link_relative_probe(state):
    safe = {
        "direction": "back", "distance": 70, "floor_found": True,
        "floor_y": 0, "delta_y": 0, "floor_type": 0,
        "wall_hit": False, "wall_distance": None, "wall_flags": 0,
    }
    game = type(state).model_validate({**state.model_dump(), "navigation_probes": [safe]})
    assert _dodge_direction_safe(game, "back")
    unsafe = {**safe, "delta_y": -80}
    game = type(state).model_validate({**state.model_dump(), "navigation_probes": [unsafe]})
    assert not _dodge_direction_safe(game, "back")
    wall = {**safe, "wall_hit": True, "wall_distance": 55}
    game = type(state).model_validate({**state.model_dump(), "navigation_probes": [wall]})
    assert not _dodge_direction_safe(game, "back")


def test_control_stick_direction_accepts_engine_backflip_bin(state):
    state.player.control_stick_direction = 2
    assert state.player.control_stick_direction == 2


def test_stick_quadrant_feedback_correction():
    assert _rotate_stick_quadrants(0, 70, 1) == (-70, 0)
    assert _rotate_stick_quadrants(0, 70, 2) == (0, -70)
    assert _rotate_stick_quadrants(0, 70, 3) == (70, 0)


def test_common_controller_imports_time_for_monotonic_deadlines():
    import zelda_ai.skills.common as common
    assert common.time.monotonic() > 0


def test_scene_exit_target_resolves_to_engine_observation(state):
    game = type(state).model_validate({
        **state.model_dump(),
        "scene_exits": [
            {"exit_index": 1, "entrance_index": 0x211, "position": [70, 0, 116], "samples": 5},
            {"exit_index": 2, "entrance_index": 0x300, "position": [-180, 0, 20], "samples": 2},
        ],
    })
    selected = _resolve_scene_exit(game, [75, 0, 110])
    assert selected is not None
    assert selected.exit_index == 1
    assert selected.position == (70.0, 0.0, 116.0)


def test_single_scene_exit_is_selected_without_inventing_door(state):
    game = type(state).model_validate({
        **state.model_dump(),
        "scene_exits": [
            {"exit_index": 1, "entrance_index": 0x211, "position": [70, 0, 116], "samples": 5},
        ],
    })
    selected = _resolve_scene_exit(game, [65, 0, 100])
    assert selected is not None
    assert selected.entrance_index == 0x211


def test_traverse_exit_rejects_unobserved_coordinate_even_with_single_exit(state):
    game = type(state).model_validate({
        **state.model_dump(),
        "navmesh": {
            "origin": [0, 0, 0], "step": 70, "half_extent": 4,
            "cells": [[0, 0, 0, 0]],
        },
        "scene_exits": [
            {"exit_index": 1, "entrance_index": 0x211,
             "position": [70, 0, 116], "samples": 5},
        ],
    })
    assert _resolve_scene_exit(game, [900, 0, 900]) is None


def test_scene_exit_resolution_uses_current_observation_only(state):
    observed = type(state).model_validate({
        **state.model_dump(),
        "navmesh": {
            "origin": [0, 0, 0], "step": 70, "half_extent": 4,
            "cells": [[0, 0, 0, 0]],
        },
        "scene_exits": [
            {"exit_index": 1, "entrance_index": 0x211,
             "position": [70, 0, 116], "samples": 5},
        ],
    })
    assert _resolve_scene_exit(observed, [70, 0, 116]) is not None

    refreshed = type(state).model_validate({
        **observed.model_dump(),
        "scene_exits": [],
    })
    assert _resolve_scene_exit(refreshed, [70, 0, 116]) is None


def test_traversal_affordance_resolves_by_direction_and_approach(state):
    game = type(state).model_validate({
        **state.model_dump(),
        "navmesh": {
            "origin": [0, 0, 0], "step": 70, "half_extent": 4,
            "cells": [[0, 0, 0, 0]],
        },
        "traversal_affordances": [
            {"kind": "stairs_or_slope_down", "direction": "down",
             "approach_position": [70, 0, 0], "target_position": [140, -40, 0],
             "distance": 70, "height_delta": -40, "wall_flags": 0},
            {"kind": "ladder_up", "direction": "up",
             "approach_position": [-70, 0, 0], "target_position": [-100, 30, 0],
             "distance": 70, "height_delta": 0, "wall_flags": 2},
        ],
    })
    down = _resolve_traversal_affordance(game, "down", [75, 0, 4])
    assert down is not None and down.kind == "stairs_or_slope_down"
    assert _resolve_traversal_affordance(game, "up", [75, 0, 4]) is None


def test_traversal_affordance_refresh_requires_same_kind(state):
    game = type(state).model_validate({
        **state.model_dump(),
        "navmesh": {
            "origin": [0, 0, 0], "step": 70, "half_extent": 4,
            "cells": [[0, 0, 0, 0]],
        },
        "traversal_affordances": [
            {"kind": "ledge_down", "direction": "down",
             "approach_position": [70, 0, 0], "target_position": [140, -100, 0],
             "distance": 70, "height_delta": -100, "wall_flags": 0},
        ],
    })
    assert _resolve_traversal_affordance(
        game, "down", [70, 0, 0], kind="ledge_down") is not None
    assert _resolve_traversal_affordance(
        game, "down", [70, 0, 0], kind="ladder_down") is None


def test_recentered_ladder_can_revalidate_from_fast_collision(state):
    original_game = type(state).model_validate({
        **state.model_dump(),
        "traversal_affordances": [{
            "kind": "ladder_down",
            "direction": "down",
            "approach_position": [70, 0, 0],
            "target_position": [110, -20, 0],
            "distance": 70,
            "height_delta": 0,
            "wall_flags": 4,
        }],
    })
    original = original_game.traversal_affordances[0]
    current = type(state).model_validate({
        **state.model_dump(),
        "player": {**state.player.model_dump(), "position": [70, 0, 0], "wall_flags": 4},
        "traversal_affordances": [],
    })
    assert _refresh_traversal_affordance(current, original) is None
    assert _local_traversal_evidence(current, "down", kind="ladder_down")


def test_ladder_fast_revalidation_does_not_accept_unrelated_floor_drop(state):
    original_game = type(state).model_validate({
        **state.model_dump(),
        "traversal_affordances": [{
            "kind": "ladder_down",
            "direction": "down",
            "approach_position": [70, 0, 0],
            "target_position": [110, -20, 0],
            "distance": 70,
            "height_delta": 0,
            "wall_flags": 4,
        }],
    })
    probe = {
        "direction": "forward",
        "distance": 70,
        "floor_found": True,
        "floor_y": -40,
        "delta_y": -40,
        "floor_type": 0,
        "wall_hit": False,
        "wall_distance": None,
        "wall_flags": 0,
    }
    current = type(state).model_validate({
        **state.model_dump(),
        "player": {**state.player.model_dump(), "position": [70, 0, 0], "wall_flags": 0},
        "navigation_probes": [probe],
        "traversal_affordances": [],
    })
    assert not _local_traversal_evidence(current, "down", kind="ladder_down")
    assert _local_traversal_evidence(current, "down", kind="stairs_or_slope_down")


def test_recentered_affordance_can_be_reclassified_near_same_geometry(state):
    original_game = type(state).model_validate({
        **state.model_dump(),
        "traversal_affordances": [{
            "kind": "ladder_down",
            "direction": "down",
            "approach_position": [70, 0, 0],
            "target_position": [110, -20, 0],
            "distance": 70,
            "height_delta": 0,
            "wall_flags": 4,
        }],
    })
    original = original_game.traversal_affordances[0]
    current = type(state).model_validate({
        **state.model_dump(),
        "player": {**state.player.model_dump(), "position": [68, 0, 2]},
        "navmesh": {
            "origin": [68, 0, 2],
            "step": 70,
            "half_extent": 4,
            "cells": [[0, 0, 0, 0]],
        },
        "traversal_affordances": [{
            "kind": "ledge_down",
            "direction": "down",
            "approach_position": [72, 0, 3],
            "target_position": [116, -30, 4],
            "distance": 4,
            "height_delta": -30,
            "wall_flags": 0,
        }],
    })
    refreshed = _refresh_traversal_affordance(current, original)
    assert refreshed is not None
    assert refreshed.kind == "ledge_down"


def test_recentered_affordance_rejects_distant_same_direction_route(state):
    original_game = type(state).model_validate({
        **state.model_dump(),
        "traversal_affordances": [{
            "kind": "ladder_down",
            "direction": "down",
            "approach_position": [70, 0, 0],
            "target_position": [110, -20, 0],
            "distance": 70,
            "height_delta": 0,
            "wall_flags": 4,
        }],
    })
    original = original_game.traversal_affordances[0]
    current = type(state).model_validate({
        **state.model_dump(),
        "player": {**state.player.model_dump(), "position": [70, 0, 0]},
        "navmesh": {
            "origin": [70, 0, 0],
            "step": 70,
            "half_extent": 4,
            "cells": [[0, 0, 0, 0]],
        },
        "traversal_affordances": [{
            "kind": "ledge_down",
            "direction": "down",
            "approach_position": [250, 0, 0],
            "target_position": [300, -100, 0],
            "distance": 180,
            "height_delta": -100,
            "wall_flags": 0,
        }],
    })
    assert _refresh_traversal_affordance(current, original) is None


def test_fast_revalidation_rejects_unrelated_ladder_far_from_original_target(state):
    original_game = type(state).model_validate({
        **state.model_dump(),
        "traversal_affordances": [{
            "kind": "ladder_down",
            "direction": "down",
            "approach_position": [70, 0, 0],
            "target_position": [110, -20, 0],
            "distance": 70,
            "height_delta": 0,
            "wall_flags": 4,
        }],
    })
    original = original_game.traversal_affordances[0]
    current = type(state).model_validate({
        **state.model_dump(),
        "player": {**state.player.model_dump(),
                   "position": [400, 0, 0],
                   "wall_flags": 4},
        "traversal_affordances": [],
    })
    assert _local_traversal_evidence(current, "down", kind="ladder_down")
    assert not _fast_revalidation_matches_original(current, original)
