from zelda_ai.models import Decision, SkillArgs
from zelda_ai.runtime import _aim_error, _aim_stick, _best_traversal_probe, _door_intent_actor, _equipment_point, _inventory_slot, _matching_actor, _menu_grid_directions, _recovery_inputs, _steer_to, _traversal_intent_direction, controller_input


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
    assert controller_input(decision("backflip", None))[0] == 0x2000
    assert controller_input(decision("backflip", None))[2] < 0
    assert controller_input(decision("roll", None))[0] == 0x8000
    assert controller_input(decision("roll", None))[2] > 0
    assert controller_input(decision("jump_attack", None))[0] == 0xA000


def test_sidestep_uses_target_plus_lateral_stick():
    left = controller_input(decision("sidestep", "left"))
    right = controller_input(decision("sidestep", "right"))
    assert left[0] == right[0] == 0x2000
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


def test_local_servo_steers_right_when_target_is_camera_right(state):
    game = state.model_copy(update={"camera_eye": (0.0, 0.0, -10.0),
        "camera_at": (0.0, 0.0, 0.0)})
    x, y, _ = _steer_to(game, (100.0, 0.0, 0.0), 0.8)
    assert x > 0
    assert abs(y) <= 1


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
