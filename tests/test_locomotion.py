from zelda_ai.models import Decision, SkillArgs
from zelda_ai.runtime import controller_input


def decision(skill, direction, duration=700, strength=0.7):
    return Decision(goal="navigate", summary="test", skill=skill,
        args=SkillArgs(direction=direction, duration_ms=duration, strength=strength, slot=None, choice_index=None, song=None),
        memory_note=None)


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
