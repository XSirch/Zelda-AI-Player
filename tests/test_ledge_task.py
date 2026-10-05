import pytest

from zelda_ai.autonomy.ledge_task import ObservedLedgeAscentTask
from zelda_ai.autonomy.locomotion import locomotor_mode
from zelda_ai.models import TraversalAffordanceObservation


def contact_ledge(state):
    state.player.bg_check_flags = 1 | 0x200
    state.camera_input_yaw = 0
    row = TraversalAffordanceObservation(kind="ledge_up", direction="up",
        approach_position=state.player.position, target_position=(0, 21, 25),
        distance=0, height_delta=21)
    state.traversal_affordances = [row]
    return row


def test_native_climb_animation_is_allowed_but_cannot_finish_before_landing(state):
    task = ObservedLedgeAscentTask.create(state, contact_ledge(state), now=0)
    assert task.reference_stick(state) == (0, 40)
    state.player.position = (0, 21, 25)
    state.player.floor_height = 21
    state.player.climbing_ledge = True
    state.player.state_flags_1 = (1 << 14) | (1 << 18)
    for index in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1 + index * .1)
        assert task.phase != "succeeded" and task.failure is None
        assert task.reference_stick(state) == (0, 0)
    state.player.climbing_ledge = False
    state.player.state_flags_1 = 0
    for index in range(3):
        state.seq += 1
        task.observe(state, consumed=False, now=.5 + index * .1)
    assert task.phase == "succeeded"


@pytest.mark.parametrize("field,value", (("climbing_ladder", True), ("hanging_ledge", True),
                                         ("state_flags_1", 1 << 27)))
def test_unrelated_attachment_or_water_interrupts_ledge_ascent(state, field, value):
    task = ObservedLedgeAscentTask.create(state, contact_ledge(state), now=0)
    setattr(state.player, field, value)
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert task.phase == "interrupted"
    assert task.reference_stick(state) == (0, 0)


def test_ledge_task_cannot_invent_distant_landing_or_reuse_old_proposal(state):
    row = contact_ledge(state)
    state.traversal_affordances = []
    with pytest.raises(ValueError, match="currently observed"):
        ObservedLedgeAscentTask.create(state, row)
    distant = row.model_copy(update={"target_position": (0, 21, 140)})
    state.traversal_affordances = [distant]
    with pytest.raises(ValueError, match="local landing"):
        ObservedLedgeAscentTask.create(state, distant)


def test_water_modes_use_native_flags_not_distance_to_water_alone(state):
    contact_ledge(state)
    state.player.y_dist_to_water = 10
    assert locomotor_mode(state) == "ground"  # Wading, no native swimming flag.
    state.player.state_flags_1 = 1 << 27
    assert locomotor_mode(state) == "swimming"
    state.player.state_flags_2 = 1 << 10
    assert locomotor_mode(state) == "submerged_ground"
    state.player.bg_check_flags = 0
    assert locomotor_mode(state) == "underwater"
    state.player.state_flags_2 |= 1 << 11
    assert locomotor_mode(state) == "diving"
    state.player.climbing_ladder = True
    assert locomotor_mode(state) == "ladder"
