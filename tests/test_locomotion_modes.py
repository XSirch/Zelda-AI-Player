import pytest

from zelda_ai.autonomy.local_tasks import LocalTask
from zelda_ai.models import TraversalAffordanceObservation


def observed_ascent(state):
    row = TraversalAffordanceObservation(kind="stairs_or_slope_up", direction="up",
        approach_position=state.player.position, target_position=(0, 21, 70),
        distance=0, height_delta=21)
    state.traversal_affordances = [row]
    return row


def test_airborne_apex_cannot_finish_a_ground_traversal(state):
    state.player.bg_check_flags = 1
    task = LocalTask.traversal(state, observed_ascent(state), now=0)
    state.player.position = (0, 21, 70)
    state.player.floor_height = 21
    # Native ledge animations can retain GROUND while JUMPING remains set.
    state.player.state_flags_1 = 1 << 18
    for index in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1 + index * .1)
    assert task.phase != "succeeded"
    state.player.state_flags_1 = 0
    state.player.bg_check_flags = 0
    for index in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.4 + index * .1)
    assert task.phase != "succeeded"
    state.player.bg_check_flags = 1
    for index in range(3):
        state.seq += 1
        task.observe(state, consumed=False, now=.7 + index * .1)
    assert task.phase == "succeeded"


@pytest.mark.parametrize("water_flags", (0, 1 << 10, 1 << 11))
def test_water_entry_hands_off_ground_task_and_releases_stick(state, water_flags):
    state.player.bg_check_flags = 1
    task = LocalTask.traversal(state, observed_ascent(state), now=0)
    state.player.state_flags_1 = 1 << 27
    state.player.state_flags_2 = water_flags
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert task.failure == "unsupported_locomotor_mode"
    assert task.phase == "interrupted"
    assert task.reference_stick(state) == (0, 0)


def test_submerged_ground_does_not_prove_dry_ground_arrival(state):
    state.player.bg_check_flags = 1
    task = LocalTask.traversal(state, observed_ascent(state), now=0)
    state.player.position = (0, 21, 70)
    state.player.floor_height = 21
    state.player.state_flags_2 = 1 << 10
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert task.phase != "succeeded"
    assert task.reference_stick(state) == (0, 0)
