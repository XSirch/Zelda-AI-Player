"""Software task contracts; fixture positions are not native gameplay evidence."""
import pytest

from zelda_ai.autonomy.container_task import ObservedContainerApproachTask
from zelda_ai.autonomy.imitation import encode_surface
from zelda_ai.models import ActorObservation, NavigationMeshSnapshot


def setup(state):
    state.player.bg_check_flags = 1
    state.player.floor_height = 0
    state.player.speed_xz = 0
    state.camera_input_yaw = 0
    state.navmesh = NavigationMeshSnapshot(step=70, half_extent=4,
        cells=[(0, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 64)])
    actor = ActorObservation(actor_uid="container", actor_id=999, category=10, category_name="chest",
        params=-1, position=(180, 0, 0), distance=180, drawn=True)
    state.nearby_actors = [actor]
    return actor


def test_floor_arrival_is_not_container_success_and_final_contact_is_bounded(state):
    actor = setup(state)
    task = ObservedContainerApproachTask.create(state, actor, now=0)
    assert task.floor_task.target == (140, 0, 0)
    assert encode_surface(state, task)[4:] == [0., 0., 0., 0., 1.]
    state.player.position = (140, 0, 0)
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.1+i/10)
    assert task.floor_task.phase == "succeeded" and task.phase == "execute"
    assert task.steering_point(state) == actor.position
    state.seq += 1
    task.observe(state, consumed=True, now=2.9)
    assert task.phase == "failed" and task.failure == "container_no_observed_prompt"


def test_only_the_current_matching_open_prompt_completes_after_consumed_stopped_frames(state):
    actor = setup(state)
    task = ObservedContainerApproachTask.create(state, actor, now=0)
    state.context_action.label = "open"
    state.context_actor = actor.model_copy(update={"actor_uid": "other-container"})
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert not task.terminal and task.verification_frames == 0
    state.context_actor = actor
    for i in range(3):
        state.seq += 1
        task.observe(state, consumed=True, now=.2+i/10)
    assert task.phase == "succeeded" and task.verification_frames == 3
    assert not task.allows_context_buttons(state)


def test_current_fast_room_actor_keeps_target_when_optional_nearby_list_is_absent(state):
    actor = setup(state)
    state.protocol, state.full_seq = 3, state.seq
    task = ObservedContainerApproachTask.create(state, actor, now=0)
    # The real bridge clears nearby_actors on fast observations, while the
    # native room_actors list remains authoritative for current UID/draw/pose.
    state.seq += 1
    state.nearby_actors = []
    state.room_actors = [actor]
    task.observe(state, consumed=True, now=.1)
    assert not task.terminal and task.current_actor(state).actor_uid == actor.actor_uid


@pytest.mark.parametrize("loss", ["not_drawn", "uid_changed", "moved"])
def test_fast_actor_loss_cannot_be_hidden_by_an_older_nearby_snapshot(state, loss):
    actor = setup(state)
    state.protocol, state.full_seq = 3, state.seq
    task = ObservedContainerApproachTask.create(state, actor, now=0)
    update = {"drawn": False} if loss == "not_drawn" else (
        {"actor_uid": "replacement"} if loss == "uid_changed" else {"position": (300, 0, 0)})
    state.seq += 1
    state.room_actors = [actor.model_copy(update=update)]
    # Even if a caller erroneously carries an old optional list, current
    # authoritative fast actor facts must win.
    task.observe(state, consumed=True, now=.1)
    assert task.failure == "container_target_lost"


@pytest.mark.parametrize("loss", ["actor", "modal", "reload", "water"])
def test_approach_revokes_control_on_evidence_or_mode_loss(state, loss):
    from zelda_ai.models import GameEvent
    actor = setup(state)
    task = ObservedContainerApproachTask.create(state, actor, now=0)
    if loss == "actor":
        state.nearby_actors = []
    elif loss == "modal":
        state.cutscene_active = True
    elif loss == "reload":
        state.events = [GameEvent(id="new-load", kind="save_loaded")]
    else:
        state.player.state_flags_1 |= 1 << 27
    state.seq += 1
    task.observe(state, consumed=True, now=.1)
    assert task.phase == "interrupted"
