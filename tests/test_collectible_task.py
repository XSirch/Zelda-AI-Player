"""Software contracts only; fixture gains are not actual game acquisitions."""
import pytest

from zelda_ai.autonomy.collectible_task import ObservedCollectibleApproachTask
from zelda_ai.autonomy.imitation import encode_surface
from zelda_ai.models import ActorObservation, GameEvent, InventoryObservation, NavigationMeshSnapshot


def setup(state, *, position=(180, 1, 0)):
    state.player.bg_check_flags, state.player.floor_height = 1, 0
    state.player.speed_xz, state.camera_input_yaw = 0, 0
    state.navmesh = NavigationMeshSnapshot(step=70, half_extent=4,
        cells=[(0, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 64)])
    actor = ActorObservation(actor_uid="collectible", actor_id=21, category=6, category_name="misc",
        name="En_Item00", params=-32768, position=position, distance=180, drawn=True)
    state.nearby_actors = [actor]
    return actor


def advance(state, task, *, consumed=True, now=.1):
    state.seq += 1
    task.observe(state, consumed=consumed, now=now)


def test_floor_arrival_does_not_claim_collection_or_read_actor_contents(state):
    actor = setup(state)
    task = ObservedCollectibleApproachTask.create(state, actor, now=0)
    assert task.floor_task.target == (140, 0, 0)
    assert task.target == (180, 0, 0) and encode_surface(state, task)[-1] == 1
    state.player.position = (140, 0, 0)
    for i in range(3):
        advance(state, task, now=.1 + i/10)
    assert not task.terminal and task.floor_task is None
    assert task.steering_point(state) == (180, 0, 0)
    actor.params = 32767
    state.player.position = (160, 0, 0)
    advance(state, task, now=.5)
    advance(state, task, now=3.1)
    assert task.phase == "failed" and task.failure == "collectible_no_observed_gain"


@pytest.mark.parametrize("missing", ["gain", "consumption", "near_contact", "uid_loss", "fresh_frames"])
def test_missing_acquisition_proof_cannot_complete(state, missing):
    actor = setup(state, position=(20, 1, 0))
    task = ObservedCollectibleApproachTask.create(state, actor, now=0)
    if missing != "gain":
        state.player.rupees += 5
    if missing == "near_contact":
        state.player.position = (-100, 0, 0)
    if missing != "uid_loss":
        state.nearby_actors = []
    for i in range(3):
        if missing == "fresh_frames":
            task.observe(state, consumed=True, now=.1 + i/10)
        else:
            advance(state, task, consumed=missing != "consumption", now=.1 + i/10)
    assert task.phase != "succeeded"


def test_consumed_near_contact_positive_delta_and_lost_uid_complete_after_three_frames(state):
    actor = setup(state, position=(20, 1, 0))
    task = ObservedCollectibleApproachTask.create(state, actor, now=0)
    state.player.rupees = 5
    # An actor that blinks is still present: no disappearance postcondition.
    state.nearby_actors = [actor.model_copy(update={"drawn": False})]
    advance(state, task)
    assert task.phase == "verify" and task.verification_frames == 0
    state.nearby_actors = []
    for i in range(3):
        advance(state, task, now=.2 + i/10)
    assert task.phase == "succeeded" and task.gain_values == {"rupees": 5}


def test_fast_actor_loss_waits_neutrally_for_actual_complete_ammo_delta(state):
    actor = setup(state, position=(20, 1, 0))
    state.protocol, state.full_seq, state.inventory = 3, state.seq, [255] * 24
    state.room_actors = [actor]
    task = ObservedCollectibleApproachTask.create(state, actor, now=0)
    state.room_actors = []
    advance(state, task)
    assert task.phase == "verify" and task.gain_seq is None
    state.inventory[0] = 0
    state.inventory_named = [InventoryObservation(slot=0, item_id=0, name="Observed ammo", ammo=5)]
    state.full_seq = state.seq + 1
    state.nearby_actors = []
    advance(state, task, now=.2)
    for i in range(2):
        advance(state, task, now=.3 + i/10)
    assert task.phase == "succeeded" and task.gain_values == {"ammo:0": 5}


def test_full_capacity_item_received_and_actor_loss_do_not_create_a_gain(state):
    actor = setup(state, position=(20, 1, 0))
    state.player.rupees = 99
    task = ObservedCollectibleApproachTask.create(state, actor, now=0)
    state.events = [GameEvent(id="pickup", kind="item_received")]
    state.nearby_actors = []
    advance(state, task)
    advance(state, task, now=1.2)
    assert task.phase == "failed" and task.failure == "collectible_no_resource_gain"
    assert not task.gain_values


@pytest.mark.parametrize("loss", ["reload", "modal", "water", "changed_uid", "hidden", "moved"])
def test_current_ownership_and_mode_loss_revoke_the_target(state, loss):
    actor = setup(state)
    task = ObservedCollectibleApproachTask.create(state, actor, now=0)
    if loss == "reload":
        state.events = [GameEvent(id="new-load", kind="save_loaded")]
    elif loss == "modal":
        state.dialogue.active = True
    elif loss == "water":
        state.player.state_flags_1 |= 1 << 27
    elif loss == "changed_uid":
        state.nearby_actors = [actor.model_copy(update={"actor_uid": "replacement"})]
    elif loss == "hidden":
        state.nearby_actors = [actor.model_copy(update={"drawn": False})]
    else:
        state.nearby_actors = [actor.model_copy(update={"position": (300, 1, 0)})]
    advance(state, task)
    assert task.phase == "interrupted"


@pytest.mark.parametrize("missing", ["drawn", "identity", "floor", "class", "path"])
def test_selection_needs_current_visible_supported_evidence(state, missing):
    actor = setup(state)
    if missing == "drawn":
        actor.drawn = False
    elif missing == "identity":
        actor.actor_uid = None
    elif missing == "floor":
        actor.position = (180, 30, 0)
    elif missing == "class":
        actor.name = "En_Elf"
    else:
        state.navmesh.cells = [(0, 0, 0., 0), (2, 0, 0., 0)]
    with pytest.raises(ValueError):
        ObservedCollectibleApproachTask.create(state, actor, now=0)


def test_protocol_three_selection_needs_actor_in_current_authoritative_room_scope(state):
    actor = setup(state)
    state.protocol, state.full_seq, state.room_actors_truncated = 3, state.seq, True
    # A full nearby list may retain a distant actor omitted from the bounded
    # room list. Its first fast packet cannot maintain that target's ownership.
    with pytest.raises(ValueError, match="current.*room"):
        ObservedCollectibleApproachTask.create(state, actor, now=0)


def test_truncated_room_scope_cannot_prove_consumption_even_on_a_complete_packet(state):
    actor = setup(state, position=(20, 1, 0))
    state.protocol, state.full_seq, state.room_actors = 3, state.seq, [actor]
    task = ObservedCollectibleApproachTask.create(state, actor, now=0)
    state.room_actors, state.nearby_actors, state.room_actors_truncated = [], [], True
    state.player.rupees = 5
    for i in range(3):
        state.full_seq = state.seq + 1
        advance(state, task, now=.1 + i/10)
    assert task.phase != "succeeded" and task.verification_frames == 0
    # Completeness of a later actual room observation can verify the loss.
    state.room_actors_truncated = False
    for i in range(3):
        state.full_seq = state.seq + 1
        advance(state, task, now=.4 + i/10)
    assert task.phase == "succeeded"
