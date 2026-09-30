from zelda_ai.autonomy.models import AgentIntent, ObjectiveCompletion
from zelda_ai.autonomy.objectives import ObjectiveTracker
from zelda_ai.models import ActorObservation, EquipmentObservation, GameEvent


def _intent(objective: str, completion: ObjectiveCompletion) -> AgentIntent:
    return AgentIntent(
        objective=objective,
        completion=completion,
        summary=objective,
        mode="explore",
        horizon_ms=60000,
    )


def test_kokiri_sword_objective_stays_open_until_equipment_is_observed(state):
    tracker = ObjectiveTracker()
    intent = _intent(
        "Obtain the Kokiri Sword",
        ObjectiveCompletion(kind="equipment", name="Kokiri Sword"),
    )
    tracker.assign(intent, state)

    assert tracker.evaluate(state).completed is False

    unrelated = state.model_copy(deep=True)
    unrelated.player.rupees += 10
    assert tracker.evaluate(unrelated).completed is False
    assert tracker.intent.objective == "Obtain the Kokiri Sword"

    acquired = unrelated.model_copy(deep=True)
    acquired.progress.equipment = [
        EquipmentObservation(
            item_id=1,
            name="Kokiri Sword",
            equipment_type="sword",
            value=1,
            equipped=False,
        )
    ]
    acquired.progress.owned_equipment = ["Kokiri Sword"]

    status = tracker.evaluate(acquired)
    assert status.completed is True
    assert status.reason == "equipment_name"

    tracker.complete(status)
    assert tracker.intent is None
    assert tracker.completed_count == 1
    assert tracker.last_completion["objective"] == "Obtain the Kokiri Sword"


def test_leave_scene_room_uses_assignment_location_when_not_explicit(state):
    tracker = ObjectiveTracker()
    tracker.assign(
        _intent(
            "Leave this house",
            ObjectiveCompletion(kind="leave_scene_room"),
        ),
        state,
    )
    assert tracker.evaluate(state).completed is False

    moved_inside = state.model_copy(deep=True)
    moved_inside.player.position = (100.0, 0.0, 0.0)
    assert tracker.evaluate(moved_inside).completed is False

    outside = moved_inside.model_copy(deep=True)
    outside.room += 1
    assert tracker.evaluate(outside).completed is True


def test_event_objective_ignores_events_already_present_at_assignment(state):
    existing = state.model_copy(deep=True)
    existing.events = [
        GameEvent(id="old-chest", kind="chest_opened", detail="already there")
    ]
    tracker = ObjectiveTracker()
    tracker.assign(
        _intent(
            "Open a chest",
            ObjectiveCompletion(kind="event_kind", event_kind="chest_opened"),
        ),
        existing,
    )
    assert tracker.evaluate(existing).completed is False

    fresh = existing.model_copy(deep=True)
    fresh.events = [
        *existing.events,
        GameEvent(id="new-chest", kind="chest_opened", detail="new"),
    ]
    assert tracker.evaluate(fresh).completed is True


def test_dialogue_actor_objective_completes_only_for_matching_speaker(state):
    tracker = ObjectiveTracker()
    tracker.assign(
        _intent(
            "Talk to Mido",
            ObjectiveCompletion(kind="dialogue_actor", actor_name="Mido"),
        ),
        state,
    )

    wrong = state.model_copy(deep=True)
    wrong.dialogue.active = True
    wrong.dialogue.speaker = ActorObservation(
        actor_uid="saria",
        actor_id=1,
        name="Saria",
        description="Saria",
        category=5,
        category_name="npc",
        params=0,
        position=(0.0, 0.0, 0.0),
        distance=10.0,
    )
    assert tracker.evaluate(wrong).completed is False

    right = wrong.model_copy(deep=True)
    right.dialogue.speaker = ActorObservation(
        actor_uid="mido",
        actor_id=2,
        name="Mido",
        description="Mido",
        category=5,
        category_name="npc",
        params=0,
        position=(0.0, 0.0, 0.0),
        distance=10.0,
    )
    assert tracker.evaluate(right).completed is True
