from zelda_ai.checkpoints import checkpoint_blocks_decision, opening_checkpoint_plan
from zelda_ai.models import Decision


def _equipment(name: str, equipment_type: str, *, equipped: bool = False, item_id: int = 1):
    return {
        "item_id": item_id,
        "name": name,
        "equipment_type": equipment_type,
        "value": 1,
        "equipped": equipped,
    }


def _game(state, *, scene_name="Kokiri Forest", story_flags=None, equipment=None, quest_items=None,
          room_actors=None, autosave=None):
    return type(state).model_validate({
        **state.model_dump(),
        "scene_name": scene_name,
        "progress": {
            **state.progress.model_dump(),
            "story_flags": story_flags or {},
            "equipment": equipment or [],
            "owned_equipment": [row["name"] for row in (equipment or [])],
            "quest_items": quest_items or [],
        },
        "room_actors": room_actors or [],
        "room_actor_count": len(room_actors or []),
        "autosave": autosave or state.autosave.model_dump(),
    })


def test_opening_checkpoint_sequence_uses_observable_progress(state):
    house = _game(state, scene_name="Link's House")
    assert opening_checkpoint_plan(house)["current"]["id"] == "leave_links_house"

    forest = _game(state)
    assert opening_checkpoint_plan(forest)["current"]["id"] == "greet_saria_once"

    greeted = _game(state, story_flags={"greeted_by_saria": True})
    plan = opening_checkpoint_plan(greeted)
    assert plan["current"]["id"] == "obtain_kokiri_sword"
    assert any("Saria" in row for row in plan["do_not_repeat"])

    sword = _game(state, story_flags={"greeted_by_saria": True},
        equipment=[_equipment("Kokiri Sword", "sword", item_id=59)])
    assert opening_checkpoint_plan(sword)["current"]["id"] == "obtain_deku_shield"

    both = _game(state, story_flags={"greeted_by_saria": True},
        equipment=[
            _equipment("Kokiri Sword", "sword", item_id=59),
            _equipment("Deku Shield", "shield", item_id=60),
        ])
    assert opening_checkpoint_plan(both)["current"]["id"] == "equip_sword_and_shield"

    equipped = _game(state, story_flags={"greeted_by_saria": True},
        equipment=[
            _equipment("Kokiri Sword", "sword", equipped=True, item_id=59),
            _equipment("Deku Shield", "shield", equipped=True, item_id=60),
        ])
    assert opening_checkpoint_plan(equipped)["current"]["id"] == "pass_mido"

    passed = _game(state, story_flags={
        "greeted_by_saria": True, "showed_mido_sword_shield": True,
    }, equipment=[
        _equipment("Kokiri Sword", "sword", equipped=True, item_id=59),
        _equipment("Deku Shield", "shield", equipped=True, item_id=60),
    ])
    assert opening_checkpoint_plan(passed)["current"]["id"] == "meet_deku_tree"

    met = _game(state, story_flags={
        "greeted_by_saria": True, "showed_mido_sword_shield": True,
        "met_deku_tree": True,
    }, equipment=[
        _equipment("Kokiri Sword", "sword", equipped=True, item_id=59),
        _equipment("Deku Shield", "shield", equipped=True, item_id=60),
    ])
    assert opening_checkpoint_plan(met)["current"]["id"] == "enter_deku_tree"

    inside = _game(state, scene_name="Inside the Deku Tree", story_flags={
        "greeted_by_saria": True, "showed_mido_sword_shield": True,
        "met_deku_tree": True,
    }, equipment=[
        _equipment("Kokiri Sword", "sword", equipped=True, item_id=59),
        _equipment("Deku Shield", "shield", equipped=True, item_id=60),
    ])
    final = opening_checkpoint_plan(inside)
    assert final["active"] is False
    assert final["completed_count"] == final["total"]


def test_kokiri_emerald_keeps_opening_plan_completed_after_leaving_dungeon(state):
    game = _game(state, scene_name="Kokiri Forest", story_flags={
        "greeted_by_saria": True, "showed_mido_sword_shield": True,
        "met_deku_tree": True,
    }, equipment=[
        _equipment("Kokiri Sword", "sword", equipped=True, item_id=59),
        _equipment("Deku Shield", "shield", equipped=True, item_id=60),
    ], quest_items=["Kokiri Emerald"])
    assert opening_checkpoint_plan(game)["active"] is False


def test_checkpoint_blocks_repeating_saria_and_premature_mido(state, decision):
    saria = {
        "actor_uid": "saria-1", "actor_id": 100, "name": "Saria", "description": "Saria",
        "category": 4, "category_name": "npc", "room": 0, "params": 0,
        "position": [20, 0, 0], "focus_position": [20, 20, 0], "distance": 20,
        "targeted": False, "drawn": True, "text_id": 0,
    }
    mido = {
        **saria, "actor_uid": "mido-1", "actor_id": 101, "name": "Mido",
        "description": "Mido", "position": [40, 0, 0], "distance": 40,
    }
    game = _game(state, story_flags={"greeted_by_saria": True},
        room_actors=[saria, mido])
    plan = opening_checkpoint_plan(game)
    assert plan["current"]["id"] == "obtain_kokiri_sword"

    talk_saria = Decision.model_validate({
        **decision.model_dump(),
        "skill": "talk_to_actor",
        "args": {
            **decision.args.model_dump(),
            "duration_ms": 5000,
            "target_actor_id": 100,
            "target_actor_params": 0,
        },
    })
    blocked = checkpoint_blocks_decision(game, talk_saria, plan)
    assert blocked and blocked["blocked"] == "saria_already_greeted"

    talk_mido = Decision.model_validate({
        **decision.model_dump(),
        "skill": "talk_to_actor",
        "args": {
            **decision.args.model_dump(),
            "duration_ms": 5000,
            "target_actor_id": 101,
            "target_actor_params": 0,
        },
    })
    blocked = checkpoint_blocks_decision(game, talk_mido, plan)
    assert blocked and blocked["blocked"] == "mido_not_current_checkpoint"


def test_autosave_and_story_flags_contract(state):
    game = _game(state,
        story_flags={"greeted_by_saria": True, "first_spoke_to_mido": False},
        autosave={
            "pending": False,
            "target_scene": -1,
            "last_scene": 85,
            "count": 3,
            "last_saved_at_ms": 123456,
        })
    assert game.progress.story_flags["greeted_by_saria"] is True
    assert game.autosave.count == 3
    assert game.autosave.last_scene == 85
