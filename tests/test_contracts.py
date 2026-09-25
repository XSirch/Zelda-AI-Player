import json

import pytest
from pydantic import ValidationError

from zelda_ai.providers.base import SYSTEM_PROMPT
from zelda_ai.models import Decision, PlayerState, Usage
from zelda_ai.providers.codex import parse_usage as codex_usage
from zelda_ai.providers.openrouter import model_info, parse_usage, reserve_cost
from zelda_ai.providers.base import ProviderFailure
from zelda_ai.runtime import controller_input, _model_state_payload, _model_world_edges, _strip_transition_ids, _model_memory_note_persistent, _model_last_decision


def test_token_subsets_are_not_double_counted():
    u = Usage(input_tokens=100, cached_input_tokens=80, output_tokens=40, reasoning_output_tokens=30)
    assert u.total_tokens == 140
    assert Usage().total_tokens is None
    assert Usage().cost_usd is None


@pytest.mark.parametrize("patch", [{"skill": "teleport"}, {"args": {"direction": None,
    "slot": None, "choice_index": None, "song": None, "target_actor_id": None,
        "target_actor_params": None, "target_position": None, "stop_distance": None,
        "item_id": None, "duration_ms": 100, "strength": .5}}, {"shell": "anything"},
    {"args": {"direction": "forward", "slot": None, "choice_index": None, "song": None, "target_actor_id": None,
        "target_actor_params": None, "target_position": None, "stop_distance": None,
        "item_id": None, "duration_ms": 2001, "strength": .5}}])
def test_decision_rejects_invalid_actions(decision, patch):
    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(), **patch})


def test_schema_requires_every_property():
    schema = Decision.model_json_schema()
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    assert set(schema["$defs"]["SkillArgs"]["required"]) == set(schema["$defs"]["SkillArgs"]["properties"])


def test_nonfinite_state_rejected(state):
    with pytest.raises(ValidationError):
        PlayerState.model_validate({**state.player.model_dump(), "position": [float("nan"), 0, 0]})


def test_native_buttons(decision):
    assert controller_input(decision) == (0, 0, 40)
    for skill, buttons in [("attack", 0x4000), ("defend", 0x2010), ("wait", 0)]:
        assert controller_input(decision.model_copy(update={"skill": skill}))[0] == buttons


@pytest.mark.parametrize("reasoning, expected", [({}, []), ({"supported_efforts": ["high", "low"]}, ["high", "low"]),
    ({"supported_efforts": ["none", "high"], "mandatory": True}, ["high"])])
def test_effort_catalog(reasoning, expected):
    info = model_info({"id": "test/model", "reasoning": reasoning, "supported_parameters": ["structured_outputs"]})
    assert info.efforts == expected
    assert info.structured_output


def test_nullable_effort_list_has_explicit_gateway_semantics():
    info = model_info({"id": "test", "reasoning": {"supported_efforts": None}})
    assert "medium" in info.efforts and "max" in info.efforts
    assert not info.structured_output


def test_usage_parsers():
    u = parse_usage({"id": "gen-1", "model": "m", "usage": {"prompt_tokens": 100,
        "completion_tokens": 50, "cost": .02, "prompt_tokens_details": {"cached_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 30}}})
    assert u.total_tokens == 150 and u.cost_usd == .02 and u.reasoning_output_tokens == 30
    c = codex_usage({"tokenUsage": {"total": {"inputTokens": 100, "outputTokens": 40,
        "cachedInputTokens": 80, "reasoningOutputTokens": 30}}}, "m")
    assert c.total_tokens == 140 and c.cost_usd is None


def test_cost_preflight_rejects_unknown_pricing():
    with pytest.raises(ProviderFailure):
        reserve_cost(model_info({"id": "test"}), "{}", 2048)
    info = model_info({"id": "test", "pricing": {"prompt": "0.000001", "completion": "0.000002"}})
    assert reserve_cost(info, json.dumps({"state": "test"}), 2048) > .004096


def test_dialogue_choice_requires_index(decision):
    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(), "skill": "choose_dialogue"})
    chosen = Decision.model_validate({**decision.model_dump(), "skill": "choose_dialogue",
        "args": {**decision.args.model_dump(), "choice_index": 1}})
    assert chosen.args.choice_index == 1


def test_game_state_accepts_complete_room_actor_observation(state):
    door = {"actor_id": 9, "name": "En_Door", "description": "Door", "category": 10,
        "category_name": "door", "room": 0, "params": 1, "position": [140, 0, -20],
        "focus_position": [140, 30, -20], "distance": 141.4, "targeted": False,
        "drawn": False, "text_id": 0}
    enriched = type(state).model_validate({**state.model_dump(),
        "room_actors": [door], "room_actor_count": 1, "room_actors_truncated": False})
    assert enriched.room_actors[0].category_name == "door"
    assert enriched.room_actors[0].drawn is False
    assert enriched.room_actors[0].room == 0
    assert enriched.room_actor_count == 1


def test_game_state_accepts_structured_dialogue_and_drawn_actor(state):
    actor = {"actor_id": 123, "category": 4, "params": 0, "position": [10, 0, 5],
        "distance": 11.2, "targeted": False, "drawn": True, "text_id": 4097}
    enriched = state.model_copy(update={
        "dialogue": {"active": True, "text_id": 4097, "text": "Hello Link",
            "state": "choice", "state_code": 4, "message_mode": 8, "can_advance": True,
            "choice_count": 2, "choice_index": 0, "choices": ["Yes", "No"], "speaker": actor},
        "context_action": {"code": 15, "label": "speak"},
        "nearby_actors": [actor],
        "entrance_index": 52,
    })
    validated = type(state).model_validate(enriched.model_dump())
    assert validated.dialogue.text == "Hello Link"
    assert validated.dialogue.choices == ["Yes", "No"]
    assert validated.nearby_actors[0].drawn


def test_traverse_requires_vertical_direction_and_accepts_terrain_probes(decision, state):
    down = Decision.model_validate({**decision.model_dump(), "skill": "traverse",
        "args": {**decision.args.model_dump(), "direction": "down", "duration_ms": 7000}})
    assert down.args.direction == "down"
    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(), "skill": "traverse",
            "args": {**decision.args.model_dump(), "direction": "left", "duration_ms": 7000}})

    enriched = type(state).model_validate({**state.model_dump(),
        "player": {**state.player.model_dump(), "wall_flags": 2,
            "climbing_ladder": True, "can_down": True},
        "navigation_probes": [
            {"direction": "forward", "distance": 70, "floor_found": True,
             "floor_y": -80, "delta_y": -80, "floor_type": 0},
            {"direction": "back", "distance": 140, "floor_found": False,
             "floor_y": None, "delta_y": None, "floor_type": None},
        ]})
    assert enriched.player.climbing_ladder
    assert enriched.navigation_probes[0].delta_y == -80


def test_compound_navigation_accepts_longer_window(decision):
    nav = Decision.model_validate({**decision.model_dump(), "skill": "navigate_to",
        "args": {**decision.args.model_dump(), "duration_ms": 8000,
            "target_position": [120.0, 0.0, -40.0], "stop_distance": 35.0}})
    assert nav.args.duration_ms == 8000
    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(),
            "args": {**decision.args.model_dump(), "duration_ms": 8000}})


def test_actor_and_equip_high_level_skills_require_targets(decision):
    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(), "skill": "approach_actor"})
    actor = Decision.model_validate({**decision.model_dump(), "skill": "talk_to_actor",
        "args": {**decision.args.model_dump(), "target_actor_id": 123,
            "target_actor_params": 4, "duration_ms": 6000}})
    assert actor.args.target_actor_id == 123

    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(), "skill": "equip_item"})
    equipped = Decision.model_validate({**decision.model_dump(), "skill": "equip_item",
        "args": {**decision.args.model_dump(), "item_id": 7, "slot": "left",
            "duration_ms": 8000}})
    assert equipped.args.item_id == 7 and equipped.args.slot == "left"


def test_game_state_accepts_semantic_scene_inventory_and_pause_ready(state):
    enriched = type(state).model_validate({**state.model_dump(),
        "scene_name": "Kokiri Forest",
        "inventory": [255, 7],
        "inventory_named": [{"slot": 1, "item_id": 7, "name": "Fairy Ocarina"}],
        "pause_menu": {"active": True, "ready": True, "state": 6, "transition_state": 0,
            "page_index": 0, "cursor_special_pos": 0, "cursor_point": [1, 0, 0, 0, 0],
            "cursor_item": [7, 999, 999, 59], "cursor_slot": [1, 0, 0, 0],
            "named_item": 7, "prompt_choice": 0}})
    assert enriched.scene_name == "Kokiri Forest"
    assert enriched.inventory_named[0].name == "Fairy Ocarina"
    assert enriched.pause_menu.ready


def test_game_state_accepts_pause_visible_progress(state):
    enriched = type(state).model_validate({**state.model_dump(), "progress": {
        "quest_items": ["Kokiri's Emerald", "Saria's Song"],
        "owned_equipment": ["Kokiri Sword", "Deku Shield"],
        "upgrade_levels": {"bullet_bag": 1, "wallet": 1},
        "heart_pieces": 2, "skull_tokens": 4, "magic_acquired": False,
        "double_magic": False, "double_defense": False, "map_index": 0,
        "dungeon_items": ["Dungeon Map"], "small_keys": 1}})
    assert "Kokiri's Emerald" in enriched.progress.quest_items
    assert "Deku Shield" in enriched.progress.owned_equipment
    assert enriched.progress.small_keys == 1


def test_fight_enemy_requires_observed_actor_target(decision):
    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(), "skill": "fight_enemy",
            "args": {**decision.args.model_dump(), "duration_ms": 8000}})
    fight = Decision.model_validate({**decision.model_dump(), "skill": "fight_enemy",
        "args": {**decision.args.model_dump(), "duration_ms": 8000,
            "target_actor_id": 37, "target_actor_params": 0}})
    assert fight.args.target_actor_id == 37


def test_actor_metadata_is_optional_but_bounded(state):
    actor = {"actor_id": 12, "name": "En_Sa", "description": "Saria", "category": 4,
        "params": 0, "position": [1, 2, 3], "distance": 4.0, "targeted": False,
        "drawn": True, "text_id": 4096}
    enriched = type(state).model_validate({**state.model_dump(), "nearby_actors": [actor]})
    assert enriched.nearby_actors[0].description == "Saria"


def test_equip_gear_requires_item_and_progress_can_mark_equipped(decision, state):
    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(), "skill": "equip_gear",
            "args": {**decision.args.model_dump(), "duration_ms": 8000}})
    gear_decision = Decision.model_validate({**decision.model_dump(), "skill": "equip_gear",
        "args": {**decision.args.model_dump(), "duration_ms": 8000, "item_id": 0x45}})
    assert gear_decision.args.item_id == 0x45

    enriched = type(state).model_validate({**state.model_dump(), "progress": {
        "quest_items": [], "owned_equipment": ["Iron Boots"],
        "equipment": [{"item_id": 0x45, "name": "Iron Boots", "equipment_type": "boots",
            "value": 2, "equipped": True}],
        "upgrade_levels": {}, "heart_pieces": 0, "skull_tokens": 0,
        "magic_acquired": False, "double_magic": False, "double_defense": False,
        "map_index": 0, "dungeon_items": [], "small_keys": 0}})
    assert enriched.progress.equipment[0].equipped


@pytest.mark.parametrize("skill,args", [
    ("follow_actor", {"target_actor_id": 10, "duration_ms": 6000}),
    ("interact_with_actor", {"target_actor_id": 10, "duration_ms": 5000}),
    ("manipulate_object", {"target_actor_id": 10, "direction": "forward", "duration_ms": 6000}),
    ("face_target", {"target_position": [10, 0, 20], "duration_ms": 3000}),
    ("shield_face", {"target_actor_id": 10, "duration_ms": 3000}),
    ("aim_at", {"target_actor_id": 10, "slot": "left", "duration_ms": 5000}),
    ("explore_area", {"duration_ms": 10000}),
])
def test_autonomy_v2_compound_skill_contracts(decision, skill, args):
    payload = {**decision.args.model_dump(), **args}
    parsed = Decision.model_validate({**decision.model_dump(), "skill": skill, "args": payload})
    assert parsed.skill == skill


def test_manipulate_object_rejects_sideways_direction(decision):
    with pytest.raises(ValidationError):
        Decision.model_validate({**decision.model_dump(), "skill": "manipulate_object",
            "args": {**decision.args.model_dump(), "target_actor_id": 10,
                "direction": "left", "duration_ms": 5000}})


def test_scene_exit_observation_and_traverse_exit_contract(decision, state):
    enriched = type(state).model_validate({
        **state.model_dump(),
        "capabilities": [*state.capabilities, "scene_exit_surfaces"],
        "scene_exits": [{
            "exit_index": 1,
            "entrance_index": 0x211,
            "position": [70.0, 0.0, 116.0],
            "samples": 4,
        }],
    })
    assert enriched.scene_exits[0].entrance_index == 0x211
    traverse = Decision.model_validate({
        **decision.model_dump(),
        "skill": "traverse_exit",
        "args": {
            **decision.args.model_dump(),
            "duration_ms": 8000,
            "target_position": [70.0, 0.0, 116.0],
        },
    })
    assert traverse.skill == "traverse_exit"
    with pytest.raises(ValidationError):
        Decision.model_validate({
            **decision.model_dump(),
            "skill": "traverse_exit",
            "args": {**decision.args.model_dump(), "duration_ms": 8000, "target_position": None},
        })


def test_planner_prompt_understands_actorless_scene_exits():
    assert 'traverse_exit(target_position)' in SYSTEM_PROMPT
    assert 'scene_exits' in SYSTEM_PROMPT
    assert "Link's House" in SYSTEM_PROMPT
    assert 'no door actor' in SYSTEM_PROMPT
    assert 'destination is unknown' in SYSTEM_PROMPT
    assert 'MUST NOT be used to infer where it leads' in SYSTEM_PROMPT
    assert 'Do not write a memory_note claiming where an untraversed transition leads' in SYSTEM_PROMPT


def test_model_sees_scene_exit_as_opaque_position_only(state):
    game = type(state).model_validate({
        **state.model_dump(),
        "capabilities": [*state.capabilities, "scene_exit_surfaces"],
        "scene_exits": [{
            "exit_index": 7,
            "entrance_index": 0x211,
            "position": [70.0, 0.0, 116.0],
            "samples": 9,
        }],
    })
    payload = _model_state_payload(game)
    assert payload["scene_exits"] == [{"position": [70.0, 0.0, 116.0]}]
    serialized = json.dumps(payload["scene_exits"])
    assert "exit_index" not in serialized
    assert "entrance_index" not in serialized
    assert "samples" not in serialized
    assert "entrance_index" not in payload



def test_model_world_edges_expose_only_empirically_observed_topology():
    rows = [{
        "id": "edge-secret",
        "from_scene": 1,
        "from_scene_name": "Origin",
        "from_room": 0,
        "from_position": [70.0, 0.0, 116.0],
        "to_scene": 2,
        "to_scene_name": "Observed Destination",
        "to_room": 0,
        "to_position": [0.0, 0.0, 0.0],
        "entrance_index": 0x211,
        "traversals": 1,
        "created_at": 1.0,
        "updated_at": 2.0,
    }]
    visible = _model_world_edges(rows)
    assert visible[0]["to_scene_name"] == "Observed Destination"
    assert visible[0]["from_position"] == [70.0, 0.0, 116.0]
    assert "entrance_index" not in visible[0]
    assert "id" not in visible[0]


def test_model_neutralizes_transition_actor_destination_labels(state):
    warp_actor = {
        "actor_uid": "warp-1",
        "actor_id": 123,
        "name": "Door_Warp1",
        "description": "Warp to Kokiri Forest",
        "category": 7,
        "category_name": "item_action",
        "room": 0,
        "params": 44,
        "position": [20, 0, 30],
        "focus_position": [20, 20, 30],
        "distance": 35.0,
        "targeted": False,
        "drawn": True,
        "text_id": 0,
    }
    game = type(state).model_validate({
        **state.model_dump(),
        "room_actors": [warp_actor],
        "room_actor_count": 1,
    })
    payload = _model_state_payload(game)
    actor = payload["room_actors"][0]
    assert actor["name"] == "Transition Object"
    assert actor["description"] == ""
    serialized = json.dumps(actor)
    assert "Kokiri Forest" not in serialized
    # Targeting identity remains available; only semantic destination clues are removed.
    assert actor["actor_id"] == 123
    assert actor["actor_uid"] == "warp-1"

    exit_actor = {**warp_actor, "actor_uid": "exit-1", "name": "Kokiri Forest Exit",
                  "description": "Loads the exterior"}
    exit_game = type(state).model_validate({
        **state.model_dump(), "room_actors": [exit_actor], "room_actor_count": 1,
    })
    exit_visible = _model_state_payload(exit_game)["room_actors"][0]
    assert exit_visible["name"] == "Transition Object"
    assert "Kokiri Forest" not in json.dumps(exit_visible)



def test_native_transition_ids_are_removed_from_model_results_and_events():
    raw = {
        "status": "completed",
        "reason": "scene_exit_traversed",
        "exit_index": 3,
        "entrance_index": 0x211,
        "nested": {
            "exit_index": 4,
            "detail": "kept",
        },
        "rows": [{"entrance_index": 0x222, "value": 7}],
    }
    visible = _strip_transition_ids(raw)
    serialized = json.dumps(visible)
    assert "exit_index" not in serialized
    assert "entrance_index" not in serialized
    assert visible["nested"]["detail"] == "kept"
    assert visible["rows"][0]["value"] == 7
    floor = _strip_transition_ids({"floor_exit_index": 7, "value": 1})
    assert floor == {"value": 1}


    nested_actor = {
        "actor_id": 77, "actor_uid": "warp-result", "category": 7,
        "category_name": "item_action", "name": "Warp Portal",
        "description": "Exit to a hidden destination", "params": 2,
    }
    actor_visible = _strip_transition_ids({
        "actor": nested_actor,
        "dialogue": {"speaker": nested_actor},
    })
    assert actor_visible["actor"]["name"] == "Transition Object"
    assert actor_visible["actor"]["description"] == ""
    assert actor_visible["dialogue"]["speaker"]["name"] == "Transition Object"
    assert "hidden destination" not in json.dumps(actor_visible)



def test_freeform_model_memory_is_never_persistent(decision):
    transition = Decision.model_validate({
        **decision.model_dump(),
        "skill": "traverse_exit",
        "args": {
            **decision.args.model_dump(),
            "target_position": [70, 0, 116],
            "duration_ms": 8000,
        },
        "memory_note": "This warp leads somewhere I am guessing.",
    })
    ordinary = Decision.model_validate({
        **decision.model_dump(),
        "skill": "wait",
        "args": {**decision.args.model_dump(), "duration_ms": 500},
        "memory_note": "A guessed world topology note hidden on wait.",
    })
    assert not _model_memory_note_persistent(transition)
    assert not _model_memory_note_persistent(ordinary)


def test_model_does_not_receive_previous_decision_guessed_prose():
    visible = _model_last_decision({
        "goal": "Go to a guessed destination",
        "summary": "I think this warp leads to Kokiri Forest",
        "skill": "traverse_exit",
        "args": {"target_position": [70, 0, 116], "entrance_index": 0x211},
        "memory_note": "Warp leads to Kokiri Forest",
    })
    assert visible == {
        "skill": "traverse_exit",
        "args": {"target_position": [70, 0, 116]},
    }
    assert "Kokiri Forest" not in json.dumps(visible)
