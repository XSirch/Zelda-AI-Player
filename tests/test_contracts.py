import json

import pytest
from pydantic import ValidationError

from zelda_ai.models import Decision, PlayerState, Usage
from zelda_ai.providers.codex import parse_usage as codex_usage
from zelda_ai.providers.openrouter import model_info, parse_usage, reserve_cost
from zelda_ai.providers.base import ProviderFailure
from zelda_ai.runtime import controller_input


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
