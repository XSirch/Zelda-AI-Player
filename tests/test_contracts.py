import math

import pytest
from pydantic import ValidationError

from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.config import Settings
from zelda_ai.models import GameState, PlayerState, RunConfig, Usage


def test_usage_token_subsets_are_not_double_counted():
    usage = Usage(
        input_tokens=100,
        cached_input_tokens=60,
        output_tokens=40,
        reasoning_output_tokens=20,
    )
    assert usage.total_tokens == 140


def test_agent_intent_schema_requires_all_nullable_fields_for_strict_output():
    schema = AgentIntent.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "skill" not in schema["properties"]
    assert "args" not in schema["properties"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_player_position_is_rejected(value):
    with pytest.raises(ValidationError):
        PlayerState(
            position=(value, 0.0, 0.0),
            yaw=0,
            health=48,
            max_health=48,
            rupees=0,
        )


def test_game_state_keeps_structured_dialogue_actor_and_progress(state):
    game = GameState.model_validate({
        **state.model_dump(),
        "dialogue": {
            "active": True,
            "text_id": 123,
            "text": "Hello Link",
            "state": "choice",
            "state_code": 2,
            "message_mode": 1,
            "can_advance": True,
            "choice_count": 2,
            "choice_index": 0,
            "choices": ["Yes", "No"],
            "speaker": {
                "actor_uid": "npc:1",
                "actor_id": 100,
                "name": "NPC",
                "description": "Observed NPC",
                "category": 4,
                "category_name": "npc",
                "room": 0,
                "params": 0,
                "position": [10, 0, 5],
                "focus_position": [10, 20, 5],
                "distance": 11.2,
                "targeted": False,
                "drawn": True,
            },
        },
        "inventory_named": [
            {"slot": 0, "item_id": 1, "name": "Observed item", "ammo": None}
        ],
        "progress": {
            "quest_items": ["observed_flag"],
            "owned_equipment": ["observed_gear"],
            "equipment": [],
            "upgrade_levels": {},
            "heart_pieces": 0,
            "skull_tokens": 0,
            "magic_acquired": False,
            "double_magic": False,
            "double_defense": False,
            "map_index": 0,
            "dungeon_items": [],
            "small_keys": 0,
            "story_flags": {},
        },
    })
    assert game.dialogue.text == "Hello Link"
    assert game.dialogue.speaker.actor_uid == "npc:1"
    assert game.inventory_named[0].name == "Observed item"
    assert game.progress.quest_items == ["observed_flag"]


def test_game_state_accepts_realtime_navigation_and_autosave_observations(state):
    game = GameState.model_validate({
        **state.model_dump(),
        "navigation_probes": [{
            "direction": "forward",
            "distance": 70,
            "floor_found": True,
            "floor_y": 0,
            "delta_y": 0,
            "floor_type": 0,
            "wall_hit": False,
            "wall_distance": None,
            "wall_flags": 0,
        }],
        "traversal_affordances": [{
            "kind": "ladder_down",
            "direction": "down",
            "approach_position": [0, 0, 50],
            "target_position": [0, -100, 50],
            "distance": 50,
            "height_delta": -100,
            "wall_flags": 0,
        }],
        "scene_exits": [{
            "exit_index": 1,
            "entrance_index": 10,
            "position": [30, 0, 0],
            "samples": 4,
            "direct_reachable": True,
        }],
        "autosave": {
            "pending": False,
            "target_scene": -1,
            "last_scene": 85,
            "count": 2,
            "last_saved_at_ms": 100,
        },
    })
    assert game.navigation_probes[0].floor_found
    assert game.traversal_affordances[0].direction == "down"
    assert game.scene_exits[0].direct_reachable
    assert game.autosave.count == 2


def test_zero_budgets_mean_unlimited_harness_limits():
    config = RunConfig(
        provider="codex",
        model="test",
        max_calls=0,
        max_tokens=0,
        max_cost_usd=0,
        max_runtime_s=0,
    )
    assert config.max_calls == 0
    assert config.max_tokens == 0
    assert config.max_cost_usd == 0
    assert config.max_runtime_s == 0


def test_blank_agent_effort_is_treated_as_unset(monkeypatch):
    monkeypatch.setenv("ZELDA_AGENT_EFFORT", "")
    settings = Settings(_env_file=None)
    assert settings.agent_effort is None
