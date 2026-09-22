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
    "slot": None, "choice_index": None, "song": None, "duration_ms": 100, "strength": .5}}, {"shell": "anything"},
    {"args": {"direction": "forward", "slot": None, "choice_index": None,
        "duration_ms": 2001, "strength": .5}}])
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
