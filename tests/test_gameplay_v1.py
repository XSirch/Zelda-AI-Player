import pytest

from conftest import packet
from test_bridge_native import Transport
from zelda_ai.bridge import Bridge
from zelda_ai.models import GameState, RunConfig
from zelda_ai.runtime import Runtime, execute_skill
from zelda_ai.simulator import DemoProvider


def with_dialogue(state, **overrides):
    dialogue = {
        "active": True,
        "text_id": 0x1001,
        "text": "Hello Link",
        "state": "done",
        "state_code": 6,
        "message_mode": 8,
        "can_advance": True,
        "choice_count": 0,
        "choice_index": 0,
        "choices": [],
        "speaker": None,
        **overrides,
    }
    return GameState.model_validate({**state.model_dump(), "seq": state.seq + 1, "dialogue": dialogue})


@pytest.mark.asyncio
async def test_dialogue_preempts_world_movement(state, decision):
    bridge = Bridge("x" * 32)
    transport = Transport()
    bridge.connection_made(transport)
    game = with_dialogue(state)
    bridge.datagram_received(packet(game), ("127.0.0.1", 5000))
    result = await execute_skill(bridge, decision, game)
    assert result["status"] == "stale"
    assert result["reason"] == "dialogue_requires_handling"
    assert transport.sent == []


@pytest.mark.asyncio
async def test_cutscene_preempts_world_movement(state, decision):
    bridge = Bridge("x" * 32)
    transport = Transport()
    bridge.connection_made(transport)
    game = GameState.model_validate({**state.model_dump(), "seq": state.seq + 1, "cutscene_active": True})
    bridge.datagram_received(packet(game), ("127.0.0.1", 5000))
    result = await execute_skill(bridge, decision, game)
    assert result["status"] == "stale"
    assert result["reason"] == "cutscene_active"
    assert transport.sent == []


def test_room_transition_is_logged_and_releases_input(store, state):
    bridge = Bridge("x" * 32, True)
    transport = Transport()
    bridge.connection_made(transport)
    bridge.datagram_received(packet(state, source="simulator"), ("127.0.0.1", 5000))
    runtime = Runtime(bridge, store, {"demo": DemoProvider()})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    runtime.state = "running"

    changed = GameState.model_validate({
        **state.model_dump(),
        "source": "simulator",
        "seq": state.seq + 1,
        "scene_epoch": state.scene_epoch + 1,
        "room": state.room + 1,
    })
    bridge.datagram_received(packet(changed, source="simulator"), ("127.0.0.1", 5000))

    assert any(event["kind"] == "world_transition" for event in runtime.recent)
    assert runtime.last_result["reason"] == "world_transition"
    assert transport.sent
    assert b'"active":false' in transport.sent[-1][0]
