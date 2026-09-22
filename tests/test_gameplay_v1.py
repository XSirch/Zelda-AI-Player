import pytest

from conftest import packet
from zelda_ai.bridge import Bridge
from zelda_ai.models import GameState, RunConfig
from zelda_ai.runtime import Runtime, execute_skill
from zelda_ai.simulator import DemoProvider


class Transport:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append((data, addr))

    def close(self):
        pass


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


@pytest.mark.asyncio
async def test_gameover_continue_is_handled_locally(store, state):
    bridge = Bridge("x" * 32, True)
    transport = Transport()
    bridge.connection_made(transport)
    game = GameState.model_validate({**state.model_dump(), "seq": state.seq + 1,
        "source": "simulator", "game_over_state": 4,
        "pause_menu": {"active": True, "ready": False, "state": 0xE,
            "transition_state": 0, "page_index": 0, "cursor_special_pos": 0,
            "cursor_point": [], "cursor_item": [], "cursor_slot": [],
            "named_item": None, "prompt_choice": 0}})
    bridge.datagram_received(packet(game, source="simulator"), ("127.0.0.1", 5000))
    runtime = Runtime(bridge, store, {})
    assert await runtime._handle_gameover(game)
    assert any(b'"buttons":32768' in payload and b'"active":true' in payload
        for payload, _ in transport.sent)
