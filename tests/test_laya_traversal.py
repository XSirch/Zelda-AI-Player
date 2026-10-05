import asyncio
from types import SimpleNamespace

import pytest

from zelda_ai.laya_traversal import fresh_collision_snapshot


def test_return_to_ground_waits_for_new_full_collision_observation(state):
    async def scenario():
        state.full_seq = 10
        states = [state.model_copy(update={"seq":state.seq+1}),
                  state.model_copy(update={"seq":state.seq+2,"full_seq":11})]
        bridge = SimpleNamespace(state=state,calls=0)
        async def next_state(seq, *, timeout):
            assert timeout > 0 and seq == bridge.state.seq
            bridge.state = states[bridge.calls]
            bridge.calls += 1
            return bridge.state
        bridge.next_state = next_state
        result = await fresh_collision_snapshot(bridge)
        assert result.full_seq == 11 and bridge.calls == 2
    asyncio.run(scenario())


def test_collision_handoff_cannot_cross_scene_or_reload_epoch(state):
    async def scenario():
        bridge = SimpleNamespace(state=state)
        async def next_state(seq, *, timeout):
            bridge.state = state.model_copy(update={"seq":state.seq+1,"full_seq":state.full_seq+1,
                                                    "scene_epoch":state.scene_epoch+1})
        bridge.next_state = next_state
        with pytest.raises(RuntimeError,match="Physical context changed"):
            await fresh_collision_snapshot(bridge)
    asyncio.run(scenario())
