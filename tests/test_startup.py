import pytest

from zelda_ai.models import GameEvent, StartupState
from zelda_ai.startup import enter_playable_save


class StartupBridge:
    def __init__(self, state):
        self.state = state
        self.connected = True
        self.released = False

    def release(self):
        self.released = True

    async def sequence_receipt(self, *args, **kwargs):
        raise AssertionError("No input is authorized for this observation")


@pytest.mark.asyncio
async def test_missing_save_fails_closed_and_releases_input(state):
    state.in_game = False
    state.capabilities = ["startup_controls_v1"]
    state.startup = StartupState(phase=2, existing_slots=[True, False, False])
    bridge = StartupBridge(state)
    result = await enter_playable_save(bridge, 1)
    assert result["reason"] == "requested_save_missing"
    assert bridge.released


@pytest.mark.asyncio
async def test_wrong_file_load_is_not_accepted_as_startup_success(state):
    state.events = [GameEvent(id="load-2", kind="save_loaded", detail="0")]
    bridge = StartupBridge(state)
    result = await enter_playable_save(bridge, 1)
    assert result["reason"] == "different_save_loaded"
    assert bridge.released


@pytest.mark.asyncio
async def test_startup_cancellation_releases_input(state):
    import asyncio

    state.in_game = False
    state.capabilities = ["startup_controls_v1"]
    state.startup = StartupState(phase=1, existing_slots=[True, True, False])
    bridge = StartupBridge(state)

    async def cancelled(*args, **kwargs):
        raise asyncio.CancelledError

    bridge.sequence_receipt = cancelled
    with pytest.raises(asyncio.CancelledError):
        await enter_playable_save(bridge, 1)
    assert bridge.released
