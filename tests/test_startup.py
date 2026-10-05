import pytest

from zelda_ai.models import GameEvent, StartupState
from zelda_ai.startup import StartupFailure, enter_playable_save


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


@pytest.mark.asyncio
async def test_startup_failure_preserves_the_interrupted_probe_and_last_observation(state):
    from zelda_ai.models import InputReceipt

    state.in_game = False
    state.capabilities = ["startup_controls_v1"]
    state.startup = StartupState(phase=3, cursor=0, existing_slots=[True, True, False])
    bridge = StartupBridge(state)
    bridge.receipts = {7: InputReceipt(seq=7, owner_epoch=1, status="accepted",
                                    first_tick=12, last_tick=12, pressed=0, released=0)}
    bridge.telemetry = lambda: {"last_seen_age_s": .351, "rejected_packets": 0}

    async def gap(*args, **kwargs):
        assert kwargs["startup_slot"] == 1
        raise RuntimeError("state_feedback_timeout")

    bridge.sequence_receipt = gap
    retained = []
    with pytest.raises(StartupFailure, match="state_feedback_timeout") as error:
        await enter_playable_save(bridge, 1, failure_report=retained.append)
    report = error.value.report
    assert bridge.released
    assert report["status"] == "failed" and report["probes"] == 1
    assert report["error_kind"] == "RuntimeError"
    assert report["trace"][0]["status"] == "interrupted"
    assert report["trace"][0]["seq"] == state.seq
    assert report["trace"][0]["receipt"] is None
    # The exact call failed before returning its receipt. Recent native evidence
    # may already contain consumption; missing call feedback is not no delivery.
    assert report["trace"][0]["delivered"] is None
    assert report["last_observation"]["startup"]["phase"] == 3
    assert report["recent_receipts"][0]["seq"] == 7
    assert report["bridge"]["last_seen_age_s"] == .351
    assert retained == [report]


@pytest.mark.asyncio
async def test_startup_report_write_failure_still_releases_input(state):
    state.in_game = False
    state.capabilities = ["startup_controls_v1"]
    state.startup = StartupState(phase=1, existing_slots=[True, True, False])
    bridge = StartupBridge(state)

    async def gap(*args, **kwargs):
        raise RuntimeError("state_feedback_timeout")

    def broken_writer(report):
        raise OSError("unit-test-report-unavailable")

    bridge.sequence_receipt = gap
    with pytest.raises(OSError, match="unit-test-report-unavailable"):
        await enter_playable_save(bridge, 1, failure_report=broken_writer)
    assert bridge.released
