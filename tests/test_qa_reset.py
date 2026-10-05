import copy

import pytest

from zelda_ai.models import InputReceipt
from zelda_ai.qa_reset import QA_RESET_BUTTONS, reset_to_title, soft_reset_config


def test_soft_reset_binding_changes_only_copied_qa_config():
    original = {"CVars": {"gSettings": {"ResetBtn": 128, "Volume": 50}}, "Window": {"Width": 800}}
    before = copy.deepcopy(original)
    modified = soft_reset_config(original)
    assert original == before
    assert modified["CVars"]["gSettings"]["ResetBtn"] == 0x3010
    assert modified["Window"] == original["Window"]
    assert modified["CVars"]["gSettings"]["Volume"] == 50


class ResetBridge:
    connected = realtime = True

    def __init__(self, state, *, effect=True, consumed=True, new_process=False):
        self.state = state.model_copy(deep=True)
        self.state.protocol = 3
        self.effect, self.consumed, self.new_process = effect, consumed, new_process
        self.releases, self.pulses = 0, []

    def release(self):
        self.releases += 1

    async def next_state(self, after_seq, **kwargs):
        assert after_seq == self.state.seq
        self.state = self.state.model_copy(deep=True)
        self.state.seq += 1
        return self.state

    async def pulse_receipt(self, **kwargs):
        self.pulses.append(kwargs)
        if self.effect:
            self.state = self.state.model_copy(deep=True)
            self.state.seq += 1
            self.state.context_epoch += 1
            self.state.in_game = False
            self.state.player = None
            self.state.startup.phase = 1
        if self.new_process:
            self.state.instance_id = "new-process"
        return InputReceipt(seq=1, owner_epoch=1, status="cancelled",
                            first_tick=10 if self.consumed else 0, last_tick=10 if self.consumed else 0,
                            pressed=QA_RESET_BUTTONS if self.consumed else 0, released=0)


@pytest.mark.asyncio
async def test_reset_requires_consumed_input_and_real_title_effect_in_same_process(state):
    bridge = ResetBridge(state)
    result = await reset_to_title(bridge, configured_buttons=QA_RESET_BUTTONS)
    assert result["success"] and result["same_native_process"]
    # A context-fenced receipt can be cancelled AFTER it caused the reset.
    assert result["consumed"] and result["receipt"]["status"] == "cancelled"
    assert bridge.pulses[0]["buttons"] == 0x3010
    assert bridge.pulses[0]["hold_ticks"] == 1
    assert bridge.releases == 2


@pytest.mark.asyncio
async def test_reset_waits_for_native_reception_without_retrying_stale_input(state):
    bridge = ResetBridge(state)

    async def missing_fresh_state(*args, **kwargs):
        raise RuntimeError("state_feedback_timeout")

    bridge.next_state = missing_fresh_state
    with pytest.raises(RuntimeError, match="state_feedback_timeout"):
        await reset_to_title(bridge, configured_buttons=QA_RESET_BUTTONS)
    assert not bridge.pulses and bridge.releases == 2


@pytest.mark.asyncio
async def test_reset_does_not_pulse_when_scene_changed_during_freshness_barrier(state):
    bridge = ResetBridge(state)

    async def scene_changed(*args, **kwargs):
        bridge.state = bridge.state.model_copy(deep=True)
        bridge.state.seq += 1
        bridge.state.scene_epoch += 1
        return bridge.state

    bridge.next_state = scene_changed
    result = await reset_to_title(bridge, configured_buttons=QA_RESET_BUTTONS)
    assert not result["success"] and result["reason"] == "context_changed_before_reset_pulse"
    assert not bridge.pulses and bridge.releases == 2


@pytest.mark.parametrize("effect,consumed,new_process,reason", (
    (False, True, False, "reset_effect_timeout"),
    (True, False, False, "title_observed_without_consumed_reset_receipt"),
    (True, True, True, "native_instance_changed"),
))
@pytest.mark.asyncio
async def test_receipt_alone_or_other_process_cannot_qualify_reset(state, effect, consumed, new_process, reason):
    bridge = ResetBridge(state, effect=effect, consumed=consumed, new_process=new_process)
    result = await reset_to_title(bridge, configured_buttons=QA_RESET_BUTTONS, budget_s=.05)
    assert not result["success"] and result["reason"] == reason
    assert bridge.releases == 2


@pytest.mark.asyncio
async def test_reset_error_releases_input_and_never_retries(state):
    bridge = ResetBridge(state)
    async def interrupted(**kwargs):
        raise RuntimeError("state feedback unavailable")
    bridge.pulse_receipt = interrupted
    with pytest.raises(RuntimeError, match="feedback"):
        await reset_to_title(bridge, configured_buttons=QA_RESET_BUTTONS)
    assert bridge.releases == 2
