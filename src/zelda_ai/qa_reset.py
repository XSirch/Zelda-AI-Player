"""QA-only normal SoH reset via its configured controller hotkey.

No native reset opcode, shell command, teleport or save writer reaches the
motor. Configure only a copied QA home and keep this helper out of gameplay.
"""
from __future__ import annotations

import asyncio
import copy
import time

QA_RESET_BUTTONS = 0x3010  # Physical START + Z + R; pinned SoH ResetHotKey mask.


def soft_reset_config(config):
    """Return a copy with the normal SoH reset binding; preserve the source."""
    result = copy.deepcopy(config)
    result.setdefault("CVars", {}).setdefault("gSettings", {})["ResetBtn"] = QA_RESET_BUTTONS
    return result


async def reset_to_title(bridge, *, configured_buttons, budget_s=8):
    if configured_buttons != QA_RESET_BUTTONS or not .05 <= budget_s <= 30:
        raise ValueError("Use the explicitly configured QA reset binding and a bounded budget")
    initial = bridge.state
    if (not bridge.connected or not bridge.realtime or not initial or initial.source != "soh"
            or initial.protocol != 3 or not initial.in_game or not initial.player):
        raise ValueError("Reset requires the current real playable QA session")
    result = {"source": "soh", "success": False, "reason": "reset_effect_timeout",
              "same_native_process": None, "consumed": None, "receipt": None,
              "initial_seq": initial.seq, "initial_scene_epoch": initial.scene_epoch,
              "initial_context_epoch": initial.context_epoch, "provider_calls": 0}
    deadline = time.monotonic() + budget_s
    bridge.release()
    try:
        # QA hashing/reporting can leave a queued, old observation. Yield to
        # native reception before the ONE reset pulse; do not retry an input.
        fresh = await bridge.next_state(initial.seq, timeout=min(1.5, budget_s))
        if fresh.instance_id != initial.instance_id:
            result.update(reason="native_instance_changed", same_native_process=False)
            return result
        if (not fresh.in_game or not fresh.player or fresh.scene_epoch != initial.scene_epoch
                or fresh.context_epoch != initial.context_epoch):
            result["reason"] = "context_changed_before_reset_pulse"
            return result
        result["command_base_seq"] = fresh.seq
        receipt = await bridge.pulse_receipt(buttons=configured_buttons, hold_ticks=1,
                                            timeout=min(1.5, max(.01, deadline - time.monotonic())))
        if receipt is not None:
            result["receipt"] = receipt.model_dump()
            result["consumed"] = (receipt.first_tick > 0
                                  and receipt.pressed & configured_buttons == configured_buttons)
        while time.monotonic() < deadline:
            game = bridge.state
            if game and game.instance_id != initial.instance_id:
                result.update(reason="native_instance_changed", same_native_process=False)
                break
            if not bridge.connected:
                result["reason"] = "bridge_unavailable"
                break
            if (game and game.seq > initial.seq and not game.in_game and game.startup.phase == 1
                    and (game.scene_epoch > initial.scene_epoch or game.context_epoch > initial.context_epoch)):
                result.update(same_native_process=True, final_seq=game.seq,
                              final_scene_epoch=game.scene_epoch, final_context_epoch=game.context_epoch)
                if result["consumed"] is True:
                    result.update(success=True, reason="observed_normal_title_reset")
                else:
                    result["reason"] = "title_observed_without_consumed_reset_receipt"
                break
            await asyncio.sleep(.05)
        return result
    finally:
        bridge.release()
