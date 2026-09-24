"""Local aim controller helpers."""
from __future__ import annotations

import asyncio
import math
import time
from ..bridge import Bridge
from ..control.feedback import consumed, feedback
from ..models import Decision, GameState
from .catalog import BUTTONS
from .common import _aim_error, _aim_stick, _matching_actor

async def _aim_at(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    if not current or not current.player:
        return {"status": "failed", "reason": "player_state_unavailable", "skill": decision.skill}
    if current.paused or current.dialogue.active or current.cutscene_active or current.game_over_state != 0:
        return {"status": "stale", "reason": "gameplay_state_blocks_aim", "skill": decision.skill}

    slot = decision.args.slot
    if slot is None:
        return {"status": "failed", "reason": "aim_slot_missing", "skill": decision.skill}
    button_index = {"left": 1, "down": 2, "right": 3}[slot]
    if len(current.equipped) <= button_index or current.equipped[button_index] in {0xFE, 0xFF}:
        return {"status": "failed", "reason": "aim_slot_empty", "slot": slot, "skill": decision.skill}
    item_id = current.equipped[button_index]
    button = BUTTONS[f"C_{slot.upper()}"]

    deadline = time.monotonic() + min(8.0, max(0.6, decision.args.duration_ms / 1000))
    first_command = None
    acknowledged = False
    sign_x = sign_y = 1
    flipped_x = flipped_y = False
    previous_abs = None
    previous_stick = (0, 0)
    stable = 0
    last_seq = -1
    before_events = {event.id for event in current.events}

    # Give the equipped aiming item a short moment to enter first-person/aim state.
    first_command = bridge.send(buttons=button, lease_ms=300)
    await asyncio.sleep(0.22)

    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not current or not bridge.connected:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
                return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
            if current.paused or current.dialogue.active or current.cutscene_active or current.game_over_state != 0:
                return {"status": "interrupted", "reason": "gameplay_state_changed", "skill": decision.skill}

            if consumed(bridge, first_command):
                acknowledged = True
            if current.seq == last_seq:
                await asyncio.sleep(0.04)
                continue
            last_seq = current.seq

            actor = None
            if decision.args.target_actor_id is not None:
                actor = _matching_actor(current, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
                if actor is None:
                    return {"status": "failed", "reason": "aim_target_actor_not_observed",
                        "item_id": item_id, "skill": decision.skill}
                target = actor.focus_position or actor.position
            else:
                target = decision.args.target_position
            error = _aim_error(current, target)
            if error is None:
                return {"status": "failed", "reason": "camera_state_unavailable",
                    "item_id": item_id, "skill": decision.skill}
            yaw_error, pitch_error = error
            abs_error = (abs(yaw_error), abs(pitch_error))

            if previous_abs is not None:
                if previous_stick[0] != 0 and abs_error[0] > previous_abs[0] + math.radians(0.5) and not flipped_x:
                    sign_x *= -1
                    flipped_x = True
                if previous_stick[1] != 0 and abs_error[1] > previous_abs[1] + math.radians(0.5) and not flipped_y:
                    sign_y *= -1
                    flipped_y = True

            if abs_error[0] <= math.radians(2.2) and abs_error[1] <= math.radians(2.2):
                stable += 1
                if stable >= 2:
                    bridge.release()  # Releasing the held C-button fires bow/slingshot/hookshot-style items.
                    await asyncio.sleep(0.35)
                    after = bridge.state
                    defeated = None
                    if after:
                        acknowledged |= consumed(bridge, first_command)
                        defeated = next((event for event in after.events
                            if event.id not in before_events and event.kind in {"enemy_defeated", "boss_defeated"}
                            and (decision.args.target_actor_id is None or
                                 event.detail == str(decision.args.target_actor_id))), None)
                    return {"status": "completed",
                        "reason": defeated.kind if defeated else "aligned_shot_released",
                        "item_id": item_id, "slot": slot,
                        "yaw_error_deg": round(math.degrees(yaw_error), 2),
                        "pitch_error_deg": round(math.degrees(pitch_error), 2),
                        "target_defeated": defeated is not None,
                        "acknowledged": acknowledged, "skill": decision.skill}
            else:
                stable = 0

            stick_x = _aim_stick(yaw_error, sign_x)
            stick_y = _aim_stick(pitch_error, sign_y)
            previous_abs = abs_error
            previous_stick = (stick_x, stick_y)
            command_id = bridge.send(buttons=button, stick_x=stick_x, stick_y=stick_y, lease_ms=260)
            if first_command is None:
                first_command = command_id
            await feedback(bridge, current, .10)
    finally:
        bridge.release()

    after = bridge.state
    return {"status": "failed", "reason": "aim_timeout", "item_id": item_id, "slot": slot,
        "acknowledged": bool(after and consumed(bridge, first_command)) or acknowledged,
        "skill": decision.skill}
