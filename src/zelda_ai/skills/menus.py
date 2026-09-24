"""Local menus controller helpers."""
from __future__ import annotations

import asyncio
import time
from ..bridge import Bridge
from ..control.feedback import consumed, is_realtime
from ..models import Decision, GameState
from .catalog import BUTTONS, SONG_IDS, SONG_NOTES
from .common import _pulse

async def _choose_dialogue(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    if not current or not bridge.connected:
        raise RuntimeError("bridge_disconnected")
    if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    target = decision.args.choice_index
    dialogue = current.dialogue
    if not dialogue.active or dialogue.choice_count < 2 or target is None or target >= dialogue.choice_count:
        return {"status": "failed", "reason": "dialogue_choice_not_available", "skill": decision.skill}

    acknowledged = False
    first_command = None
    for _ in range(5):
        current = bridge.state
        if not current or (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
            return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
        if not current.dialogue.active or current.dialogue.choice_count < 2:
            return {"status": "interrupted", "reason": "dialogue_changed", "skill": decision.skill}
        if current.dialogue.choice_index == target:
            break
        stick_y = -60 if current.dialogue.choice_index < target else 60
        if is_realtime(bridge):
            acknowledged |= await bridge.pulse(stick_y=stick_y, hold_ticks=2)
            continue
        command_id = bridge.send(stick_y=stick_y, lease_ms=120)
        if first_command is None:
            first_command = command_id
        await asyncio.sleep(0.14)
        bridge.release()
        await asyncio.sleep(0.14)
        sample = bridge.state
        if sample and first_command is not None:
            acknowledged |= consumed(bridge, first_command)
    current = bridge.state
    if not current or current.dialogue.choice_index != target:
        return {"status": "failed", "reason": "choice_cursor_did_not_reach_target",
            "acknowledged": acknowledged, "skill": decision.skill}

    if is_realtime(bridge):
        acknowledged |= await bridge.pulse(buttons=BUTTONS["A"])
        return {"status": "completed" if acknowledged else "failed", "reason": "choice_input_consumed",
            "choice_index": target, "acknowledged": acknowledged, "skill": decision.skill}
    command_id = bridge.send(buttons=BUTTONS["A"], lease_ms=120)
    if first_command is None:
        first_command = command_id
    await asyncio.sleep(0.14)
    bridge.release()
    await asyncio.sleep(0.2)
    after = bridge.state
    if after and first_command is not None:
        acknowledged |= consumed(bridge, first_command)
    return {"status": "completed" if acknowledged else "failed",
        "reason": "choice_confirmed" if acknowledged else "input_not_acknowledged",
        "choice_index": target, "acknowledged": acknowledged, "skill": decision.skill}


async def _play_song(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    song = decision.args.song
    if not current or not bridge.connected:
        raise RuntimeError("bridge_disconnected")
    if song not in SONG_NOTES:
        return {"status": "failed", "reason": "unknown_song", "skill": decision.skill}
    if current.ocarina_mode == 0:
        return {"status": "failed", "reason": "ocarina_not_active", "skill": decision.skill}
    before_event_ids = {event.id for event in current.events}
    acknowledged = False
    first_command = None
    for note in SONG_NOTES[song]:
        current = bridge.state
        if not current or (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
            return {"status": "interrupted", "reason": "world_changed", "skill": decision.skill}
        if current.ocarina_mode == 0:
            return {"status": "interrupted", "reason": "ocarina_closed", "skill": decision.skill}
        if is_realtime(bridge):
            delivered = await bridge.pulse(buttons=BUTTONS[note], hold_ticks=2)
            acknowledged |= delivered
            if not delivered:
                return {"status": "failed", "reason": "ocarina_note_not_consumed", "skill": decision.skill}
            continue
        command_id = bridge.send(buttons=BUTTONS[note], lease_ms=90)
        if first_command is None:
            first_command = command_id
        await asyncio.sleep(0.11)
        bridge.release()
        await asyncio.sleep(0.07)
        sample = bridge.state
        if sample and first_command is not None:
            acknowledged |= consumed(bridge, first_command)
    await asyncio.sleep(0.35)
    after = bridge.state
    if after and first_command is not None:
        acknowledged |= consumed(bridge, first_command)
    expected = SONG_IDS[song]
    confirmed = bool(after and any(
        event.id not in before_event_ids and event.kind == "ocarina_song_action" and event.detail == str(expected)
        for event in after.events))
    return {"status": "completed" if acknowledged else "failed",
        "reason": "song_confirmed" if confirmed else ("notes_sent" if acknowledged else "input_not_acknowledged"),
        "song": song, "song_id": expected, "confirmed": confirmed,
        "acknowledged": acknowledged, "skill": decision.skill}


def _inventory_slot(game: GameState, item_id: int) -> int | None:
    for item in game.inventory_named:
        if item.item_id == item_id:
            return item.slot
    for slot, value in enumerate(game.inventory):
        if value == item_id:
            return slot
    return None


def _menu_grid_directions(current_slot: int, target_slot: int, width: int = 6) -> list[str]:
    current_row, current_col = divmod(current_slot, width)
    target_row, target_col = divmod(target_slot, width)
    preferred = []
    if current_col != target_col:
        preferred.append("right" if target_col > current_col else "left")
    if current_row != target_row:
        preferred.append("down" if target_row > current_row else "up")
    for direction in ("right", "down", "left", "up"):
        if direction not in preferred:
            preferred.append(direction)
    return preferred


async def _wait_pause_ready(bridge: Bridge, timeout_s: float = 2.5) -> GameState | None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        current = bridge.state
        if current and current.pause_menu.active and current.pause_menu.ready:
            return current
        await asyncio.sleep(0.08)
    return None


async def _equip_item(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    item_id = decision.args.item_id
    c_slot = decision.args.slot
    if not current or not current.player or item_id is None or c_slot is None:
        return {"status": "failed", "reason": "equip_arguments_or_state_missing", "skill": decision.skill}
    if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    if current.game_over_state != 0:
        return {"status": "stale", "reason": "game_over_active", "skill": decision.skill}
    if current.dialogue.active or current.cutscene_active:
        return {"status": "stale", "reason": "gameplay_state_blocks_pause", "skill": decision.skill}

    target_slot = _inventory_slot(current, item_id)
    if target_slot is None or target_slot >= 24:
        return {"status": "failed", "reason": "item_not_owned_or_not_assignable",
            "item_id": item_id, "skill": decision.skill}
    button_index = {"left": 1, "down": 2, "right": 3}[c_slot]
    if len(current.equipped) > button_index and current.equipped[button_index] == item_id:
        return {"status": "completed", "reason": "already_equipped", "item_id": item_id,
            "slot": c_slot, "skill": decision.skill}

    acknowledged = False
    opened_here = not current.pause_menu.active
    if opened_here:
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["START"], hold_ms=90, settle_s=0.12)
    current = await _wait_pause_ready(bridge)
    if current is None:
        return {"status": "failed", "reason": "pause_menu_not_ready",
            "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}

    # R is the stable right-page control in the pinned SoH revision. Cycle until the item page (0).
    for _ in range(5):
        if current.pause_menu.page_index == 0:
            break
        old_page = current.pause_menu.page_index
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["R"], hold_ms=80, settle_s=0.16)
        current = await _wait_pause_ready(bridge, 1.5)
        if current is None:
            return {"status": "failed", "reason": "pause_page_transition_timeout",
                "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}
        if current.pause_menu.page_index == old_page:
            continue
    if current.pause_menu.page_index != 0:
        return {"status": "failed", "reason": "item_page_unreachable",
            "item_id": item_id, "page": current.pause_menu.page_index,
            "acknowledged": acknowledged, "skill": decision.skill}

    # If the cursor sits on a page-switch sentinel, move it back toward the item grid.
    if current.pause_menu.cursor_special_pos == 10:
        acknowledged |= await _pulse(bridge, stick_x=60, hold_ms=100, settle_s=0.12)
        current = await _wait_pause_ready(bridge, 1.0) or current
    elif current.pause_menu.cursor_special_pos == 11:
        acknowledged |= await _pulse(bridge, stick_x=-60, hold_ms=100, settle_s=0.12)
        current = await _wait_pause_ready(bridge, 1.0) or current

    visited = {}
    for _ in range(32):
        current = bridge.state or current
        slots = current.pause_menu.cursor_slot
        if not current.pause_menu.active or not current.pause_menu.ready or not slots:
            current = await _wait_pause_ready(bridge, 1.0)
            if current is None:
                break
            slots = current.pause_menu.cursor_slot
        cursor = slots[0] if slots else -1
        if cursor == target_slot:
            break
        if cursor < 0 or cursor >= 24:
            return {"status": "failed", "reason": "invalid_item_cursor",
                "cursor": cursor, "item_id": item_id, "skill": decision.skill}

        moved = False
        for direction in _menu_grid_directions(cursor, target_slot):
            key = (cursor, direction)
            if visited.get(key, 0) >= 2:
                continue
            visited[key] = visited.get(key, 0) + 1
            x = {"left": -60, "right": 60}.get(direction, 0)
            y = {"up": 60, "down": -60}.get(direction, 0)
            acknowledged |= await _pulse(bridge, stick_x=x, stick_y=y, hold_ms=100, settle_s=0.12)
            sample = await _wait_pause_ready(bridge, 0.9)
            if sample is None:
                continue
            new_cursor = sample.pause_menu.cursor_slot[0] if sample.pause_menu.cursor_slot else cursor
            current = sample
            if new_cursor != cursor:
                moved = True
                break
        if not moved:
            return {"status": "failed", "reason": "item_cursor_stuck",
                "cursor": cursor, "target_slot": target_slot, "item_id": item_id,
                "acknowledged": acknowledged, "skill": decision.skill}

    current = bridge.state or current
    cursor = current.pause_menu.cursor_slot[0] if current.pause_menu.cursor_slot else -1
    if cursor != target_slot:
        return {"status": "failed", "reason": "item_cursor_target_not_reached",
            "cursor": cursor, "target_slot": target_slot, "item_id": item_id,
            "acknowledged": acknowledged, "skill": decision.skill}

    acknowledged |= await _pulse(bridge, buttons=BUTTONS[f"C_{c_slot.upper()}"],
        hold_ms=90, settle_s=0.15)

    equip_deadline = time.monotonic() + 2.5
    equipped = False
    while time.monotonic() < equip_deadline:
        sample = bridge.state
        if sample and len(sample.equipped) > button_index and sample.equipped[button_index] == item_id:
            equipped = True
            current = sample
            break
        await asyncio.sleep(0.08)
    if not equipped:
        return {"status": "failed", "reason": "equip_not_confirmed",
            "item_id": item_id, "slot": c_slot, "acknowledged": acknowledged,
            "skill": decision.skill}

    # Wait out the pause equip animation before closing. Failure to close is reported but does not
    # erase the verified equipment success.
    ready = await _wait_pause_ready(bridge, 2.5)
    if ready is not None:
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["START"], hold_ms=90, settle_s=0.15)
        close_deadline = time.monotonic() + 1.5
        while time.monotonic() < close_deadline:
            sample = bridge.state
            if sample and not sample.pause_menu.active:
                return {"status": "completed", "reason": "item_equipped",
                    "item_id": item_id, "slot": c_slot, "pause_closed": True,
                    "acknowledged": acknowledged, "skill": decision.skill}
            await asyncio.sleep(0.08)
    return {"status": "completed", "reason": "item_equipped_pause_left_open",
        "item_id": item_id, "slot": c_slot, "pause_closed": False,
        "acknowledged": acknowledged, "skill": decision.skill}


def _gear_observation(game: GameState, item_id: int):
    return next((row for row in game.progress.equipment if row.item_id == item_id), None)


def _equipment_point(item_id: int) -> int | None:
    # Vanilla equipment item IDs are four contiguous rows of three:
    # swords 0x3B..0x3D, shields 0x3E..0x40, tunics 0x41..0x43, boots 0x44..0x46.
    if not 0x3B <= item_id <= 0x46:
        return None
    offset = item_id - 0x3B
    return (offset // 3) * 4 + (offset % 3) + 1


async def _equip_gear(bridge: Bridge, decision: Decision, observation: GameState) -> dict:
    current = bridge.state
    item_id = decision.args.item_id
    if not current or not current.player or item_id is None:
        return {"status": "failed", "reason": "equip_gear_arguments_or_state_missing", "skill": decision.skill}
    if (current.instance_id, current.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    if current.game_over_state != 0:
        return {"status": "stale", "reason": "game_over_active", "skill": decision.skill}
    if current.dialogue.active or current.cutscene_active:
        return {"status": "stale", "reason": "gameplay_state_blocks_pause", "skill": decision.skill}

    gear = _gear_observation(current, item_id)
    target_point = _equipment_point(item_id)
    if gear is None or target_point is None:
        return {"status": "failed", "reason": "gear_not_owned_or_not_equipable",
            "item_id": item_id, "skill": decision.skill}
    if gear.equipped:
        return {"status": "completed", "reason": "already_equipped",
            "item_id": item_id, "equipment_type": gear.equipment_type, "skill": decision.skill}

    acknowledged = False
    if not current.pause_menu.active:
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["START"], hold_ms=90, settle_s=0.12)
    current = await _wait_pause_ready(bridge)
    if current is None:
        return {"status": "failed", "reason": "pause_menu_not_ready",
            "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}

    # R cycles pages in the pinned SoH build. Stop on PAUSE_EQUIP == 3.
    for _ in range(5):
        if current.pause_menu.page_index == 3:
            break
        old_page = current.pause_menu.page_index
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["R"], hold_ms=80, settle_s=0.16)
        current = await _wait_pause_ready(bridge, 1.5)
        if current is None:
            return {"status": "failed", "reason": "equipment_page_transition_timeout",
                "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}
        if current.pause_menu.page_index == old_page:
            continue
    if current.pause_menu.page_index != 3:
        return {"status": "failed", "reason": "equipment_page_unreachable",
            "item_id": item_id, "page": current.pause_menu.page_index,
            "acknowledged": acknowledged, "skill": decision.skill}

    if current.pause_menu.cursor_special_pos == 10:
        acknowledged |= await _pulse(bridge, stick_x=60, hold_ms=100, settle_s=0.12)
        current = await _wait_pause_ready(bridge, 1.0) or current
    elif current.pause_menu.cursor_special_pos == 11:
        acknowledged |= await _pulse(bridge, stick_x=-60, hold_ms=100, settle_s=0.12)
        current = await _wait_pause_ready(bridge, 1.0) or current

    visited = {}
    for _ in range(28):
        current = bridge.state or current
        if not current.pause_menu.active or not current.pause_menu.ready:
            current = await _wait_pause_ready(bridge, 1.0)
            if current is None:
                break
        points = current.pause_menu.cursor_point
        cursor = points[3] if len(points) > 3 else -1
        if cursor == target_point:
            break
        if cursor < 0 or cursor >= 16:
            return {"status": "failed", "reason": "invalid_equipment_cursor",
                "cursor": cursor, "target_point": target_point, "item_id": item_id,
                "skill": decision.skill}

        moved = False
        for direction in _menu_grid_directions(cursor, target_point, width=4):
            key = (cursor, direction)
            if visited.get(key, 0) >= 2:
                continue
            visited[key] = visited.get(key, 0) + 1
            x = {"left": -60, "right": 60}.get(direction, 0)
            y = {"up": 60, "down": -60}.get(direction, 0)
            acknowledged |= await _pulse(bridge, stick_x=x, stick_y=y, hold_ms=100, settle_s=0.12)
            sample = await _wait_pause_ready(bridge, 0.9)
            if sample is None:
                continue
            new_cursor = sample.pause_menu.cursor_point[3] if len(sample.pause_menu.cursor_point) > 3 else cursor
            current = sample
            if new_cursor != cursor:
                moved = True
                break
        if not moved:
            return {"status": "failed", "reason": "equipment_cursor_stuck",
                "cursor": cursor, "target_point": target_point, "item_id": item_id,
                "acknowledged": acknowledged, "skill": decision.skill}

    current = bridge.state or current
    cursor = current.pause_menu.cursor_point[3] if len(current.pause_menu.cursor_point) > 3 else -1
    if cursor != target_point:
        return {"status": "failed", "reason": "equipment_cursor_target_not_reached",
            "cursor": cursor, "target_point": target_point, "item_id": item_id,
            "acknowledged": acknowledged, "skill": decision.skill}

    acknowledged |= await _pulse(bridge, buttons=BUTTONS["A"], hold_ms=90, settle_s=0.12)
    deadline = time.monotonic() + 2.0
    verified = None
    while time.monotonic() < deadline:
        sample = bridge.state
        if sample:
            candidate = _gear_observation(sample, item_id)
            if candidate is not None and candidate.equipped:
                verified = candidate
                current = sample
                break
        await asyncio.sleep(0.08)
    if verified is None:
        return {"status": "failed", "reason": "gear_equip_not_confirmed",
            "item_id": item_id, "acknowledged": acknowledged, "skill": decision.skill}

    ready = await _wait_pause_ready(bridge, 2.0)
    if ready is not None:
        acknowledged |= await _pulse(bridge, buttons=BUTTONS["START"], hold_ms=90, settle_s=0.15)
        close_deadline = time.monotonic() + 1.5
        while time.monotonic() < close_deadline:
            sample = bridge.state
            if sample and not sample.pause_menu.active:
                return {"status": "completed", "reason": "gear_equipped",
                    "item_id": item_id, "equipment_type": verified.equipment_type,
                    "pause_closed": True, "acknowledged": acknowledged, "skill": decision.skill}
            await asyncio.sleep(0.08)
    return {"status": "completed", "reason": "gear_equipped_pause_left_open",
        "item_id": item_id, "equipment_type": verified.equipment_type,
        "pause_closed": False, "acknowledged": acknowledged, "skill": decision.skill}
