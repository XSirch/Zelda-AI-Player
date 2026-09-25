"""Local executor controller helpers."""
from __future__ import annotations

import asyncio
import math
import time
from ..bridge import Bridge
from ..control.authority import owned
from ..control.feedback import consumed, feedback, is_realtime
from ..models import Decision, GameState
from ..navmesh import primitive_move_safe
from .catalog import BUTTONS, DIALOGUE_SKILLS, MENU_SKILLS, controller_input
from .common import _is_door_actor, _matching_actor, _perform_dodge
from .menus import _choose_dialogue, _equip_gear, _equip_item, _play_song
from .navigation import _explore_area, _face_target, _follow_actor, _navigate_local
from .interactions import _interact_with_door, _manipulate_object
from .aim import _aim_at
from .combat import _fight_enemy
from .traversal import _traverse_local

async def execute_skill(bridge: Bridge, decision: Decision, observation: GameState,
                        combat_profile: dict | None = None) -> dict:
    if decision.args.target_actor_id is not None and decision.args.target_actor_uid is None:
        actor = _matching_actor(observation, decision.args.target_actor_id, decision.args.target_actor_params)
        if actor and actor.actor_uid:
            decision = decision.model_copy(update={"args": decision.args.model_copy(
                update={"target_actor_uid": actor.actor_uid})})
    with owned(bridge, decision.skill):
        return await _execute_skill(bridge, decision, observation, combat_profile=combat_profile)


async def _execute_skill(bridge: Bridge, decision: Decision, observation: GameState,
                         combat_profile: dict | None = None) -> dict:
    before = bridge.state
    if not before or not bridge.connected:
        raise RuntimeError("bridge_disconnected")
    if (before.instance_id, before.scene_epoch) != (observation.instance_id, observation.scene_epoch):
        return {"status": "stale", "reason": "world_changed_during_inference", "skill": decision.skill}
    if not before.in_game:
        return {"status": "stale", "reason": "game_not_ready", "skill": decision.skill}

    if decision.skill == "traverse":
        return await _traverse_local(bridge, decision, observation, decision.args.direction)

    if decision.skill == "equip_item":
        return await _equip_item(bridge, decision, observation)
    if decision.skill == "equip_gear":
        return await _equip_gear(bridge, decision, observation)
    if decision.skill == "navigate_to":
        return await _navigate_local(bridge, decision, observation, actor_mode=False)
    if decision.skill == "approach_actor":
        return await _navigate_local(bridge, decision, observation, actor_mode=True)
    if decision.skill == "follow_actor":
        return await _follow_actor(bridge, decision, observation)
    if decision.skill == "talk_to_actor":
        return await _navigate_local(bridge, decision, observation, actor_mode=True, talk=True)
    if decision.skill == "interact_with_actor":
        target = _matching_actor(before, decision.args.target_actor_id, decision.args.target_actor_params, decision.args.target_actor_uid)
        if _is_door_actor(target):
            return await _interact_with_door(bridge, decision, observation)
        return await _navigate_local(bridge, decision, observation, actor_mode=True, interact=True)
    if decision.skill == "aim_at":
        return await _aim_at(bridge, decision, observation)
    if decision.skill == "face_target":
        return await _face_target(bridge, decision, observation)
    if decision.skill == "shield_face":
        return await _face_target(bridge, decision, observation, shield=True)
    if decision.skill == "fight_enemy":
        return await _fight_enemy(bridge, decision, observation, profile=combat_profile)
    if decision.skill == "explore_area":
        return await _explore_area(bridge, decision, observation)
    if decision.skill == "manipulate_object":
        return await _manipulate_object(bridge, decision, observation)
    if before.paused and decision.skill not in MENU_SKILLS:
        return {"status": "stale", "reason": "pause_menu_active", "skill": decision.skill}
    if not before.paused and decision.skill in {"menu_move", "menu_confirm", "menu_cancel", "menu_assign"}:
        return {"status": "stale", "reason": "pause_menu_not_active", "skill": decision.skill}
    if before.dialogue.active and decision.skill not in DIALOGUE_SKILLS and decision.skill != "wait":
        return {"status": "stale", "reason": "dialogue_requires_handling", "skill": decision.skill}
    if before.cutscene_active and not before.dialogue.active and decision.skill not in {"wait", "continue_gameover"}:
        return {"status": "stale", "reason": "cutscene_active", "skill": decision.skill}
    if decision.skill == "choose_dialogue":
        return await _choose_dialogue(bridge, decision, observation)
    if decision.skill == "play_song":
        return await _play_song(bridge, decision, observation)

    if decision.skill == "move" and decision.args.direction and not primitive_move_safe(before, decision.args.direction):
        return {"status": "failed", "reason": "unsafe_navigation_probe",
            "acknowledged": False, "effect_confirmed": False, "skill": decision.skill}

    buttons, x, y = controller_input(decision)
    if is_realtime(bridge) and decision.skill in {"attack", "interact", "advance_dialogue", "camera_center",
            "backflip", "sidestep", "jump_attack", "roll", "pause_toggle", "menu_confirm", "menu_cancel",
            "menu_assign", "continue_gameover", "menu_move"}:
        if decision.skill in {"backflip", "sidestep"}:
            direction = "back" if decision.skill == "backflip" else decision.args.direction
            dodge = await _perform_dodge(bridge, before, direction)
            row = dodge["receipt"]
            delivered = bool(row and row.first_tick > 0 and
                             row.status not in {"rejected", "cancelled", "superseded"})
            return {"status": "completed" if dodge["confirmed"] else "failed",
                "reason": dodge["reason"], "distance": dodge["max_distance"],
                "expected_distance": dodge["expected_distance"], "hopping_seen": dodge["hopping_seen"],
                "hop_direction": dodge["hop_direction"],
                "expected_hop_direction": dodge["expected_hop_direction"],
                "stick_x": dodge["stick_x"], "stick_y": dodge["stick_y"],
                "control_stick_direction": dodge.get("control_stick_direction"),
                "acknowledged": delivered, "ack_stage": "consumed",
                "effect_confirmed": dodge["confirmed"], "skill": decision.skill}
        delivered = await bridge.pulse(buttons=buttons, stick_x=x, stick_y=y,
            hold_ticks=1, edge_buttons=buttons & ~BUTTONS["Z"] if decision.skill == "jump_attack" else buttons)
        return {"status": "completed" if delivered else "failed",
            "reason": "input_consumed_effect_unconfirmed" if delivered else "input_not_consumed",
            "acknowledged": delivered, "ack_stage": "consumed", "effect_confirmed": False,
            "skill": decision.skill}
    start_yaw = before.player.yaw if before.player else None
    target_turn_units = None
    if decision.skill == "turn" and start_yaw is not None:
        target_turn_units = max(3600, min(18200,
            int(decision.args.duration_ms * 9.1 * max(0.35, decision.args.strength))))
    effective_duration_ms = (min(decision.args.duration_ms, 180)
        if decision.skill in MENU_SKILLS | {"advance_dialogue"}
        else min(decision.args.duration_ms, 2000))
    deadline = time.monotonic() + effective_duration_ms / 1000
    status, reason, acknowledged = "completed", "duration_elapsed", False
    first_command = None
    try:
        while time.monotonic() < deadline:
            current = bridge.state
            if not bridge.connected or not current:
                raise RuntimeError("bridge_disconnected")
            if (current.instance_id, current.scene_epoch) != (before.instance_id, before.scene_epoch):
                status, reason = "interrupted", "world_changed"
                break
            if not current.in_game:
                status, reason = "interrupted", "game_not_ready"
                break
            if current.paused and decision.skill not in MENU_SKILLS:
                status, reason = "interrupted", "pause_menu_opened"
                break
            if not current.paused and decision.skill in {"menu_move", "menu_confirm", "menu_cancel", "menu_assign"}:
                status, reason = "interrupted", "pause_menu_closed"
                break
            if (not before.dialogue.active and current.dialogue.active and
                    decision.skill not in DIALOGUE_SKILLS):
                status, reason = "interrupted", "dialogue_opened"
                break
            if (not before.cutscene_active and current.cutscene_active and
                    not current.dialogue.active and decision.skill != "wait"):
                status, reason = "interrupted", "cutscene_started"
                break
            if target_turn_units is not None and current.player:
                delta = ((current.player.yaw - start_yaw + 32768) % 65536) - 32768
                if abs(delta) >= target_turn_units:
                    reason = "heading_reached"
                    break
            command_id = bridge.send(buttons=buttons, stick_x=x, stick_y=y, lease_ms=300)
            if first_command is None:
                first_command = command_id
            acknowledged |= consumed(bridge, first_command)
            await feedback(bridge, current, min(0.1, max(0, deadline - time.monotonic())))
    finally:
        bridge.release()

    await asyncio.sleep(0.22)
    after = bridge.state
    distance, damage, yaw_delta = None, None, None
    if after:
        acknowledged |= first_command is not None and consumed(bridge, first_command)
    if before.player and after and after.player and before.scene_epoch == after.scene_epoch:
        distance = math.dist(before.player.position, after.player.position)
        damage = max(0, before.player.health - after.player.health)
        yaw_delta = ((after.player.yaw - before.player.yaw + 32768) % 65536) - 32768
    if status == "completed" and not acknowledged and decision.skill != "wait":
        status, reason = "failed", "input_not_acknowledged"
    elif status == "completed" and decision.skill == "move" and distance is not None and distance < 1:
        status, reason = "failed", "no_displacement_observed"
    elif status == "completed" and decision.skill == "turn" and yaw_delta is not None and abs(yaw_delta) < 900:
        status, reason = "failed", "no_heading_change_observed"
    return {"status": status, "reason": reason, "distance": distance, "yaw_delta": yaw_delta,
        "health_lost": damage, "acknowledged": acknowledged, "skill": decision.skill}
