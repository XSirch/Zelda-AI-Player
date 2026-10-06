"""Audited neutral cooldown waits; no policy sampling or game-progress credit."""
from __future__ import annotations

import asyncio
import time

from .autonomy.execution import physical_context
from .autonomy.locomotion import camera_modal_active, grounded
from .g1 import write_json


async def wait_for_replan(controller, bridge, directory, *, retry_at, deadline, observe=None):
    if getattr(controller,"training_enabled",False):
        raise ValueError("Frozen planning wait cannot discard a training transition")
    initial = bridge.state
    if initial.source != "soh" or initial.protocol != 3 or not grounded(initial.player):
        raise ValueError("Cooldown waiting requires native dry-ground telemetry")
    if getattr(controller,"local_task",None) is not None:
        raise ValueError("A local task still owns control")
    directory.mkdir(parents=True,exist_ok=True)
    started = time.monotonic()
    until = min(retry_at, started+30, deadline-.5)
    context, epoch = physical_context(initial), initial.scene_epoch
    loads = {e.id for e in initial.events if e.kind in {"save_loaded","player_died","game_over"}}
    requested, receipts, frames = {}, {}, []
    controller.pending = controller.pending_interaction_probe = None
    bridge.release()
    reason = "cooldown_elapsed"

    def capture_receipts():
        for seq, row in bridge.receipts.items():
            if seq in requested and row.first_tick > 0:
                receipts[seq] = row.model_dump()

    try:
        while time.monotonic() < until:
            game = bridge.state
            if not bridge.connected:
                reason = "bridge_unavailable"
                break
            if (physical_context(game) != context or game.scene_epoch != epoch
                    or any(e.id not in loads and e.kind in {"save_loaded","player_died","game_over"} for e in game.events)):
                reason = "physical_context_changed"
                break
            if (not game.in_game or not game.player or game.player.health <= 0 or game.game_over_state
                    or not grounded(game.player) or camera_modal_active(game.player)
                    or game.dialogue.active or game.pause_menu.active or game.paused or game.cutscene_active):
                reason = "control_mode_changed"
                break
            if observe:
                observe(game)
            seq = bridge.send(buttons=0,stick_x=0,stick_y=0,lease_ms=150)
            requested[seq] = {"observation_seq":game.seq,"buttons":0,"stick":[0,0]}
            # Keep delivery and macro/campaign supervision live. Neutral input
            # cannot create a target, movement success or durable-progress reset.
            controller.executor.observe(game,{})
            capture_receipts()
            if len(frames)<660:
                frames.append({"seq":game.seq,"scene":game.scene,"room":game.room,
                    "scene_epoch":game.scene_epoch,"position":game.player.position,
                    "speed_xz":game.player.speed_xz,"health":game.player.health,
                    "last_consumed":game.last_applied_command_seq})
            if controller.executor.state == "blocked":
                reason = f"supervisor:{controller.executor.reason}"
                break
            remaining = max(.001,until-time.monotonic())
            try:
                await bridge.next_state(game.seq,timeout=min(.1,remaining))
            except TimeoutError:
                pass  # Reobserve this bridge; no relaunch or lease extension.
            await asyncio.sleep(min(.04,max(0,until-time.monotonic())))
    except asyncio.CancelledError:
        reason = "planning_wait_cancelled"
        raise
    except (RuntimeError,ValueError,OSError) as exc:
        reason = f"{type(exc).__name__}:{str(exc)[:180]}"
        raise
    finally:
        capture_receipts()
        bridge.release()
        report = {"source":"soh","reason":reason,"game_progress":False,
            "started":started,"requested_retry_at":retry_at,"until":until,
            "elapsed_seconds":time.monotonic()-started,"context":context,
            "requested":requested,"receipts":receipts,"frames":frames,
            "consumed_actions":len(receipts),"raw_button_actions":sum(bool(r["pressed"]) for r in receipts.values()),
            "input_source":"neutral_planning_cooldown","provider_calls":0,"training_updates":0}
        await asyncio.to_thread(write_json,directory/'wait.json',report)
    if report["raw_button_actions"]:
        raise RuntimeError("Non-neutral native receipt during planning wait")
    return report
