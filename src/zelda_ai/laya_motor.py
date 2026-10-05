"""Frozen raw-analog execution with one caller-owned strategic objective."""
from __future__ import annotations

import asyncio
import time

from .autonomy.imitation import encode_surface
from .g1 import write_json


async def execute_frozen_task(controller, bridge, task, policy, directory, *, encoder=encode_surface,
                              on_started=None):
    if controller.training_enabled or policy is None:
        raise ValueError("Frozen execution requires an explicit candidate and disabled training")
    directory.mkdir(parents=True, exist_ok=True)
    initial = bridge.state.model_copy(deep=True)
    objective = controller.intent.model_copy(deep=True)
    starting_updates = controller.policy.updates
    policy.invalidate()
    # Old consumed walking commands cannot pay the next attached task's
    # consumption predicate. The controller and locked objective are retained.
    controller.pending = None
    controller.start_local_task(task, stick_policy=policy)
    if on_started:
        on_started()
    requested, frames = {}, []
    first_seq = bridge.command_seq

    def active():
        game = bridge.state
        return (controller.local_task is not None and bridge.connected
                and time.monotonic() < task.deadline + .5
                and not (game and (game.dialogue.active or game.pause_menu.active or game.paused)))

    def publish():
        game, owned = bridge.state, controller.local_task
        if (owned and game.player and owned.phase in {"prepare", "execute"}
                and controller.last_setpoint.reason == "local_task"):
            requested[bridge.command_seq] = {"observation_seq": game.seq, "features": encoder(game, owned),
                "stick": [controller.last_setpoint.stick_x, controller.last_setpoint.stick_y],
                "buttons": controller.last_setpoint.buttons}
        if game and len(frames) < 660:
            frames.append({"state": game.model_dump(), "task": task.snapshot(),
                           "reason": controller.last_setpoint.reason})

    failure, cancelled = None, False
    try:
        await controller.run(active, publish)
    except (RuntimeError, ValueError, OSError) as exc:
        failure = f"{type(exc).__name__}:{str(exc)[:180]}"
    except asyncio.CancelledError:
        failure, cancelled = "stage_cancelled", True
    finally:
        bridge.release()
        policy.invalidate()
    receipts = {s: r.model_dump() for s, r in bridge.receipts.items()
                if s > first_seq and r.first_tick > 0}
    consumed = sum(s in receipts for s in requested)
    buttons = sum(bool(row["pressed"]) for row in receipts.values())
    unchanged, updates = controller.intent == objective, controller.policy.updates - starting_updates
    success = bool(failure is None and task.phase == "succeeded" and consumed > 0
                   and buttons == 0 and unchanged and updates == 0)
    report = {"source": "soh", "success": success, "task": task.snapshot(), "initial": initial.model_dump(),
              "final": bridge.state.model_dump(), "frames": frames, "requested": requested,
              "receipts": receipts, "consumed_actions": consumed, "raw_button_actions": buttons,
              "objective": objective.model_dump(), "final_objective": controller.intent.model_dump(),
              "objective_unchanged": unchanged, "run_updates": updates, "provider_calls": 0,
              "reference_blend": 0, "demonstration_labels": 0, "motor_failure": failure}
    await asyncio.to_thread(write_json, directory / "motor.json", report)
    if cancelled:
        raise asyncio.CancelledError
    return report
