"""Audited local causal-button stages; no Laya/PPO button training or labels."""
from __future__ import annotations

import asyncio
import time

from .autonomy.controller import BUTTON_MASKS
from .g1 import write_json

CAUSAL_SOURCES = {"interaction_probe", "interaction_memory", "dialogue_probe", "dialogue_memory"}


async def execute_context_task(controller, bridge, task, directory, *, on_started=None, observe=None):
    if controller.training_enabled:
        raise ValueError("Frozen contextual QA cannot train the policy")
    directory.mkdir(parents=True, exist_ok=True)
    initial, objective = bridge.state.model_copy(deep=True), controller.intent.model_copy(deep=True)
    starting_updates, first_seq = controller.policy.updates, bridge.command_seq
    controller.pending = None
    controller.start_local_task(task, stick_policy=lambda game, owned: (0, 0))
    if on_started:
        on_started()
    requested, frames = {}, []

    def active():
        return bool(controller.local_task is not None and not task.terminal and bridge.connected
                    and time.monotonic() < task.deadline + .5)

    def publish():
        game, setpoint = bridge.state, controller.last_setpoint
        if observe:
            observe(game)
        seq = bridge.command_seq
        if seq > first_seq:
            trainable = bool((controller.pending or {}).get("trainable", False))
            probe = controller.pending_interaction_probe
            owned_button = bool(setpoint.reason in CAUSAL_SOURCES and setpoint.buttons
                and probe and BUTTON_MASKS.get(probe.get("button")) == setpoint.buttons
                and seq in probe.get("command_seqs", ())
                and setpoint.buttons & (setpoint.buttons - 1) == 0
                and not setpoint.buttons & ~0xE01F and not setpoint.stick_x and not setpoint.stick_y
                and not trainable)
            requested[seq] = {"observation_seq": game.seq, "source": setpoint.reason,
                "stick": [setpoint.stick_x, setpoint.stick_y], "buttons": setpoint.buttons,
                "trainable": trainable, "causal_button": owned_button,
                "probe_key": probe.get("key") if owned_button else None,
                "probe_kind": probe.get("kind") if owned_button else None,
                "probe_at": probe.get("at") if owned_button else None}
            if owned_button:
                task.register_button(seq, setpoint.buttons)
        if len(frames) < 660:
            frames.append({"state": game.model_dump(), "task": task.snapshot(), "reason": setpoint.reason})

    failure, cancelled = None, False
    try:
        await controller.run(active, publish)
    except (RuntimeError, ValueError, OSError) as exc:
        failure = f"{type(exc).__name__}:{str(exc)[:180]}"
    except asyncio.CancelledError:
        failure, cancelled = "stage_cancelled", True
    finally:
        bridge.release()
        probe = controller.pending_interaction_probe
        if probe:
            if (probe.get("known") and not task.observed_effect and any(
                    bridge.command_consumed(seq) for seq in probe.get("command_seqs", ()))):
                controller.interaction_memory.record_interaction_failure(probe["key"], probe["button"])
            controller.pending_interaction_probe = None
    receipts = {seq: row.model_dump() for seq, row in bridge.receipts.items()
                if seq > first_seq and row.first_tick > 0}
    consumed = sum(seq in receipts for seq in requested)
    buttons = sum(bool(row["pressed"]) for row in receipts.values())
    causal = sum(bool(row["pressed"]) and requested.get(seq, {}).get("causal_button", False)
                 and requested[seq]["buttons"] == row["pressed"] for seq, row in receipts.items())
    unchanged, updates = controller.intent == objective, controller.policy.updates - starting_updates
    success = bool(not failure and task.phase == "succeeded" and task.consumed and consumed
                   and buttons == causal and unchanged and updates == 0)
    report = {"source": "soh", "success": success, "task": task.snapshot(),
        "initial": initial.model_dump(), "final": bridge.state.model_dump(), "frames": frames,
        "requested": requested, "receipts": receipts, "consumed_actions": consumed,
        "raw_button_actions": buttons, "causal_button_actions": causal, "unowned_button_actions": buttons-causal,
        "input_source": "causal_context_controller", "candidate_button_actions": 0,
        "objective": objective.model_dump(), "final_objective": controller.intent.model_dump(),
        "objective_unchanged": unchanged, "run_updates": updates, "provider_calls": 0,
        "reference_blend": 0, "demonstration_labels": 0, "motor_failure": failure,
        "interaction_memory": controller.interaction_memory.snapshot()}
    await asyncio.to_thread(write_json, directory / "motor.json", report)
    if cancelled:
        raise asyncio.CancelledError
    return report
