"""Empirical, bounded startup through normal physical controller input.

The native fence admits buttons only on the requested existing save and its
safe confirmation. No save-state, teleport, memory write or shell access is
available to this controller. Save selection is explicit operator configuration.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Callable

from .autonomy.controller import BUTTON_MASKS


class StartupFailure(RuntimeError):
    """Retain bounded causal evidence when startup fails before returning."""

    def __init__(self, cause, trace, bridge):
        reason = str(cause)[:160]
        super().__init__(reason)
        game = bridge.state
        self.report = {
            "status": "failed", "reason": reason, "error_kind": type(cause).__name__,
            "probes": len(trace), "trace": trace,
            "last_observation": {
                "seq": game.seq if game else None,
                "scene_epoch": game.scene_epoch if game else None,
                "context_epoch": game.context_epoch if game else None,
                "in_game": game.in_game if game else None,
                "startup": game.startup.model_dump() if game else None,
            },
            "bridge": bridge.telemetry() if hasattr(bridge, "telemetry") else None,
            "recent_receipts": [row.model_dump() for row in list(getattr(bridge, "receipts", {}).values())[-6:]],
        }


async def enter_playable_save(bridge, slot: int, *, budget_s=60.0, failure_report: Callable[[dict], None] | None = None):
    if not 0 <= slot <= 2:
        raise ValueError("Save slot must be 0..2")
    trace = []
    try:
        return await _enter_playable_save(bridge, slot, budget_s=budget_s, trace=trace)
    except (RuntimeError, OSError) as exc:
        failure = StartupFailure(exc, trace, bridge)
        if failure_report:
            failure_report(failure.report)
        raise failure from exc
    finally:
        bridge.release()


async def _enter_playable_save(bridge, slot: int, *, budget_s, trace):
    deadline = time.monotonic() + budget_s
    probes = 0
    last_seq = -1
    pending = None
    next_probe_at = 0.0
    button_index = {}
    axis_sign = {}
    # Physical probe order only; no button is assigned a UI meaning beforehand.
    buttons = tuple(BUTTON_MASKS)
    while time.monotonic() < deadline:
        game = bridge.state
        if not bridge.connected or game is None:
            await asyncio.sleep(0.05)
            continue
        if game.in_game and game.player:
            loaded = next((event.detail for event in reversed(game.events) if event.kind == "save_loaded"), None)
            if loaded is not None and loaded != str(slot):
                return {"status": "blocked", "reason": "different_save_loaded", "trace": trace}
            return {"status": "loaded", "slot": slot + 1, "probes": probes, "trace": trace}
        if "startup_controls_v1" not in game.capabilities:
            return {"status": "blocked", "reason": "startup_adapter_upgrade_required", "trace": trace}
        observation = game.startup
        if not observation.existing_slots[slot]:
            return {"status": "blocked", "reason": "requested_save_missing", "trace": trace}
        if game.seq == last_seq:
            await asyncio.sleep(0.05)
            continue
        last_seq = game.seq
        if pending is not None:
            if pending["delivered"] and observation.phase == pending["phase"] and observation.cursor != pending["cursor"] and pending["stick_y"]:
                delta = observation.cursor - pending["cursor"]
                # Infer axis effect from actual cursor motion, not knowledge of
                # a Zelda menu. Reject wraps, animations and mode changes.
                if abs(delta) == 1:
                    axis_sign[observation.phase] = (1 if delta > 0 else -1) * pending["stick_y"]
            pending = None
        if observation.phase not in {1, 2, 3} or time.monotonic() < next_probe_at:
            await asyncio.sleep(0.05)
            continue
        phase = observation.phase
        target_cursor = slot if phase == 2 else 0
        button = 0
        stick_y = 0
        if phase in {2, 3} and observation.cursor != target_cursor:
            direction = 1 if target_cursor > observation.cursor else -1
            sign = axis_sign.get(phase)
            stick_y = direction * sign if sign is not None else 80
        else:
            index = button_index.get(phase, 0)
            button = BUTTON_MASKS[buttons[index % len(buttons)]]
            button_index[phase] = index + 1
        # Record before awaiting: a telemetry gap must not erase the failed
        # physical probe or be confused with a model/portal failure.
        attempt = {"phase": phase, "cursor": observation.cursor, "buttons": button,
                   "stick_y": stick_y, "seq": game.seq, "delivered": None,
                   "receipt": None, "status": "awaiting_receipt"}
        trace.append(attempt)
        probe_started = time.monotonic()
        try:
            receipt = await bridge.sequence_receipt([
                {"buttons": button, "stick_x": 0, "stick_y": stick_y, "ticks": 1},
                {"buttons": 0, "stick_x": 0, "stick_y": 0, "ticks": 1},
            ], edge_buttons=button, startup_slot=slot, timeout=0.8)
        except BaseException:
            attempt["status"] = "interrupted"
            raise
        finally:
            attempt["elapsed_ms"] = (time.monotonic() - probe_started) * 1000
        delivered = bool(receipt and receipt.first_tick > 0)
        pending = {"phase": phase, "cursor": observation.cursor,
            "buttons": button, "stick_y": stick_y, "delivered": delivered}
        attempt.update(delivered=delivered, receipt=receipt.model_dump() if receipt else None,
                       status="receipt_observed" if receipt else "receipt_missing")
        probes += 1
        next_probe_at = time.monotonic() + 0.3
        if probes >= 40:
            break
    return {"status": "blocked", "reason": "startup_no_effect_or_budget", "probes": probes, "trace": trace}
