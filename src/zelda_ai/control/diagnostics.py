"""Provider-free controller diagnostics. Results are volatile and never benchmark/learning data."""
from __future__ import annotations

import math
import time
from collections.abc import Iterable

from ..bridge import Bridge
from ..control.authority import ControlRevoked
from ..models import GameState, InputReceipt
from ..skills.common import _perform_dodge

A, B, Z = 0x8000, 0x4000, 0x2000
ACTIONS = {"tap_a", "tap_b", "target", "forward", "back", "backflip", "stress_a", "stress_b"}


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))
    return round(ordered[index], 2)


def direction_is_safe(game: GameState, direction: str) -> bool:
    """Conservative short-motion guard. A probe is evidence, not a global path proof."""
    probes = [p for p in game.navigation_probes
        if p.direction == direction and p.distance <= 70.0 and p.floor_found and p.delta_y is not None]
    if not probes:
        return False
    probe = min(probes, key=lambda p: p.distance)
    if abs(probe.delta_y) > 22.0:
        return False
    if probe.wall_hit and probe.wall_distance is not None and probe.wall_distance < 32.0:
        return False
    return True


def summarize_receipts(receipts: Iterable[InputReceipt | None], *, edge_button: int = 0,
                       expected_edges: int = 0) -> dict:
    attempts = list(receipts)
    rows = [row for row in attempts if row is not None]
    consumed = [row for row in rows if row.first_tick > 0]
    latencies = [float(row.client_to_consume_ms) for row in consumed if row.client_to_consume_ms is not None]
    native_latencies = [float(row.apply_latency_ms) for row in consumed if row.apply_latency_ms is not None]
    presses = sum(1 for row in consumed if edge_button and row.pressed & edge_button)
    releases = sum(1 for row in consumed if edge_button and row.released & edge_button)
    return {
        "commands": len(attempts),
        "consumed": len(consumed),
        "completed": sum(1 for row in rows if row.status == "completed"),
        "lost": max(0, len(attempts) - len(consumed)),
        "expected_edges": expected_edges,
        "presses": presses,
        "releases": releases,
        "duplicate_presses": max(0, presses - expected_edges),
        "duplicate_releases": max(0, releases - expected_edges),
        "latency_ms": {
            "p50": _percentile(latencies, .50),
            "p95": _percentile(latencies, .95),
            "p99": _percentile(latencies, .99),
            "min": round(min(latencies), 3) if latencies else None,
            "max": round(max(latencies), 3) if latencies else None,
        },
        "native_queue_ms": {
            "p50": _percentile(native_latencies, .50),
            "p95": _percentile(native_latencies, .95),
            "p99": _percentile(native_latencies, .99),
        },
    }


def _ready(game: GameState | None) -> bool:
    return bool(game and game.in_game and game.player and not game.paused and not game.dialogue.active
        and not game.cutscene_active and game.game_over_state == 0 and game.ocarina_mode == 0)


async def _wait_consumed(bridge: Bridge, seq: int, current: GameState,
                         timeout: float = .35) -> tuple[InputReceipt | None, GameState]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        row = bridge.receipts.get(seq)
        if row and (row.first_tick > 0 or row.status in {"rejected", "cancelled", "superseded"}):
            return row, current
        current = await bridge.next_state(current.seq, timeout=min(.2, max(.001, deadline-time.monotonic())))
    return bridge.receipts.get(seq), current


async def _motion(bridge: Bridge, direction: str, *, seconds: float = 1.0) -> tuple[list[InputReceipt | None], str]:
    receipts: list[InputReceipt | None] = []
    current = bridge.state
    deadline = time.monotonic() + seconds
    reason = "motion_window_complete"
    while time.monotonic() < deadline:
        if not _ready(current):
            reason = "gameplay_state_changed"
            break
        if not direction_is_safe(current, direction):
            reason = "terrain_probe_not_safe"
            break
        seq = bridge.send(stick_y=55 if direction == "forward" else -55, lease_ms=160)
        try:
            row, current = await _wait_consumed(bridge, seq, current)
        except RuntimeError:
            receipts.append(bridge.receipts.get(seq))
            reason = "state_feedback_timeout"
            break
        receipts.append(row)
        if not row or row.first_tick == 0:
            reason = row.reason if row else "input_not_consumed"
            break
    bridge.release()
    return receipts, reason


async def run_input_diagnostic(bridge: Bridge, action: str) -> dict:
    if action not in ACTIONS:
        raise ValueError("Unknown input diagnostic")
    start = bridge.state
    if not _ready(start):
        raise ValueError("Input diagnostics require normal unpaused gameplay")
    started = time.monotonic()
    receipts: list[InputReceipt | None] = []
    edge_button = 0
    expected_edges = 0
    reason = "completed"
    effect_confirmed = None
    hopping_seen = False
    hop_direction = None
    expected_hop_direction = None
    max_distance = 0.0
    expected_distance = 0.0
    stick_x = None
    stick_y = None

    try:
        with bridge.input_scope(f"diagnostic:{action}"):
            if action in {"tap_a", "tap_b", "target"}:
                edge_button = {"tap_a": A, "tap_b": B, "target": Z}[action]
                expected_edges = 1
                receipts.append(await bridge.pulse_receipt(buttons=edge_button, hold_ticks=1))
            elif action in {"stress_a", "stress_b"}:
                edge_button = A if action == "stress_a" else B
                expected_edges = 20
                for _ in range(expected_edges):
                    if not _ready(bridge.state):
                        reason = "gameplay_state_changed"
                        break
                    receipts.append(await bridge.pulse_receipt(buttons=edge_button, hold_ticks=1))
                    row = receipts[-1]
                    if not row or row.first_tick == 0 or row.status != "completed":
                        reason = row.reason if row else "input_not_consumed"
                        break
            elif action in {"forward", "back"}:
                receipts, reason = await _motion(bridge, action)
            elif action == "backflip":
                if not direction_is_safe(bridge.state, "back"):
                    reason = "terrain_probe_not_safe"
                else:
                    edge_button = A
                    expected_edges = 1
                    dodge = await _perform_dodge(bridge, start, "back")
                    receipts.append(dodge["receipt"])
                    effect_confirmed = dodge["confirmed"]
                    hopping_seen = dodge["hopping_seen"]
                    hop_direction = dodge["hop_direction"]
                    expected_hop_direction = dodge["expected_hop_direction"]
                    max_distance = dodge["max_distance"]
                    expected_distance = dodge["expected_distance"]
                    stick_x = dodge["stick_x"]
                    stick_y = dodge["stick_y"]
                    reason = dodge["reason"]
    except ControlRevoked:
        reason = "control_revoked"
    except RuntimeError as exc:
        reason = str(exc) or "controller_runtime_error"

    after = bridge.state
    summary = summarize_receipts(receipts, edge_button=edge_button, expected_edges=expected_edges)
    if reason == "completed":
        if summary["lost"]:
            reason = "input_not_consumed"
        elif expected_edges and (summary["presses"] != expected_edges or summary["releases"] != expected_edges):
            reason = "edge_mismatch"
    status = "completed" if reason in {"completed", "motion_window_complete", "dodge_confirmed"} and summary["lost"] == 0 else (
        "interrupted" if reason in {"gameplay_state_changed", "state_feedback_timeout",
            "terrain_probe_not_safe", "bridge_disconnected", "state_feedback_stale", "control_revoked"} else "failed")
    distance = None
    if start and start.player and after and after.player and start.scene_epoch == after.scene_epoch:
        distance = round(math.dist(start.player.position, after.player.position), 2)
    return {
        "action": action, "status": status, "reason": reason, **summary,
        "duration_ms": round((time.monotonic()-started)*1000, 2),
        "effect_confirmed": effect_confirmed, "hopping_seen": hopping_seen,
        "hop_direction": hop_direction, "expected_hop_direction": expected_hop_direction,
        "max_distance": max_distance, "expected_distance": expected_distance, "distance": distance,
        "stick_x": stick_x, "stick_y": stick_y,
        "start_position": list(start.player.position) if start and start.player else None,
        "end_position": list(after.player.position) if after and after.player else None,
    }
