"""Real SoH motor qualification. Never constructs or calls a cognition provider."""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid

from .app import bridge_secret
from .autonomy.controller import ContinuousController
from .autonomy.models import AgentIntent, ObjectiveCompletion
from .bridge import Bridge, bind_bridge
from .config import Settings


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


async def qualify_motor(settings: Settings, *, seconds=120.0, wait_s=8.0, probe_only=False, save_slot=None):
    directory = settings.data_dir / "qualification" / f"motor-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True)
    bridge = Bridge(bridge_secret(settings), allow_simulator=False)
    await bind_bridge(bridge, settings.bridge_port)
    transport = bridge.transport
    report = {"schema_version": 1, "status": "blocked", "source": "soh",
        "observation_profile": "instrumented_local_v1", "provider_calls": 0,
        "training_enabled": False, "physical_episodes": 0, "successful_episodes": 0,
        "g1_qualified": False, "required_g1_episodes": 100,
        "reason": "bridge_unavailable", "seconds_budget": seconds}
    try:
        deadline = time.monotonic() + wait_s
        while time.monotonic() < deadline and not bridge.connected:
            await asyncio.sleep(0.05)
        game = bridge.state
        report["bridge"] = bridge.telemetry()
        if game is None:
            return report
        report["baseline"] = game.model_dump()
        if save_slot is not None and not probe_only and not (game.in_game and game.player):
            from .startup import enter_playable_save
            report["startup"] = await enter_playable_save(bridge, save_slot - 1)
            game = bridge.state
        if not game.in_game or not game.player:
            report["reason"] = report.get("startup", {}).get("reason", "no_playable_save_or_startup_observation")
            return report
        if probe_only:
            report["status"], report["reason"] = "observed", "read_only_probe"
            return report
        if game.protocol != 3 or not bridge.realtime:
            report["reason"] = "native_input_receipts_required"
            return report
        if game.game_over_state or game.player.health <= 0:
            report["reason"] = "game_over_requires_recovery"
            return report
        report["episode_initial"] = game.model_dump()
        original = settings.data_dir / "ml" / "raw-controller-ppo-rnd-v3.pt"
        paths = [original, original.parent / "route-graph-v1.json", original.parent / "room-map-v1.json"]
        before = {path.name: sha256(path) for path in paths}
        controller = ContinuousController(bridge, original if original.is_file() else directory / "untrained.pt",
            training_enabled=False, route_graph_path=paths[1], room_map_path=paths[2])
        controller.executor.room_budget_s = min(120.0, seconds)
        initial_context = (game.instance_id, game.scene, game.room)
        controller.set_intent(AgentIntent(objective="Leave the currently observed room through a physical portal",
            summary="Leave the currently observed room through a physical portal", mode="explore",
            completion=ObjectiveCompletion(kind="leave_scene_room", scene=game.scene, room=game.room)))
        started = time.monotonic()
        last_print = [started]

        def crossed():
            current = bridge.state
            return bool(current and current.player and current.instance_id == initial_context[0]
                and current.in_game and not current.cutscene_active
                and (current.scene, current.room) != initial_context[1:]
                and current.player.health > 0 and not current.game_over_state)

        def active():
            return time.monotonic() - started < seconds and not crossed() and bridge.connected

        def publish():
            if time.monotonic() - last_print[0] >= 5.0:
                last_print[0] = time.monotonic()
                print(json.dumps({"elapsed_s": round(time.monotonic() - started, 1),
                    "scene": bridge.state.scene, "room": bridge.state.room,
                    "position": bridge.state.player.position if bridge.state.player else None,
                    "execution": controller.executor.state, "motor": controller.last_setpoint.reason}), flush=True)

        report["physical_episodes"] = 1
        await controller.run(active, publish)
        report["elapsed_s"] = round(time.monotonic() - started, 3)
        report["final_state"] = bridge.state.model_dump() if bridge.state else None
        report["controller"] = controller.telemetry()
        report["artifacts_unchanged"] = before == {path.name: sha256(path) for path in paths}
        report["artifact_sha256"] = before
        report["successful_episodes"] = int(crossed())
        report["status"] = "episode_success" if crossed() else "blocked"
        report["reason"] = "observed_portal_crossing" if crossed() else controller.executor.reason or "episode_timeout_or_disconnect"
        (directory / "trace.json").write_text(json.dumps(list(controller.executor.trace), indent=2), encoding="utf-8")
        return report
    finally:
        bridge.release()
        transport.close()
        report["report_file"] = str(directory / "report.json")
        (directory / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in report.items()
            if key not in {"baseline", "episode_initial", "final_state", "controller", "bridge", "startup"}}, ensure_ascii=False), flush=True)
