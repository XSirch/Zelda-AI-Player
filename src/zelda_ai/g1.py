"""Automatic, isolated physical G1 qualification. No cognition provider is used.

The process supervisor is QA infrastructure, outside the controller. Save loads
use normal startup inputs; position variations use CURRENT collision links.
Neither supervisor nor motor can teleport, write game state or script a quest.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import random
import shutil
import subprocess
import time
import uuid
from collections import Counter, deque
from pathlib import Path

from .app import bridge_secret
from .autonomy.controller import ContinuousController
from .autonomy.features import camera_relative_stick
from .autonomy.models import AgentIntent, ObjectiveCompletion
from .autonomy.navigation import observed_local_path
from .bridge import Bridge, bind_bridge
from .qualification import sha256
from .startup import enter_playable_save

ARTIFACTS = ("raw-controller-ppo-rnd-v3.pt", "route-graph-v1.json", "room-map-v1.json")
G1_EPISODES = 100
G1_SUCCESSES = 99
RUNNER_VERSION = "physical-g1-v1"


def write_json(path: Path, value):
    """Publish one complete evidence file, never a partially written report."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def isolated_config(config):
    """Disable physical-device mappings only in the QA copy, not the user's config."""
    settings = config.setdefault("CVars", {}).setdefault("gSettings", {})
    controllers = settings.setdefault("Controllers", {})
    port = controllers.setdefault("Port1", {})
    port["HasConfig"] = 1  # Native ControlDeck must not reinstall default mappings.
    for section in ("Buttons", "LeftStick", "RightStick"):
        values = port.setdefault(section, {})
        for key in list(values):
            if key.endswith("MappingIds"):
                values[key] = ""
    for key in ("GyroMappingId", "RumbleMappingIds", "LEDMappingIds"):
        port[key] = ""
    return config


def physical_mappings_disabled(config):
    port = config.get("CVars", {}).get("gSettings", {}).get("Controllers", {}).get("Port1", {})
    return port.get("HasConfig") == 1 and all(
        not value for section in ("Buttons", "LeftStick", "RightStick")
        for key, value in port.get(section, {}).items() if key.endswith("MappingIds"))


def context(game):
    return (game.instance_id, game.scene, game.room, game.mirrored_world,
            game.player.age if game.player else None)


def playable(game):
    return bool(game and game.source == "soh" and game.in_game and game.player
                and game.player.health > 0 and not game.game_over_state
                and not game.cutscene_active and not game.paused
                and not game.dialogue.active and not game.pause_menu.active)


def portal_crossed(initial, final):
    if not playable(initial) or not playable(final):
        return False
    old, new = context(initial), context(final)
    if old[0] != new[0] or old[3:] != new[3:] or old[1:3] == new[1:3]:
        return False
    # Death/save-load/process restart is never a physical portal crossing.
    return (final.seq > initial.seq and final.scene_epoch > initial.scene_epoch
            and not any(e.kind in {"save_loaded", "game_over", "player_died"}
                        and e.id not in {old.id for old in initial.events} for e in final.events))


def variation_target(game, variant: int):
    """Choose a reachable, nearby observed cell. No room coordinates are stored."""
    if variant == 0 or not playable(game) or not game.navmesh.available:
        return None
    mesh, position = game.navmesh, game.player.position
    desired = (variant - 1) * math.pi / 2
    candidates = []
    for x, z, y, _ in mesh.cells:
        point = (mesh.origin[0] + x * mesh.step, y, mesh.origin[2] + z * mesh.step)
        dx, dz = point[0] - position[0], point[2] - position[2]
        distance = math.hypot(dx, dz)
        if not 35 <= distance <= 145 or abs(y - position[1]) > 25:
            continue
        # Keep setup out of observed portals. Crossing here would invalidate it.
        if any(math.dist(point, exit.position) < 55 for exit in game.scene_exits):
            continue
        path = observed_local_path(game, point, minimum_gain=5)
        if path is None or path["partial"] or path["path_length"] > 180:
            continue
        alignment = math.cos(math.atan2(dx, dz) - desired)
        candidates.append((alignment, -abs(distance - 70), point))
    return max(candidates)[2] if candidates else None


async def settle(bridge, *, deadline):
    """Require three fresh playable frames and a stopped player after release."""
    count, last_seq = 0, -1
    while time.monotonic() < deadline:
        game = bridge.state
        if game and game.seq != last_seq:
            last_seq = game.seq
            count = count + 1 if playable(game) and game.player.speed_xz < 0.1 else 0
            if count >= 3:
                return game
        await asyncio.sleep(.05)
    raise RuntimeError("playable_settle_timeout")


async def _move_observed_cell(bridge, target, *, deadline):
    initial = bridge.state.model_copy(deep=True)
    trace = deque(maxlen=160)
    start = time.monotonic()
    status = "baseline" if target is None else "budget_exhausted"
    try:
        while target is not None and time.monotonic() < deadline:
            game = bridge.state
            if not playable(game) or context(game) != context(initial):
                status = "setup_context_or_modal_changed"
                break
            if math.dist(game.player.position, target) <= 16:
                status = "reached_observed_cell"
                break
            path = observed_local_path(game, target, minimum_gain=2)
            waypoint = path["waypoint"] if path else target
            # Stop if the mesh no longer proves a path, except at the final cell.
            if path is None and math.dist(game.player.position, target) > game.navmesh.step * .8:
                status = "setup_path_lost"
                break
            dx = waypoint[0] - game.player.position[0]
            dz = waypoint[2] - game.player.position[2]
            stick = camera_relative_stick(game, game.player, math.atan2(dx, dz))
            seq = bridge.send(stick_x=round(stick[0] * 60), stick_y=round(stick[1] * 60), lease_ms=150)
            await asyncio.sleep(.05)
            trace.append({"seq": game.seq, "position": game.player.position,
                          "camera_input_yaw": game.camera_input_yaw, "target": target,
                          "waypoint": waypoint, "command_seq": seq,
                          "stick": [round(v * 60) for v in stick],
                          "consumed": bridge.command_consumed(seq)})
    finally:
        bridge.release()
    final = await settle(bridge, deadline=min(deadline + 2, time.monotonic() + 2))
    delta = [final.player.position[i] - initial.player.position[i] for i in range(3)]
    displacement = math.hypot(delta[0], delta[2])
    heading = math.atan2(delta[0], delta[2]) % (2 * math.pi) if displacement >= 12 else None
    return {"status": status, "target": target,
            "valid": context(initial) == context(final) and playable(final),
            "elapsed_s": round(time.monotonic() - start, 3),
            "initial": initial.model_dump(), "final": final.model_dump(), "displacement": delta,
            "heading_bin": int((heading + math.pi / 4) % (2 * math.pi) / (math.pi / 2))
                           if heading is not None else None,
            "consumed_commands": sum(bridge.command_consumed(row["command_seq"]) for row in trace),
            "trace": list(trace)}


def target_alignment(game, target, variant):
    if target is None:
        return -1.0
    dx, dz = target[0] - game.player.position[0], target[2] - game.player.position[2]
    return math.cos(math.atan2(dx, dz) - (variant - 1) * math.pi / 2)


async def prepare_variation(bridge, variant, *, deadline):
    initial = bridge.state.model_copy(deep=True)
    target = variation_target(initial, variant)
    legs = []
    # If the requested direction is blocked at spawn, first relocate through an
    # observed cell in the opposite direction. Then re-observe before probing
    # the requested direction. This is a QA transform check, not a route macro.
    if variant and target_alignment(initial, target, variant) < .65:
        opposite = (variant + 1) % 4 + 1
        relocation = variation_target(initial, opposite)
        if target_alignment(initial, relocation, opposite) >= .65:
            legs.append(await _move_observed_cell(bridge, relocation, deadline=deadline))
            if legs[-1]["valid"] and time.monotonic() < deadline - 1:
                target = variation_target(bridge.state, variant)
    if not legs or legs[-1]["valid"]:
        legs.append(await _move_observed_cell(bridge, target, deadline=deadline))
    final = bridge.state
    return {"variant": variant, "status": legs[-1]["status"],
            "valid": all(leg["valid"] for leg in legs) and context(initial) == context(final),
            "initial": initial.model_dump(), "final": final.model_dump(),
            "heading_bin": legs[-1]["heading_bin"],
            "heading_bins": sorted({leg["heading_bin"] for leg in legs
                                    if leg["heading_bin"] is not None and leg["consumed_commands"] > 0}),
            "consumed_commands": sum(leg["consumed_commands"] for leg in legs), "legs": legs}


class OwnedProcess:
    """Launch and stop ONLY the child created by this QA supervisor."""

    def __init__(self, executable, home, environment):
        self.log = (home / "process.log").open("wb")
        try:
            self.child = subprocess.Popen([str(executable)], cwd=home, env=environment,
                                          stdout=self.log, stderr=subprocess.STDOUT)
        except BaseException:
            self.log.close()
            raise

    async def close(self):
        if self.child.poll() is None:
            if os.name == "nt":
                import ctypes
                from ctypes import wintypes
                callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

                @callback_type
                def close_window(window, _):
                    owner = wintypes.DWORD()
                    ctypes.windll.user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
                    if owner.value == self.child.pid:
                        ctypes.windll.user32.PostMessageW(window, 0x0010, 0, 0)  # ordinary WM_CLOSE
                    return True

                user32 = ctypes.windll.user32
                user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
                user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
                user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
                user32.EnumWindows(close_window, 0)
            else:
                self.child.terminate()
            try:
                await asyncio.to_thread(self.child.wait, timeout=5)
            except subprocess.TimeoutExpired:
                self.child.kill()
                await asyncio.to_thread(self.child.wait, timeout=3)
        self.log.close()


def summarize(manifest, episodes, *, finished=False, artifacts_unchanged=True):
    """Fail-closed gate. Pilots, duplicates and nominal variations do not qualify."""
    valid = [e for e in episodes if e.get("success") and e.get("source") == "soh"
             and e.get("settled_crossing") and e.get("consumed_commands", 0) > 0
             and 0 < e.get("total_elapsed_s", math.inf) <= 120
             and e.get("run_updates") == 0 and e.get("artifacts_unchanged")
             and e.get("setup_valid") and e.get("objective_unchanged")]
    poses = {(e["initial_context"][1], e["initial_context"][2],
              *(round(v / 20) for v in e["initial_position"])) for e in valid}
    cameras = {int(((e["initial_camera_yaw"] or 0) % 65536 + 4096) % 65536 / 8192)
               for e in valid if e.get("initial_camera_yaw") is not None}
    rooms = {tuple(e["initial_context"][1:3]) for e in valid}
    headings = {heading for e in valid for heading in e.get("setup_heading_bins", [])
                if e.get("setup_consumed_commands", 0) > 0}
    unique = len({e.get("episode_id") for e in episodes}) == len(episodes)
    checks = {"finished": finished, "exactly_100_attempts": len(episodes) == G1_EPISODES
              and manifest["planned_episodes"] == G1_EPISODES,
              "at_least_99_valid_successes": len(valid) >= G1_SUCCESSES,
              "unique_attempts": unique, "artifacts_unchanged": artifacts_unchanged,
              "no_provider_calls": manifest["provider_calls"] == 0,
              "physical_device_mappings_disabled": manifest.get("physical_device_mappings_disabled", False),
              "one_native_build_and_revision": len({(e.get("native_build"), e.get("upstream_revision"))
                                                    for e in valid}) == 1
               and all(e.get("native_build", "legacy") != "legacy"
                       and e.get("upstream_revision", "unknown") != "unknown" for e in valid),
              "at_least_four_observed_poses": len(poses) >= 4,
              "at_least_two_camera_bins": len(cameras) >= 2,
              "at_least_two_room_contexts": len(rooms) >= 2,
              "four_consumed_setup_heading_bins": headings == {0, 1, 2, 3}}
    return {"schema_version": 1, "suite_id": manifest["suite_id"],
            "status": "qualified" if all(checks.values()) else "finished_unqualified" if finished else "running",
            "g1_qualified": all(checks.values()), "source": "soh",
            "observation_profile": "instrumented_local_v1", "provider_calls": 0,
            "training_enabled": False, "attempted_episodes": len(episodes),
            "successful_episodes": len(valid), "checks": checks,
            "coverage": {"poses": len(poses), "camera_bins": sorted(cameras),
                         "room_contexts": sorted(rooms), "setup_heading_bins": sorted(headings)},
            "failures": dict(Counter(e.get("reason", "invalid_success_evidence")
                                     for e in episodes if e not in valid)),
            "episodes": episodes}


def prepare_suite(settings, executable: Path, source_home: Path, *, episodes, seed, save_slot, seconds):
    executable, source_home = executable.resolve(strict=True), source_home.resolve(strict=True)
    if not executable.is_file() or not (source_home / "Save" / f"file{save_slot}.sav").is_file():
        raise ValueError("Existing executable and explicitly selected native save are required")
    originals = [settings.data_dir.resolve() / "ml" / name for name in ARTIFACTS]
    if not all(path.is_file() for path in originals):
        raise ValueError("G1 requires an existing frozen policy, route graph and room map")
    directory = settings.data_dir.resolve() / "qualification" / f"g1-{uuid.uuid4().hex[:12]}"
    directory.mkdir(parents=True)
    seed_home, frozen = directory / "seed-home", directory / "frozen"
    seed_home.mkdir()
    frozen.mkdir()
    shutil.copytree(source_home / "Save", seed_home / "Save")
    config = source_home / "shipofharkinian.json"
    write_json(seed_home / config.name, isolated_config(json.loads(config.read_text(encoding="utf-8"))))
    if (source_home / "mods").is_dir():
        shutil.copytree(source_home / "mods", seed_home / "mods")
    for original in originals:
        shutil.copy2(original, frozen / original.name)
    assets = sorted(executable.parent.glob("*.o2r"))
    if not (executable.parent / "soh.o2r").is_file() or not (executable.parent / "oot.o2r").is_file():
        raise ValueError("Real local SoH assets are required; no assets will be downloaded")
    # Local hashes only; no ROM/save/config bytes appear in the public index.
    source_files = [*Path(__file__).parent.rglob("*.py"),
                    *Path(__file__).resolve().parents[2].joinpath("native").glob("*")]
    pinned = [executable, config, *assets, *originals, *sorted((source_home / "Save").glob("*.sav")),
              *[p for p in source_files if p.is_file()],
              *[p for p in (source_home / "mods").rglob("*") if p.is_file()]]
    manifest = {"schema_version": 1, "runner_version": RUNNER_VERSION, "suite_id": directory.name,
                "planned_episodes": episodes, "seed": seed, "save_slot": save_slot,
                "seconds_budget": seconds, "budget_scope": "reset_launch_startup_setup_motor_and_settle",
                "source": "soh", "observation_profile": "instrumented_local_v1",
                "provider_calls": 0, "training_enabled": False,
                "input_policy": "harness_controller_only_no_operator_game_input",
                "physical_device_mappings_disabled": True,
                "executable": str(executable), "source_home": str(source_home),
                "pinned_sha256": {str(path): sha256(path) for path in pinned},
                "frozen_sha256": {name: sha256(frozen / name) for name in ARTIFACTS},
                "fixture_sha256": {str(path.relative_to(seed_home)): sha256(path)
                                   for path in seed_home.rglob("*") if path.is_file()}}
    write_json(directory / "manifest.json", manifest)
    return directory, manifest


async def _motor_episode(bridge, frozen, directory, *, deadline):
    initial = bridge.state.model_copy(deep=True)
    # A cold optimizer/CUDA/checkpoint load must not starve native packets and
    # expire Bridge.connected before the first motor tick. Loading still pays
    # the existing scenario deadline; the receive loop remains responsive.
    controller = await asyncio.to_thread(
        ContinuousController, bridge, frozen / ARTIFACTS[0], training_enabled=False,
        route_graph_path=frozen / ARTIFACTS[1], room_map_path=frozen / ARTIFACTS[2])
    controller.executor.room_budget_s = max(1, deadline - time.monotonic())
    objective = AgentIntent(objective="Leave the currently observed room through a physical portal",
                            summary="Leave the currently observed room through a physical portal", mode="explore",
                            completion=ObjectiveCompletion(kind="leave_scene_room", scene=initial.scene, room=initial.room))
    controller.set_intent(objective)
    first_command = bridge.command_seq
    full_trace = deque(maxlen=2400)  # <=120 s at the native 20 Hz motor cadence
    last_print = time.monotonic()

    def active():
        return (time.monotonic() < deadline and bridge.connected
                and not portal_crossed(initial, bridge.state))

    def publish():
        nonlocal last_print
        if controller.executor.trace:
            full_trace.append(controller.executor.trace[-1])
        if time.monotonic() - last_print >= 10:
            last_print = time.monotonic()
            print(json.dumps({"phase": "motor", "scene": bridge.state.scene,
                              "position": bridge.state.player.position if bridge.state.player else None,
                              "reason": controller.last_setpoint.reason}), flush=True)

    # Leave a bounded verification reserve inside the complete scenario budget.
    motor_deadline = deadline - .4

    def bounded_active():
        return time.monotonic() < motor_deadline and active()

    motor_failure = None
    try:
        await controller.run(bounded_active, publish)
    except RuntimeError as exc:
        # The controller releases authority in its finally path. Preserve
        # any already-consumed inputs instead of losing the failed episode.
        motor_failure = f"RuntimeError:{str(exc)[:160]}"
    final = bridge.state
    verified_crossing = motor_failure is None and portal_crossed(initial, final)
    if verified_crossing:
        try:
            final = await settle(bridge, deadline=deadline)
            verified_crossing = portal_crossed(initial, final)
        except RuntimeError:
            verified_crossing = False
    receipts = [r.model_dump() for s, r in bridge.receipts.items() if s > first_command and r.first_tick > 0]
    telemetry = controller.telemetry()
    write_json(directory / "motor.json", {"initial": initial.model_dump(),
               "final": final.model_dump() if final else None, "controller": telemetry,
               "objective_unchanged": controller.intent == objective, "trace": list(full_trace), "receipts": receipts,
               "motor_failure": motor_failure})
    return {"settled_crossing": verified_crossing, "consumed_commands": len(receipts),
            "objective_unchanged": controller.intent == objective,
            "run_updates": telemetry["learning"]["run_updates"],
            "initial_context": context(initial), "initial_position": initial.player.position,
            "initial_camera_yaw": initial.camera_input_yaw,
            "final_context": context(final) if final else None,
            "final_position": final.player.position if final and final.player else None,
            "reason": "observed_settled_portal_crossing" if verified_crossing
                      else motor_failure or controller.executor.reason or "motor_budget_or_disconnect"}


async def qualify_g1(settings, executable, source_home, *, episodes=100, seed=1042026, save_slot=2, seconds=120):
    if not 1 <= episodes <= 100 or not 10 <= seconds <= 120 or save_slot not in {1, 2, 3}:
        raise ValueError("G1 allows 1..100 attempts, 10..120 seconds and an explicitly selected existing save")
    directory, manifest = prepare_suite(settings, executable, source_home, episodes=episodes,
                                        seed=seed, save_slot=save_slot, seconds=seconds)
    frozen, records = directory / "frozen", []
    token = bridge_secret(settings)
    rng = random.Random(seed)
    variants = [i % 5 for i in range((episodes + 1) // 2)]
    rng.shuffle(variants)
    process = bridge = None
    unchanged = True
    try:
        for index in range(episodes):
            episode_dir = directory / f"episode-{index + 1:03}"
            episode_dir.mkdir()
            started = time.monotonic()
            deadline = started + seconds
            row = {"episode_id": f"{manifest['suite_id']}:{index + 1}", "index": index + 1,
                   "source": "soh", "success": False, "setup_valid": False,
                   "reason": "episode_not_started", "artifacts_unchanged": False,
                   "pair_role": "outbound" if index % 2 == 0 else "revisit"}
            try:
                if index % 2 == 0 or process is None or not playable(bridge.state):
                    if bridge:
                        bridge.close()
                    if process:
                        await process.close()
                    home = episode_dir / "native-home"
                    shutil.copytree(directory / "seed-home", home)
                    bridge = Bridge(token, allow_simulator=False)
                    # Bind once, then give the actual chosen port to the native process.
                    await bind_bridge(bridge, 0)
                    port = bridge.transport.get_extra_info("sockname")[1]
                    environment = dict(os.environ)
                    environment.update(ZELDA_BRIDGE_TOKEN=token, ZELDA_BRIDGE_PORT=str(port), SHIP_HOME=str(home))
                    process = OwnedProcess(Path(manifest["executable"]), home, environment)
                    while not bridge.connected and time.monotonic() < min(deadline, started + 15):
                        if process.child.poll() is not None:
                            raise RuntimeError("owned_soh_process_exited")
                        await asyncio.sleep(.05)
                    if not bridge.connected or not bridge.realtime or bridge.state.protocol != 3:
                        raise RuntimeError("real_native_bridge_unavailable")
                    startup = await enter_playable_save(bridge, save_slot - 1,
                                                       budget_s=max(.1, min(60, deadline - time.monotonic())))
                    write_json(episode_dir / "startup.json", startup)
                    if startup["status"] != "loaded":
                        raise RuntimeError(startup["reason"])
                await settle(bridge, deadline=min(deadline, time.monotonic() + 8))
                variant = variants[index // 2]
                # Different legitimate preparation direction on the return visit.
                if index % 2 and variant:
                    variant = (variant + 1) % 4 + 1
                setup = await prepare_variation(bridge, variant, deadline=min(deadline - 2, time.monotonic() + 8))
                write_json(episode_dir / "setup.json", setup)
                row.update(setup_valid=setup["valid"], setup_status=setup["status"],
                           setup_heading_bin=setup["heading_bin"], setup_heading_bins=setup["heading_bins"],
                           setup_consumed_commands=setup["consumed_commands"])
                if not setup["valid"]:
                    raise RuntimeError("invalid_physical_setup")
                if time.monotonic() >= deadline:
                    raise RuntimeError("complete_scenario_budget_exhausted")
                row.update(await _motor_episode(bridge, frozen, episode_dir, deadline=deadline))
                row["success"] = row["settled_crossing"] and row["consumed_commands"] > 0 and row["run_updates"] == 0
                row["native_build"] = bridge.state.bridge_build
                row["upstream_revision"] = bridge.state.upstream_revision
            except (RuntimeError, ValueError, OSError) as exc:
                row["reason"] = f"{type(exc).__name__}:{str(exc)[:200]}"
            finally:
                if bridge:
                    bridge.release()
                row["total_elapsed_s"] = round(time.monotonic() - started, 3)
                row["artifacts_unchanged"] = manifest["frozen_sha256"] == {name: sha256(frozen / name) for name in ARTIFACTS}
                pinned_ok = all(sha256(Path(path)) == value for path, value in manifest["pinned_sha256"].items())
                fixture_ok = all(sha256(directory / "seed-home" / path) == value
                                 for path, value in manifest["fixture_sha256"].items())
                unchanged = unchanged and row["artifacts_unchanged"] and pinned_ok and fixture_ok
                write_json(episode_dir / "report.json", row)
                records.append(row)
                write_json(directory / "report.json", summarize(manifest, records, artifacts_unchanged=unchanged))
                print(json.dumps({"suite": directory.name, **row}), flush=True)
            if not unchanged:
                raise RuntimeError("pinned_artifact_changed_batch_stopped")
    finally:
        if bridge:
            with contextlib.suppress(RuntimeError):
                bridge.close()
        if process:
            await process.close()
        unchanged = unchanged and all(sha256(Path(path)) == value
                                       for path, value in manifest["pinned_sha256"].items())
        unchanged = unchanged and all(sha256(frozen / name) == value
                                       for name, value in manifest["frozen_sha256"].items())
        mappings_ok = all(physical_mappings_disabled(json.loads(path.read_text(encoding="utf-8")))
                          for path in directory.glob("episode-*/native-home/shipofharkinian.json"))
        manifest["physical_device_mappings_disabled"] = mappings_ok
        write_json(directory / "manifest.json", manifest)
        report = summarize(manifest, records, finished=len(records) == episodes, artifacts_unchanged=unchanged)
        report["report_file"] = str(directory / "report.json")
        write_json(directory / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "episodes"}), flush=True)
    return report
