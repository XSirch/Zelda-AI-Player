"""Isolated reference data or frozen candidate attached descents; zero providers."""
from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import time
from pathlib import Path

from .app import bridge_secret
from .autonomy.controller import ContinuousController
from .autonomy.laya_ladder_policy import LayaLadderPolicy
from .bridge import Bridge, bind_bridge
from .config import Settings
from .g1 import ARTIFACTS, OwnedProcess, prepare_suite, settle, write_json
from .laya_curriculum import prepare_observed_descent, prepare_portal_start
from .laya_data import digest
from .laya_ladder_data import export_ladders
from .laya_training import verify_base
from .startup import enter_playable_save


async def warm_reference(frozen):
    """Pay cold GPU/optimizer initialization before any native process exists."""
    bridge = Bridge("local-warmup-without-native-peer", allow_simulator=False)
    started = time.monotonic()
    try:
        controller = await asyncio.to_thread(ContinuousController, bridge, frozen / ARTIFACTS[0],
            training_enabled=False, route_graph_path=frozen / ARTIFACTS[1], room_map_path=frozen / ARTIFACTS[2])
        if bridge.command_seq != 0 or controller.training_enabled or controller.policy.updates != controller.starting_updates:
            raise RuntimeError("Warmup must not send inputs or train")
        return {"elapsed_ms": (time.monotonic() - started) * 1000, "native_commands": 0, "run_updates": 0}
    finally:
        bridge.close()


async def ladder_curriculum(settings, executable, source_home, *, sessions=3, save_slot=2,
                            seed=4103300, python=None, base=None, candidate=None):
    if not 3 <= sessions <= 12 or save_slot not in (1, 2, 3):
        raise ValueError("Use 3..12 separate native sessions and a normal save slot")
    if candidate and (not python or not base):
        raise ValueError("Candidate evaluation needs the isolated interpreter and pinned base")
    directory, manifest = prepare_suite(settings, executable, source_home,
        episodes=sessions, seed=seed, save_slot=save_slot, seconds=120)
    candidate_hashes = ({name: digest(candidate / name) for name in ("candidate.json", "heads.safetensors")}
                        if candidate else {})
    manifest.update(runner_version="laya-attached-descent-v1", native_sessions=sessions,
        initial_profile="copied_selected_save_normal_startup_then_observed_portal_and_ladder_attachment",
        scope="one_attached_descent_per_native_instance_no_approach_credit", candidate_sha256=candidate_hashes,
        decision_budget_ms=100, actuation_grace_ms=50, promotion="disabled")
    write_json(directory / "manifest.json", manifest)
    warmup = await warm_reference(directory / "frozen")
    manifest["reference_warmup_before_native_launch"] = warmup
    write_json(directory / "manifest.json", manifest)
    policy = (await LayaLadderPolicy.start(python, base, candidate, directory / "candidate-worker.log")
              if candidate else None)
    records, preparations = [], []
    try:
        for index in range(sessions):
            trial = directory / f"native-session-{index + 1}"
            trial.mkdir()
            home = trial / "native-home"
            shutil.copytree(directory / "seed-home", home)
            bridge = Bridge(bridge_secret(settings), allow_simulator=False)
            await bind_bridge(bridge, 0)
            environment = dict(os.environ)
            environment.update(ZELDA_BRIDGE_TOKEN=bridge.token,
                               ZELDA_BRIDGE_PORT=str(bridge.transport.get_extra_info("sockname")[1]))
            process = OwnedProcess(Path(manifest["executable"]), home, environment)
            record = {"session": index + 1, "source": "soh", "attempted": False, "success": False,
                "reason": "not_started", "candidate_actions": 0, "reference_actions": 0,
                "portal_actions": 0,
                "provider_calls": 0, "run_updates": 0, "reference_blend": 0}
            if policy:
                policy.invalidate()
            try:
                async with asyncio.timeout(180):
                    started = time.monotonic()
                    while not bridge.connected and time.monotonic() - started < 15:
                        if process.child.poll() is not None:
                            raise RuntimeError("Owned native process exited")
                        await asyncio.sleep(.05)
                    if not bridge.connected or not bridge.realtime or bridge.state.protocol != 3:
                        raise RuntimeError("Real native bridge unavailable")
                    startup = await enter_playable_save(bridge, save_slot - 1, budget_s=60)
                    write_json(trial / "startup.json", startup)
                    if startup["status"] != "loaded":
                        raise RuntimeError(startup.get("reason", "Normal save startup failed"))
                    await settle(bridge, deadline=time.monotonic() + 8)
                    portal = await prepare_portal_start(bridge, directory / "frozen", trial / "portal-preparation")
                    preparations.append(portal)
                    record["portal_actions"] = portal["consumed_commands"]
                    record["reference_actions"] = portal["consumed_commands"]
                    if not portal["success"]:
                        raise RuntimeError("Portal preparation failed")
                    descent = await prepare_observed_descent(bridge, directory / "frozen",
                        trial / "descent-preparation", attached_policy=policy, collect_attached_labels=policy is None)
                    preparations.append(descent)
                    stages = [stage for attempt in descent["attempts"] for stage in attempt["stages"]]
                    attached = [stage for stage in stages if "ladder-descent-task" in stage["task_version"]
                                or "attached-descent-reference" in stage["task_version"]]
                    record.update(attempted=bool(attached), success=descent["success"] and bool(attached),
                        reason=descent["reason"], candidate_actions=descent["candidate_actions"],
                        reference_actions=record["portal_actions"] + descent["consumed_actions"] - descent["candidate_actions"],
                        provider_calls=descent["provider_calls"],
                        run_updates=sum(attempt["run_updates"] for attempt in descent["attempts"]),
                        objective_unchanged=all(attempt["objective_unchanged"] for attempt in descent["attempts"]))
            except (ValueError, RuntimeError, OSError, TimeoutError) as exc:
                record["reason"] = f"{type(exc).__name__}:{str(exc)[:160]}"
            finally:
                bridge.close()
                await process.close()
                if policy:
                    policy.invalidate()
            records.append(record)
            write_json(trial / "report.json", record)
            write_json(directory / "progress.json", {"records": records, "worker": policy.metrics if policy else None})
            print({"directory": str(directory), **record}, flush=True)
            if not all(digest(Path(path)) == sha for path, sha in manifest["pinned_sha256"].items()):
                raise RuntimeError("Pinned source or original artifacts changed")
            if not all(digest(directory / "frozen" / name) == sha for name, sha in manifest["frozen_sha256"].items()):
                raise RuntimeError("Frozen motor or maps changed")
            if candidate and not all(digest(candidate / name) == sha for name, sha in candidate_hashes.items()):
                raise RuntimeError("Frozen attached candidate changed")
            if policy and policy.failure:
                raise RuntimeError(f"Attached inference worker failed: {policy.failure}")
        report = {"source": "soh", "records": records, "preparations": preparations,
            "planned_trials": sessions, "attempted_trials": sum(r["attempted"] for r in records),
            "successful_trials": sum(r["success"] for r in records),
            "candidate_actions": sum(r["candidate_actions"] for r in records),
            "reference_actions": sum(r["reference_actions"] for r in records),
            "worker": policy.metrics if policy else None, "provider_calls": 0, "training_updates": 0,
            "reference_warmup_before_native_launch": warmup,
            "promotion": "disabled", "g2_complete": False, "dataset": None, "dataset_error": None}
        write_json(directory / "report.json", report)
        if policy is None:
            try:
                report["dataset"] = export_ladders(directory, directory / "dataset")
            except ValueError as exc:
                report["dataset_error"] = str(exc)
                write_json(directory / "report.json", report)
                raise
            write_json(directory / "report.json", report)
        else:
            verify_base(base)
        print({"report": str(directory / "report.json"), "attempted": report["attempted_trials"],
               "successful": report["successful_trials"], "dataset": str(directory / "dataset") if not policy else None},
              flush=True)
        return directory
    finally:
        if policy:
            await policy.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--source-home", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--save-slot", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--seed", type=int, default=4103300)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    asyncio.run(ladder_curriculum(Settings(), **vars(args)))


if __name__ == "__main__":
    main()
