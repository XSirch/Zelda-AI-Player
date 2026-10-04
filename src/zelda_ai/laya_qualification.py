"""Frozen Laya walking candidates evaluated through normal, isolated SoH input."""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

from .app import bridge_secret
from .autonomy.laya_policy import LayaWalkingPolicy
from .autonomy.local_tasks import LocalTask
from .bridge import Bridge, bind_bridge
from .config import Settings
from .g1 import OwnedProcess, prepare_suite, prepare_variation, settle, variation_target, write_json
from .laya_data import digest
from .laya_training import verify_base
from .startup import enter_playable_save
from .surface_curriculum import FAMILIES, run_task, select_surface


async def prepare_task(bridge, frozen, directory, family, variant):
    if family == "recovery_cell":
        target = variation_target(bridge.state, variant)
        if target is None:
            raise ValueError("No currently observed reachable cell")
        return LocalTask.observed_cell(bridge.state, target)
    direction = "up" if family == "surface_up" else "down"
    for scan in range(4):
        try:
            return LocalTask.traversal(bridge.state, select_surface(bridge.state, direction, variant))
        except ValueError:
            try:
                opposite = select_surface(bridge.state, "down" if direction == "up" else "up", variant)
            except ValueError:
                setup = await prepare_variation(bridge, 3, deadline=time.monotonic() + 8)
                write_json(directory / f"surface-setup-{scan}.json", setup)
            else:
                result, _ = await run_task(
                    bridge,
                    frozen,
                    LocalTask.traversal(bridge.state, opposite),
                    directory / f"relocalize-{scan}",
                )
                if not result["success"]:
                    raise RuntimeError("Opposite surface preparation failed")
    raise ValueError("No currently observed short walking surface")


async def qualify_laya(
    settings,
    executable,
    source_home,
    python,
    base,
    candidate,
    *,
    baseline=None,
    trials=3,
    seed=4102100,
    save_slot=2,
):
    if not 3 <= trials <= 300 or trials % 3:
        raise ValueError("Use 3..300 trials, divisible by three; three is a pilot only")
    directory, manifest = prepare_suite(
        settings, executable, source_home, episodes=trials, seed=seed, save_slot=save_slot, seconds=120
    )
    variants = [random.Random(seed + i).randrange(1, 5) for i in range(trials)]
    stages = [("before", baseline), ("after", candidate)] if baseline else [("candidate", candidate)]
    manifests = {
        stage: {name: digest(path / name) for name in ("candidate.json", "heads.safetensors")}
        for stage, path in stages
    }
    manifest.update(
        runner_version="frozen-laya-walking-v1",
        candidate_sha256=manifests,
        scope="short_observed_walking_only",
        decision_budget_ms=100,
        actuation_grace_ms=50,
        preparation_controller="reference_only_before_evaluation",
        promotion="disabled",
    )
    write_json(directory / "manifest.json", manifest)
    records = []
    for stage, path in stages:
        policy = await LayaWalkingPolicy.start(python, base, path, directory / f"{stage}-worker.log")
        try:
            for index in range(trials):
                trial_dir = directory / stage / f"trial-{index + 1:03}"
                trial_dir.mkdir(parents=True)
                home = trial_dir / "native-home"
                shutil.copytree(directory / "seed-home", home)
                bridge = Bridge(bridge_secret(settings), allow_simulator=False)
                await bind_bridge(bridge, 0)
                environment = dict(os.environ)
                environment.update(
                    ZELDA_BRIDGE_TOKEN=bridge.token,
                    ZELDA_BRIDGE_PORT=str(bridge.transport.get_extra_info("sockname")[1]),
                )
                process = OwnedProcess(Path(manifest["executable"]), home, environment)
                started = time.monotonic()
                family = FAMILIES[index % 3]
                record = {
                    "stage": stage,
                    "index": index + 1,
                    "family": family,
                    "source": "soh",
                    "success": False,
                    "reason": "not_started",
                    "provider_calls": 0,
                    "run_updates": 0,
                    "reference_blend": 0,
                }
                policy.invalidate()
                try:
                    async with asyncio.timeout(120):
                        while not bridge.connected and time.monotonic() - started < 15:
                            if process.child.poll() is not None:
                                raise RuntimeError("Native process exited")
                            await asyncio.sleep(0.05)
                        if not bridge.connected or not bridge.realtime or bridge.state.protocol != 3:
                            raise RuntimeError("Real native input-receipt bridge unavailable")
                        startup = await enter_playable_save(bridge, save_slot - 1, budget_s=60)
                        write_json(trial_dir / "startup.json", startup)
                        if startup["status"] != "loaded":
                            raise RuntimeError(startup.get("reason", "Save not loaded"))
                        await settle(bridge, deadline=time.monotonic() + 8)
                        setup = await prepare_variation(
                            bridge, variants[index], deadline=time.monotonic() + 8
                        )
                        write_json(trial_dir / "setup.json", setup)
                        if not setup["valid"]:
                            raise RuntimeError("Invalid physical preparation")
                        task = await prepare_task(
                            bridge, directory / "frozen", trial_dir, family, variants[index]
                        )
                        result, _ = await run_task(
                            bridge, directory / "frozen", task, trial_dir / "motor", policy=policy
                        )
                        record.update(
                            {
                                k: result[k]
                                for k in (
                                    "success",
                                    "consumed_actions",
                                    "objective_unchanged",
                                    "run_updates",
                                    "reference_blend",
                                )
                            }
                        )
                        record.update(
                            reason="success" if result["success"] else result["task"]["failure"],
                            initial_position=result["initial"]["player"]["position"],
                            final_position=result["final"]["player"]["position"],
                        )
                except (OSError, ValueError, RuntimeError, asyncio.TimeoutError) as exc:
                    record["reason"] = f"{type(exc).__name__}:{str(exc)[:180]}"
                finally:
                    policy.invalidate()
                    bridge.close()
                    await process.close()
                    record["elapsed_s"] = round(time.monotonic() - started, 3)
                    record["artifacts_unchanged"] = all(
                        digest(Path(p)) == h for p, h in manifest["pinned_sha256"].items()
                    )
                    record["artifacts_unchanged"] &= all(
                        digest(directory / "frozen" / p) == h for p, h in manifest["frozen_sha256"].items()
                    )
                    record["artifacts_unchanged"] &= all(
                        digest(directory / "seed-home" / p) == h
                        for p, h in manifest["fixture_sha256"].items()
                    )
                    record["candidate_unchanged"] = all(
                        digest(path / name) == h for name, h in manifests[stage].items()
                    )
                    record["worker_failure"] = policy.failure
                    records.append(record)
                    write_json(trial_dir / "report.json", record)
                    write_json(directory / "progress.json", {"records": records, "worker": policy.metrics})
                    print(record, flush=True)
                if policy.failure or not record["artifacts_unchanged"] or not record["candidate_unchanged"]:
                    raise RuntimeError("Frozen evaluation failed integrity or inference worker")
        finally:
            await policy.close()
            write_json(directory / f"{stage}-worker-metrics.json", policy.metrics)
    counts = {
        stage: {
            family: {
                "attempted": sum(r["stage"] == stage and r["family"] == family for r in records),
                "successful": sum(
                    r["stage"] == stage and r["family"] == family and r["success"] for r in records
                ),
            }
            for family in FAMILIES
        }
        for stage, _ in stages
    }
    report = {
        "source": "soh",
        "records": records,
        "families": counts,
        "failure_reasons": dict(Counter(r["reason"] for r in records if not r["success"])),
        "provider_calls": 0,
        "training_updates": 0,
        "promotion": "disabled",
        "g2_complete": False,
        "sword_acquisition_evaluated": False,
    }
    verify_base(base)
    write_json(directory / "report.json", report)
    print({"directory": str(directory), "families": counts}, flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description="Frozen local Laya walking qualification; zero providers")
    parser.add_argument("executable", type=Path)
    for name in ("source-home", "python", "base", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=4102100)
    parser.add_argument("--save-slot", type=int, choices=(1, 2, 3), default=2)
    args = parser.parse_args()
    asyncio.run(
        qualify_laya(
            Settings(),
            args.executable,
            args.source_home,
            args.python,
            args.base,
            args.candidate,
            baseline=args.baseline,
            trials=args.trials,
            seed=args.seed,
            save_slot=args.save_slot,
        )
    )


if __name__ == "__main__":
    main()
