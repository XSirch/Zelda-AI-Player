"""Real walking collection or frozen evaluation in continuous native sessions."""

from __future__ import annotations

import argparse
import asyncio
import math
import os
import random
import shutil
import time
import uuid
from pathlib import Path

from .app import bridge_secret
from .autonomy.laya_policy import LayaWalkingPolicy
from .autonomy.local_tasks import LocalTask
from .autonomy.navigation import observed_local_path
from .bridge import Bridge, bind_bridge
from .config import Settings
from .g1 import OwnedProcess, prepare_suite, settle, write_json
from .laya_data import digest, export_surfaces
from .startup import enter_playable_save
from .surface_curriculum import run_task


def observed_target(game, rng):
    if not game.navmesh.available or not game.player:
        raise ValueError("Current observed local collision cells are required")
    mesh, position = game.navmesh, game.player.position
    direction = rng.uniform(-math.pi, math.pi)
    candidates = []
    for x, z, y, _ in mesh.cells:
        point = (mesh.origin[0] + x * mesh.step, y, mesh.origin[2] + z * mesh.step)
        dx, dz = point[0] - position[0], point[2] - position[2]
        distance, height = math.hypot(dx, dz), abs(y - position[1])
        if not 40 <= distance <= 130 or height > 25:
            continue
        if any(math.dist(point, exit.position) < 55 for exit in game.scene_exits):
            continue
        path = observed_local_path(game, point, minimum_gain=0)
        if not path or path["partial"] or path["target_gap"] > 4 or path["path_length"] > 180:
            continue
        candidates.append((height <= 4, math.cos(math.atan2(dx, dz) - direction), -distance, point))
    if not candidates:
        raise ValueError("No currently observed reachable short walking target")
    return max(candidates)[-1]


async def collect(
    settings,
    executable,
    source_home,
    *,
    sessions=3,
    tasks=20,
    seed=4102200,
    save_slot=2,
    policy=None,
    candidate=None,
):
    if not 3 <= sessions <= 12 or not 3 <= tasks <= 40:
        raise ValueError("Use 3..12 native sessions and 3..40 tasks per session")
    if (policy is None) != (candidate is None):
        raise ValueError("Frozen candidate evaluation requires both policy and checkpoint provenance")
    directory, manifest = prepare_suite(
        settings,
        executable,
        source_home,
        episodes=sessions,
        seed=seed,
        save_slot=save_slot,
        seconds=60 + tasks * 14,
    )
    manifest.update(
        runner_version="laya-reference-walking-v1" if policy is None else "laya-continuous-walking-eval-v1",
        native_sessions=sessions,
        tasks_per_session=tasks,
        scope="current_observed_walking_cells_only",
        controller="reference_demonstrations" if policy is None else "frozen_laya_candidate",
        training_updates=0,
        promotion="disabled",
        episode_grouping="native_instance",
        decision_budget_ms=100 if policy is not None else None,
        actuation_grace_ms=50 if policy is not None else None,
    )
    expected_candidate = (
        {name: digest(candidate / name) for name in ("candidate.json", "heads.safetensors")}
        if candidate
        else None
    )
    manifest["candidate_sha256"] = expected_candidate
    write_json(directory / "manifest.json", manifest)
    records = []
    for session in range(sessions):
        home = directory / f"native-session-{session + 1}"
        shutil.copytree(directory / "seed-home", home)
        bridge = Bridge(bridge_secret(settings), allow_simulator=False)
        await bind_bridge(bridge, 0)
        environment = dict(os.environ)
        environment.update(
            ZELDA_BRIDGE_TOKEN=bridge.token,
            ZELDA_BRIDGE_PORT=str(bridge.transport.get_extra_info("sockname")[1]),
        )
        process = OwnedProcess(Path(manifest["executable"]), home, environment)
        rng = random.Random(seed + session)
        if policy is not None:
            policy.invalidate()
        try:
            async with asyncio.timeout(60 + tasks * 14):
                deadline = time.monotonic() + 15
                while not bridge.connected and time.monotonic() < deadline:
                    if process.child.poll() is not None:
                        raise RuntimeError("Native process exited")
                    await asyncio.sleep(0.05)
                if not bridge.connected or not bridge.realtime or bridge.state.protocol != 3:
                    raise RuntimeError("Real native input-receipt bridge unavailable")
                startup = await enter_playable_save(bridge, save_slot - 1, budget_s=60)
                write_json(home / "startup.json", startup)
                if startup["status"] != "loaded":
                    raise RuntimeError(startup.get("reason", "Native save not loaded"))
                await settle(bridge, deadline=time.monotonic() + 8)
                for index in range(tasks):
                    trial = (
                        directory
                        / ("demonstrations" if policy is None else "candidate")
                        / f"trial-{session * tasks + index + 1:04}"
                    )
                    trial.mkdir(parents=True)
                    record = {
                        "session": session + 1,
                        "index": index + 1,
                        "success": False,
                        "source": "soh",
                        "provider_calls": 0,
                        "controller": "reference" if policy is None else "candidate",
                    }
                    try:
                        task = LocalTask.observed_cell(bridge.state, observed_target(bridge.state, rng))
                        result, rows = await run_task(
                            bridge,
                            directory / "frozen",
                            task,
                            trial / "motor",
                            policy=policy,
                            demonstrations=policy is None,
                        )
                        record.update(
                            success=result["success"],
                            consumed_actions=result["consumed_actions"],
                            positive_labels=len(rows),
                            reason=result["task"]["failure"],
                            run_updates=result["run_updates"],
                            objective_unchanged=result["objective_unchanged"],
                            reference_blend=result["reference_blend"],
                            initial_position=result["initial"]["player"]["position"],
                            final_position=result["final"]["player"]["position"],
                        )
                    except (OSError, ValueError, RuntimeError) as exc:
                        record["reason"] = f"{type(exc).__name__}:{str(exc)[:180]}"
                    records.append(record)
                    write_json(trial / "report.json", record)
                    write_json(
                        directory / "progress.json",
                        {"records": records, "worker": policy.metrics if policy is not None else None},
                    )
                    print({"directory": str(directory), **record}, flush=True)
                    if policy is not None and policy.failure:
                        raise RuntimeError(f"Inference worker failed: {policy.failure}")
        finally:
            bridge.close()
            await process.close()
            if policy is not None:
                policy.invalidate()
        if not all(digest(Path(p)) == h for p, h in manifest["pinned_sha256"].items()):
            raise RuntimeError("Pinned source or original artifacts changed during collection")
        if not all(digest(directory / "frozen" / p) == h for p, h in manifest["frozen_sha256"].items()):
            raise RuntimeError("Frozen motor artifacts changed")
        if expected_candidate and not all(
            digest(candidate / name) == h for name, h in expected_candidate.items()
        ):
            raise RuntimeError("Frozen Laya candidate changed")
    exported = (
        export_surfaces(directory, directory / "dataset", group_native_sessions=True)
        if policy is None
        else None
    )
    write_json(
        directory / "report.json",
        {
            "source": "soh",
            "records": records,
            "dataset": exported,
            "provider_calls": 0,
            "training_updates": 0,
            "promotion": "disabled",
            "worker": policy.metrics if policy is not None else None,
            "g2_complete": False,
            "sword_acquisition_evaluated": False,
        },
    )
    if exported:
        print({"dataset": str(directory / "dataset"), "records": exported["records"]}, flush=True)
    else:
        print(
            {
                "report": str(directory / "report.json"),
                "attempted": len(records),
                "successful": sum(r["success"] for r in records),
            },
            flush=True,
        )
    return directory / ("dataset" if exported else "report.json")


async def evaluate(settings, executable, source_home, *, python, base, candidate, **options):
    from .laya_training import verify_base

    log = settings.data_dir / "laya" / f"walking-eval-{uuid.uuid4().hex[:12]}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    policy = await LayaWalkingPolicy.start(python, base, candidate, log)
    try:
        result = await collect(
            settings, executable, source_home, policy=policy, candidate=candidate, **options
        )
        verify_base(base)
        return result
    finally:
        await policy.close()


def main():
    parser = argparse.ArgumentParser(
        description="Real walking collection or frozen candidate evaluation; zero providers"
    )
    parser.add_argument("executable", type=Path)
    parser.add_argument("--source-home", type=Path, required=True)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=4102200)
    parser.add_argument("--save-slot", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--python", type=Path)
    args = parser.parse_args()
    if args.candidate and (not args.base or not args.python):
        parser.error("--candidate requires --base and --python")
    runner = evaluate if args.candidate else collect
    extras = {"candidate": args.candidate, "base": args.base, "python": args.python} if args.candidate else {}
    asyncio.run(
        runner(
            Settings(),
            args.executable,
            args.source_home,
            sessions=args.sessions,
            tasks=args.tasks,
            seed=args.seed,
            save_slot=args.save_slot,
            **extras,
        )
    )


if __name__ == "__main__":
    main()
