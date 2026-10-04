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
from .g1 import OwnedProcess, _motor_episode, context, portal_crossed, prepare_suite, settle, write_json
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


async def prepare_portal_start(bridge, frozen, directory):
    """QA preparation through real input, separately attributed from Laya."""
    directory.mkdir(parents=True, exist_ok=True)
    initial = bridge.state.model_copy(deep=True)
    result = await _motor_episode(bridge, frozen, directory, deadline=time.monotonic() + 120)
    success = bool(result["settled_crossing"] and result["consumed_commands"] > 0
                   and result["objective_unchanged"] and result["run_updates"] == 0
                   and portal_crossed(initial, bridge.state))
    report = {**result, "success": success, "source": "soh", "provider_calls": 0,
              "controller": "frozen_v3_motor_qa_preparation",
              "candidate_actions": 0, "included_in_laya_training": False}
    write_json(directory / "report.json", report)
    return report


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
    cross_initial_portal=False,
):
    if not 3 <= sessions <= 12 or not 3 <= tasks <= 40:
        raise ValueError("Use 3..12 native sessions and 3..40 tasks per session")
    if (policy is None) != (candidate is None):
        raise ValueError("Frozen candidate evaluation requires both policy and checkpoint provenance")
    session_budget = 60 + tasks * 14 + (120 if cross_initial_portal else 0)
    directory, manifest = prepare_suite(
        settings,
        executable,
        source_home,
        episodes=sessions,
        seed=seed,
        save_slot=save_slot,
        seconds=session_budget,
    )
    manifest.update(
        runner_version="laya-reference-walking-v2" if policy is None else "laya-continuous-walking-eval-v2",
        native_sessions=sessions,
        tasks_per_session=tasks,
        scope="current_observed_walking_cells_only",
        controller="reference_demonstrations" if policy is None else "frozen_laya_candidate",
        training_updates=0,
        promotion="disabled",
        episode_grouping="native_instance",
        decision_budget_ms=100 if policy is not None else None,
        actuation_grace_ms=50 if policy is not None else None,
        initial_preparation="normal_input_portal_crossing" if cross_initial_portal else "none",
        preparation_controller="frozen_v3_motor" if cross_initial_portal else None,
        preparation_in_candidate_results=False,
        preparation_in_training=False,
    )
    expected_candidate = (
        {name: digest(candidate / name) for name in ("candidate.json", "heads.safetensors")}
        if candidate
        else None
    )
    manifest["candidate_sha256"] = expected_candidate
    write_json(directory / "manifest.json", manifest)
    records, preparations = [], []
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
            async with asyncio.timeout(session_budget):
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
                preparation = None
                if cross_initial_portal:
                    preparation = await prepare_portal_start(
                        bridge, directory / "frozen", home / "portal-preparation",
                    )
                    preparations.append({"session": session + 1, **preparation})
                    write_json(directory / "preparations.json", preparations)
                    if policy is not None:
                        policy.invalidate()
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
                        "attempted": False,
                        "consumed_actions": 0,
                        "run_updates": 0,
                        "reference_blend": 0,
                        "objective_unchanged": None,
                        "initial_context": context(bridge.state),
                    }
                    try:
                        if preparation is not None and not preparation["success"]:
                            raise RuntimeError("initial_portal_preparation_failed")
                        task = LocalTask.observed_cell(bridge.state, observed_target(bridge.state, rng))
                        record["attempted"] = True
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
                            final_context=context(bridge.state),
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
    report = {
        "source": "soh",
        "records": records,
        "preparations": preparations,
        "planned_tasks": sessions * tasks,
        "attempted_tasks": sum(r["attempted"] for r in records),
        "successful_tasks": sum(r["success"] for r in records),
        "dataset": None,
        "dataset_error": None,
        "provider_calls": 0,
        "training_updates": 0,
        "promotion": "disabled",
        "worker": policy.metrics if policy is not None else None,
        "g2_complete": False,
        "sword_acquisition_evaluated": False,
    }
    # Preserve native failures even when too few successful sessions remain
    # for a causal train/validation/test export. Invalid exports still fail.
    write_json(directory / "report.json", report)
    exported = None
    if policy is None:
        try:
            exported = export_surfaces(directory, directory / "dataset", group_native_sessions=True)
        except ValueError as exc:
            report["dataset_error"] = str(exc)[:180]
            write_json(directory / "report.json", report)
            raise
        report["dataset"] = exported
        write_json(directory / "report.json", report)
    if exported:
        print({"dataset": str(directory / "dataset"), "records": exported["records"]}, flush=True)
    else:
        print(
            {
                "report": str(directory / "report.json"),
                "planned": sessions * tasks,
                "attempted": report["attempted_tasks"],
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
    parser.add_argument("--cross-initial-portal", action="store_true",
                        help="Use the frozen motor for a verified normal-input portal preparation; "
                             "exclude preparation from Laya results/training")
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
            cross_initial_portal=args.cross_initial_portal,
            **extras,
        )
    )


if __name__ == "__main__":
    main()
