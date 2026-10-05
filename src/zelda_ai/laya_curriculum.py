"""Real walking collection or frozen evaluation in continuous native sessions."""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import time
import uuid
from pathlib import Path

from .app import bridge_secret
from .autonomy.attached_descent import AttachedDescentTask
from .autonomy.descent_task import ObservedDescentTask
from .autonomy.ladder_task import LadderDescentTask
from .autonomy.laya_ladder_policy import encode_ladder
from .autonomy.laya_policy import LayaWalkingPolicy
from .autonomy.local_tasks import LocalTask
from .autonomy.navigation import observed_local_path
from .bridge import Bridge, bind_bridge
from .config import Settings
from .g1 import OwnedProcess, _motor_episode, context, portal_crossed, prepare_suite, settle, write_json
from .laya_data import LADDER_PROFILE, digest, export_surfaces
from .qa_reset import QA_RESET_BUTTONS
from .qa_session import QaProcessPool, evaluation_reuses_process
from .startup import enter_playable_save
from .surface_curriculum import run_task, warm_reference


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
              "kind": "portal",
              "controller": "frozen_v3_motor_qa_preparation",
              "candidate_actions": 0, "included_in_laya_training": False}
    write_json(directory / "report.json", report)
    return report


async def prepare_observed_descent(bridge, frozen, directory, *, attached_policy=None, collect_attached_labels=False):
    """Try a current landing proposal; never infer a hidden route or button."""
    directory.mkdir(parents=True, exist_ok=True)
    if attached_policy is not None and collect_attached_labels:
        raise ValueError("Candidate evaluation cannot collect reference labels")
    initial_context = context(bridge.state)
    report = {"success": False, "source": "soh", "provider_calls": 0, "kind": "descent",
              "controller": "observed_descent_reference_qa_preparation",
              "candidate_actions": 0, "included_in_laya_training": False,
              "reason": "no_supported_observed_landing", "consumed_actions": 0, "attempts": [],
              "initial": bridge.state.model_dump(), "rejected_proposals": []}
    if attached_policy is not None:
        report["controller"] = "walking_reference_preparation_then_attached_candidate"
    attempted = []
    for index in range(3):
        # The prior attempt may reveal a different opening. Only reconsider
        # CURRENT native proposals, with a bounded local cooldown for failed
        # landing regions. This list never becomes a learned route/edge.
        rows = sorted((r for r in bridge.state.traversal_affordances if r.kind == "ledge_down"),
                      key=lambda r: r.distance)
        task = None
        for row in rows:
            if any(math.hypot(row.target_position[0] - p[0], row.target_position[2] - p[2]) <= 140
                   and abs(row.target_position[1] - p[1]) <= 24 for p in attempted):
                continue
            try:
                task = ObservedDescentTask.create(bridge.state, row)
            except ValueError as exc:
                report["rejected_proposals"].append({"attempt": index + 1, "observation_seq": bridge.state.seq,
                    "target": row.target_position, "reason": str(exc)[:160]})
                continue
            break
        if task is None or context(bridge.state) != initial_context:
            break
        attempted.append(task.target)
        result, _ = await run_task(bridge, frozen, task, directory / f"attempt-{index + 1}" / "motor")
        walking_task = task
        stages = [result]
        if result["task"]["failure"] == "unsupported_locomotor_mode" and bridge.state.player.climbing_ladder:
            try:
                task_type = LadderDescentTask if attached_policy is not None else AttachedDescentTask
                attached = task_type.continue_from(bridge.state, task)
            except ValueError as exc:
                report["rejected_proposals"].append({"attempt": index + 1, "observation_seq": bridge.state.seq,
                    "target": task.target, "reason": str(exc)[:160]})
            else:
                options = ({"policy": attached_policy, "demonstrations": collect_attached_labels,
                            "feature_encoder": encode_ladder, "feature_profile": LADDER_PROFILE}
                           if attached_policy is not None or collect_attached_labels else {})
                result, _ = await run_task(bridge, frozen, attached,
                    directory / f"attempt-{index + 1}" / "attached-motor", **options)
                if attached_policy is not None:
                    report["candidate_actions"] += result["consumed_actions"]
                task = attached
                stages.append(result)
        checks = (
            (context(bridge.state) == initial_context, "context_changed"),
            (task.landing_verified(bridge.state), "landing_not_verified"),
            (all(stage.get("source") == "soh" for stage in stages), "invalid_source"),
            (result["consumed_actions"] > 0, "missing_consumed_input"),
            (all(stage["objective_unchanged"] for stage in stages), "objective_changed"),
            (all(stage["run_updates"] == 0 for stage in stages), "evaluation_trained"),
            (all(stage.get("reference_blend") == 0 for stage in stages), "reference_blend"),
            (all(stage.get("provider_calls") == 0 for stage in stages), "provider_calls"),
        )
        success = bool(result["success"] and all(valid for valid, _ in checks))
        reason = ("observed_verified_landing" if success else result["task"]["failure"]
                  or next((reason for valid, reason in checks if not valid), "unverified_descent"))
        attempt = dict(
            success=success, reason=reason, consumed_actions=sum(stage["consumed_actions"] for stage in stages),
            objective_unchanged=all(stage["objective_unchanged"] for stage in stages),
            run_updates=sum(stage["run_updates"] for stage in stages),
            provider_calls=(sum(stage["provider_calls"] for stage in stages)
                if all(type(stage.get("provider_calls")) is int and stage["provider_calls"] >= 0
                       for stage in stages) else None),
            reference_blend=(0 if all(stage.get("reference_blend") == 0 for stage in stages) else None),
            initial_position=stages[0]["initial"]["player"]["position"],
            final_position=result["final"]["player"]["position"],
            landing_target=task.target, upper_floor_approach=walking_task.approach,
            task_version=task.version,
            stages=[{"task_version": stage["task"]["version"], "success": stage["success"],
                     "failure": stage["task"]["failure"], "consumed_actions": stage["consumed_actions"]}
                    for stage in stages],
        )
        report["attempts"].append(attempt)
        report["consumed_actions"] += attempt["consumed_actions"]
        calls = [a["provider_calls"] for a in report["attempts"]]
        report["provider_calls"] = sum(calls) if all(type(c) is int and c >= 0 for c in calls) else None
        report.update(success=attempt["success"], reason=attempt["reason"])
        write_json(directory / "report.json", report)
        if attempt["success"]:
            break
    report["final"] = bridge.state.model_dump()
    write_json(directory / "report.json", report)
    return report


def verify_collection_integrity(directory, manifest, candidate, expected_candidate):
    if not all(digest(Path(p)) == h for p, h in manifest["pinned_sha256"].items()):
        raise RuntimeError("Pinned source or original artifacts changed during collection")
    if not all(digest(directory / "frozen" / p) == h for p, h in manifest["frozen_sha256"].items()):
        raise RuntimeError("Frozen motor artifacts changed")
    if not all(digest(directory / "seed-home" / p) == h for p, h in manifest.get("fixture_sha256", {}).items()):
        raise RuntimeError("Seed fixture changed")
    if expected_candidate and not all(
        digest(candidate / name) == h for name, h in expected_candidate.items()
    ):
        raise RuntimeError("Frozen Laya candidate changed")


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
    descend_observed_ledge=False,
    reuse_process=None,
):
    if not 3 <= sessions <= 12 or not 3 <= tasks <= 40:
        raise ValueError("Use 3..12 native sessions and 3..40 tasks per session")
    if (policy is None) != (candidate is None):
        raise ValueError("Frozen candidate evaluation requires both policy and checkpoint provenance")
    if descend_observed_ledge and not cross_initial_portal:
        raise ValueError("Observed descent preparation requires cross_initial_portal")
    reuse_process = evaluation_reuses_process(evaluating=policy is not None, requested=reuse_process)
    session_budget = 60 + tasks * 14 + (120 if cross_initial_portal else 0) + (75 if descend_observed_ledge else 0)
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
        runner_version="laya-reference-walking-v4" if policy is None else "laya-continuous-walking-eval-v4",
        native_sessions=1 if reuse_process else sessions,
        actual_native_sessions=0,
        reset_episodes=sessions if reuse_process else 0,
        process_reuse="normal_soh_controller_reset" if reuse_process else "independent_native_groups",
        qa_reset_buttons=QA_RESET_BUTTONS if reuse_process else None,
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
        observed_descent_preparation=descend_observed_ledge,
    )
    expected_candidate = (
        {name: digest(candidate / name) for name in ("candidate.json", "heads.safetensors")}
        if candidate
        else None
    )
    manifest["candidate_sha256"] = expected_candidate
    write_json(directory / "manifest.json", manifest)
    manifest["reference_warmup_before_native_launch"] = await warm_reference(directory / "frozen")
    write_json(directory / "manifest.json", manifest)
    records, preparations = [], []
    async with QaProcessPool(manifest["executable"], directory / "seed-home", bridge_secret(settings),
                             reuse=reuse_process, bridge_factory=Bridge, process_factory=OwnedProcess,
                             bind=bind_bridge) as pool:
        for session in range(sessions):
            home = directory / f"native-session-{session + 1}"
            bridge, process = pool.bridge, pool.process
            rng = random.Random(seed + session)
            if policy is not None:
                policy.invalidate()
            stage = "normal_reset_or_launch"
            try:
                async with asyncio.timeout(session_budget):
                    bridge, process, reset = await pool.acquire(home)
                    manifest["actual_native_sessions"] = pool.launches
                    write_json(directory / "manifest.json", manifest)
                    stage = "await_native_bridge"
                    deadline = time.monotonic() + 15
                    while not bridge.connected and time.monotonic() < deadline:
                        if process.child.poll() is not None:
                            raise RuntimeError("Native process exited")
                        await asyncio.sleep(0.05)
                    if not bridge.connected or not bridge.realtime or bridge.state.protocol != 3:
                        raise RuntimeError("Real native input-receipt bridge unavailable")
                    stage = "normal_save_startup"
                    prior_loads = {e.id for e in bridge.state.events if e.kind == "save_loaded"}
                    startup = await enter_playable_save(bridge, save_slot - 1, budget_s=60,
                        failure_report=lambda report: write_json(home / "startup.json", report))
                    write_json(home / "startup.json", startup)
                    if startup["status"] != "loaded":
                        raise RuntimeError(startup.get("reason", "Native save not loaded"))
                    stage = "settle_after_startup"
                    await settle(bridge, deadline=time.monotonic() + 8)
                    if reset is not None:
                        fresh_loads = [e.model_dump() for e in bridge.state.events
                                       if e.kind == "save_loaded" and e.id not in prior_loads
                                       and e.detail == str(save_slot - 1)]
                        write_json(home / "reload.json", {"events": fresh_loads,
                                   "same_native_instance": bridge.state.instance_id})
                        if not fresh_loads:
                            raise RuntimeError("Reset episode lacks a new normal save-load event")
                    preparation = None
                    if cross_initial_portal:
                        stage = "initial_physical_preparation"
                        preparation = await prepare_portal_start(
                            bridge, directory / "frozen", home / "portal-preparation",
                        )
                        preparations.append({"session": session + 1, **preparation})
                        if preparation["success"] and descend_observed_ledge:
                            preparation = await prepare_observed_descent(
                                bridge, directory / "frozen", home / "descent-preparation",
                            )
                            preparations.append({"session": session + 1, **preparation})
                        write_json(directory / "preparations.json", preparations)
                        if policy is not None:
                            policy.invalidate()
                    stage = "walking_tasks"
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
                                kind = preparation.get("kind", "portal")
                                raise RuntimeError(f"initial_{kind}_preparation_failed")
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
            except (OSError, ValueError, RuntimeError, TimeoutError) as exc:
                game = bridge.state if bridge else None
                # Persist the pre-input phase too, before release/owned shutdown.
                # Missing state/receipts stay unknown; never retain raw tokens or
                # infer an unconsumed command from absent feedback.
                write_json(home / "session-failure.json", {
                    "status": "failed", "phase": stage, "error_kind": type(exc).__name__,
                    "reason": str(exc)[:160], "bridge": bridge.telemetry() if bridge else None,
                    "last_observation": ({
                        "seq": game.seq, "protocol": game.protocol, "capabilities": game.capabilities,
                        "scene_epoch": game.scene_epoch, "context_epoch": game.context_epoch,
                        "scene": game.scene, "room": game.room, "in_game": game.in_game,
                    } if game else None),
                    "native_process_exit_code": process.child.poll() if process else None,
                    "command_seq_at_failure": bridge.command_seq if bridge else None,
                })
                raise
            finally:
                await pool.finish_episode()
                if policy is not None:
                    policy.invalidate()
            await asyncio.to_thread(verify_collection_integrity, directory, manifest, candidate, expected_candidate)
    await asyncio.to_thread(verify_collection_integrity, directory, manifest, candidate, expected_candidate)
    report = {
        "source": "soh",
        "actual_native_sessions": pool.launches,
        "process_reuse": manifest["process_reuse"],
        "integrity_after_shutdown": True,
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
    parser.add_argument("--fresh-processes", action="store_true",
                        help="Use independent native instances for frozen evaluation instead of normal resets")
    parser.add_argument("--cross-initial-portal", action="store_true",
                        help="Use the frozen motor for a verified normal-input portal preparation; "
                             "exclude preparation from Laya results/training")
    parser.add_argument("--descend-observed-ledge", action="store_true",
                        help="After portal preparation, try a bounded current-observed landing "
                             "with reference control; excludes descent from Laya results/training")
    args = parser.parse_args()
    if args.candidate and (not args.base or not args.python):
        parser.error("--candidate requires --base and --python")
    if args.descend_observed_ledge and not args.cross_initial_portal:
        parser.error("--descend-observed-ledge requires --cross-initial-portal")
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
            descend_observed_ledge=args.descend_observed_ledge,
            reuse_process=False if args.fresh_processes else None,
            **extras,
        )
    )


if __name__ == "__main__":
    main()
