"""Frozen Laya portal QA with ordinary saves, reset and current collision only."""
from __future__ import annotations

import argparse
import asyncio
import math
import time
import uuid
from pathlib import Path

from .app import bridge_secret
from .autonomy.controller import ContinuousController
from .autonomy.laya_ladder_policy import LayaLadderPolicy
from .autonomy.laya_policy import LayaWalkingPolicy
from .autonomy.models import AgentIntent, ObjectiveCompletion
from .autonomy.portal_task import ObservedPortalTask
from .bridge import Bridge, bind_bridge
from .config import Settings
from .g1 import ARTIFACTS, OwnedProcess, prepare_suite, settle, write_json
from .laya_curriculum import verify_collection_integrity
from .laya_data import digest
from .laya_motor import execute_frozen_task
from .laya_training import verify_base
from .laya_traversal import traversal_episode
from .qa_session import QaProcessPool
from .startup import enter_playable_save
from .surface_curriculum import warm_reference


def select_task(game, *, budget_s):
    # Only the currently visible floor exits compete. No scene ID, quest
    # coordinates, destination lookup or unseen doorway informs this choice.
    rejected = []
    for row in sorted(game.scene_exits, key=lambda e: math.dist(game.player.position, e.position)):
        try:
            return ObservedPortalTask.create(game, row, budget_s=budget_s), rejected
        except ValueError as exc:
            rejected.append({"exit_index": row.exit_index, "reason": str(exc)})
    raise ValueError(f"No currently supported observed walking portal: {rejected}")


def verified_result(task, *, consumed_actions, raw_buttons, objective_unchanged, run_updates):
    return bool(task.phase == "succeeded" and task.crossing_context is not None
                and task.verification_frames >= 3 and consumed_actions > 0 and raw_buttons == 0
                and objective_unchanged and run_updates == 0)


async def motor_episode(bridge, frozen, policy, directory, *, seconds, on_started=None):
    # Keep cold checkpoint/map work outside native reception and before the
    # task's origin sequence. No game command or label is generated here.
    controller = await asyncio.to_thread(ContinuousController, bridge, frozen / ARTIFACTS[0],
        training_enabled=False, route_graph_path=frozen / ARTIFACTS[1], room_map_path=frozen / ARTIFACTS[2])
    await bridge.next_state(bridge.state.seq, timeout=1.5)
    task, rejected = select_task(bridge.state, budget_s=seconds)
    objective = AgentIntent(objective="Obtain the Kokiri Sword", summary="Obtain the Kokiri Sword",
                            mode="explore", completion=ObjectiveCompletion(kind="equipment", name="Kokiri Sword"))
    controller.set_intent(objective)
    controller.executor.room_budget_s = seconds + 2
    report = await execute_frozen_task(controller, bridge, task, policy, directory, on_started=on_started)
    report["success"] &= verified_result(task, consumed_actions=report["consumed_actions"],
        raw_buttons=report["raw_button_actions"], objective_unchanged=report["objective_unchanged"],
        run_updates=report["run_updates"])
    report["rejected_proposals"] = rejected
    await asyncio.to_thread(write_json, directory / "motor.json", report)
    return report


async def evaluate(settings, executable, source_home, *, python, base, candidate,
                   episodes=3, seed=4105600, save_slot=2, seconds=30, ladder_candidate=None, post_descent_walks=0):
    if not 1 <= episodes <= 12 or not 5 <= seconds <= 30:
        raise ValueError("Use 1..12 episodes and 5..30 seconds per walking portal")
    if not 0 <= post_descent_walks <= 5 or post_descent_walks and not ladder_candidate:
        raise ValueError("Use 0..5 observed post-descent walks with an explicit attached candidate")
    verify_base(base)
    episode_budget = 75 + seconds + (20 if ladder_candidate else 0) + 12 * post_descent_walks
    directory, manifest = prepare_suite(settings, executable, source_home, episodes=episodes,
        seed=seed, save_slot=save_slot, seconds=episode_budget)
    expected = {name: digest(candidate / name) for name in ("candidate.json", "heads.safetensors")}
    ladder_expected = ({name: digest(ladder_candidate / name) for name in ("candidate.json", "heads.safetensors")}
                       if ladder_candidate else None)
    manifest.update(runner_version="laya-observed-portal-eval-v3", candidate_sha256=expected,
                    native_sessions=1, reset_episodes=episodes, promotion="disabled",
                    scope="current_observed_dry_floor_exit_with_partial_collision_approach",
                    episode_grouping="native_instance", decision_budget_ms=100,
                    training_updates=0, reference_blend=0)
    manifest.update(ladder_candidate_sha256=ladder_expected,
        scope="current_observed_portal_then_dry_approach_and_native_attachment" if ladder_candidate else manifest["scope"],
        handoff_budget="original_descent_deadline_preserved" if ladder_candidate else None)
    manifest.update(post_descent_walks=post_descent_walks,
                    collision_handoff="fresh_full_same_context" if post_descent_walks else None)
    manifest["warmup"] = await warm_reference(directory / "frozen")
    write_json(directory / "manifest.json", manifest)
    log = settings.data_dir / "laya" / f"portal-eval-{uuid.uuid4().hex[:12]}.log"
    policy = await LayaWalkingPolicy.start(python, base, candidate, log)
    records, instance, ladder_policy = [], None, None
    report = {"source": "soh", "planned_tasks": episodes, "records": records,
              "provider_calls": 0, "training_updates": 0, "promotion": "disabled"}

    def verify_all():
        verify_collection_integrity(directory, manifest, candidate, expected)
        if ladder_expected and not all(digest(ladder_candidate / name) == sha for name, sha in ladder_expected.items()):
            raise RuntimeError("Frozen attached candidate changed")

    try:
        if ladder_candidate:
            ladder_policy = await LayaLadderPolicy.start(python, base, ladder_candidate, directory / "ladder-worker.log")
        async with QaProcessPool(manifest["executable"], directory / "seed-home", bridge_secret(settings),
            reuse=True, bridge_factory=Bridge, process_factory=OwnedProcess, bind=bind_bridge) as pool:
            for index in range(episodes):
                home = directory / f"episode-{index + 1}"
                row = {"index": index + 1, "source": "soh", "attempted": False, "success": False}
                records.append(row)
                try:
                    async with asyncio.timeout(episode_budget):
                        bridge, process, reset = await pool.acquire(home)
                        deadline = time.monotonic() + 15
                        while not bridge.connected and time.monotonic() < deadline:
                            if process.child.poll() is not None:
                                raise RuntimeError("Owned native process exited")
                            await asyncio.sleep(.05)
                        if not bridge.connected or not bridge.realtime or bridge.state.protocol != 3:
                            raise RuntimeError("Real native bridge unavailable")
                        instance = instance or bridge.state.instance_id
                        prior = {e.id for e in bridge.state.events if e.kind == "save_loaded"}
                        startup = await enter_playable_save(bridge, save_slot - 1, budget_s=60,
                            failure_report=lambda r: write_json(home / "startup.json", r))
                        write_json(home / "startup.json", startup)
                        if startup["status"] != "loaded":
                            raise RuntimeError(startup.get("reason", "normal_save_not_loaded"))
                        await settle(bridge, deadline=time.monotonic() + 8)
                        if bridge.state.instance_id != instance:
                            raise RuntimeError("Native instance changed")
                        new_loads = [e.id for e in bridge.state.events if e.kind == "save_loaded"
                                     and e.id not in prior and e.detail == str(save_slot - 1)]
                        if reset and not new_loads:
                            raise RuntimeError("Reset episode lacks a fresh normal save load")
                        row.update(instance_id=instance, pid=process.child.pid, reset=reset,
                                   fresh_load_events=new_loads, initial_scene=bridge.state.scene)
                        options = {"seconds": seconds, "on_started": lambda: row.update(attempted=True)}
                        if ladder_policy:
                            result = await traversal_episode(bridge, directory / "frozen", policy, ladder_policy,
                                home / "motor", select_portal=select_task, post_descent_walks=post_descent_walks,
                                seed=seed+index, **options)
                        else:
                            result = await motor_episode(bridge, directory / "frozen", policy, home / "motor", **options)
                        row.update(success=result["success"],
                                   reason=result.get("reason", (result["task"] or {}).get("failure")),
                                   consumed_actions=result["consumed_actions"], run_updates=result["run_updates"],
                                   objective_unchanged=result["objective_unchanged"],
                                   raw_button_actions=result["raw_button_actions"],
                                   initial_context=result.get("initial_context", (result["task"] or {}).get("context")),
                                   final_context=result.get("final_context", (result["task"] or {}).get("crossing_context")))
                        if ladder_policy:
                            row.update(attached_ladder=result["attached_ladder"], landing_verified=result["landing_verified"],
                                post_descent_walks_successful=result["post_descent_walks_successful"],
                                stages=[{k:s[k] for k in ("name", "success", "consumed_actions", "motor_failure")}
                                        for s in result["stages"]])
                except (RuntimeError, ValueError, OSError, TimeoutError) as exc:
                    row["reason"] = f"{type(exc).__name__}:{str(exc)[:180]}"
                    # Preserve the actual selection/bridge state, including
                    # unattempted episodes. They never disappear from the denominator.
                    if pool.bridge and pool.bridge.state:
                        await asyncio.to_thread(write_json, home / "failure-state.json", pool.bridge.state.model_dump())
                finally:
                    await pool.finish_episode()
                    policy.invalidate()
                    if ladder_policy:
                        ladder_policy.invalidate()
                    await asyncio.to_thread(write_json, directory / "report.json", report)
                    print({"suite": directory.name, **row}, flush=True)
                await asyncio.to_thread(verify_all)
            report["native_processes"] = pool.launches
    finally:
        try:
            if ladder_policy:
                await ladder_policy.close()
        finally:
            await policy.close()
        await asyncio.to_thread(verify_all)
        verify_base(base)
        report.update(attempted_tasks=sum(bool(r["attempted"]) for r in records),
                      successful_tasks=sum(bool(r["success"]) for r in records),
                      integrity_after_shutdown=True, worker=policy.metrics,
                      ladder_worker=ladder_policy.metrics if ladder_policy else None)
        write_json(directory / "report.json", report)
    return directory / "report.json"


def main():
    parser = argparse.ArgumentParser(description="Frozen Laya portal pilot; current observations and zero providers")
    parser.add_argument("executable", type=Path)
    for name in ("source-home", "python", "base", "candidate"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=30)
    parser.add_argument("--seed", type=int, default=4105600)
    parser.add_argument("--save-slot", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--ladder-candidate", type=Path,
                        help="Also evaluate current observed descent and attached handoff under the same objective")
    parser.add_argument("--post-descent-walks", type=int, default=0,
                        help="0..5 current-collision walks after released dry landing; requires --ladder-candidate")
    args = vars(parser.parse_args())
    print(asyncio.run(evaluate(Settings(), **args)), flush=True)


if __name__ == "__main__":
    main()
