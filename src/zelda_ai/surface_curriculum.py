"""Isolated physical curriculum and before/after evaluation; zero providers.

Only the QA supervisor loads copied native saves. Task parameters and paths
come from the current collision observations, never a walkthrough or savestate.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import shutil
import time
from collections import Counter
from pathlib import Path

from .app import bridge_secret
from .autonomy.controller import ContinuousController
from .autonomy.imitation import VERSION, SurfacePolicy, encode_surface
from .autonomy.local_tasks import LocalTask, walking_surface_supported
from .autonomy.models import AgentIntent, ObjectiveCompletion
from .bridge import Bridge, bind_bridge
from .g1 import (
    ARTIFACTS,
    OwnedProcess,
    _motor_episode,
    prepare_suite,
    prepare_variation,
    settle,
    variation_target,
    write_json,
)
from .qualification import sha256
from .startup import enter_playable_save

FAMILIES = ("surface_up", "surface_down", "recovery_cell")


def select_surface(game, direction, variant):
    rows = [row for row in game.traversal_affordances
            if row.kind == f"stairs_or_slope_{direction}" and row.distance <= 140
            and 4 < abs(row.height_delta) <= 30 and walking_surface_supported(game, row)]
    rows.sort(key=lambda row: row.target_position)
    if not rows:
        raise ValueError("No currently observed short walking surface")
    return rows[variant % len(rows)]


async def run_task(bridge, frozen, task, directory, *, policy=None, demonstrations=False,
                   feature_encoder=encode_surface, feature_profile=None):
    directory.mkdir(parents=True, exist_ok=True)
    controller = ContinuousController(bridge, frozen / ARTIFACTS[0], training_enabled=False,
        route_graph_path=frozen / ARTIFACTS[1], room_map_path=frozen / ARTIFACTS[2])
    initial = bridge.state.model_copy(deep=True)
    intent = AgentIntent(objective="Leave the currently observed room through a physical portal",
        summary="Leave the currently observed room through a physical portal", mode="explore",
        completion=ObjectiveCompletion(kind="leave_scene_room", scene=initial.scene, room=initial.room))
    controller.set_intent(intent)
    controller.start_local_task(task, stick_policy=policy)
    requested, frames = {}, []

    def publish():
        game = bridge.state
        seq = bridge.command_seq
        active = controller.local_task
        if active and active.phase in {"prepare", "execute"} and controller.last_setpoint.reason == "local_task":
            requested[seq] = {"command_seq": seq, "observation_seq": game.seq,
                "scene_epoch": game.scene_epoch, "features": feature_encoder(game, active),
                "action": [controller.last_setpoint.stick_x / 80, controller.last_setpoint.stick_y / 80],
                "buttons": controller.last_setpoint.buttons, "source": "soh",
                "controller": "reference" if policy is None else "candidate", "reference_blend": 0,
                "reference_calibrated": getattr(active, "learned_stick", None) is not None,
                "ladder_attached": game.player.climbing_ladder}
        if controller.executor.trace:
            frames.append({"state": game.model_dump(), "motor": controller.executor.trace[-1]})

    await controller.run(lambda: controller.local_task is not None and bridge.connected
        and time.monotonic() < task.deadline + .5, publish)
    await settle(bridge, deadline=time.monotonic() + 3)
    receipts = {seq: receipt.model_dump() for seq, receipt in bridge.receipts.items()
                if seq in requested and receipt.first_tick > 0}
    successful = bool(controller.last_local_task and controller.last_local_task["phase"] == "succeeded")
    rows = [{**row, "first_tick": receipts[seq]["first_tick"], "successful_episode": successful}
            for seq, row in requested.items() if seq in receipts]
    report = {"source": "soh", "success": successful, "task": controller.last_local_task,
        "feature_profile": feature_profile,
        "initial": initial.model_dump(), "final": bridge.state.model_dump(), "frames": frames,
        "receipts": list(receipts.values()), "demonstrations": rows if demonstrations else [],
        "actions": rows, "intent": intent.model_dump(), "final_intent": controller.intent.model_dump(),
        "requested_actions": len(requested), "consumed_actions": len(receipts),
        "provider_calls": 0, "run_updates": controller.policy.updates - controller.starting_updates,
        "reference_blend": 0, "objective_unchanged": controller.intent == intent}
    write_json(directory / "task.json", report)
    return report, rows if demonstrations and successful else []


def aggregate(records, family, stage):
    rows = [r for r in records if r["family"] == family and r["stage"] == stage]
    successes = sum(bool(r.get("success") and r.get("consumed_actions", 0) > 0
                         and r.get("objective_unchanged") and r.get("run_updates") == 0
                         and r.get("reference_blend") == 0 and r.get("artifacts_unchanged")
                         and r.get("source") == "soh") for r in rows)
    unique = len({r.get("index") for r in rows}) == len(rows)
    return {"attempted": len(rows), "successful": successes,
            "g2_qualified": len(rows) == 100 and successes >= 95 and unique
              and not any(r.get("critical_control_bug", False) for r in rows),
            "failure_reasons": dict(Counter(r.get("reason", "invalid_evidence")
                for r in rows if not r.get("success"))),
            "initial_pose_bins": len({tuple(round(v / 10) for v in r["initial_position"])
                                      for r in rows if r.get("initial_position")})}


async def surface_curriculum(settings, executable, source_home, *, save_slot=2,
                             demonstration_trials=24, evaluation_trials=12, seed=4102026,
                             retention_trials=10):
    if (not 3 <= demonstration_trials <= 100 or not 3 <= evaluation_trials <= 300
            or evaluation_trials % 3 or not 0 <= retention_trials <= 100):
        raise ValueError("Use 3..100 demonstrations, evaluation divisible by 3 (3..300), retention 0..100")
    directory, manifest = prepare_suite(settings, executable, source_home,
        episodes=evaluation_trials, seed=seed, save_slot=save_slot, seconds=120)
    manifest.update(runner_version="surface-curriculum-v1", curriculum_version=VERSION,
                    demonstration_trials=demonstration_trials, evaluation_trials=evaluation_trials,
                    retention_trials=retention_trials,
                    evaluation_scope="short_observed_walking_surfaces_and_cell_after_input_withholding",
                    promotion="disabled_pending_complete_skill_and_retention_gates",
                    budget_scope="post_launch_startup_setup_motor_settle; shutdown_and_reporting_excluded")
    write_json(directory / "manifest.json", manifest)
    frozen, records, demos = directory / "frozen", [], []
    token, rng = bridge_secret(settings), random.Random(seed)
    variants = [rng.randrange(4) + 1 for _ in range(max(demonstration_trials, evaluation_trials, retention_trials))]
    initial_policy = SurfacePolicy(seed=seed)
    candidate_path = directory / "surface-candidate.pt"
    candidate = None
    unchanged = True

    async def trial(stage, index, family, policy):
        nonlocal unchanged
        trial_dir = directory / stage / f"trial-{index + 1:03}"
        trial_dir.mkdir(parents=True)
        home = trial_dir / "native-home"
        shutil.copytree(directory / "seed-home", home)
        bridge = Bridge(token, allow_simulator=False)
        await bind_bridge(bridge, 0)
        environment = dict(os.environ)
        environment.update(ZELDA_BRIDGE_TOKEN=token,
                           ZELDA_BRIDGE_PORT=str(bridge.transport.get_extra_info("sockname")[1]))
        process = OwnedProcess(Path(manifest["executable"]), home, environment)
        started = time.monotonic()
        record = {"stage": stage, "index": index + 1, "family": family, "source": "soh",
                  "success": False, "reason": "not_started", "provider_calls": 0,
                  "reference_blend": 0, "run_updates": 0, "objective_unchanged": False}
        collected = []
        try:
            async with asyncio.timeout(120):
                while not bridge.connected and time.monotonic() - started < 15:
                    if process.child.poll() is not None:
                        raise RuntimeError("owned_process_exited")
                    await asyncio.sleep(.05)
                if not bridge.connected or not bridge.realtime or bridge.state.protocol != 3:
                    raise RuntimeError("real_native_bridge_unavailable")
                startup = await enter_playable_save(bridge, save_slot - 1, budget_s=60)
                write_json(trial_dir / "startup.json", startup)
                if startup["status"] != "loaded":
                    raise RuntimeError(startup.get("reason", "save_not_loaded"))
                await settle(bridge, deadline=time.monotonic() + 8)
                # Broad setup directions are empirical, not guaranteed exact poses.
                setup = await prepare_variation(bridge, variants[index], deadline=time.monotonic() + 8)
                write_json(trial_dir / "setup.json", setup)
                if not setup["valid"]:
                    raise RuntimeError("invalid_physical_setup")
                if family == "portal_retention":
                    record.update(await _motor_episode(bridge, frozen, trial_dir, deadline=time.monotonic() + 105))
                    record["success"] = record["settled_crossing"] and record["consumed_commands"] > 0
                    record["consumed_actions"] = record["consumed_commands"]
                elif family in {"surface_up", "surface_down"}:
                    # Relocalize from current geometry: an elevated start may
                    # already expose descent and need descent before a new ascent.
                    direction = "up" if family == "surface_up" else "down"
                    row = None
                    for scan in range(4):
                        try:
                            row = select_surface(bridge.state, direction, variants[index])
                            break
                        except ValueError:
                            try:
                                opposite = select_surface(bridge.state, "down" if direction == "up" else "up", variants[index])
                            except ValueError:
                                setup2 = await prepare_variation(bridge, 3, deadline=time.monotonic() + 8)
                                write_json(trial_dir / f"surface-setup-{scan}.json", setup2)
                            else:
                                preparation, _ = await run_task(bridge, frozen, LocalTask.traversal(bridge.state, opposite),
                                                               trial_dir / f"relocalize-{scan}")
                                if not preparation["success"]:
                                    raise RuntimeError("opposite_surface_setup_failed")
                    if row is None:
                        row = select_surface(bridge.state, direction, variants[index])
                    task = LocalTask.traversal(bridge.state, row)
                else:
                    # Adverse QA: consumed neutral input causes a bounded failed
                    # attempt; the next task must recover without strategic churn.
                    target = variation_target(bridge.state, variants[index])
                    if target is None:
                        target = next((variation_target(bridge.state, v) for v in range(1, 5)
                                       if variation_target(bridge.state, v) is not None), None)
                    if target is None:
                        raise ValueError("No currently observed reachable recovery cell")
                    failed, _ = await run_task(bridge, frozen, LocalTask.observed_cell(bridge.state, target),
                        trial_dir / "withheld-input", policy=lambda *args: (0, 0))
                    if failed["success"] or failed["task"]["failure"] != "no_geometric_progress":
                        raise RuntimeError("adverse_setup_did_not_prove_stall")
                    task = LocalTask.observed_cell(bridge.state, target)
                if family != "portal_retention":
                    result, collected = await run_task(bridge, frozen, task, trial_dir / "motor",
                                                       policy=policy, demonstrations=stage == "demonstrations")
                    record.update({key: result[key] for key in ("success", "consumed_actions",
                        "objective_unchanged", "run_updates", "reference_blend")})
                    record.update(reason="success" if result["success"] else result["task"]["failure"],
                                  initial_position=result["initial"]["player"]["position"],
                                  final_position=result["final"]["player"]["position"],
                                  native_build=bridge.state.bridge_build,
                                  upstream_revision=bridge.state.upstream_revision)
        except (OSError, RuntimeError, ValueError) as exc:
            record["reason"] = f"{type(exc).__name__}:{str(exc)[:180]}"
        finally:
            bridge.close()
            await process.close()
            record["elapsed_s"] = round(time.monotonic() - started, 3)
            unchanged = unchanged and all(sha256(Path(path)) == expected
                for path, expected in manifest["pinned_sha256"].items())
            unchanged = unchanged and all(sha256(directory / "seed-home" / path) == expected
                for path, expected in manifest["fixture_sha256"].items())
            unchanged = unchanged and all(sha256(frozen / name) == expected
                for name, expected in manifest["frozen_sha256"].items())
            record["artifacts_unchanged"] = unchanged
            write_json(trial_dir / "report.json", record)
            records.append(record)
            write_json(directory / "progress.json", {"source": "soh", "records": records})
            print(json.dumps({"directory": str(directory), **record}), flush=True)
        if not unchanged:
            raise RuntimeError("Pinned artifact changed; curriculum stopped")
        return record, collected

    for index in range(demonstration_trials):
        _, rows = await trial("demonstrations", index, FAMILIES[index % 3], None)
        demos.extend(rows)
    dataset = directory / "demonstrations.json"
    write_json(dataset, {"version": VERSION, "source": "soh", "rows": demos})
    initial_policy.save(directory / "surface-before.pt", dataset_sha256=None)
    if not all(aggregate(records, family, "demonstrations")["successful"] for family in FAMILIES):
        raise RuntimeError(f"All three families need real demonstrations; inspect {directory}")
    candidate = SurfacePolicy(seed=seed)
    training = candidate.fit(demos)
    candidate.save(candidate_path, dataset_sha256=sha256(dataset))
    checkpoints = {name: sha256(directory / name) for name in ("surface-before.pt", "surface-candidate.pt")}
    # Paired families/variants, disjoint from the demonstration seed schedule.
    evaluation_seed = seed + 100003
    rng = random.Random(evaluation_seed)
    variants = [rng.randrange(4) + 1 for _ in range(max(evaluation_trials, retention_trials))]
    for stage, policy in (("before", initial_policy), ("after", SurfacePolicy.load(candidate_path))):
        for index in range(evaluation_trials):
            await trial(stage, index, FAMILIES[index % 3], policy)
    for index in range(retention_trials):
        await trial("retention", index, "portal_retention", None)
    evaluation = {family: {stage: aggregate(records, family, stage) for stage in ("before", "after")}
                  for family in FAMILIES}
    retained = aggregate(records, "portal_retention", "retention")
    unchanged = unchanged and all(sha256(directory / name) == expected for name, expected in checkpoints.items())
    report = {"schema_version": 1, "source": "soh", "suite_id": directory.name,
        "provider_calls": 0, "evaluation_training_updates": 0, "promotion": "not_promoted",
        "evaluation_seed": evaluation_seed, "training": training, "families": evaluation,
        "portal_retention": retained, "artifacts_unchanged": unchanged,
        "candidate_sha256": checkpoints["surface-candidate.pt"],
        "before_sha256": checkpoints["surface-before.pt"], "dataset_sha256": sha256(dataset),
        "g2_partial": unchanged and all(evaluation[f]["after"]["g2_qualified"] for f in FAMILIES),
        "g2_complete": False, "records": records}
    write_json(directory / "report.json", report)
    print(json.dumps({"report": str(directory / "report.json"), "families": evaluation,
                      "portal_retention": retained, "promotion": "not_promoted"}), flush=True)
    return report
