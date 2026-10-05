"""Only explicitly collected successful calibrated reference descents are labels."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from .laya_data import (
    HISTORY_LIMIT,
    LADDER_PROFILE,
    STICK_BINS,
    bounded_state,
    digest,
    load_dataset,
    quantize,
    validate_record,
)


def export_ladders(suite: Path, output: Path):
    if output.exists():
        raise FileExistsError("Export into a new ladder dataset directory")
    episodes, skipped, hashes = {}, Counter(), {}
    for path in sorted(suite.glob("native-session-*/descent-preparation/attempt-*/attached-motor/task.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if not task.get("success"):
            skipped["failed_task"] += 1
            continue
        if (task.get("source") != "soh" or task.get("provider_calls") != 0 or task.get("run_updates") != 0
                or not task.get("objective_unchanged") or task.get("reference_blend") != 0
                or task.get("feature_profile") != LADDER_PROFILE
                or task["task"]["version"] != "observed-attached-descent-reference-v1"
                or task["task"]["verification_frames"] < 3):
            raise ValueError("Invalid attached reference provenance")
        initial, final = task["initial"], task["final"]
        instance = initial.get("instance_id")
        if (not instance or instance != final.get("instance_id") or initial["scene_epoch"] != final["scene_epoch"]
                or any(initial[key] != final[key] for key in ("scene", "room", "mirrored_world"))
                or initial["source"] != "soh" or final["source"] != "soh"
                or not initial["player"]["climbing_ladder"] or final["player"]["climbing_ladder"]
                or final["player"]["hanging_ledge"] or final["player"]["climbing_ledge"]
                or not final["player"]["bg_check_flags"] & 1 or final["player"]["speed_xz"] >= .1
                or abs(final["player"]["position"][1] - task["task"]["target"][1]) > 4
                or abs(final["player"]["position"][1] - final["player"]["floor_height"]) > 4
                or initial["player"]["age"] != final["player"]["age"]):
            raise ValueError("Attached labels need one unchanged native instance and a released landing")
        episode = hashlib.sha256(instance.encode("utf-8")).hexdigest()
        if episode in episodes:
            raise ValueError("One successful attached trial per native instance is required")
        receipts = {row["seq"]: row for row in task["receipts"]}
        history, rows, last = [], [], -1
        for action in task["demonstrations"]:
            if not action.get("reference_calibrated") or not action.get("ladder_attached"):
                skipped["uncalibrated_or_unattached_action"] += 1
                continue
            seq = action["observation_seq"]
            if seq < last or receipts.get(action["command_seq"], {}).get("first_tick", 0) != action["first_tick"]:
                raise ValueError("Missing consumed receipt or noncausal ladder action")
            history = (history + [action["features"]])[-HISTORY_LIMIT:]
            last = seq
            stick = [v * 80 for v in action["action"]]
            row = {**{k: action[k] for k in ("source", "controller", "first_tick", "buttons",
                    "successful_episode", "reference_blend", "command_seq", "observation_seq",
                    "reference_calibrated", "ladder_attached")},
                "profile": LADDER_PROFILE, "episode_id": episode, "family": "ladder_down",
                "state": bounded_state(history, profile=LADDER_PROFILE), "executed_stick": stick,
                "labels": [quantize(v) for v in stick]}
            validate_record(row, profile=LADDER_PROFILE)
            rows.append(row)
        if rows:
            episodes[episode] = rows
            hashes[path.relative_to(suite).as_posix()] = digest(path)
    if len(episodes) < 3:
        raise ValueError("At least three successful native ladder sessions required")
    splits = {name: [] for name in ("train", "validation", "test")}
    for index, (_, rows) in enumerate(sorted(episodes.items())):
        split = "test" if index == len(episodes) - 1 else "validation" if index == len(episodes) - 2 else "train"
        splits[split].extend(rows)
    output.mkdir(parents=True)
    for split, rows in splits.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(row, separators=(",", ":")) + "\n"
            for row in rows), encoding="utf-8")
    manifest = {"profile": LADDER_PROFILE, "split_unit": "episode", "episode_grouping": "native_instance",
        "source": "soh", "provider_calls": 0, "records": {k: len(v) for k, v in splits.items()},
        "episodes": {k: len({r["episode_id"] for r in v}) for k, v in splits.items()},
        "sha256": {k: digest(output / f"{k}.jsonl") for k in splits}, "source_task_sha256": hashes,
        "stick_bins": list(STICK_BINS), "excluded": dict(skipped),
        "coverage": "observed_attached_ladder_descent_only_no_approach_buttons_combat_or_campaign"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    load_dataset(output)
    return manifest
