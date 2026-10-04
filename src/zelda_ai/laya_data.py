"""Audited real-game walking demonstrations for an isolated Laya pilot.

Only successful reference tasks with consumed native inputs are positive labels.
Episodes, rather than individual frames, own the train/validation/test split.
This narrow profile contains no button, ladder, combat or campaign labels.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

PROFILE = "laya-observed-walking-v1"
FEATURE_NAMES = ("goal_heading_sin", "goal_heading_cos", "mirrored", "goal_distance",
                 "goal_height", "speed", "surface_up", "surface_down", "observed_cell")
STICK_BINS = (-80, -60, -40, -20, 0, 20, 40, 60, 80)
HISTORY_LIMIT = 4
QUESTIONS = {
    axis: {"type": "choice", "instructions": f"Choose the executed N64 analog {axis} for this observed walking task.",
           "criteria": {str(v): None for v in STICK_BINS}}
    for axis in ("stick_x", "stick_y")
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bounded_state(history: list[list[float]]) -> dict:
    if not 1 <= len(history) <= HISTORY_LIMIT:
        raise ValueError("Use one to four observed feature frames")
    for row in history:
        if len(row) != len(FEATURE_NAMES) or any(not math.isfinite(v) or abs(v) > 1 for v in row):
            raise ValueError("Invalid bounded walking telemetry")
    return {"profile": PROFILE, "features": list(FEATURE_NAMES),
            "history_oldest_first": [[round(v, 4) for v in row] for row in history]}


def quantize(value: float) -> int:
    if not math.isfinite(value) or abs(value) > 80:
        raise ValueError("Invalid executed N64 stick")
    return min(STICK_BINS, key=lambda v: (abs(v - value), abs(v), v))


def validate_record(row: dict) -> None:
    if (row.get("profile") != PROFILE or row.get("source") != "soh"
            or row.get("controller") != "reference" or row.get("first_tick", 0) <= 0
            or row.get("buttons") != 0 or not row.get("successful_episode")
            or row.get("reference_blend") != 0 or not row.get("episode_id")
            or row.get("observation_seq", -1) < 0 or row.get("command_seq", 0) <= 0):
        raise ValueError("Only successful consumed real reference walking actions may be labels")
    state = row.get("state", {})
    if state != bounded_state(state.get("history_oldest_first", [])):
        raise ValueError("Unexpected state fields or incompatible observation profile")
    action = row.get("executed_stick", [])
    if len(action) != 2 or row.get("labels") != [quantize(v) for v in action]:
        raise ValueError("Labels must represent the actual executed stick")


def load_dataset(directory: Path) -> tuple[dict, dict[str, list[dict]]]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("profile") != PROFILE or manifest.get("split_unit") != "episode":
        raise ValueError("Incompatible dataset manifest")
    splits, owners, commands = {}, {}, set()
    for name in ("train", "validation", "test"):
        path = directory / f"{name}.jsonl"
        if digest(path) != manifest["sha256"][name]:
            raise ValueError("Dataset checksum mismatch")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        if not rows:
            raise ValueError("Every reserved split must contain real records")
        last_seq = {}
        for row in rows:
            validate_record(row)
            ep = row["episode_id"]
            if ep in owners and owners[ep] != name:
                raise ValueError("Episode leakage between train and reserved evaluation")
            owners[ep] = name
            command = (ep, row["command_seq"])
            if command in commands:
                raise ValueError("Duplicate native action in the dataset")
            commands.add(command)
            if row["observation_seq"] < last_seq.get(ep, -1):
                raise ValueError("Observation history must be causal")
            last_seq[ep] = row["observation_seq"]
        splits[name] = rows
    return manifest, splits


def majority_baseline(training: list[dict], reserved: list[dict]) -> dict:
    """A no-learning comparator selected strictly from the training split."""
    if not training or not reserved:
        raise ValueError("Training and reserved records are required")
    stick = [Counter(r["labels"][i] for r in training).most_common(1)[0][0] for i in (0, 1)]
    return {"stick": stick,
            "axis_accuracy": sum(r["labels"][i] == stick[i] for r in reserved for i in (0, 1))/(2*len(reserved)),
            "joint_accuracy": sum(r["labels"] == stick for r in reserved)/len(reserved),
            "stick_mae_native_units": sum(abs(r["executed_stick"][i]-stick[i])
                                          for r in reserved for i in (0, 1))/(2*len(reserved))}


def export_surfaces(suite: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Export into a new dataset directory")
    by_family, skipped = defaultdict(list), Counter()
    for path in sorted((suite / "demonstrations").glob("trial-*/motor/task.json")):
        task = json.loads(path.read_text(encoding="utf-8"))
        if not task.get("success"):
            skipped["failed_task"] += 1
            continue
        if (task.get("source") != "soh" or task.get("provider_calls") != 0
                or task.get("run_updates") != 0 or not task.get("objective_unchanged")
                or task.get("reference_blend") != 0):
            raise ValueError("Invalid demonstration task provenance")
        family = task["task"]["kind"]
        if family not in {"stairs_or_slope_up", "stairs_or_slope_down", "observed_cell"}:
            raise ValueError("Unsupported task family")
        receipts = {r["seq"]: r for r in task["receipts"]}
        episode = digest(path)
        history, rows, last = [], [], -1
        for action in task["demonstrations"]:
            seq = action["observation_seq"]
            receipt = receipts.get(action["command_seq"], {})
            if seq < last or receipt.get("first_tick", 0) != action.get("first_tick"):
                raise ValueError("Missing consumed receipt or noncausal action")
            last = seq
            history = (history + [action["features"]])[-HISTORY_LIMIT:]
            stick = [v * 80 for v in action["action"]]
            row = {**{k: action[k] for k in ("source", "controller", "first_tick", "buttons",
                                           "successful_episode", "reference_blend", "command_seq",
                                           "observation_seq")},
                   "profile": PROFILE, "episode_id": episode, "family": family,
                   "state": bounded_state(history), "executed_stick": stick,
                   "labels": [quantize(v) for v in stick]}
            validate_record(row)
            rows.append(row)
        if rows:
            by_family[family].append((episode, rows))
    splits = {name: [] for name in ("train", "validation", "test")}
    for family, episodes in by_family.items():
        if len(episodes) < 3:
            raise ValueError(f"At least three real successful episodes required per family: {family}")
        ordered = sorted(episodes)
        # Last two whole episodes are reserved; no frame shuffle or family leakage.
        for i, (_, rows) in enumerate(ordered):
            name = "test" if i == len(ordered)-1 else "validation" if i == len(ordered)-2 else "train"
            splits[name].extend(rows)
    if any(not rows for rows in splits.values()):
        raise ValueError("Insufficient real gameplay demonstrations")
    output.mkdir(parents=True)
    for name, rows in splits.items():
        (output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows),
            encoding="utf-8")
    manifest = {"profile": PROFILE, "split_unit": "episode", "source": "soh",
                "provider_calls": 0, "families": dict(Counter(r["family"] for r in sum(splits.values(), []))),
                "records": {k: len(v) for k, v in splits.items()},
                "episodes": {k: len({r["episode_id"] for r in v}) for k, v in splits.items()},
                "excluded": dict(skipped), "stick_bins": list(STICK_BINS),
                "sha256": {k: digest(output / f"{k}.jsonl") for k in splits},
                "coverage": "short_observed_walking_only_no_buttons_ladders_combat_or_campaign"}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    load_dataset(output)
    return manifest
