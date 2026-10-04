"""Audit local physical G1 evidence and export only a public, whitelisted index."""
import argparse
import hashlib
import json
import math
import statistics
from datetime import datetime
from pathlib import Path

from zelda_ai.g1 import ARTIFACTS, portal_crossed, summarize, write_json
from zelda_ai.models import GameState


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def audit(suite):
    manifest, report = load(suite / "manifest.json"), load(suite / "report.json")
    records = []
    for row in report["episodes"]:
        episode = suite / f"episode-{row['index']:03}"
        if row != load(episode / "report.json"):
            raise ValueError(f"Aggregate/per-episode report mismatch: {episode.name}")
        if row.get("success"):
            motor = load(episode / "motor.json")
            initial, final = (GameState.model_validate(motor[key]) for key in ("initial", "final"))
            if not portal_crossed(initial, final) or not motor["objective_unchanged"]:
                raise ValueError(f"Invalid physical crossing: {episode.name}")
            if (motor["controller"]["learning"]["run_updates"] != 0
                    or motor["controller"]["learning"]["training_enabled"]):
                raise ValueError(f"Evaluation trained: {episode.name}")
            consumed = [receipt for receipt in motor["receipts"] if receipt["first_tick"] > 0]
            if not consumed or len(consumed) != row["consumed_commands"]:
                raise ValueError(f"Consumed-input evidence mismatch: {episode.name}")
            if list(initial.player.position) != row["initial_position"] or list(final.player.position) != row["final_position"]:
                raise ValueError(f"Observed-position mismatch: {episode.name}")
        records.append(row)
    pinned_ok = all(Path(path).is_file() and digest(Path(path)) == value
                    for path, value in manifest["pinned_sha256"].items())
    frozen_ok = all(digest(suite / "frozen" / name) == manifest["frozen_sha256"][name] for name in ARTIFACTS)
    recomputed = summarize(manifest, records, finished=report["checks"]["finished"],
                           artifacts_unchanged=pinned_ok and frozen_ok)
    if recomputed["checks"] != report["checks"]:
        raise ValueError("Current artifact hashes or gate evidence disagree with the recorded result")
    durations = sorted(e["total_elapsed_s"] for e in records)
    yaws, modal_rows = [], 0
    for episode in suite.glob("episode-*/motor.json"):
        for row in load(episode)["trace"]:
            if row["camera_input_yaw"] is not None:
                yaws.append(row["camera_input_yaw"])
            modal_rows += bool(row["dialogue_active"] or row["pause_active"])
    first_evidence = min(suite.glob("episode-*/report.json"), key=lambda path: path.stat().st_mtime)
    date = datetime.fromtimestamp(first_evidence.stat().st_mtime).astimezone().date().isoformat()
    return {"schema_version": 1, "date": date, "suite_id": manifest["suite_id"],
            "source": "soh", "observation_profile": manifest["observation_profile"],
            "scope": "native Arquivo 2; child, normal world; initial house and physical portal revisits",
            "zero_shot": False, "training_enabled": False, "provider_calls": manifest["provider_calls"],
            "g1_qualified": recomputed["g1_qualified"], "checks": recomputed["checks"],
            "attempted_episodes": len(records), "successful_episodes": recomputed["successful_episodes"],
            "coverage": recomputed["coverage"], "failures": recomputed["failures"],
            "seed": manifest["seed"], "seconds_budget": manifest["seconds_budget"],
            "budget_scope": manifest["budget_scope"],
            "timing_s": {"total_scenarios": round(sum(durations), 3), "min": min(durations),
                         "median": statistics.median(durations),
                         "p95_nearest_rank": durations[math.ceil(len(durations) * .95) - 1], "max": max(durations)},
            "native_builds": sorted({e.get("native_build", "unknown") for e in records}),
            "native_instances_observed": len({e["initial_context"][0] for e in records if "initial_context" in e}),
            "upstream_revisions": sorted({e.get("upstream_revision", "unknown") for e in records}),
            "physical_device_mappings_disabled": manifest["physical_device_mappings_disabled"],
            "frozen_sha256": manifest["frozen_sha256"],
            "executable_sha256": manifest["pinned_sha256"][manifest["executable"]],
            "local_evidence_sha256": {name: digest(suite / name) for name in ("manifest.json", "report.json")},
            "motor_camera_bins_45_degrees": sorted({int(((yaw % 65536 + 4096) % 65536) / 8192) for yaw in yaws}),
            "modal_trace_samples": modal_rows,
            "episodes": [{key: value for key, value in e.items() if key in {
                "index", "episode_id", "success", "reason", "pair_role", "initial_position", "final_position",
                "initial_camera_yaw", "setup_heading_bins", "setup_status", "total_elapsed_s", "consumed_commands",
                "objective_unchanged", "artifacts_unchanged", "run_updates", "settled_crossing"}}
                | {"report_sha256": digest(suite / f"episode-{e['index']:03}" / "report.json"),
                   "motor_sha256": digest(suite / f"episode-{e['index']:03}" / "motor.json")
                                    if (suite / f"episode-{e['index']:03}" / "motor.json").is_file() else None,
                   "setup_sha256": digest(suite / f"episode-{e['index']:03}" / "setup.json")
                                    if (suite / f"episode-{e['index']:03}" / "setup.json").is_file() else None}
                for e in records]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = audit(args.suite.resolve(strict=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, index)
    print(json.dumps({key: index[key] for key in ("suite_id", "g1_qualified", "attempted_episodes", "successful_episodes")}))
