"""Audit real local curriculum evidence and publish only a small safe index."""
import argparse
import json
from pathlib import Path

from zelda_ai.autonomy.imitation import VERSION, SurfacePolicy
from zelda_ai.autonomy.ml_policy import torch
from zelda_ai.g1 import ARTIFACTS, context, playable, portal_crossed
from zelda_ai.models import GameState
from zelda_ai.qualification import sha256
from zelda_ai.surface_curriculum import FAMILIES, aggregate


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert report["source"] == "soh" and report["provider_calls"] == 0
    assert report["promotion"] == "not_promoted" and report["evaluation_training_updates"] == 0
    assert report["artifacts_unchanged"] and not report["g2_complete"]
    for path, expected in manifest["pinned_sha256"].items():
        assert sha256(Path(path)) == expected, f"Pinned file changed: {path}"
    for name in ARTIFACTS:
        assert sha256(directory / "frozen" / name) == manifest["frozen_sha256"][name]
    for path, expected in manifest["fixture_sha256"].items():
        assert sha256(directory / "seed-home" / path) == expected
    dataset = directory / "demonstrations.json"
    assert sha256(dataset) == report["dataset_sha256"]
    rows = json.loads(dataset.read_text(encoding="utf-8"))["rows"]
    for row in rows:
        assert row["source"] == "soh" and row["controller"] == "reference"
        assert row["first_tick"] > 0 and row["successful_episode"]
        assert row["buttons"] == 0 and row["reference_blend"] == 0
    for name, key in (("surface-before.pt", "before_sha256"), ("surface-candidate.pt", "candidate_sha256")):
        assert sha256(directory / name) == report[key]
        SurfacePolicy.load(directory / name)
    checkpoint = torch.load(directory / "surface-candidate.pt", weights_only=True)
    assert checkpoint["dataset_sha256"] == report["dataset_sha256"]
    records, identifiers, demo_rows = [], set(), []
    for row in report["records"]:
        stage, index = row["stage"], row["index"]
        assert stage in {"demonstrations", "before", "after", "retention"} and 1 <= index <= 300
        assert (stage, index) not in identifiers
        identifiers.add((stage, index))
        assert row["family"] in (*FAMILIES, "portal_retention") and row["artifacts_unchanged"]
        path = directory / stage / f"trial-{index:03}"
        saved = json.loads((path / "report.json").read_text(encoding="utf-8"))
        assert saved == row
        if row["family"] == "portal_retention":
            if row["success"]:
                data = json.loads((path / "motor.json").read_text(encoding="utf-8"))
                assert portal_crossed(GameState.model_validate(data["initial"]),
                                      GameState.model_validate(data["final"]))
                assert data["objective_unchanged"] and data["controller"]["learning"]["run_updates"] == 0
                assert any(receipt["first_tick"] > 0 for receipt in data["receipts"])
        elif (path / "motor" / "task.json").exists():
            data = json.loads((path / "motor" / "task.json").read_text(encoding="utf-8"))
            assert data["provider_calls"] == 0 and data["run_updates"] == 0
            assert data["intent"] == data["final_intent"] and data["objective_unchanged"]
            assert data["reference_blend"] == 0
            receipts = {r["seq"]: r for r in data["receipts"]}
            assert len(receipts) == data["consumed_actions"] == row["consumed_actions"]
            for action in data["actions"]:
                receipt = receipts[action["command_seq"]]
                assert receipt["first_tick"] == action["first_tick"] > 0
                assert action["buttons"] == 0 and action["reference_blend"] == 0
                assert action["source"] == "soh"
            assert len(data["actions"]) == len(receipts)
            assert row["success"] == data["success"] == (data["task"]["phase"] == "succeeded")
            if data["success"]:
                initial, final = GameState.model_validate(data["initial"]), GameState.model_validate(data["final"])
                assert playable(initial) and playable(final) and context(initial) == context(final)
                assert final.seq > initial.seq and final.scene_epoch == initial.scene_epoch
                task = data["task"]
                assert task["consumed"] and task["verification_frames"] >= 3 and task["failure"] is None
                assert abs(final.player.position[1] - task["target"][1]) <= 4
                assert final.player.speed_xz < .1
                assert receipts
            if stage == "demonstrations" and data["success"]:
                demo_rows.extend(data["demonstrations"])
            if row["family"] == "recovery_cell":
                failed = json.loads((path / "withheld-input" / "task.json").read_text(encoding="utf-8"))
                assert not failed["success"] and failed["task"]["failure"] == "no_geometric_progress"
                assert failed["intent"] == data["intent"]
        records.append({key: row[key] for key in ("stage", "index", "family", "success", "reason", "elapsed_s")})
    assert rows == demo_rows
    for family in FAMILIES:
        for stage in ("before", "after"):
            assert report["families"][family][stage] == aggregate(report["records"], family, stage)
    assert sum(r["stage"] == "demonstrations" for r in records) == manifest["demonstration_trials"]
    assert sum(r["stage"] == "before" for r in records) == manifest["evaluation_trials"]
    assert sum(r["stage"] == "after" for r in records) == manifest["evaluation_trials"]
    assert sum(r["stage"] == "retention" for r in records) == manifest["retention_trials"]
    assert report["portal_retention"] == aggregate(report["records"], "portal_retention", "retention")
    return {"schema_version": 1, "date": "2026-10-04", "suite_id": report["suite_id"],
        "source": "soh", "observation_profile": "instrumented_local_v1", "candidate_version": VERSION,
        "scope": manifest["evaluation_scope"], "provider_calls": 0, "evaluation_training_updates": 0,
        "promotion": "not_promoted", "g2_complete": False, "artifacts_unchanged": True,
        "save_slot": manifest["save_slot"], "seed": manifest["seed"], "evaluation_seed": report["evaluation_seed"],
        "training": report["training"], "families": report["families"],
        "portal_retention": report["portal_retention"], "candidate_sha256": report["candidate_sha256"],
        "before_sha256": report["before_sha256"], "dataset_sha256": report["dataset_sha256"],
        "records": records}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit(args.directory)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Audited {len(result['records'])} physical trials: {args.output}")
