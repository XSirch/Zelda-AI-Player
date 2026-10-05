import json

import pytest

from zelda_ai.laya_data import LADDER_PROFILE, load_dataset
from zelda_ai.laya_ladder_data import export_ladders


def write_reference(suite, index, *, bad=None):
    player = {"climbing_ladder": True, "hanging_ledge": False, "climbing_ledge": False,
              "position": [0, 100, 0], "floor_height": 100, "bg_check_flags": 1, "speed_xz": 0, "age": "child"}
    initial = {"instance_id": f"instance-{index}", "scene_epoch": 2, "scene": 1, "room": 0,
               "mirrored_world": False, "source": "soh", "player": player}
    final = {**initial, "player": {**player, "climbing_ladder": False,
                                   "position": [0, -80, 0], "floor_height": -80}}
    action = {"source": "soh", "controller": "reference", "first_tick": 4, "command_seq": 2,
              "observation_seq": 3, "buttons": 0, "successful_episode": True, "reference_blend": 0,
              "reference_calibrated": True, "ladder_attached": True,
              "features": [-.6, 0, 0, 1, 0, 1, 0, 1, 1], "action": [0, -.75]}
    task = {"success": True, "source": "soh", "provider_calls": 0, "run_updates": 0,
            "objective_unchanged": True, "reference_blend": 0, "feature_profile": LADDER_PROFILE,
            "task": {"version": "observed-attached-descent-reference-v1", "verification_frames": 3,
                     "target": [0, -80, 0]}, "initial": initial, "final": final,
            "receipts": [{"seq": 2, "first_tick": 4}], "demonstrations": [action]}
    if bad == "receipt":
        task["receipts"] = []
    elif bad == "candidate":
        action["controller"] = "candidate"
    elif bad == "airborne":
        final["player"]["bg_check_flags"] = 0
    elif bad == "instance":
        final["instance_id"] = "different"
    elif bad == "uncalibrated":
        action["reference_calibrated"] = False
    path = suite / f"native-session-{index}" / "descent-preparation" / "attempt-1" / "attached-motor" / "task.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(task), encoding="utf-8")


def test_export_splits_whole_actual_instance_ids_and_keeps_executed_labels(tmp_path):
    for index in range(3):
        write_reference(tmp_path / "suite", index)
    manifest = export_ladders(tmp_path / "suite", tmp_path / "dataset")
    assert manifest["episode_grouping"] == "native_instance"
    assert manifest["records"] == {"train": 1, "validation": 1, "test": 1}
    _, splits = load_dataset(tmp_path / "dataset")
    assert len({r["episode_id"] for rows in splits.values() for r in rows}) == 3
    assert all(r["labels"] == [0, -60] for rows in splits.values() for r in rows)


@pytest.mark.parametrize("bad", ("receipt", "candidate", "airborne", "instance", "uncalibrated"))
def test_export_refuses_noncausal_unverified_or_exploration_labels(tmp_path, bad):
    for index in range(3):
        write_reference(tmp_path / "suite", index, bad=bad if index == 1 else None)
    with pytest.raises(ValueError):
        export_ladders(tmp_path / "suite", tmp_path / "dataset")
