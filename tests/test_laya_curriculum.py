import asyncio
import json
from types import SimpleNamespace

import pytest

from zelda_ai import laya_curriculum


@pytest.mark.parametrize("failure", (None, "same_room", "reload", "no_consumption", "training"))
def test_portal_preparation_requires_actual_transition_and_is_not_laya_evidence(
    state, tmp_path, monkeypatch, failure,
):
    class Bridge:
        pass

    bridge = Bridge()
    bridge.state = state

    async def motor(actual_bridge, frozen, directory, *, deadline):
        assert directory.is_dir()
        if failure != "same_room":
            actual_bridge.state = state.model_copy(deep=True)
            actual_bridge.state.scene += 1
            actual_bridge.state.scene_epoch += 1
            actual_bridge.state.seq += 1
        if failure == "reload":
            actual_bridge.state.instance_id = "other-instance"
        return {"settled_crossing": True, "consumed_commands": 0 if failure == "no_consumption" else 12,
                "objective_unchanged": True, "run_updates": 1 if failure == "training" else 0}

    monkeypatch.setattr(laya_curriculum, "_motor_episode", motor)
    directory = tmp_path / "preparation"
    result = asyncio.run(laya_curriculum.prepare_portal_start(bridge, tmp_path, directory))
    assert result["success"] is (failure is None)
    assert result["controller"] == "frozen_v3_motor_qa_preparation"
    assert result["candidate_actions"] == 0
    assert result["included_in_laya_training"] is False
    assert json.loads((directory / "report.json").read_text(encoding="utf-8")) == result


def test_failed_preparation_never_runs_candidate_and_survives_dataset_rejection(
    state, tmp_path, monkeypatch,
):
    directory = tmp_path / "suite"
    directory.mkdir()
    (directory / "seed-home").mkdir()
    processes = []

    def prepare(*args, **kwargs):
        return directory, {"executable": "unused", "pinned_sha256": {}, "frozen_sha256": {}}

    class Bridge:
        connected = realtime = True

        def __init__(self, *args, **kwargs):
            self.state = state.model_copy(deep=True)
            self.state.protocol = 3
            self.token = "unit-test-token"
            self.transport = SimpleNamespace(get_extra_info=lambda _: ("127.0.0.1", 0))

        def close(self):
            pass

    class Process:
        def __init__(self, *args):
            self.child = SimpleNamespace(poll=lambda: None)
            self.closed = False
            processes.append(self)

        async def close(self):
            self.closed = True

    async def bind(*args):
        pass

    async def startup(*args, **kwargs):
        return {"status": "loaded"}

    async def settle(*args, **kwargs):
        return state

    async def failed(*args):
        return {"success": False, "candidate_actions": 0}

    def rejected_export(*args, **kwargs):
        raise ValueError("No successful native sessions to export")

    monkeypatch.setattr(laya_curriculum, "prepare_suite", prepare)
    monkeypatch.setattr(laya_curriculum, "bridge_secret", lambda _: "unit-test-token")
    monkeypatch.setattr(laya_curriculum, "Bridge", Bridge)
    monkeypatch.setattr(laya_curriculum, "OwnedProcess", Process)
    monkeypatch.setattr(laya_curriculum, "bind_bridge", bind)
    monkeypatch.setattr(laya_curriculum, "enter_playable_save", startup)
    monkeypatch.setattr(laya_curriculum, "settle", settle)
    monkeypatch.setattr(laya_curriculum, "prepare_portal_start", failed)
    monkeypatch.setattr(laya_curriculum, "run_task", lambda *a, **kw: pytest.fail("invalid start ran a task"))
    monkeypatch.setattr(laya_curriculum, "export_surfaces", rejected_export)
    with pytest.raises(ValueError, match="No successful native sessions"):
        asyncio.run(laya_curriculum.collect(None, tmp_path, tmp_path, sessions=3, tasks=3,
                                            cross_initial_portal=True))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert report["planned_tasks"] == len(report["records"]) == 9
    assert report["attempted_tasks"] == report["successful_tasks"] == 0
    assert report["dataset"] is None
    assert report["dataset_error"] == "No successful native sessions to export"
    assert all(not r["attempted"] and r["consumed_actions"] == 0 for r in report["records"])
    assert all("initial_portal_preparation_failed" in r["reason"] for r in report["records"])
    assert len(processes) == 3 and all(p.closed for p in processes)
