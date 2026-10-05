import asyncio
import json
from types import SimpleNamespace

import pytest

from zelda_ai import laya_curriculum
from zelda_ai.models import NavigationMeshSnapshot, TraversalAffordanceObservation


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


@pytest.mark.parametrize("failure_kind", ("portal", "descent"))
def test_failed_preparation_never_runs_candidate_and_survives_dataset_rejection(
    state, tmp_path, monkeypatch, failure_kind,
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
        return {"success": False, "candidate_actions": 0, "kind": failure_kind}

    async def portal(*args):
        return {"success": True, "candidate_actions": 0, "kind": "portal"}

    def rejected_export(*args, **kwargs):
        raise ValueError("No successful native sessions to export")

    monkeypatch.setattr(laya_curriculum, "prepare_suite", prepare)
    monkeypatch.setattr(laya_curriculum, "bridge_secret", lambda _: "unit-test-token")
    monkeypatch.setattr(laya_curriculum, "Bridge", Bridge)
    monkeypatch.setattr(laya_curriculum, "OwnedProcess", Process)
    monkeypatch.setattr(laya_curriculum, "bind_bridge", bind)
    monkeypatch.setattr(laya_curriculum, "enter_playable_save", startup)
    monkeypatch.setattr(laya_curriculum, "settle", settle)
    monkeypatch.setattr(laya_curriculum, "prepare_portal_start", failed if failure_kind == "portal" else portal)
    monkeypatch.setattr(laya_curriculum, "prepare_observed_descent", failed)
    monkeypatch.setattr(laya_curriculum, "run_task", lambda *a, **kw: pytest.fail("invalid start ran a task"))
    monkeypatch.setattr(laya_curriculum, "export_surfaces", rejected_export)
    with pytest.raises(ValueError, match="No successful native sessions"):
        asyncio.run(laya_curriculum.collect(None, tmp_path, tmp_path, sessions=3, tasks=3,
                                            cross_initial_portal=True,
                                            descend_observed_ledge=failure_kind == "descent"))
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    assert report["planned_tasks"] == len(report["records"]) == 9
    assert report["attempted_tasks"] == report["successful_tasks"] == 0
    assert report["dataset"] is None
    assert report["dataset_error"] == "No successful native sessions to export"
    assert all(not r["attempted"] and r["consumed_actions"] == 0 for r in report["records"])
    assert all(f"initial_{failure_kind}_preparation_failed" in r["reason"] for r in report["records"])
    assert len(processes) == 3 and all(p.closed for p in processes)


def descent_observation(state):
    state.player.position = (0, 100, 0)
    state.player.floor_height = 100
    state.player.bg_check_flags = 1
    state.player.speed_xz = 0
    state.navmesh = NavigationMeshSnapshot(origin=(0, 100, 0), step=70, half_extent=4,
        cells=[(0, 0, 100., 85), (1, 0, 100., 0), (-1, 0, 100., 0),
               (0, 1, 100., 0), (0, -1, 100., 0)])
    return [TraversalAffordanceObservation(kind="ledge_down", direction="down",
        target_position=(x, 0, z), approach_position=(x * 2 / 3, 100, z * 2 / 3),
        distance=140, height_delta=-100) for x, z in ((210, 0), (-210, 0), (0, 210), (0, -210))]


def test_descent_reconsiders_only_current_proposals_and_caps_failed_attempts(state, tmp_path, monkeypatch):
    rows = descent_observation(state)
    duplicate = rows[0].model_copy(update={"target_position": (220, 0, 0)})
    state.traversal_affordances = [rows[0], duplicate]
    bridge = SimpleNamespace(state=state)
    attempted = []

    async def failed(actual_bridge, frozen, task, directory):
        assert task.target in [r.target_position for r in actual_bridge.state.traversal_affordances]
        attempted.append(task.target)
        state.seq += 1
        task.observe(state, consumed=True, now=task.started + 3)
        assert task.failure == "no_geometric_progress"
        if len(attempted) == 1:
            # New local observation; the old target is no longer present.
            state.traversal_affordances = [duplicate, *rows[1:]]
        return {"success": False, "task": task.snapshot(), "consumed_actions": 1,
                "source": "soh", "reference_blend": 0, "provider_calls": 0,
                "objective_unchanged": True, "run_updates": 0,
                "initial": {"player": {"position": task.origin}},
                "final": {"player": {"position": state.player.position}}}, []

    monkeypatch.setattr(laya_curriculum, "run_task", failed)
    result = asyncio.run(laya_curriculum.prepare_observed_descent(bridge, tmp_path, tmp_path / "descent"))
    assert attempted == [r.target_position for r in rows[:3]]
    assert len(result["attempts"]) == result["consumed_actions"] == 3
    assert not result["success"] and result["reason"] == "no_geometric_progress"
    assert result["candidate_actions"] == 0 and not result["included_in_laya_training"]
    assert json.loads((tmp_path / "descent" / "report.json").read_text()) == json.loads(json.dumps(result))


@pytest.mark.parametrize("failure", (None, "not_stopped", "context", "no_receipt", "objective", "training",
                                    "simulator", "reference_blend", "provider"))
def test_descent_preparation_requires_verified_ground_and_unmodified_evaluation(
    state, tmp_path, monkeypatch, failure,
):
    state.traversal_affordances = descent_observation(state)[:1]
    bridge = SimpleNamespace(state=state)

    async def landed(actual_bridge, frozen, task, directory):
        state.player.position = task.target
        state.player.floor_height = task.target[1]
        for index in range(3):
            state.seq += 1
            task.observe(state, consumed=True, now=task.started + .1 * (index + 1))
        assert task.phase == "succeeded"
        if failure == "not_stopped":
            state.player.speed_xz = 1
        if failure == "context":
            state.instance_id = "other-process"
        return {"success": True, "task": task.snapshot(),
                "source": "simulator" if failure == "simulator" else "soh",
                "reference_blend": int(failure == "reference_blend"),
                "provider_calls": int(failure == "provider"),
                "consumed_actions": 0 if failure == "no_receipt" else 3,
                "objective_unchanged": failure != "objective", "run_updates": int(failure == "training"),
                "initial": {"player": {"position": task.origin}},
                "final": {"player": {"position": state.player.position}}}, []

    monkeypatch.setattr(laya_curriculum, "run_task", landed)
    result = asyncio.run(laya_curriculum.prepare_observed_descent(bridge, tmp_path, tmp_path / "descent"))
    assert result["success"] is (failure is None)
    assert len(result["attempts"]) == 1
    assert result["candidate_actions"] == 0 and not result["included_in_laya_training"]
    reasons = {None: "observed_verified_landing", "not_stopped": "landing_not_verified",
               "context": "context_changed", "no_receipt": "missing_consumed_input",
               "objective": "objective_changed", "training": "evaluation_trained", "simulator": "invalid_source",
               "reference_blend": "reference_blend", "provider": "provider_calls"}
    assert result["reason"] == reasons[failure]
    assert result["provider_calls"] == int(failure == "provider")
