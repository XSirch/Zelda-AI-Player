import asyncio
import json
from types import SimpleNamespace

import pytest

import zelda_ai.laya_exploration as module
from zelda_ai.models import GameEvent, NavigationMeshSnapshot


def setup(state, monkeypatch, *, mutate=False, effect=None):
    state.player.bg_check_flags = 1
    state.camera_input_yaw = 0
    state.navmesh = NavigationMeshSnapshot(step=70, half_extent=4,
        cells=[(0, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 68), (3, 0, 0., 64)])
    bridge = SimpleNamespace(state=state, release=lambda: None)
    policy = SimpleNamespace(invalidate=lambda: None)
    counts = {"controllers": 0, "intents": 0, "stages": 0}
    controller = SimpleNamespace(policy=SimpleNamespace(updates=0), starting_updates=0,
        executor=SimpleNamespace(state="executing", room_budget_s=0, reason=""))

    def create(*args, **kwargs):
        assert kwargs["training_enabled"] is False
        counts["controllers"] += 1
        return controller

    def set_intent(intent):
        controller.intent = intent
        counts["intents"] += 1

    async def fresh(bridge):
        if counts["stages"] >= 2:
            raise ValueError("no_more_observed_tasks")
        return bridge.state

    async def execute(owned, bridge, task, policy, path, *, on_started, observe, **kwargs):
        assert owned is controller and controller.executor.room_budget_s == 120
        counts["stages"] += 1
        if on_started:
            on_started()
        # Software integration fixture only; these moves are never native evidence.
        state.player.position = task.target
        state.navmesh.origin = task.target
        state.seq += 1
        observe(state)
        task.phase = "succeeded"
        if mutate:
            controller.intent.objective = "Replace the locked goal"
        if effect:
            effect(state)
        return {"success": True, "task": task.snapshot(), "consumed_actions": 1,
                "raw_button_actions": 0, "objective_unchanged": True,
                "run_updates": 0, "motor_failure": None}

    controller.set_intent = set_intent
    monkeypatch.setattr(module, "ContinuousController", create)
    monkeypatch.setattr(module, "fresh_collision_snapshot", fresh)
    monkeypatch.setattr(module, "execute_frozen_task", execute)
    return bridge, policy, counts


def test_continuous_tasks_share_one_locked_controller_and_local_success_is_not_goal_completion(state, monkeypatch, tmp_path):
    bridge, policy, counts = setup(state, monkeypatch)
    result = asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
        tmp_path / "motor", seconds=10))
    assert counts == {"controllers": 1, "intents": 1, "stages": 2}
    assert all(stage["success"] for stage in result["stages"])
    assert not result["success"] and not result["objective_completed"]
    assert result["objective_unchanged"] and result["planning"]["occupied_macro_regions"] == 3
    assert result["reason"] == "ValueError:no_more_observed_tasks"
    diagnostic = json.loads((tmp_path / "motor/planning-state.json").read_text(encoding="utf-8"))
    assert diagnostic["schema"] == "observed-exploration-planning-state-v1"
    assert len(diagnostic["planning"]["visited_regions"]) == 3


def test_mutating_the_original_intent_cannot_mutate_completion_lock(state, monkeypatch, tmp_path):
    bridge, policy, counts = setup(state, monkeypatch, mutate=True)
    result = asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
        tmp_path / "motor", seconds=10))
    assert counts["stages"] == 1 and not result["success"]
    assert result["reason"] == "objective_lock_changed" and not result["objective_unchanged"]
    assert result["objective"]["objective"] == "Obtain the Kokiri Sword"


def test_already_owned_equipment_is_not_new_acquisition_evidence(state, monkeypatch, tmp_path):
    bridge, policy, counts = setup(state, monkeypatch)
    state.progress.owned_equipment = ["Kokiri Sword"]
    result = asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
        tmp_path / "motor", seconds=10))
    assert counts["stages"] == 0 and result["objective_completed"]
    assert not result["success"] and not result["objective_acquired_in_run"]
    assert result["reason"] == "objective_already_satisfied_at_start"


@pytest.mark.parametrize("replacement", ["save", "instance", "death", "game_over"])
def test_equipment_in_a_replaced_episode_does_not_count_as_acquired(state, monkeypatch, tmp_path, replacement):
    def effect(observed):
        observed.progress.owned_equipment = ["Kokiri Sword"]
        if replacement == "instance":
            observed.instance_id = "replacement-native-instance"
        else:
            kind = {"save": "save_loaded", "death": "player_died", "game_over": "game_over"}[replacement]
            observed.events = [GameEvent(id="new-lifetime-event", kind=kind)]

    bridge, policy, counts = setup(state, monkeypatch, effect=effect)
    result = asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
        tmp_path / "motor", seconds=10))
    assert counts["stages"] == 1 and result["objective_completed"]
    assert result["reason"] == "episode_identity_or_load_changed"
    assert not result["success"] and not result["objective_acquired_in_run"]


def test_invalid_lifetime_remains_invalid_after_native_event_rolls_off(state, monkeypatch, tmp_path):
    bridge, policy, counts = setup(state, monkeypatch)
    execute = module.execute_frozen_task

    async def with_reload(*args, **kwargs):
        result = await execute(*args, **kwargs)
        state.events = [GameEvent(id="new-load", kind="save_loaded")]
        kwargs["observe"](state)
        state.events = []
        state.progress.owned_equipment = ["Kokiri Sword"]
        return result

    monkeypatch.setattr(module, "execute_frozen_task", with_reload)
    result = asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
        tmp_path / "motor", seconds=10))
    assert counts["stages"] == 1 and not result["episode_identity_unchanged"]
    assert result["objective_completed"] and not result["objective_acquired_in_run"]
    assert not result["success"] and result["reason"] == "episode_identity_or_load_changed"


def test_new_equipment_in_the_same_native_episode_counts_as_completion(state, monkeypatch, tmp_path):
    def effect(observed):
        observed.progress.owned_equipment = ["Kokiri Sword"]

    bridge, policy, counts = setup(state, monkeypatch, effect=effect)
    result = asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
        tmp_path / "motor", seconds=10))
    assert counts["stages"] == 1 and result["objective_completed"]
    assert result["reason"] == "objective_telemetry_completed"
    assert result["success"] and result["objective_acquired_in_run"]


def test_motor_failure_cannot_qualify_a_run_with_observed_equipment_progress(state, monkeypatch, tmp_path):
    def effect(observed):
        observed.progress.owned_equipment = ["Kokiri Sword"]

    bridge, policy, counts = setup(state, monkeypatch, effect=effect)
    execute = module.execute_frozen_task

    async def with_failure(*args, **kwargs):
        result = await execute(*args, **kwargs)
        result.update(success=False, motor_failure="RuntimeError:local_candidate_failed")
        return result

    monkeypatch.setattr(module, "execute_frozen_task", with_failure)
    result = asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
        tmp_path / "motor", seconds=10))
    assert counts["stages"] == 1 and result["objective_acquired_in_run"]
    assert result["reason"] == "RuntimeError:local_candidate_failed"
    assert not result["success"] and not result["runtime_clean"]


@pytest.mark.parametrize("seconds", [0, 9, 301])
def test_continuous_budget_is_bounded_before_controller_construction(state, monkeypatch, tmp_path, seconds):
    bridge, policy, counts = setup(state, monkeypatch)
    with pytest.raises(ValueError):
        asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
            tmp_path / "motor", seconds=seconds))
    assert counts["controllers"] == 0


def test_context_dialogue_walk_handoff_releases_close_and_restores_observer(state, monkeypatch, tmp_path):
    # Software integration only: no artificial state is native gameplay evidence.
    state.player.bg_check_flags = 1
    state.player.floor_height = state.player.position[1]
    state.camera_input_yaw = 0
    state.context_action.code, state.context_action.label = 15, "speak"
    state.navmesh = NavigationMeshSnapshot(step=70, half_extent=4,
        cells=[(0, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 68), (3, 0, 0., 64)])
    calls, kinds = [], []
    def previous_observer(game, old):
        calls.append("previous_observer")

    bridge = SimpleNamespace(state=state, on_state=previous_observer, release=lambda: calls.append("release"))
    policy = SimpleNamespace(invalidate=lambda: None)
    controller = SimpleNamespace(policy=SimpleNamespace(updates=0), starting_updates=0,
        executor=SimpleNamespace(state="executing", room_budget_s=0, reason=""), local_task=None,
        interaction_memory=SimpleNamespace(snapshot=lambda: {}))
    assignments = []

    def set_intent(intent):
        controller.intent = intent
        assignments.append(intent.model_copy(deep=True))

    def closed(old, game):
        calls.append("dialogue_closed")
        controller.dialogue_reentry_guard = {"until": module.time.monotonic()+8,
            "scene": game.scene, "room": game.room, "anchor": tuple(game.player.position)}

    async def fresh(bridge):
        if len(kinds) == 3:
            raise ValueError("end_of_software_fixture")
        return bridge.state

    async def execute(owned, bridge, task, path, **kwargs):
        assert owned is controller
        controller.local_task = task
        kinds.append(task.kind)
        old = state.model_copy(deep=True)
        state.seq += 1
        if task.kind == "observed_context_interaction":
            state.dialogue.active = True
        elif task.kind == "observed_linear_dialogue":
            state.dialogue.active = False
        else:
            assert kinds == ["observed_context_interaction", "observed_linear_dialogue", "observed_cell"]
            assert "dialogue_closed" in calls
        bridge.on_state(state, old)
        if task.kind == "observed_linear_dialogue":
            assert calls[-3:] == ["previous_observer", "dialogue_closed", "release"]
        if task.kind == "observed_cell":
            assert not task.terminal  # The re-entry guard preserves stick escape.
            old = state.model_copy(deep=True)
            state.player.position = task.target
            state.context_action.code, state.context_action.label = 0, "none"
            state.seq += 1
            bridge.on_state(state, old)
        task.phase, task.consumed = "succeeded", True
        task.observed_effect = "dialogue_closed" if isinstance(task, module.LinearDialogueTask) else "dialogue_started"
        controller.local_task = None
        button_count = int(isinstance(task, module.ObservedContextTask))
        return {"success": True, "task": task.snapshot(), "consumed_actions": 1,
            "raw_button_actions": button_count, "causal_button_actions": button_count,
            "unowned_button_actions": 0, "objective_unchanged": True,
            "run_updates": 0, "motor_failure": None}

    async def walking(owned, bridge, task, policy, path, **kwargs):
        return await execute(owned, bridge, task, path, **kwargs)

    controller.set_intent, controller.note_dialogue_closed = set_intent, closed
    monkeypatch.setattr(module, "ContinuousController", lambda *args, **kwargs: controller)
    monkeypatch.setattr(module, "fresh_collision_snapshot", fresh)
    monkeypatch.setattr(module, "execute_context_task", execute)
    monkeypatch.setattr(module, "execute_frozen_task", walking)
    result = asyncio.run(module.exploration_episode(bridge, tmp_path / "frozen", policy, policy,
        tmp_path / "motor", seconds=10, contextual_interactions=True))
    assert bridge.on_state is previous_observer
    assert len(assignments) == 1 and controller.intent == assignments[0]
    assert result["objective_unchanged"] and result["run_updates"] == 0
    assert result["causal_button_actions"] == 2 and result["unowned_button_actions"] == 0
    assert not result["success"] and result["reason"] == "ValueError:end_of_software_fixture"
