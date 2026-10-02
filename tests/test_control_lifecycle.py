import asyncio
import json
import threading
import time

import pytest

from zelda_ai.autonomy.controller import BUTTON_MASKS, ContinuousController, Setpoint
from zelda_ai.autonomy.models import AgentIntent, ObjectiveCompletion
from zelda_ai.autonomy.runtime import AutonomyRuntime
from zelda_ai.bridge import Bridge
from zelda_ai.models import ActorObservation, ModelInfo, NavigationProbe, RunConfig, SceneExitObservation, Usage
from zelda_ai.providers.base import InferenceResult


class Transport:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append(json.loads(data))


def connected(state):
    bridge = Bridge("x" * 32)
    bridge.state = state
    bridge.peer = ("127.0.0.1", 9999)
    bridge.last_seen = time.monotonic()
    bridge.transport = Transport()
    return bridge


def test_camera_cut_neutralizes_then_reprojects_same_world_heading(
    tmp_path, state
):
    game = state.model_copy(deep=True)
    game.camera_input_yaw = 0
    controller = ContinuousController(
        connected(game),
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    controller.last_guidance = {
        "active": True,
        "stick": (0.0, 1.0),
        "strength": 1.0,
        "target": (0.0, 0.0, 300.0),
        "world_yaw": 0.0,
        "camera_yaw": 0.0,
    }
    controller.pending = {
        "policy_stick": [0.0, 0.0],
        "guidance_strength": 1.0,
        "guidance_stick": [0.0, 1.0],
        "stick": [0.0, 1.0],
        "trainable": True,
    }
    controller.last_setpoint = Setpoint(
        stick_x=0,
        stick_y=80,
        reason="ml_policy",
    )
    controller.route_graph.active_path = ["source", "waypoint", "target"]

    # Establish the current camera basis without changing the route.
    controller._refresh_camera_relative_setpoint(game)
    original_target = controller.last_guidance["target"]
    original_path = list(controller.route_graph.active_path)

    rotated = game.model_copy(deep=True)
    rotated.camera_input_yaw = 0x4000
    controller._refresh_camera_relative_setpoint(rotated)

    assert controller.camera_motion_state == "cut"
    assert controller.last_setpoint.reason == "camera_cut"
    assert controller.last_setpoint.stick_x == 0
    assert controller.last_setpoint.stick_y == 0
    assert controller.pending["trainable"] is False
    assert controller.last_guidance["target"] == original_target
    assert controller.route_graph.active_path == original_path

    # Two guard ticks absorb the abrupt indoor camera switch. Once stable, the
    # same world-north heading is reprojected against a camera facing east.
    controller._refresh_camera_relative_setpoint(rotated)
    controller._refresh_camera_relative_setpoint(rotated)
    controller._refresh_camera_relative_setpoint(rotated)

    assert controller.camera_motion_state == "stable"
    assert controller.last_setpoint.reason == "ml_policy"
    assert controller.last_setpoint.stick_x < -75
    assert abs(controller.last_setpoint.stick_y) <= 1
    assert controller.last_guidance["target"] == original_target
    assert controller.route_graph.active_path == original_path


def test_camera_cut_does_not_override_interaction_authority(tmp_path, state):
    game = state.model_copy(deep=True)
    game.camera_input_yaw = 0
    controller = ContinuousController(
        connected(game),
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    controller.last_setpoint = Setpoint(
        buttons=BUTTON_MASKS["A"],
        stick_x=0,
        stick_y=0,
        reason="interaction_memory",
    )

    controller._refresh_camera_relative_setpoint(game)
    rotated = game.model_copy(deep=True)
    rotated.camera_input_yaw = 0x4000
    controller._refresh_camera_relative_setpoint(rotated)

    assert controller.camera_motion_state == "cut"
    assert controller.last_setpoint.reason == "interaction_memory"
    assert controller.last_setpoint.buttons == BUTTON_MASKS["A"]
    assert controller.last_setpoint.stick_x == 0
    assert controller.last_setpoint.stick_y == 0


class SlowCognition:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def status(self):
        return {"connected": True}

    async def models(self):
        return [ModelInfo(id="test", name="test")]

    async def think(self, config, prompt):
        self.entered.set()
        await self.release.wait()
        return InferenceResult(
            AgentIntent.bootstrap().model_dump_json(),
            Usage(input_tokens=3, output_tokens=4, actual_model="test"),
        )


class ObjectiveSequenceCognition:
    def __init__(self, intents):
        self.intents = list(intents)
        self.calls = 0
        self.prompts = []
        self.called = asyncio.Event()

    async def status(self):
        return {"connected": True}

    async def models(self):
        return [ModelInfo(id="test", name="test")]

    async def think(self, config, prompt):
        self.prompts.append(json.loads(prompt))
        index = min(self.calls, len(self.intents) - 1)
        intent = self.intents[index]
        self.calls += 1
        self.called.set()
        return InferenceResult(
            intent.model_dump_json(),
            Usage(input_tokens=3, output_tokens=4, actual_model="test"),
        )


class CountingCognition:
    def __init__(self):
        self.calls = 0
        self.prompts = []
        self.called = asyncio.Event()

    async def status(self):
        return {"connected": True}

    async def models(self):
        return [ModelInfo(id="test", name="test")]

    async def think(self, config, prompt):
        self.calls += 1
        self.prompts.append(json.loads(prompt))
        self.called.set()
        return InferenceResult(
            AgentIntent.bootstrap().model_dump_json(),
            Usage(input_tokens=3, output_tokens=4, actual_model="test"),
        )


def unlimited():
    return RunConfig(
        provider="codex",
        model="test",
        max_calls=0,
        max_tokens=0,
        max_runtime_s=0,
        max_cost_usd=0,
    )


@pytest.mark.asyncio
async def test_motor_keeps_sending_raw_input_while_cognition_is_blocked(tmp_path, store, state):
    bridge = connected(state)
    provider = SlowCognition()
    runtime = AutonomyRuntime(bridge, store, {"codex": provider}, tmp_path / "ml")

    await runtime.start(unlimited())
    await asyncio.wait_for(provider.entered.wait(), timeout=1.0)

    before = len(bridge.transport.sent)
    await asyncio.sleep(0.22)
    after = len(bridge.transport.sent)

    assert runtime.state == "running"
    assert after > before
    active = [row for row in bridge.transport.sent[before:] if row.get("active") is True]
    assert active
    assert any(
        row.get("stick_x") != 0 or row.get("stick_y") != 0 or row.get("buttons") != 0
        for row in active
    )

    # Stop must revoke input without waiting for the blocked provider.
    await asyncio.wait_for(runtime.control("stop"), timeout=0.3)
    assert runtime.state == "stopped"
    assert not bridge.authority.enabled
    provider.release.set()


@pytest.mark.asyncio
async def test_stop_neutralizes_realtime_panel_immediately(tmp_path, store, state):
    bridge = connected(state)
    provider = SlowCognition()
    runtime = AutonomyRuntime(bridge, store, {"codex": provider}, tmp_path / "ml")

    await runtime.start(unlimited())
    await asyncio.wait_for(provider.entered.wait(), timeout=1.0)
    await asyncio.sleep(0.12)
    await runtime.control("stop")

    snapshot = runtime.snapshot()
    assert snapshot["input"]["buttons"] == 0
    assert snapshot["input"]["stick_x"] == 0
    assert snapshot["input"]["stick_y"] == 0
    assert snapshot["input"]["button_names"] == []
    provider.release.set()


@pytest.mark.asyncio
async def test_controller_checkpoint_is_reused_between_runs(tmp_path, store, state):
    bridge = connected(state)
    provider = SlowCognition()
    runtime = AutonomyRuntime(bridge, store, {"codex": provider}, tmp_path / "ml")
    checkpoint = tmp_path / "ml" / "raw-controller-ppo-rnd-v3.pt"

    await runtime.start(unlimited())
    await asyncio.wait_for(provider.entered.wait(), timeout=1.0)
    runtime.controller.policy.save()
    assert checkpoint.is_file()
    await runtime.control("stop")
    route_graph = tmp_path / "ml" / "route-graph-v1.json"
    provider.release.set()
    await runtime._settle_previous_controller()
    assert route_graph.is_file()

    second_provider = SlowCognition()
    runtime.providers["codex"] = second_provider
    await runtime.start(unlimited())
    assert runtime.controller.policy.checkpoint == checkpoint
    assert runtime.controller.route_graph.path == route_graph
    assert runtime.controller.route_graph.stats()["nodes"] >= 1
    await runtime.control("stop")
    second_provider.release.set()


def test_contextual_interaction_probes_one_button_and_learns_success(
    tmp_path, state
):
    game = state.model_copy(deep=True)
    game.context_action.code = 1
    game.context_action.label = "open"
    game.context_actor = ActorObservation(
        actor_uid="door-1",
        actor_id=9,
        name="Door",
        category=10,
        category_name="door",
        params=0,
        position=(0.0, 0.0, 20.0),
        distance=20.0,
    )
    bridge = connected(game)
    controller = ContinuousController(
        bridge,
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    sample = {
        "stick": [0.0, 1.0],
        "policy_stick": [0.0, 0.0],
        "buttons": [0.0 for _ in range(9)],
        "log_prob": 0.0,
        "value": 0.0,
        "guidance_strength": 0.9,
        "button_quiet_strength": 0.0,
    }
    setpoint, executed, overridden = controller._interaction_override(
        game,
        {"exit_active": True},
        Setpoint(stick_y=80, reason="ml_policy"),
        sample,
    )

    assert overridden is True
    assert setpoint.reason == "interaction_probe"
    assert setpoint.stick_x == 0
    assert setpoint.stick_y == 0
    assert executed["stick"] == [0.0, 0.0]
    assert sum(1 for value in executed["buttons"] if value > 0.5) == 1
    probe = dict(controller.pending_interaction_probe)
    assert probe["button"] != "START"

    transitioned = game.model_copy(deep=True)
    transitioned.room = 1

    class Reward:
        breakdown = {"new_world_transition": 0.8}

    controller._observe_interaction_outcome(transitioned, Reward())
    assert controller.pending_interaction_probe is None
    assert controller.route_graph.interaction_button(probe["key"]) == probe["button"]


def test_dialogue_close_releases_button_and_blocks_immediate_reentry(
    tmp_path, state
):
    old = state.model_copy(deep=True)
    old.dialogue.active = True
    old.dialogue.can_advance = True
    old.dialogue.text_id = 321
    old.dialogue.text = "Final page"
    old.dialogue.speaker = ActorObservation(
        actor_uid="shopkeeper-1",
        actor_id=77,
        name="Shopkeeper",
        description="Shopkeeper",
        category=5,
        category_name="npc",
        params=0,
        position=(0.0, 0.0, 20.0),
        distance=20.0,
    )

    closed = old.model_copy(deep=True)
    closed.seq += 1
    closed.dialogue.active = False
    closed.dialogue.can_advance = False
    closed.dialogue.text = ""

    controller = ContinuousController(
        connected(closed),
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    controller.last_setpoint = Setpoint(
        buttons=BUTTON_MASKS["A"],
        stick_x=40,
        stick_y=20,
        reason="dialogue_probe",
    )
    controller.last_buttons = tuple(
        1.0 if name == "A" else 0.0
        for name in (
            "A", "B", "Z", "R", "START",
            "C_UP", "C_LEFT", "C_DOWN", "C_RIGHT",
        )
    )
    controller.pending_interaction_probe = {
        "kind": "dialogue",
        "key": "dialogue:advance",
        "button": "A",
        "at": 0.0,
    }

    controller.note_dialogue_closed(old, closed)

    assert controller.dialogue_reentry_guard is not None
    assert controller.last_setpoint.buttons == 0
    assert controller.last_setpoint.reason == "dialogue_disengage"
    assert controller.route_graph.interaction_button("dialogue:advance") == "A"

    sample = {
        "stick": [0.5, -0.25],
        "policy_stick": [0.5, -0.25],
        "buttons": [1.0 for _ in range(9)],
        "log_prob": 0.0,
        "value": 0.0,
        "guidance_strength": 0.8,
        "button_quiet_strength": 0.0,
    }
    guarded, guarded_sample, overridden = controller._dialogue_reentry_override(
        closed,
        Setpoint(
            buttons=0xFFFF,
            stick_x=40,
            stick_y=-20,
            reason="ml_policy",
        ),
        sample,
    )

    assert overridden is True
    assert guarded.reason == "dialogue_disengage"
    assert guarded.buttons == 0
    assert guarded.stick_x == 40
    assert guarded.stick_y == -20
    assert not any(guarded_sample["buttons"])

    escaped = closed.model_copy(deep=True)
    escaped.player.position = (
        closed.player.position[0] + 140.0,
        closed.player.position[1],
        closed.player.position[2],
    )
    released, released_sample, overridden = controller._dialogue_reentry_override(
        escaped,
        Setpoint(buttons=BUTTON_MASKS["A"], stick_x=20, reason="ml_policy"),
        sample,
    )
    assert overridden is False
    assert controller.dialogue_reentry_guard is None
    assert released.buttons == BUTTON_MASKS["A"]
    assert released_sample["buttons"] == sample["buttons"]


def test_room_dwell_prefers_observed_door_without_native_scene_exit(
    tmp_path, state
):
    game = state.model_copy(deep=True)
    game.scene_exits = []
    game.room_actors = [
        ActorObservation(
            actor_uid="room-door",
            actor_id=9,
            name="Door",
            category=10,
            category_name="door",
            params=0,
            position=(160.0, 0.0, 0.0),
            distance=160.0,
        )
    ]
    controller = ContinuousController(
        connected(game),
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    controller.reward_tracker.local_dwell_seconds = 25.0

    assert controller.route_graph.has_observed_escape(game) is True
    assert controller._should_prefer_observed_exit(
        game,
        actor_is_door=False,
    ) is True


def test_extreme_room_dwell_stops_redundant_ppo_training(
    tmp_path, state
):
    controller = ContinuousController(
        connected(state),
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    normal_sample = {"guidance_strength": 0.8}

    assert controller._ppo_transition_trainable(
        interaction_override=False,
        guidance={"exit_active": False},
        sample=normal_sample,
    ) is True

    controller.reward_tracker.local_dwell_seconds = 181.0
    assert controller._ppo_transition_trainable(
        interaction_override=False,
        guidance={"exit_active": False},
        sample=normal_sample,
    ) is False

    controller.reward_tracker.local_dwell_seconds = 0.0
    assert controller._ppo_transition_trainable(
        interaction_override=False,
        guidance={"exit_active": True},
        sample={"guidance_strength": 1.0},
    ) is False


def test_trackable_objective_prefers_exit_after_persistent_room_failures(
    tmp_path, state
):
    game = state.model_copy(deep=True)
    game.scene_exits = [
        SceneExitObservation(
            exit_index=1,
            entrance_index=2,
            position=(100.0, 0.0, 0.0),
            samples=4,
            direct_reachable=True,
        )
    ]
    controller = ContinuousController(
        connected(game),
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    controller.set_intent(
        AgentIntent(
            objective="Obtain the Kokiri Sword",
            completion=ObjectiveCompletion(
                kind="equipment",
                name="Kokiri Sword",
            ),
            summary="Obtain the Kokiri Sword",
            mode="explore",
            horizon_ms=60000,
        )
    )
    age = "adult" if game.player.age == "adult" else "child"
    prefix = (
        f"{int(bool(game.mirrored_world))}:{age}:"
        f"{game.scene}:{game.room}:"
    )
    controller.route_graph.frontier_failures[prefix + "1:0:0"] = 3
    controller.route_graph.frontier_failures[prefix + "2:0:0"] = 3

    assert controller.reward_tracker.local_dwell_seconds == 0.0
    assert controller.route_graph.room_failure_pressure(game) >= 6
    assert controller._should_prefer_observed_exit(
        game,
        actor_is_door=False,
    ) is True

    controller.set_intent(AgentIntent.bootstrap())
    assert controller._should_prefer_observed_exit(
        game,
        actor_is_door=False,
    ) is False


def test_active_linear_dialogue_neutralizes_movement_and_learns_advance(
    tmp_path, state
):
    game = state.model_copy(deep=True)
    game.dialogue.active = True
    game.dialogue.can_advance = True
    game.dialogue.choice_count = 0
    game.dialogue.text_id = 100
    game.dialogue.text = "Page one"

    bridge = connected(game)
    controller = ContinuousController(
        bridge,
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    sample = {
        "stick": [-0.8, 0.4],
        "policy_stick": [-0.8, 0.4],
        "buttons": [0.0, 0.0, 1.0] + [0.0 for _ in range(6)],
        "log_prob": 0.0,
        "value": 0.0,
        "guidance_strength": 0.8,
        "button_quiet_strength": 0.0,
    }
    setpoint, executed, overridden = controller._dialogue_override(
        game,
        Setpoint(buttons=0x2000, stick_x=-64, stick_y=32, reason="ml_policy"),
        sample,
    )

    assert overridden is True
    assert setpoint.reason == "dialogue_probe"
    assert setpoint.stick_x == 0
    assert setpoint.stick_y == 0
    assert executed["stick"] == [0.0, 0.0]
    assert sum(1 for value in executed["buttons"] if value > 0.5) == 1
    probe = dict(controller.pending_interaction_probe)
    assert probe["kind"] == "dialogue"
    assert probe["button"] != "START"

    advanced = game.model_copy(deep=True)
    advanced.dialogue.text_id = 101
    advanced.dialogue.text = "Page two"

    class Reward:
        breakdown = {}

    controller._observe_interaction_outcome(advanced, Reward())
    assert controller.pending_interaction_probe is None
    assert (
        controller.route_graph.interaction_button("dialogue:advance")
        == probe["button"]
    )

    controller.interaction_last_probe_at = 0.0
    replay, replay_sample, replay_override = controller._dialogue_override(
        advanced,
        Setpoint(stick_x=80, reason="ml_policy"),
        sample,
    )
    assert replay_override is True
    assert replay.reason == "dialogue_memory"
    assert replay.stick_x == 0
    assert replay.stick_y == 0
    assert replay.buttons == BUTTON_MASKS[probe["button"]]
    assert replay_sample["stick"] == [0.0, 0.0]


def test_active_dialogue_that_cannot_advance_holds_all_inputs(tmp_path, state):
    game = state.model_copy(deep=True)
    game.dialogue.active = True
    game.dialogue.can_advance = False
    game.dialogue.choice_count = 0
    game.dialogue.text_id = 200
    game.dialogue.text = "Waiting"

    controller = ContinuousController(
        connected(game),
        tmp_path / "policy.pt",
        training_enabled=True,
    )
    sample = {
        "stick": [1.0, -1.0],
        "policy_stick": [1.0, -1.0],
        "buttons": [1.0 for _ in range(9)],
        "log_prob": 0.0,
        "value": 0.0,
        "guidance_strength": 0.0,
        "button_quiet_strength": 0.0,
    }
    setpoint, executed, overridden = controller._dialogue_override(
        game,
        Setpoint(buttons=0xFFFF, stick_x=80, stick_y=-80, reason="ml_policy"),
        sample,
    )

    assert overridden is True
    assert setpoint.reason == "dialogue_wait"
    assert setpoint.buttons == 0
    assert setpoint.stick_x == 0
    assert setpoint.stick_y == 0
    assert executed["stick"] == [0.0, 0.0]
    assert not any(executed["buttons"])


@pytest.mark.asyncio
async def test_route_memory_is_saved_during_active_long_run(
    tmp_path, store, state, monkeypatch
):
    monkeypatch.setattr(
        ContinuousController,
        "route_save_interval_s",
        0.02,
    )
    bridge = connected(state)
    provider = CountingCognition()
    runtime = AutonomyRuntime(
        bridge,
        store,
        {"codex": provider},
        tmp_path / "ml",
    )

    await runtime.start(unlimited())
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    route_graph = tmp_path / "ml" / "route-graph-v1.json"

    for _ in range(30):
        if route_graph.is_file():
            break
        await asyncio.sleep(0.01)

    assert runtime.state == "running"
    assert route_graph.is_file()
    assert runtime.controller.route_graph.stats()["nodes"] >= 1
    assert runtime.controller.route_save_error == ""

    await runtime.control("stop")


def test_paused_observations_are_not_promoted_to_memory(tmp_path, store, state):
    bridge = connected(state)
    runtime = AutonomyRuntime(bridge, store, {}, tmp_path / "ml")
    runtime.run_id = store.new_run(
        unlimited().model_dump(), "soh", "paused-isolation"
    )
    runtime.state = "paused"

    changed = state.model_copy(deep=True)
    changed.seq += 1
    changed.dialogue.active = True
    changed.dialogue.text_id = 999
    changed.dialogue.text = "Human-triggered paused dialogue"

    runtime.on_state(changed, state)
    assert store.recall(runtime.namespace) == []


@pytest.mark.asyncio
async def test_trackable_objective_stays_locked_until_completion(
    tmp_path, store, state, monkeypatch
):
    import zelda_ai.autonomy.runtime as runtime_module
    from zelda_ai.models import EquipmentObservation

    monkeypatch.setattr(runtime_module, "COGNITION_EVENT_DEBOUNCE_S", 0.01)
    monkeypatch.setattr(runtime_module, "COGNITION_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(runtime_module, "COGNITION_IDLE_POLL_S", 0.01)

    sword = AgentIntent(
        objective="Obtain the Kokiri Sword",
        completion=ObjectiveCompletion(
            kind="equipment",
            name="Kokiri Sword",
        ),
        summary="Obtain the Kokiri Sword",
        mode="navigate",
        # Even if cognition tries to micromanage a waypoint, the objective lock
        # strips it and lets the local motor own navigation.
        target_position=(999.0, 0.0, 999.0),
        horizon_ms=60000,
    )
    shield = AgentIntent(
        objective="Obtain the Deku Shield",
        completion=ObjectiveCompletion(
            kind="equipment",
            name="Deku Shield",
        ),
        summary="Obtain the Deku Shield",
        mode="explore",
        horizon_ms=60000,
    )
    provider = ObjectiveSequenceCognition([sword, shield])
    bridge = connected(state)
    runtime = AutonomyRuntime(
        bridge,
        store,
        {"codex": provider},
        tmp_path / "ml",
    )

    await runtime.start(unlimited())
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    for _ in range(100):
        if runtime.objective_tracker.trackable:
            break
        await asyncio.sleep(0.01)

    assert provider.calls == 1
    assert runtime.objective_tracker.intent.objective == "Obtain the Kokiri Sword"
    assert runtime.thought == "Obtain the Kokiri Sword"
    assert runtime.controller.intent.objective == "Obtain the Kokiri Sword"
    assert runtime.controller.intent.mode == "explore"
    assert runtime.controller.intent.target_position is None

    # Strategic noise that previously caused replanning must not replace the
    # objective. This includes room changes, hard stuck requests and unrelated
    # durable progress.
    transitioned = state.model_copy(deep=True)
    transitioned.seq += 1
    transitioned.room += 1
    transitioned.scene_epoch += 1
    bridge.state = transitioned
    runtime.on_state(transitioned, state)
    runtime._request_cognition("local_area_stuck")

    unrelated = transitioned.model_copy(deep=True)
    unrelated.seq += 1
    unrelated.progress.quest_items = ["Unrelated Quest Item"]
    bridge.state = unrelated
    runtime.on_state(unrelated, transitioned)
    await asyncio.sleep(0.08)

    assert provider.calls == 1
    assert runtime.objective_tracker.intent.objective == "Obtain the Kokiri Sword"
    assert runtime.objective_replan_suppressed >= 2

    # Only the measurable completion predicate unlocks the next model call.
    acquired = unrelated.model_copy(deep=True)
    acquired.seq += 1
    acquired.progress.equipment = [
        EquipmentObservation(
            item_id=1,
            name="Kokiri Sword",
            equipment_type="sword",
            value=1,
            equipped=False,
        )
    ]
    acquired.progress.owned_equipment = ["Kokiri Sword"]
    bridge.state = acquired
    provider.called.clear()
    runtime.on_state(acquired, unrelated)

    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    for _ in range(100):
        if (
            runtime.objective_tracker.intent
            and runtime.objective_tracker.intent.objective
            == "Obtain the Deku Shield"
        ):
            break
        await asyncio.sleep(0.01)

    assert provider.calls == 2
    assert provider.prompts[-1]["trigger_reasons"] == ["objective_completed"]
    assert runtime.objective_tracker.completed_count == 1
    assert runtime.objective_tracker.intent.objective == "Obtain the Deku Shield"
    assert runtime.thought == "Obtain the Deku Shield"

    await runtime.control("stop")


@pytest.mark.asyncio
async def test_cognition_is_sparse_and_ignores_context_noise(tmp_path, store, state, monkeypatch):
    import zelda_ai.autonomy.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "COGNITION_EVENT_DEBOUNCE_S", 0.01)
    monkeypatch.setattr(runtime_module, "COGNITION_MIN_INTERVAL_S", 0.0)

    bridge = connected(state)
    provider = CountingCognition()
    runtime = AutonomyRuntime(bridge, store, {"codex": provider}, tmp_path / "ml")
    await runtime.start(unlimited())

    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    assert provider.calls == 1
    assert provider.prompts[0]["trigger_reasons"] == ["run_started"]

    # Transient contextual affordance changes used to trigger expensive replans.
    old = bridge.state
    noisy = old.model_copy(deep=True)
    noisy.seq += 1
    noisy.context_action.code = min(255, old.context_action.code + 1)
    noisy.context_action.label = "open"
    bridge.state = noisy
    runtime.on_state(noisy, old)
    await asyncio.sleep(0.08)
    assert provider.calls == 1
    assert not runtime.cognition_trigger.is_set()

    # There is no horizon/periodic refresh anymore.
    await asyncio.sleep(0.12)
    assert provider.calls == 1

    # A real world transition is strategic and causes exactly one replan.
    provider.called.clear()
    transitioned = noisy.model_copy(deep=True)
    transitioned.seq += 1
    transitioned.room += 1
    transitioned.scene_epoch += 1
    bridge.state = transitioned
    runtime.on_state(transitioned, noisy)
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    assert provider.calls == 2
    assert "world_transition" in provider.prompts[-1]["trigger_reasons"]

    await runtime.control("stop")


@pytest.mark.asyncio
async def test_hard_blocked_guidance_triggers_early_sparse_replan(
    tmp_path, store, state, monkeypatch
):
    import zelda_ai.autonomy.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "COGNITION_EVENT_DEBOUNCE_S", 0.01)
    monkeypatch.setattr(runtime_module, "COGNITION_MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(runtime_module, "COGNITION_BLOCKED_AFTER_S", 0.03)
    monkeypatch.setattr(runtime_module, "COGNITION_BLOCKED_COOLDOWN_S", 0.0)
    monkeypatch.setattr(runtime_module, "COGNITION_IDLE_POLL_S", 0.01)

    blocked_state = state.model_copy(deep=True)
    blocked_state.navigation_probes = [
        NavigationProbe(
            direction="forward",
            distance=70.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=True,
            wall_distance=25.0,
        )
    ]
    bridge = connected(blocked_state)
    provider = CountingCognition()
    runtime = AutonomyRuntime(
        bridge,
        store,
        {"codex": provider},
        tmp_path / "ml",
    )
    await runtime.start(unlimited())
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    assert provider.calls == 1

    # CountingCognition sets its event when think() starts, before the runtime
    # necessarily applies the returned initial intent. Wait for that first turn
    # to settle so it cannot overwrite the explicit blocked target below.
    for _ in range(100):
        if runtime.cognition_state == "acting" and runtime.last_cognition_at > 0:
            break
        await asyncio.sleep(0.01)
    assert runtime.cognition_state == "acting"

    runtime.controller.set_intent(
        AgentIntent.bootstrap().model_copy(update={
            "mode": "navigate",
            "target_position": (0.0, 0.0, 500.0),
            "objective": "Reach the observed point.",
            "summary": "Try the observed point.",
        })
    )
    provider.called.clear()

    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    assert provider.calls == 2
    assert "guidance_blocked" in provider.prompts[-1]["trigger_reasons"]
    assert provider.prompts[-1]["motor"]["guidance"]["blocked"] is True
    assert provider.prompts[-1]["motor"]["guidance"]["detour"] is None

    await runtime.control("stop")


@pytest.mark.asyncio
async def test_linear_signpost_does_not_replan_unless_intent_waited_for_it(tmp_path, store, state, monkeypatch):
    import zelda_ai.autonomy.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "COGNITION_EVENT_DEBOUNCE_S", 0.01)
    monkeypatch.setattr(runtime_module, "COGNITION_MIN_INTERVAL_S", 0.0)

    bridge = connected(state)
    provider = CountingCognition()
    runtime = AutonomyRuntime(bridge, store, {"codex": provider}, tmp_path / "ml")
    await runtime.start(unlimited())
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    assert provider.calls == 1

    signpost = state.model_copy(deep=True)
    signpost.seq += 1
    signpost.dialogue.active = True
    signpost.dialogue.text_id = 123
    signpost.dialogue.text = "A signpost"
    signpost.dialogue.choice_count = 0
    signpost.dialogue.speaker = None
    bridge.state = signpost
    runtime.on_state(signpost, state)
    await asyncio.sleep(0.05)
    assert provider.calls == 1

    closed = signpost.model_copy(deep=True)
    closed.seq += 1
    closed.dialogue.active = False
    bridge.state = closed
    runtime.on_state(closed, signpost)
    await asyncio.sleep(0.05)
    assert provider.calls == 1

    # If cognition explicitly chose observe/dialogue while text was active,
    # resolving it gets one call so that "wait" cannot become permanent.
    runtime.controller.set_intent(
        AgentIntent.bootstrap().model_copy(update={"mode": "observe"})
    )
    reopened = closed.model_copy(deep=True)
    reopened.seq += 1
    reopened.dialogue.active = True
    reopened.dialogue.text_id = 124
    reopened.dialogue.text = "Another signpost"
    bridge.state = reopened
    runtime.on_state(reopened, closed)
    resolved = reopened.model_copy(deep=True)
    resolved.seq += 1
    resolved.dialogue.active = False
    bridge.state = resolved
    provider.called.clear()
    runtime.on_state(resolved, reopened)
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    assert provider.calls == 2
    assert "dialogue_resolved" in provider.prompts[-1]["trigger_reasons"]

    await runtime.control("stop")


@pytest.mark.asyncio
async def test_completion_creates_champion_and_evaluation_keeps_it_frozen(
    tmp_path, store, state
):
    bridge = connected(state)
    provider = CountingCognition()
    runtime = AutonomyRuntime(
        bridge,
        store,
        {"codex": provider},
        tmp_path / "ml",
    )

    await runtime.start(unlimited())
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    await runtime.halt("completed", "game_completed")

    catalog = runtime.refresh_champions()
    assert catalog["count"] == 1
    champion = catalog["latest"]
    assert champion["id"] == "completion-0001"
    assert champion["run_id"] == runtime.run_id

    champion_path = tmp_path / "ml" / "champions" / "completion-0001.pt"
    assert champion_path.is_file()
    frozen_bytes = champion_path.read_bytes()
    assert champion["route_graph_file"] == "completion-0001.routes.json"
    champion_routes = (
        tmp_path / "ml" / "champions" / champion["route_graph_file"]
    )
    assert champion_routes.is_file()
    frozen_route_bytes = champion_routes.read_bytes()
    assert champion["room_map_file"] == "completion-0001.room-map.json"
    champion_room_map = (
        tmp_path / "ml" / "champions" / champion["room_map_file"]
    )
    assert champion_room_map.is_file()
    frozen_room_map_bytes = champion_room_map.read_bytes()

    evaluation_provider = CountingCognition()
    runtime.providers["codex"] = evaluation_provider
    evaluation = unlimited().model_copy(
        update={"run_mode": "evaluation"}
    )
    await runtime.start(evaluation)
    await asyncio.wait_for(evaluation_provider.called.wait(), timeout=1.0)

    assert runtime.config.run_mode == "evaluation"
    assert runtime.config.champion_id == "completion-0001"
    assert runtime.controller.training_enabled is False
    assert runtime.controller.policy.checkpoint == champion_path
    assert runtime.controller.route_graph.writable is False
    assert runtime.controller.route_graph.path == champion_routes
    assert runtime.controller.room_map.writable is False
    assert runtime.controller.room_map.path == champion_room_map
    snapshot = runtime.snapshot()
    assert snapshot["run_mode"] == "evaluation"
    assert snapshot["active_champion"]["id"] == "completion-0001"
    assert snapshot["learning"]["training_enabled"] is False
    assert snapshot["learning"]["run_updates"] == 0

    await asyncio.sleep(0.2)
    await runtime.control("stop")
    assert champion_path.read_bytes() == frozen_bytes
    assert champion_routes.read_bytes() == frozen_route_bytes
    assert champion_room_map.read_bytes() == frozen_room_map_bytes


@pytest.mark.asyncio
async def test_new_run_waits_for_completion_champion_capture(
    tmp_path, store, state
):
    bridge = connected(state)
    provider = CountingCognition()
    runtime = AutonomyRuntime(
        bridge,
        store,
        {"codex": provider},
        tmp_path / "ml",
    )
    await runtime.start(unlimited())
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)

    original_capture = runtime.champions.capture
    capture_started = threading.Event()
    release_capture = threading.Event()

    def slow_capture(
        source,
        metadata,
        route_graph_source=None,
        room_map_source=None,
    ):
        capture_started.set()
        if not release_capture.wait(timeout=2.0):
            raise RuntimeError("test capture release timed out")
        return original_capture(
            source,
            metadata,
            route_graph_source,
            room_map_source,
        )

    runtime.champions.capture = slow_capture
    completion = asyncio.create_task(
        runtime.halt("completed", "game_completed")
    )
    assert await asyncio.to_thread(capture_started.wait, 1.0)
    assert runtime.snapshot()["champions"]["capture_pending"] is True

    next_start = asyncio.create_task(runtime.start(unlimited()))
    await asyncio.sleep(0.05)
    assert next_start.done() is False

    release_capture.set()
    await asyncio.wait_for(completion, timeout=2.0)
    await asyncio.wait_for(next_start, timeout=2.0)
    assert runtime.snapshot()["champions"]["capture_pending"] is False
    assert runtime.state == "running"

    await runtime.control("stop")


def test_local_area_dwell_overrides_recent_micro_progress_for_stuck_detection():
    age, reason = AutonomyRuntime._stuck_signal({
        "seconds_since_useful_progress": 8.0,
        "exploration": {"local_dwell_seconds": 125.0},
    })
    assert age == 125.0
    assert reason == "local_area_stuck"

    age, reason = AutonomyRuntime._stuck_signal({
        "seconds_since_useful_progress": 140.0,
        "exploration": {"local_dwell_seconds": 20.0},
    })
    assert age == 140.0
    assert reason == "motor_stuck"


@pytest.mark.asyncio
async def test_reaching_structured_waypoint_triggers_one_sparse_replan(
    tmp_path, store, state, monkeypatch
):
    import zelda_ai.autonomy.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "COGNITION_EVENT_DEBOUNCE_S", 0.01)
    monkeypatch.setattr(runtime_module, "COGNITION_MIN_INTERVAL_S", 0.0)

    bridge = connected(state)
    provider = CountingCognition()
    runtime = AutonomyRuntime(
        bridge,
        store,
        {"codex": provider},
        tmp_path / "ml",
    )
    await runtime.start(unlimited())
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    assert provider.calls == 1

    target_intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (50.0, 0.0, 0.0),
        "objective": "Reach observed waypoint.",
        "summary": "Move to observed waypoint.",
    })
    runtime.controller.set_intent(target_intent)

    reached = state.model_copy(deep=True)
    reached.seq += 1
    reached.player.position = (10.0, 0.0, 0.0)
    bridge.state = reached
    provider.called.clear()
    runtime.on_state(reached, state)
    await asyncio.wait_for(provider.called.wait(), timeout=1.0)
    assert provider.calls == 2
    assert "intent_target_reached" in provider.prompts[-1]["trigger_reasons"]

    # The same completed waypoint is deduplicated.
    again = reached.model_copy(deep=True)
    again.seq += 1
    bridge.state = again
    runtime.on_state(again, reached)
    await asyncio.sleep(0.08)
    assert provider.calls == 2

    await runtime.control("stop")
