import asyncio
import json
import threading
import time

import pytest

from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.autonomy.runtime import AutonomyRuntime
from zelda_ai.bridge import Bridge
from zelda_ai.models import ModelInfo, NavigationProbe, RunConfig, Usage
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
    checkpoint = tmp_path / "ml" / "raw-controller-ppo-rnd-v2.pt"

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
    snapshot = runtime.snapshot()
    assert snapshot["run_mode"] == "evaluation"
    assert snapshot["active_champion"]["id"] == "completion-0001"
    assert snapshot["learning"]["training_enabled"] is False
    assert snapshot["learning"]["run_updates"] == 0

    await asyncio.sleep(0.2)
    await runtime.control("stop")
    assert champion_path.read_bytes() == frozen_bytes
    assert champion_routes.read_bytes() == frozen_route_bytes


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

    def slow_capture(source, metadata):
        capture_started.set()
        if not release_capture.wait(timeout=2.0):
            raise RuntimeError("test capture release timed out")
        return original_capture(source, metadata)

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
