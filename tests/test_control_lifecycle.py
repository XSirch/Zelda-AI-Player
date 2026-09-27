import asyncio
import json
import time

import pytest

from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.autonomy.runtime import AutonomyRuntime
from zelda_ai.bridge import Bridge
from zelda_ai.models import ModelInfo, RunConfig, Usage
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
    provider.release.set()
    await runtime._settle_previous_controller()

    second_provider = SlowCognition()
    runtime.providers["codex"] = second_provider
    await runtime.start(unlimited())
    assert runtime.controller.policy.checkpoint == checkpoint
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
