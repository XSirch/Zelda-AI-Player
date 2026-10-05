import asyncio
import json
import time
from types import SimpleNamespace

from zelda_ai import g1
from zelda_ai.bridge import Bridge
from zelda_ai.laya_ladder_curriculum import warm_reference
from zelda_ai.models import InputReceipt


def test_cold_controller_loading_does_not_block_native_heartbeat(state, tmp_path, monkeypatch):
    async def scenario():
        bridge = Bridge("unit-test-only", allow_simulator=False)
        bridge.state = state
        bridge.last_seen = time.monotonic()
        observed = []

        class ColdController:
            def __init__(self, actual_bridge, *args, **kwargs):
                # Represents the age accumulated by a synchronous cold load.
                # A live receiver can refresh it only if its loop remains free.
                actual_bridge.last_seen = time.monotonic() - 3
                time.sleep(.03)
                self.executor = SimpleNamespace(trace=[], reason=None)

            def set_intent(self, intent):
                self.intent = intent

            async def run(self, active, publish):
                observed.append(active())

            def telemetry(self):
                return {"learning": {"run_updates": 0}}

        async def heartbeat():
            while True:
                bridge.last_seen = time.monotonic()
                await asyncio.sleep(.001)

        monkeypatch.setattr(g1, "ContinuousController", ColdController)
        receiver = asyncio.create_task(heartbeat())
        try:
            await g1._motor_episode(bridge, tmp_path, tmp_path, deadline=time.monotonic() + 5)
            assert observed == [True]
        finally:
            receiver.cancel()
            await asyncio.gather(receiver, return_exceptions=True)
    asyncio.run(scenario())


def test_reference_warmup_has_no_native_peer_training_or_inputs(tmp_path, monkeypatch):
    from zelda_ai import surface_curriculum

    observed = []

    def constructor(bridge, *args, **kwargs):
        observed.append((bridge.state, bridge.peer, kwargs["training_enabled"]))
        return SimpleNamespace(training_enabled=False, policy=SimpleNamespace(updates=7), starting_updates=7)

    monkeypatch.setattr(surface_curriculum, "ContinuousController", constructor)
    result = asyncio.run(warm_reference(tmp_path))
    assert observed == [(None, None, False)]
    assert result["native_commands"] == result["run_updates"] == 0
    assert result["elapsed_ms"] >= 0


def test_motor_exception_preserves_consumed_receipts_without_crediting_a_crossing(state, tmp_path, monkeypatch):
    async def scenario():
        bridge = Bridge("unit-test-only", allow_simulator=False)
        bridge.state, bridge.last_seen = state, time.monotonic()

        class FailingController:
            def __init__(self, *args, **kwargs):
                self.executor = SimpleNamespace(trace=[], reason=None)

            def set_intent(self, intent):
                self.intent = intent

            async def run(self, active, publish):
                bridge.receipts[1] = InputReceipt(seq=1, owner_epoch=0, status="consumed", first_tick=2,
                                                  last_tick=2, pressed=0, released=0)
                state.scene += 1
                state.scene_epoch += 1
                raise RuntimeError("state_feedback_timeout")

            def telemetry(self):
                return {"learning": {"run_updates": 0}}

        monkeypatch.setattr(g1, "ContinuousController", FailingController)
        result = await g1._motor_episode(bridge, tmp_path, tmp_path, deadline=time.monotonic() + 5)
        assert not result["settled_crossing"]
        assert result["consumed_commands"] == 1
        assert result["reason"] == "RuntimeError:state_feedback_timeout"
        saved = json.loads((tmp_path / "motor.json").read_text(encoding="utf-8"))
        assert saved["receipts"][0]["seq"] == 1
        assert saved["motor_failure"] == result["reason"]
    asyncio.run(scenario())
