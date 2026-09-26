import asyncio
import time

import pytest
from fastapi.testclient import TestClient

from conftest import free_udp_port, packet
from test_bridge_native import Transport
from zelda_ai.app import create_app
from zelda_ai.bridge import Bridge
from zelda_ai.config import Settings
from zelda_ai.models import RunConfig, SwitchConfig
from zelda_ai.runtime import Runtime
from zelda_ai.simulator import DemoProvider
from zelda_ai.store import Store


def test_metrics_and_memory_isolation(store):
    config = RunConfig(provider="codex", model="m").model_dump()
    run = store.new_run(config, "soh", "fingerprint")
    segment = store.segment(run, config, "n1")
    call = store.begin_call(run, segment, {})
    store.finish_call(call, status="completed", usage={"input_tokens": 100, "cached_input_tokens": 90,
        "output_tokens": 40, "reasoning_output_tokens": 30}, latency_ms=50)
    metrics = store.metrics(run)
    assert metrics["total_tokens"] == 140 and metrics["cost_usd"] is None
    store.remember("n1", 85, "A tentative route")
    store.remember("n1", 85, "A tentative route")
    assert len(store.recall("n1", 85)) == 1
    assert store.recall("n2", 85) == []


def test_restart_does_not_resume_run(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.sqlite3"
    first = Store(url)
    config = RunConfig(provider="demo", model="deterministic-demo").model_dump()
    run = first.new_run(config, "simulator", "hash")
    first.close()
    second = Store(url)
    assert second.detail(run)["status"] == "interrupted"
    second.close()


@pytest.mark.asyncio
async def test_model_switch_queued_between_decisions(store, state):
    bridge = Bridge("x" * 32, True)
    bridge.connection_made(Transport())
    bridge.datagram_received(packet(state, source="simulator"), ("127.0.0.1", 5000))
    runtime = Runtime(bridge, store, {"demo": DemoProvider()})
    runtime.config = RunConfig(provider="demo", model="deterministic-demo")
    runtime.run_id = store.new_run(runtime.config.model_dump(), "simulator", "hash")
    runtime.namespace = runtime.new_namespace(runtime.config)
    runtime.segment_id = store.segment(runtime.run_id, runtime.config.model_dump(), runtime.namespace)
    runtime.state = "running"
    old = runtime.segment_id
    await runtime.switch(SwitchConfig(provider="demo", model="deterministic-demo"))
    assert runtime.segment_id == old and runtime.pending_switch
    runtime.apply_switch()
    assert runtime.segment_id != old and store.detail(runtime.run_id)["mixed"]
    assert len(store.detail(runtime.run_id)["segments"]) == 2
    await runtime.halt("stopped", "test")


def test_api_controls_security_and_end_to_end_demo(tmp_path):
    settings = Settings(data_dir=tmp_path, bridge_port=free_udp_port(), allow_simulator=True,
        codex_command="codex-test-not-installed", openrouter_api_key="")
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        boot = client.get("/api/bootstrap").json()
        headers = {"X-Zelda-Session": boot["session_token"]}
        assert client.post("/api/control/pause", json={}).status_code == 403
        assert client.post("/api/diagnostics/input", json={"action": "tap_a"}).status_code == 403
        assert client.get("/api/status", headers={"Origin": "https://attacker.invalid"}).status_code == 403
        assert client.get("/api/status", headers={"Host": "attacker.invalid"}).status_code == 400
        for _ in range(30):
            if client.get("/api/status").json()["bridge"]["connected"]: break
            time.sleep(.02)
        assert client.get("/api/status").json()["bridge"]["connected"]
        legacy_diag = client.post("/api/diagnostics/input", json={"action": "tap_a"}, headers=headers)
        assert legacy_diag.status_code == 400 and "BRIDGE V2" in legacy_diag.text
        payload = RunConfig(provider="demo", model="deterministic-demo", max_calls=2).model_dump()
        start = client.post("/api/runs", json=payload, headers=headers)
        assert start.status_code == 200, start.text
        run_id = start.json()["run_id"]
        for _ in range(100):
            status = client.get("/api/status").json()
            if status["status"] == "paused": break
            time.sleep(.05)
        assert status["reason"] == "call_budget_reached", status
        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["source"] == "simulator" and detail["metrics"]["calls"] == 2
        assert all(c["status"] == "completed" for c in detail["calls"])
        assert detail["metrics"]["known_cost_usd"] == 0
        assert any(e["kind"] == "skill_result" for e in detail["events"])
        assert client.post("/api/hints", json={"text": "Try right"}, headers=headers).status_code == 200
        assert client.get(f"/api/runs/{run_id}").json()["assisted"]
        assert client.post("/api/control/stop", json={}, headers=headers).status_code == 200


def test_semantic_traversal_failure_does_not_increase_stuck_score(store, state):
    from zelda_ai.models import Decision, SkillArgs
    runtime = Runtime(Bridge("x" * 32, True), store, {})
    runtime.stuck_score = 5
    decision = Decision(goal="descend", summary="go down", skill="traverse",
        args=SkillArgs(direction="down", duration_ms=8000, strength=.7,
            slot=None, choice_index=None, song=None, target_actor_id=None,
            target_actor_uid=None, target_actor_params=None, target_position=None,
            stop_distance=None, item_id=None),
        memory_note=None)
    runtime._update_stuck(decision, {"status": "failed", "reason": "traversal_timeout"})
    assert runtime.stuck_score == 3
    assert not any(row["kind"] == "stuck_detected" for row in runtime.recent)


@pytest.mark.asyncio
async def test_auto_unstick_skips_semantic_traversal_failure(store, state):
    bridge = Bridge("x" * 32, True)
    runtime = Runtime(bridge, store, {})
    runtime.stuck_score = 8
    runtime.last_result = {"status": "failed", "reason": "traversal_affordance_lost"}
    recovered = await runtime._auto_unstick(state)
    assert recovered is False
    assert runtime.stuck_score == 6


def test_checkpoint_repeat_block_does_not_increase_stuck_score(store, state):
    from zelda_ai.models import Decision, SkillArgs
    runtime = Runtime(Bridge("x" * 32, True), store, {})
    runtime.stuck_score = 7
    decision = Decision(goal="repeat", summary="repeat", skill="talk_to_actor",
        args=SkillArgs(direction=None, duration_ms=5000, strength=.7,
            slot=None, choice_index=None, song=None, target_actor_id=100,
            target_actor_uid=None, target_actor_params=0, target_position=None,
            stop_distance=None, item_id=None),
        memory_note=None)
    runtime._update_stuck(decision, {"status": "failed", "reason": "checkpoint_repeat_blocked"})
    assert runtime.stuck_score == 5
    assert not any(row["kind"] == "stuck_detected" for row in runtime.recent)


def test_idle_checkpoint_recomputes_from_new_observed_state(store, state):
    bridge = Bridge("x" * 32, True)
    runtime = Runtime(bridge, store, {})
    runtime.state = "stopped"

    before = type(state).model_validate({
        **state.model_dump(),
        "capabilities": [*state.capabilities, "story_progress_v1"],
        "scene_name": "Kokiri Forest",
        "progress": {
            **state.progress.model_dump(),
            "story_flags": {"greeted_by_saria": False},
        },
    })
    runtime.on_state(before, None)
    assert runtime.checkpoint_plan["current"]["id"] == "greet_saria_once"

    after = type(state).model_validate({
        **before.model_dump(),
        "seq": before.seq + 1,
        "progress": {
            **before.progress.model_dump(),
            "story_flags": {"greeted_by_saria": True},
        },
    })
    runtime.on_state(after, before)
    assert runtime.checkpoint_plan["current"]["id"] == "obtain_kokiri_sword"


def test_snapshot_hides_checkpoint_when_bridge_disconnected(store, state):
    bridge = Bridge("x" * 32, True)
    runtime = Runtime(bridge, store, {})
    runtime.checkpoint_plan = {"plan_id": "stale", "current": {"id": "stale"}}
    assert runtime.snapshot()["checkpoint_plan"] is None


def test_offmesh_jump_failure_is_semantic_replan_not_stuck(store, state):
    from zelda_ai.models import Decision, SkillArgs
    runtime = Runtime(Bridge("x" * 32, True), store, {})
    runtime.stuck_score = 6
    decision = Decision(goal="cross gap", summary="cross gap", skill="navigate_to",
        args=SkillArgs(direction=None, duration_ms=8000, strength=.8,
            slot=None, choice_index=None, song=None, target_actor_id=None,
            target_actor_uid=None, target_actor_params=None,
            target_position=[300.0, 0.0, 0.0], stop_distance=35.0, item_id=None),
        memory_note=None)
    runtime._update_stuck(decision, {"status": "failed", "reason": "offmesh_jump_not_triggered"})
    assert runtime.stuck_score == 4
    assert not any(row["kind"] == "stuck_detected" for row in runtime.recent)
