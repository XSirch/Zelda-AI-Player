import time

from fastapi.testclient import TestClient

from conftest import free_udp_port
from zelda_ai.app import create_app
from zelda_ai.config import Settings
from zelda_ai.models import RunConfig
from zelda_ai.store import Store


def test_metrics_and_memory_isolation(store):
    config = RunConfig(provider="codex", model="m").model_dump()
    run = store.new_run(config, "soh", "fingerprint")
    segment = store.segment(run, config, "n1")
    call = store.begin_call(run, segment, {})
    store.finish_call(call, status="completed", usage={
        "input_tokens": 100,
        "cached_input_tokens": 90,
        "output_tokens": 40,
        "reasoning_output_tokens": 30,
    }, latency_ms=50)
    metrics = store.metrics(run)
    assert metrics["input_tokens"] == 100
    assert metrics["output_tokens"] == 40
    assert metrics["total_tokens"] == 140 and metrics["cost_usd"] is None
    assert metrics["usage_by_model"][0]["provider"] == "codex"
    assert metrics["usage_by_model"][0]["model"] == "m"
    assert metrics["usage_by_model"][0]["input_tokens"] == 100
    assert metrics["usage_by_model"][0]["output_tokens"] == 40
    assert metrics["usage_by_model"][0]["cost_usd"] is None

    time.sleep(.01)
    store.update_run(run, status="completed", reason="game_completed")
    store.end_segments(run)
    final_metrics = store.metrics(run)
    assert final_metrics["ended_at"] is not None
    assert final_metrics["elapsed_s"] >= 0
    final_elapsed = final_metrics["elapsed_s"]
    time.sleep(.01)
    assert store.metrics(run)["elapsed_s"] == final_elapsed

    store.remember("n1", 85, "A tentative route")
    store.remember("n1", 85, "A tentative route")
    assert len(store.recall("n1", 85)) == 1
    assert store.recall("n2", 85) == []


def test_mixed_run_benchmark_keeps_known_cost_without_inventing_total(store):
    codex = RunConfig(provider="codex", model="gpt-test").model_dump()
    run = store.new_run(codex, "soh", "fingerprint")
    codex_segment = store.segment(run, codex, "n1")
    codex_call = store.begin_call(run, codex_segment, {})
    store.finish_call(codex_call, status="completed", usage={
        "input_tokens": 80,
        "output_tokens": 20,
        "cost_usd": None,
        "actual_model": "gpt-test",
    }, latency_ms=25)

    openrouter = RunConfig(provider="openrouter", model="vendor/model").model_dump()
    openrouter_segment = store.segment(run, openrouter, "n1")
    openrouter_call = store.begin_call(run, openrouter_segment, {})
    store.finish_call(openrouter_call, status="completed", usage={
        "input_tokens": 120,
        "output_tokens": 30,
        "cost_usd": 0.05,
        "actual_model": "vendor/model",
    }, latency_ms=30)

    store.update_run(run, status="completed", mixed=True, reason="game_completed")
    store.end_segments(run)
    metrics = store.metrics(run)

    assert metrics["input_tokens"] == 200
    assert metrics["output_tokens"] == 50
    assert metrics["total_tokens"] == 250
    assert metrics["known_cost_usd"] == 0.05
    assert metrics["cost_usd"] is None
    assert len(metrics["usage_by_model"]) == 2
    by_provider = {row["provider"]: row for row in metrics["usage_by_model"]}
    assert by_provider["codex"]["cost_usd"] is None
    assert by_provider["openrouter"]["cost_usd"] == 0.05


def test_restart_does_not_resume_run(tmp_path):
    url = f"sqlite:///{tmp_path}/restart.sqlite3"
    first = Store(url)
    config = RunConfig(provider="demo", model="deterministic-demo").model_dump()
    run = first.new_run(config, "simulator", "hash")
    first.close()
    second = Store(url)
    assert second.detail(run)["status"] == "interrupted"
    second.close()


def test_api_controls_security_and_end_to_end_demo(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        bridge_port=free_udp_port(),
        allow_simulator=True,
        codex_command="codex-test-not-installed",
        openrouter_api_key="",
        agent_provider="demo",
        agent_model="deterministic-demo",
    )
    with TestClient(create_app(settings), base_url="http://127.0.0.1:8787") as client:
        boot = client.get("/api/bootstrap").json()
        headers = {"X-Zelda-Session": boot["session_token"]}

        assert client.post("/api/start", json={}).status_code == 403
        assert client.get(
            "/api/status", headers={"Origin": "https://attacker.invalid"}
        ).status_code == 403
        assert client.get(
            "/api/status", headers={"Host": "attacker.invalid"}
        ).status_code == 400

        for _ in range(50):
            if client.get("/api/status").json()["connection"]["game"]:
                break
            time.sleep(.02)
        status_before = client.get("/api/status").json()
        assert status_before["connection"]["game"]
        assert status_before["connection"]["source"] == "simulator"
        assert status_before["run_mode"] == "train"
        assert status_before["champions"]["count"] == 0
        assert client.get("/api/champions").json()["count"] == 0
        assert client.post(
            "/api/evaluate", json={}, headers=headers
        ).status_code == 400

        started = client.post("/api/start", json={}, headers=headers)
        assert started.status_code == 200, started.text
        run_id = started.json()["run_id"]

        status = started.json()
        for _ in range(80):
            status = client.get("/api/status").json()
            if (
                status["thought"]["state"] in {"acting", "thinking"}
                and status["input"]["reason"] == "ml_policy"
            ):
                break
            time.sleep(.05)

        assert status["status"] == "running"
        assert status["input"]["reason"] == "ml_policy"
        assert isinstance(status["input"]["stick_x"], int)
        assert isinstance(status["input"]["stick_y"], int)
        assert "button_names" in status["input"]
        assert status["thought"]["state"] in {"acting", "thinking"}
        assert "learning" in status
        assert "usage" in status
        assert "total_tokens" in status["usage"]
        assert "calls" in status["usage"]
        assert "quota" in status["usage"]
        assert "trigger" in status["thought"]
        assert "guidance" in status["thought"]

        detail = client.get(f"/api/runs/{run_id}").json()
        assert detail["source"] == "simulator"
        assert any(event["kind"] == "run_started" for event in detail["events"])

        stopped = client.post("/api/stop", json={}, headers=headers)
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "stopped"
        assert stopped.json()["input"]["buttons"] == 0
        assert stopped.json()["input"]["stick_x"] == 0
        assert stopped.json()["input"]["stick_y"] == 0

        assert client.post(
            "/api/diagnostics/input", json={"action": "tap_a"}, headers=headers
        ).status_code == 404
        assert client.post(
            "/api/hints", json={"text": "Try right"}, headers=headers
        ).status_code == 404
