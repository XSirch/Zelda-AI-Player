import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.autonomy.prompt import AUTONOMY_SYSTEM_PROMPT
from zelda_ai.models import RunConfig
from zelda_ai.providers.base import ProviderFailure
from zelda_ai.providers.codex import CodexProvider, parse_codex_version, summarize_rate_limits
from zelda_ai.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
async def test_openrouter_valid_intent_and_effort():
    intent = AgentIntent.bootstrap()
    bodies = []
    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "gen", "model": "test/m", "usage": {
            "prompt_tokens": 50, "completion_tokens": 20, "cost": .01},
            "choices": [{"finish_reason": "stop", "message": {"content": intent.model_dump_json()}}]})
    provider = OpenRouterProvider("secret", httpx.MockTransport(respond))
    result = await provider.think(RunConfig(provider="openrouter", model="test/m", effort="high"), "{}")
    assert result.usage.cost_usd == .01
    assert bodies[0]["reasoning"] == {"effort": "high", "exclude": True}
    assert bodies[0]["response_format"]["json_schema"]["strict"] is True
    schema = bodies[0]["response_format"]["json_schema"]["schema"]
    completion_ref = schema["properties"]["completion"]["$ref"]
    completion_name = completion_ref.rsplit("/", 1)[-1]
    completion_schema = schema["$defs"][completion_name]
    assert set(completion_schema["required"]) == set(
        completion_schema["properties"]
    )
    assert not bodies[0]["provider"]["allow_fallbacks"]
    await provider.close()


@pytest.mark.asyncio
async def test_billable_incomplete_result_preserves_usage():
    count = 0
    def respond(request):
        nonlocal count
        count += 1
        return httpx.Response(200, json={"usage": {"prompt_tokens": 10, "completion_tokens": 30, "cost": .2},
            "choices": [{"finish_reason": "length", "message": {"content": ""}}]})
    provider = OpenRouterProvider("secret", httpx.MockTransport(respond))
    with pytest.raises(ProviderFailure) as error:
        await provider.think(RunConfig(provider="openrouter", model="test"), "{}")
    assert error.value.usage.cost_usd == .2 and count == 1
    await provider.close()


@pytest.mark.asyncio
async def test_http_failure_no_retry_or_key_leak():
    count = 0
    def respond(request):
        nonlocal count
        count += 1
        return httpx.Response(429, json={"error": "secret must not appear"})
    provider = OpenRouterProvider("very-secret", httpx.MockTransport(respond))
    with pytest.raises(ProviderFailure) as error:
        await provider.think(RunConfig(provider="openrouter", model="test"), "{}")
    assert count == 1 and "429" in str(error.value) and "secret" not in str(error.value)
    assert error.value.usage.cost_usd is None
    await provider.close()


@pytest.mark.asyncio
async def test_codex_jsonrpc_protocol(tmp_path, monkeypatch):
    script = tmp_path / "fake_codex.py"
    output = AgentIntent.bootstrap().model_dump_json()
    script.write_text('''import json, sys
for line in sys.stdin:
    req=json.loads(line)
    if "id" not in req: continue
    method=req["method"]
    result={}
    if method=="account/read": result={"account":{"type":"chatgpt","planType":"test"}}
    if method=="model/list": result={"data":[{"model":"test","displayName":"Test","supportedReasoningEfforts":[{"reasoningEffort":"high"}],"defaultReasoningEffort":"high"}]}
    if method=="thread/start":
        assert req["params"]["ephemeral"] is True
        result={"thread":{"id":"thread-test"}}
    if method=="turn/start":
        assert req["params"]["effort"]=="high"
        schema=req["params"]["outputSchema"]
        assert schema["additionalProperties"] is False
        target=schema["properties"]["target_position"]["anyOf"][0]
        assert target["items"]=={"type":"number"}
        assert target["minItems"]==3 and target["maxItems"]==3
        assert "prefixItems" not in target
        result={"turn":{"id":"turn-test"}}
    print(json.dumps({"id":req["id"],"result":result}),flush=True)
    if method=="turn/start":
        events=[("thread/tokenUsage/updated",{"tokenUsage":{"total":{"inputTokens":100,"cachedInputTokens":60,"outputTokens":40,"reasoningOutputTokens":20}}}),
        ("error",{"turnId":"turn-test","willRetry":True,"error":{"message":"temporary retry"}}),
        ("item/completed",{"item":{"type":"reasoning","content":"NEVER_STORE_THIS"}}),
        ("item/completed",{"item":{"type":"agentMessage","text":''' + repr(output) + '''}}),
        ("turn/completed",{"turn":{"id":"turn-test","status":"completed"}})]
        for event,data in events:
            data["threadId"]="thread-test"
            print(json.dumps({"method":event,"params":data}),flush=True)
''', encoding="utf-8")
    original = asyncio.create_subprocess_exec
    async def fake_exec(*args, **kwargs):
        return await original(sys.executable, str(script), **kwargs)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    provider = CodexProvider(sys.executable, tmp_path / "codex", timeout=5)
    try:
        assert (await provider.status())["connected"]
        assert (await provider.models())[0].efforts == ["high"]
        result = await provider.think(RunConfig(provider="codex", model="test", effort="high"), "{}")
        assert result.usage.total_tokens == 140 and result.usage.cost_usd is None
        assert result.text == output and "NEVER_STORE_THIS" not in result.text
        assert provider.rpc.notifications.empty()
    finally:
        await provider.close()


@pytest.mark.asyncio
async def test_codex_surfaces_terminal_turn_error_message(tmp_path, monkeypatch):
    script = tmp_path / "fake_codex_failure.py"
    script.write_text(r'''import json, sys
for line in sys.stdin:
    req=json.loads(line)
    if "id" not in req: continue
    method=req["method"]
    result={}
    if method=="account/read": result={"account":{"type":"chatgpt","planType":"test"}}
    if method=="thread/start": result={"thread":{"id":"thread-fail"}}
    if method=="turn/start": result={"turn":{"id":"turn-fail","status":"inProgress"}}
    print(json.dumps({"id":req["id"],"result":result}),flush=True)
    if method=="turn/start":
        error={"message":"The 'gpt-6-astra' model requires a newer version of Codex."}
        print(json.dumps({"method":"error","params":{"threadId":"thread-fail","turnId":"turn-fail","willRetry":False,"error":error}}),flush=True)
        print(json.dumps({"method":"turn/completed","params":{"threadId":"thread-fail","turn":{"id":"turn-fail","status":"failed","error":error}}}),flush=True)
''', encoding="utf-8")
    original = asyncio.create_subprocess_exec
    async def fake_exec(*args, **kwargs):
        return await original(sys.executable, str(script), **kwargs)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    provider = CodexProvider(sys.executable, tmp_path / "codex", timeout=5)
    try:
        with pytest.raises(ProviderFailure, match="requires a newer version of Codex"):
            await provider.think(RunConfig(provider="codex", model="gpt-6-astra"), "{}")
    finally:
        await provider.close()


def test_windows_npm_shim_avoids_shell(tmp_path, monkeypatch):
    from zelda_ai.providers.codex import executable_prefix
    shim = tmp_path / "codex.cmd"
    shim.write_text("@echo off", encoding="utf-8")
    entry = tmp_path / "node_modules/@openai/codex/bin/codex.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// fixture", encoding="utf-8")
    monkeypatch.setattr("zelda_ai.providers.codex.shutil.which", lambda name: sys.executable if name == "node" else str(shim))
    assert executable_prefix("codex") == [sys.executable, str(entry)]


def test_parse_codex_version_for_astra_era():
    assert parse_codex_version("codex-cli 0.154.0") == ("0.154.0", (0, 154, 0))
    assert parse_codex_version("zelda_ai_player/0.155.0-alpha.16 (Windows)") == ("0.155.0", (0, 155, 0))
    assert parse_codex_version("unknown") == (None, None)


def test_autonomy_prompt_never_exposes_skill_or_raw_button_selection():
    assert "You do NOT choose controller skills." in AUTONOMY_SYSTEM_PROMPT
    assert "You do NOT choose raw N64 buttons." in AUTONOMY_SYSTEM_PROMPT
    assert "fight_enemy" not in AUTONOMY_SYSTEM_PROMPT
    assert "navigate_to" not in AUTONOMY_SYSTEM_PROMPT


def test_codex_quota_normalizes_used_to_remaining_without_assuming_slots():
    payload = {
        "rateLimits": {
            "limitId": "codex",
            "planType": "plus",
            "primary": {
                "usedPercent": 31,
                "windowDurationMins": 10080,
                "resetsAt": 1790000000,
            },
            "secondary": None,
            "credits": {
                "hasCredits": False,
                "unlimited": False,
                "balance": "0",
            },
        },
        "rateLimitsByLimitId": {
            "codex": {
                "limitId": "codex",
                "limitName": "Codex",
                "primary": {
                    "usedPercent": 31,
                    "windowDurationMins": 10080,
                    "resetsAt": 1790000000,
                },
                "secondary": None,
            }
        },
    }
    quota = summarize_rate_limits(payload)
    assert quota["available"] is True
    assert len(quota["windows"]) == 1
    assert quota["windows"][0]["used_percent"] == 31
    assert quota["windows"][0]["remaining_percent"] == 69
    assert quota["windows"][0]["window_duration_mins"] == 10080
    assert quota["credits"]["balance"] == "0"


def test_codex_quota_keeps_multiple_duration_windows_when_returned():
    payload = {
        "rateLimits": {
            "limitId": "codex",
            "primary": {"usedPercent": 25, "windowDurationMins": 300},
            "secondary": {"usedPercent": 18, "windowDurationMins": 10080},
        }
    }
    quota = summarize_rate_limits(payload)
    assert [row["window_duration_mins"] for row in quota["windows"]] == [300, 10080]
    assert [row["remaining_percent"] for row in quota["windows"]] == [75, 82]
