import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from zelda_ai.models import RunConfig
from zelda_ai.providers.base import ProviderFailure
from zelda_ai.providers.codex import CodexProvider
from zelda_ai.providers.openrouter import OpenRouterProvider


@pytest.mark.asyncio
async def test_openrouter_valid_decision_and_effort(decision):
    bodies = []
    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "gen", "model": "test/m", "usage": {
            "prompt_tokens": 50, "completion_tokens": 20, "cost": .01},
            "choices": [{"finish_reason": "stop", "message": {"content": decision.model_dump_json()}}]})
    provider = OpenRouterProvider("secret", httpx.MockTransport(respond))
    result = await provider.decide(RunConfig(provider="openrouter", model="test/m", effort="high"), "{}")
    assert result.usage.cost_usd == .01
    assert bodies[0]["reasoning"] == {"effort": "high", "exclude": True}
    assert bodies[0]["response_format"]["json_schema"]["strict"] is True
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
        await provider.decide(RunConfig(provider="openrouter", model="test"), "{}")
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
        await provider.decide(RunConfig(provider="openrouter", model="test"), "{}")
    assert count == 1 and "429" in str(error.value) and "secret" not in str(error.value)
    assert error.value.usage.cost_usd is None
    await provider.close()


@pytest.mark.asyncio
async def test_codex_jsonrpc_protocol(tmp_path, decision, monkeypatch):
    script = tmp_path / "fake_codex.py"
    output = decision.model_dump_json()
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
        assert req["params"]["outputSchema"]["additionalProperties"] is False
        result={"turn":{"id":"turn-test"}}
    print(json.dumps({"id":req["id"],"result":result}),flush=True)
    if method=="turn/start":
        events=[("thread/tokenUsage/updated",{"tokenUsage":{"total":{"inputTokens":100,"cachedInputTokens":60,"outputTokens":40,"reasoningOutputTokens":20}}}),
        ("item/completed",{"item":{"type":"reasoning","content":"NEVER_STORE_THIS"}}),
        ("item/completed",{"item":{"type":"agentMessage","text":''' + repr(output) + '''}}),
        ("turn/completed",{"turn":{"id":"turn-test","status":"completed"}})]
        for event,data in events:
            data["threadId"]="thread-test"
            print(json.dumps({"method":event,"params":data}),flush=True)
''')
    original = asyncio.create_subprocess_exec
    async def fake_exec(*args, **kwargs):
        return await original(sys.executable, str(script), **kwargs)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    provider = CodexProvider(sys.executable, tmp_path / "codex", timeout=5)
    try:
        assert (await provider.status())["connected"]
        assert (await provider.models())[0].efforts == ["high"]
        result = await provider.decide(RunConfig(provider="codex", model="test", effort="high"), "{}")
        assert result.usage.total_tokens == 140 and result.usage.cost_usd is None
        assert result.text == output and "NEVER_STORE_THIS" not in result.text
        assert provider.rpc.notifications.empty()
    finally:
        await provider.close()


def test_windows_npm_shim_avoids_shell(tmp_path, monkeypatch):
    from zelda_ai.providers.codex import executable_prefix
    shim = tmp_path / "codex.cmd"
    shim.write_text("@echo off")
    entry = tmp_path / "node_modules/@openai/codex/bin/codex.js"
    entry.parent.mkdir(parents=True)
    entry.write_text("// fixture")
    monkeypatch.setattr("zelda_ai.providers.codex.shutil.which", lambda name: sys.executable if name == "node" else str(shim))
    assert executable_prefix("codex") == [sys.executable, str(entry)]
