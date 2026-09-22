"""Codex app-server over stdio; ChatGPT auth stays inside the official CLI."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import shutil
from pathlib import Path

from ..models import Decision, ModelInfo, RunConfig, Usage
from .base import InferenceResult, ProviderFailure, SYSTEM_PROMPT


ASTRA_MIN_CODEX_VERSION = (0, 153, 0)
_VERSION_RE = re.compile(r"(?<!\\d)(\\d+)\\.(\\d+)\\.(\\d+)(?:[-+][0-9A-Za-z.-]+)?")


def parse_codex_version(value: str | None) -> tuple[str | None, tuple[int, int, int] | None]:
    if not value:
        return None, None
    match = _VERSION_RE.search(value)
    if not match:
        return None, None
    normalized = ".".join(match.groups())
    return normalized, tuple(int(part) for part in match.groups())


def parse_usage(params: dict, model: str) -> Usage:
    # Each decision uses a fresh bounded thread; total includes all inference within that thread.
    info = params.get("tokenUsage") or {}
    counts = info.get("total") or info.get("last") or {}
    return Usage(input_tokens=counts.get("inputTokens"), cached_input_tokens=counts.get("cachedInputTokens"),
        output_tokens=counts.get("outputTokens"), reasoning_output_tokens=counts.get("reasoningOutputTokens"),
        actual_model=model, cost_usd=None)


def executable_prefix(command: str) -> list[str]:
    executable = shutil.which(command) or (command if Path(command).is_file() else None)
    if not executable:
        raise ProviderFailure("Codex CLI not found. Install it and configure ZELDA_CODEX_COMMAND.")
    if Path(executable).suffix.lower() in {".cmd", ".bat"}:
        # npm on Windows installs a batch shim. Invoke its official JS entrypoint without a shell.
        script = Path(executable).parent / "node_modules" / "@openai" / "codex" / "bin" / "codex.js"
        node = shutil.which("node")
        if not node or not script.is_file():
            raise ProviderFailure("Unsupported Codex batch shim. Configure a native codex.exe path instead.")
        return [node, str(script)]
    return [str(Path(executable).resolve())]


class JsonRpcProcess:
    def __init__(self, command: str, home: Path):
        self.command, self.home = command, home
        self.process: asyncio.subprocess.Process | None = None
        self.reader: asyncio.Task | None = None
        self.pending: dict[int, asyncio.Future] = {}
        self.notifications: asyncio.Queue = asyncio.Queue(maxsize=4096)
        self.next_id = 0
        self.server_version: str | None = None
        self.start_lock = asyncio.Lock()
        self.write_lock = asyncio.Lock()

    async def start(self):
        async with self.start_lock:
            if self.process and self.process.returncode is None:
                return
            self.home.mkdir(parents=True, exist_ok=True)
            workspace = self.home.parent / "agent-workspace"
            workspace.mkdir(exist_ok=True)
            prefix = executable_prefix(self.command)
            # Isolated profile avoids loading the user's projects, plugins, MCPs or global agent memory.
            environment = dict(os.environ)
            environment["CODEX_HOME"] = str(self.home)
            for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY", "ZELDA_BRIDGE_TOKEN"):
                environment.pop(key, None)
            args = [*prefix, "app-server", "-c", "features.shell_tool=false", "-c", "features.unified_exec=false",
                "-c", "features.apply_patch_freeform=false", "-c", 'web_search="disabled"',
                "-c", 'sandbox_mode="read-only"', "-c", 'approval_policy="never"']
            try:
                self.process = await asyncio.create_subprocess_exec(*args, stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
                    cwd=str(workspace), env=environment, limit=4 * 1024 * 1024)
                self.reader = asyncio.create_task(self._read())
                initialized = await self.request("initialize", {"clientInfo": {"name": "zelda_ai_player", "version": "0.1.0"},
                    "capabilities": {"experimentalApi": False}}, timeout=20)
                server_info = initialized.get("serverInfo") or {}
                raw_version = server_info.get("version") or initialized.get("userAgent")
                self.server_version, _ = parse_codex_version(raw_version)
                await self.write({"method": "initialized", "params": {}})
            except (OSError, asyncio.TimeoutError, ProviderFailure):
                await self.close()
                raise ProviderFailure("Could not initialize Codex app-server. Check CLI version and executable path.") from None

    async def write(self, message: dict):
        if not self.process or not self.process.stdin or self.process.returncode is not None:
            raise ProviderFailure("Codex app-server is not running.")
        async with self.write_lock:
            self.process.stdin.write((json.dumps(message, separators=(",", ":")) + "\n").encode())
            await self.process.stdin.drain()

    async def request(self, method: str, params: dict | None = None, timeout: float = 20):
        self.next_id += 1
        request_id = self.next_id
        future = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self.write({"id": request_id, "method": method, "params": params or {}})
            return await asyncio.wait_for(future, timeout)
        finally:
            self.pending.pop(request_id, None)

    async def _read(self):
        try:
            assert self.process and self.process.stdout
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if "id" in message and "method" not in message:
                    future = self.pending.get(message["id"])
                    if future and not future.done():
                        if "error" in message:
                            # Do not publish provider payloads that may contain account data.
                            future.set_exception(ProviderFailure(f"Codex RPC error {message['error'].get('code')}"))
                        else:
                            future.set_result(message.get("result", {}))
                elif "id" in message:
                    # There are no game tools in this profile: reject unexpected approval/tool requests.
                    await self.write({"id": message["id"], "error": {
                        "code": -32601, "message": "This client accepts structured game decisions only."}})
                else:
                    method = message.get("method", "")
                    # Never retain raw reasoning or auth payloads in the gameplay event queue.
                    if method in {"turn/completed", "thread/tokenUsage/updated", "item/completed", "model/rerouted"}:
                        if method == "item/completed" and message.get("params", {}).get("item", {}).get("type") != "agentMessage":
                            continue
                        await self.notifications.put(message)
        except (OSError, ValueError, asyncio.CancelledError):
            pass
        finally:
            for future in list(self.pending.values()):
                if not future.done():
                    future.set_exception(ProviderFailure("Codex app-server disconnected."))

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 3)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        if self.reader:
            self.reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.reader
        self.process = None
        self.server_version = None


class CodexProvider:
    def __init__(self, command: str, home: Path, timeout: float = 180):
        self.rpc = JsonRpcProcess(command, home)
        self.timeout = timeout
        self.lock = asyncio.Lock()

    async def status(self) -> dict:
        try:
            await self.rpc.start()
            version_text, version = parse_codex_version(self.rpc.server_version)
            astra_cli_ready = version is not None and version >= ASTRA_MIN_CODEX_VERSION
            result = await self.rpc.request("account/read", {"refreshToken": False})
            account = result.get("account") or {}
            connected = account.get("type") == "chatgpt"
            limits = None
            if connected:
                with contextlib.suppress(ProviderFailure, asyncio.TimeoutError):
                    limits = await self.rpc.request("account/rateLimits/read")
            if connected:
                message = "ChatGPT autenticado."
                if version_text:
                    message += f" Codex CLI {version_text}."
                if version is not None and not astra_cli_ready:
                    message += " Astra requer Codex CLI 0.153.0 ou mais recente."
            else:
                message = "Conecte a conta ChatGPT pelo painel."
            return {"connected": connected, "auth_type": account.get("type"),
                "plan": account.get("planType"), "limits": limits, "cli_version": version_text,
                "astra_cli_ready": astra_cli_ready, "message": message}
        except (ProviderFailure, asyncio.TimeoutError) as exc:
            version_text, version = parse_codex_version(self.rpc.server_version)
            return {"connected": False, "cli_version": version_text,
                "astra_cli_ready": version is not None and version >= ASTRA_MIN_CODEX_VERSION,
                "message": str(exc) or "Codex timed out."}

    async def login(self, device: bool = False) -> dict:
        await self.rpc.start()
        result = await self.rpc.request("account/login/start", {"type": "chatgptDeviceCode" if device else "chatgpt"},
            timeout=30)
        return {key: result.get(key) for key in ("type", "loginId", "authUrl", "verificationUrl", "userCode")}

    async def models(self) -> list[ModelInfo]:
        await self.rpc.start()
        result, cursor = [], None
        for _ in range(20):
            params = {"limit": 100, "includeHidden": False}
            if cursor:
                params["cursor"] = cursor
            page = await self.rpc.request("model/list", params)
            for row in page.get("data", []):
                result.append(ModelInfo(id=row["model"], name=row.get("displayName", row["model"]),
                    efforts=[e["reasoningEffort"] for e in row.get("supportedReasoningEfforts", [])],
                    effort_source="catalog", default_effort=row.get("defaultReasoningEffort")))
            cursor = page.get("nextCursor")
            if not cursor:
                return result
        raise ProviderFailure("Codex model catalog exceeded pagination limit.")

    async def decide(self, config: RunConfig, prompt: str) -> InferenceResult:
        async with self.lock:
            await self.rpc.start()
            account = await self.rpc.request("account/read", {"refreshToken": False})
            if (account.get("account") or {}).get("type") != "chatgpt":
                raise ProviderFailure("This provider requires ChatGPT login; API-key billing is not substituted.")
            while not self.rpc.notifications.empty():
                self.rpc.notifications.get_nowait()
            thread = await self.rpc.request("thread/start", {"model": config.model, "modelProvider": "openai",
                "ephemeral": True, "approvalPolicy": "never", "sandbox": "read-only",
                "baseInstructions": SYSTEM_PROMPT})
            thread_id = thread["thread"]["id"]
            turn_id = None
            text, usage = "", Usage(actual_model=config.model)
            try:
                params = {"threadId": thread_id, "model": config.model,
                    "input": [{"type": "text", "text": prompt}],
                    "outputSchema": Decision.model_json_schema()}
                if config.effort:
                    params["effort"] = config.effort
                response = await self.rpc.request("turn/start", params)
                turn_id = response["turn"]["id"]
                async with asyncio.timeout(self.timeout):
                    while True:
                        event = await self.rpc.notifications.get()
                        data = event.get("params", {})
                        if data.get("threadId") != thread_id:
                            continue
                        if event["method"] == "thread/tokenUsage/updated":
                            usage = parse_usage(data, usage.actual_model or config.model)
                        elif event["method"] == "model/rerouted":
                            usage.actual_model = data.get("toModel", config.model)
                        elif event["method"] == "item/completed":
                            item = data.get("item") or {}
                            if item.get("type") == "agentMessage":
                                text = item.get("text", "")
                        elif event["method"] == "turn/completed" and data.get("turn", {}).get("id") == turn_id:
                            if data["turn"].get("status") != "completed":
                                raise ProviderFailure("Codex turn failed or was interrupted.", usage)
                            return InferenceResult(text, usage)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                if turn_id:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(self.rpc.request("turn/interrupt", {
                            "threadId": thread_id, "turnId": turn_id}, timeout=5))
                raise
            finally:
                # Unsubscribe; the app-server manages ephemeral thread cleanup.
                with contextlib.suppress(Exception):
                    await self.rpc.request("thread/unsubscribe", {"threadId": thread_id}, timeout=5)

    async def close(self):
        await self.rpc.close()
