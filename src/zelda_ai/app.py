from __future__ import annotations

import asyncio
import contextlib
import hmac
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, SecretStr
from starlette.middleware.trustedhost import TrustedHostMiddleware

from . import __version__
from .bridge import Bridge, bind_bridge
from .config import Settings
from .models import RunConfig, SwitchConfig
from .providers import CodexProvider, OpenRouterProvider, ProviderFailure
from .runtime import Runtime, SKILL_CATALOG
from .store import Store

ALLOWED_ORIGINS = {f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (8787, 5173)}


def bridge_secret(settings: Settings) -> str:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    token = settings.bridge_token.get_secret_value()
    path = settings.data_dir / "bridge.token"
    if not token:
        if path.exists():
            token = path.read_text().strip()
        else:
            token = secrets.token_urlsafe(32)
            path.write_text(token)
            path.chmod(0o600)
    if len(token) < 24:
        raise ValueError("ZELDA_BRIDGE_TOKEN must contain at least 24 characters.")
    return token


class KeyInput(BaseModel):
    key: SecretStr = Field(min_length=10, max_length=500)


class LoginInput(BaseModel):
    device: bool = False


class HintInput(BaseModel):
    text: str = Field(min_length=1, max_length=400)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    session_token = secrets.token_urlsafe(32)

    @asynccontextmanager
    async def lifespan(app):
        token = bridge_secret(settings)
        store = Store(settings.db_url)
        bridge = Bridge(token, settings.allow_simulator)
        providers = {"codex": CodexProvider(settings.codex_command, settings.codex_home, settings.decision_timeout_s),
            "openrouter": OpenRouterProvider(settings.openrouter_api_key.get_secret_value())}
        simulator_task = simulator_transport = None
        try:
            await bind_bridge(bridge, settings.bridge_port)
            if settings.allow_simulator:
                from .simulator import DemoProvider, Simulator
                providers["demo"] = DemoProvider()
                simulator = Simulator(settings.bridge_port, token)
                simulator_transport, _ = await asyncio.get_running_loop().create_datagram_endpoint(
                    lambda: simulator, local_addr=("127.0.0.1", 0))
                simulator_task = asyncio.create_task(simulator.run())
            runtime = Runtime(bridge, store, providers)
            app.state.runtime, app.state.providers, app.state.store = runtime, providers, store
            yield
        finally:
            runtime = getattr(app.state, "runtime", None)
            if runtime:
                await runtime.halt("stopped", "server_shutdown")
            bridge.close()
            if simulator_task:
                simulator_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await simulator_task
            if simulator_transport:
                simulator_transport.close()
            for provider in providers.values():
                await provider.close()
            store.close()

    app = FastAPI(title="Zelda AI Player", version=__version__, lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1"])

    @app.middleware("http")
    async def local_control_boundary(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and origin not in ALLOWED_ORIGINS:
            return JSONResponse({"detail": "Untrusted browser origin"}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"}:
            supplied = request.headers.get("x-zelda-session", "")
            if not hmac.compare_digest(supplied, session_token):
                return JSONResponse({"detail": "Reload the local panel to renew its session"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ValueError)
    async def bad_request(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.exception_handler(ProviderFailure)
    async def provider_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=502)

    @app.get("/api/bootstrap")
    async def bootstrap():
        return {"session_token": session_token, "version": __version__, "simulator": settings.allow_simulator,
            "bridge_port": settings.bridge_port}

    @app.get("/api/status")
    async def status():
        return app.state.runtime.snapshot()

    @app.get("/api/providers")
    async def provider_status():
        names = list(app.state.providers)
        results = await asyncio.gather(*(app.state.providers[name].status() for name in names))
        return [{"id": name, **result} for name, result in zip(names, results)]

    @app.get("/api/models/{provider}")
    async def models(provider: str):
        if provider not in app.state.providers:
            raise HTTPException(404, "Unknown provider")
        return [row.model_dump() for row in await app.state.providers[provider].models()]

    @app.post("/api/auth/codex")
    async def login(body: LoginInput):
        result = await app.state.providers["codex"].login(body.device)
        for key in ("authUrl", "verificationUrl"):
            if result.get(key):
                url = urlparse(result[key])
                if url.scheme != "https" or url.hostname not in {"auth.openai.com", "chatgpt.com", "auth.chatgpt.com"}:
                    raise HTTPException(502, "Codex returned an unexpected login URL")
        return result

    @app.post("/api/auth/openrouter")
    async def connect_openrouter(body: KeyInput):
        provider = app.state.providers["openrouter"]
        if app.state.runtime.state == "running":
            raise ValueError("Pause the agent before replacing a provider credential")
        old_key = provider.api_key
        provider.api_key = body.key.get_secret_value()
        result = await provider.status()
        if not result["connected"]:
            provider.api_key = old_key
            raise HTTPException(400, result["message"])
        return {"connected": True, "persistence": "process_memory_only"}

    @app.post("/api/runs")
    async def start_run(config: RunConfig):
        await app.state.runtime.start(config)
        return app.state.runtime.snapshot()

    @app.post("/api/control/{action}")
    async def control(action: str):
        await app.state.runtime.control(action)
        return app.state.runtime.snapshot()

    @app.post("/api/model")
    async def switch_model(config: SwitchConfig):
        await app.state.runtime.switch(config)
        return app.state.runtime.snapshot()

    @app.post("/api/hints")
    async def hint(body: HintInput):
        app.state.runtime.hint(body.text)
        return {"queued": True, "run_is_assisted": True}

    @app.get("/api/runs")
    async def list_runs():
        return app.state.store.list_runs()

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        result = app.state.store.detail(run_id)
        if result is None:
            raise HTTPException(404, "Run not found")
        return result

    @app.get("/api/skills")
    async def skills():
        return SKILL_CATALOG

    @app.websocket("/api/events")
    async def live_events(websocket: WebSocket):
        if websocket.headers.get("origin") not in ALLOWED_ORIGINS:
            await websocket.close(code=1008)
            return
        await websocket.accept()
        try:
            auth = await asyncio.wait_for(websocket.receive_json(), 5)
            if not hmac.compare_digest(str(auth.get("token", "")), session_token):
                await websocket.close(code=1008)
                return
        except (ValueError, asyncio.TimeoutError, WebSocketDisconnect):
            await websocket.close()
            return
        queue = asyncio.Queue(maxsize=1)
        runtime = app.state.runtime
        runtime.subscribers.add(queue)
        try:
            while True:
                await websocket.send_json(runtime.snapshot())
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(queue.get(), 1)
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            runtime.subscribers.discard(queue)

    dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    # src/zelda_ai -> project root is parents[2].
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="panel")
    else:
        @app.get("/")
        async def frontend_missing():
            return JSONResponse({"message": "Backend ativo. Em web/, execute npm install e npm run dev; "
                "ou npm run build para servir o painel nesta porta."})
    return app
