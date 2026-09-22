"""Explicit integration-test simulator, never a fallback for SoH or a language model."""
import asyncio
import json
import time
import uuid

from .models import Decision, ModelInfo, SkillArgs, Usage
from .providers.base import InferenceResult


class DemoProvider:
    async def status(self):
        return {"connected": True, "message": "Simulador determinístico; não usa IA nem joga Zelda."}

    async def models(self):
        return [ModelInfo(id="deterministic-demo", name="Teste local — não é um modelo de IA")]

    async def decide(self, config, prompt):
        await asyncio.sleep(0.1)
        data = json.loads(prompt)
        direction = "right" if (data.get("last_result") or {}).get("status") == "failed" else "forward"
        decision = Decision(goal="Testar o ciclo de controle", summary="Movimento sintético para validar a integração.",
            skill="move", args=SkillArgs(direction=direction, duration_ms=500, strength=0.5, slot=None, choice_index=None, song=None),
            memory_note=None)
        return InferenceResult(decision.model_dump_json(), Usage(input_tokens=0, output_tokens=0,
            cached_input_tokens=0, reasoning_output_tokens=0, cost_usd=0, actual_model="deterministic-demo"))

    async def close(self):
        pass


class Simulator(asyncio.DatagramProtocol):
    def __init__(self, port: int, token: str):
        self.port, self.token = port, token
        self.transport = None
        self.instance = "sim-" + uuid.uuid4().hex
        self.seq = self.command_seq = 0
        self.x = self.z = 0.0
        self.stick_x = self.stick_y = 0
        self.expires_at = 0.0

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data, addr):
        try:
            command = json.loads(data)
            if addr != ("127.0.0.1", self.port) or command.get("token") != self.token:
                return
            if command.get("instance_id") != self.instance or command["seq"] <= self.command_seq:
                return
            self.command_seq = command["seq"]
            self.stick_x = command["stick_x"] if command["active"] else 0
            self.stick_y = command["stick_y"] if command["active"] else 0
            self.expires_at = time.monotonic() + min(command["lease_ms"], 500) / 1000
        except (ValueError, KeyError):
            pass

    async def run(self):
        while True:
            if time.monotonic() < self.expires_at:
                self.x += self.stick_x * 0.05
                self.z += self.stick_y * 0.05
            self.seq += 1
            state = {"token": self.token, "protocol": 1, "source": "simulator", "instance_id": self.instance,
                "seq": self.seq, "scene_epoch": 1, "scene": 0, "room": 0, "in_game": True,
                "player": {"position": [self.x, 0, self.z], "yaw": 0, "health": 48,
                    "max_health": 48, "rupees": 0}, "last_command_seq": self.command_seq,
                "upstream_revision": "simulator-v1"}
            self.transport.sendto(json.dumps(state).encode(), ("127.0.0.1", self.port))
            await asyncio.sleep(0.1)
