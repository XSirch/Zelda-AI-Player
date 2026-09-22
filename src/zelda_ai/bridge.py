"""Authenticated, loopback-only UDP bridge. Input commands have a short native lease."""
from __future__ import annotations

import asyncio
import hmac
import json
import time
from collections.abc import Callable

from pydantic import ValidationError

from .models import GameState


class Bridge(asyncio.DatagramProtocol):
    def __init__(self, token: str, allow_simulator: bool = False):
        self.token = token
        self.allow_simulator = allow_simulator
        self.transport: asyncio.DatagramTransport | None = None
        self.state: GameState | None = None
        self.peer: tuple[str, int] | None = None
        self.last_seen = 0.0
        self.command_seq = 0
        self.changed = asyncio.Event()
        self.on_state: Callable[[GameState, GameState | None], None] | None = None
        self.rejected_packets = 0

    def connection_made(self, transport):
        self.transport = transport

    @property
    def connected(self) -> bool:
        return self.state is not None and time.monotonic() - self.last_seen < 2

    def datagram_received(self, data: bytes, addr: tuple[str, int]):
        try:
            if addr[0] != "127.0.0.1" or len(data) > 60000:
                raise ValueError("Invalid peer or packet size")
            packet = json.loads(data)
            token = packet.pop("token", None)
            if not isinstance(token, str) or not self.token or not hmac.compare_digest(token, self.token):
                raise ValueError("Unauthorized bridge")
            state = GameState.model_validate(packet)
            if state.source == "simulator" and not self.allow_simulator:
                raise ValueError("Simulator disabled")
            previous = self.state
            if previous and state.instance_id == previous.instance_id and state.seq <= previous.seq:
                raise ValueError("Duplicate or reordered state")
            if previous and self.connected:
                if state.instance_id != previous.instance_id or addr != self.peer:
                    raise ValueError("Another game instance already owns the bridge")
                if state.seq <= previous.seq:
                    raise ValueError("Duplicate or reordered state")
            # A live input lease expires before ownership can pass to another instance.
            self.state, self.peer, self.last_seen = state, addr, time.monotonic()
            self.command_seq = max(self.command_seq, state.last_command_seq)
            self.changed.set()
            if self.on_state:
                self.on_state(state, previous)
        except (ValueError, TypeError, AttributeError, ValidationError, UnicodeError):
            self.rejected_packets += 1

    def status(self) -> dict:
        return {"connected": self.connected, "last_seen_age_s": round(time.monotonic() - self.last_seen, 2)
            if self.last_seen else None, "rejected_packets": self.rejected_packets,
            "state": self.state.model_dump() if self.state else None}

    def send(self, *, buttons: int = 0, stick_x: int = 0, stick_y: int = 0,
             lease_ms: int = 300, active: bool = True) -> int:
        if not self.connected or not self.peer or not self.transport or not self.state:
            raise RuntimeError("SoH bridge is disconnected")
        if not 0 <= buttons <= 65535 or not -80 <= stick_x <= 80 or not -80 <= stick_y <= 80:
            raise ValueError("Invalid controller input")
        if not 0 <= lease_ms <= 500:
            raise ValueError("Input lease must be at most 500 ms")
        self.command_seq += 1
        command = {"protocol": 1, "token": self.token, "instance_id": self.state.instance_id,
            "scene_epoch": self.state.scene_epoch, "base_seq": self.state.seq,
            "seq": self.command_seq, "buttons": buttons, "stick_x": stick_x,
            "stick_y": stick_y, "lease_ms": lease_ms, "active": active}
        self.transport.sendto(json.dumps(command, separators=(",", ":")).encode(), self.peer)
        return self.command_seq

    def release(self):
        if self.connected:
            self.send(lease_ms=0, active=False)

    def close(self):
        self.release()
        if self.transport:
            self.transport.close()


async def bind_bridge(bridge: Bridge, port: int):
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: bridge, local_addr=("127.0.0.1", port))
