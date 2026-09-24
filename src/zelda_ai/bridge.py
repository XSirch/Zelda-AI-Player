"""Authenticated loopback transport; fast observations never wait for a model call."""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import math
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from contextlib import contextmanager

from pydantic import ValidationError

from .control.authority import InputAuthority
from .models import GameEvent, GameState, InputReceipt, RealtimeState

logger = logging.getLogger(__name__)


class Bridge(asyncio.DatagramProtocol):
    def __init__(self, token: str, allow_simulator: bool = False):
        self.authority = InputAuthority()
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
        self.last_validation_error: dict[str, list[str]] | None = None
        self._full: GameState | None = None
        self._observer_state: GameState | None = None
        self._owner_base = 1
        self._event_cursor = 0
        self._events: deque[GameEvent] = deque(maxlen=64)
        self.receipts: OrderedDict[int, InputReceipt] = OrderedDict()
        self._intervals: deque[float] = deque(maxlen=200)
        self._apply_latencies: deque[float] = deque(maxlen=200)
        self._latency_seen: deque[int] = deque(maxlen=512)
        self.input_tick_ms = 50.0
        self.event_gaps = 0
        self.fast_base_misses = 0
        self.dropped_samples = 0
        self._last_resync = 0.0

    def connection_made(self, transport):
        self.transport = transport

    @property
    def connected(self) -> bool:
        return self.state is not None and time.monotonic() - self.last_seen < 2

    @property
    def realtime(self) -> bool:
        return bool(self.state and self.state.protocol == 2 and
                    "input_sequence" in self.state.capabilities)

    def _diagnostic(self, exc: ValidationError):
        errors = exc.errors(include_url=False, include_context=False, include_input=False)
        allowed = set(GameState.model_fields) | set(RealtimeState.model_fields)
        fields = {e["loc"][0] if e["loc"] and e["loc"][0] in allowed else "<unknown>" for e in errors}
        diagnostic = {"fields": sorted(fields)[:6], "types": sorted({e["type"] for e in errors})[:6]}
        if diagnostic != self.last_validation_error:
            logger.warning("SoH state rejected: %s", diagnostic)
        self.last_validation_error = diagnostic

    def _observe_ack(self, *, request_full=False, instance=None, peer=None):
        if not self.transport or not (peer or self.peer):
            return
        if request_full and time.monotonic() - self._last_resync < 0.1:
            return
        if request_full:
            self._last_resync = time.monotonic()
        self.transport.sendto(json.dumps({"protocol": 2, "kind": "observe_ack", "token": self.token,
            "instance_id": instance or self.state.instance_id, "event_cursor": self._event_cursor,
            "request_full": request_full}, separators=(",", ":")).encode(), peer or self.peer)

    def _merge_fast(self, fast: RealtimeState) -> GameState | None:
        base = self._full
        if (not base or base.instance_id != fast.instance_id or base.full_seq != fast.full_seq or
                (base.scene_epoch, base.context_epoch, base.scene, base.room) !=
                (fast.scene_epoch, fast.context_epoch, fast.scene, fast.room)):
            self.fast_base_misses += 1
            return None
        values = {name: getattr(fast, name) for name in RealtimeState.model_fields if name != "kind"}
        # Enrich labels only for the same actor lifetime. Never carry a disappeared actor forward.
        metadata = {a.actor_uid: a for a in base.room_actors if a.actor_uid}
        def enrich(actor):
            old = metadata.get(actor.actor_uid) if actor else None
            if old and actor and not actor.name:
                return actor.model_copy(update={"name": old.name, "description": old.description})
            return actor
        values["room_actors"] = [enrich(a) for a in fast.room_actors]
        for name in ("target_actor", "target_candidate", "context_actor"):
            values[name] = enrich(values[name])
        values["nearby_actors"] = []
        # All changed fields were validated as RealtimeState; unchanged fields are from the exact full_seq.
        return base.model_copy(update=values)

    def _ingest_events(self, state: GameState, bootstrap: bool) -> bool:
        if state.protocol == 1:
            return False
        if bootstrap:
            # Events predating connection are history, not victories/actions from a new run.
            self._events.clear()
            self._event_cursor = state.event_seq
            self._observe_ack(instance=state.instance_id)
            return False
        changed = False
        if state.event_floor > self._event_cursor + 1:
            self.event_gaps += 1
            self._events.append(GameEvent(id=f"gap-{state.seq}", kind="bridge_event_gap",
                detail=f"{self._event_cursor + 1}:{state.event_floor - 1}"))
            self._event_cursor = state.event_floor - 1
            changed = True
        for event in state.events:
            try:
                seq = int(event.id)
            except ValueError:
                continue
            if seq == self._event_cursor + 1:
                self._events.append(event)
                self._event_cursor = seq
                changed = True
            elif seq > self._event_cursor + 1:
                break  # Do not ACK an unreceived prefix; native journal resends it.
        if changed:
            self._observe_ack(instance=state.instance_id)
        return changed

    def datagram_received(self, data: bytes, addr: tuple[str, int]):
        try:
            if addr[0] != "127.0.0.1" or len(data) > 60000:
                raise ValueError("Invalid peer or packet size")
            packet = json.loads(data)
            token = packet.pop("token", None)
            if not isinstance(token, str) or not self.token or not hmac.compare_digest(token, self.token):
                raise ValueError("Unauthorized bridge")
            try:
                incoming = (RealtimeState.model_validate(packet) if packet.get("kind") == "fast"
                            else GameState.model_validate(packet))
            except ValidationError as exc:
                self._diagnostic(exc)
                raise ValueError("Invalid state") from None
            if incoming.source == "simulator" and not self.allow_simulator:
                raise ValueError("Simulator disabled")
            previous = self.state
            bootstrap = previous is None or previous.instance_id != incoming.instance_id
            if previous and self.connected and (bootstrap or addr != self.peer):
                raise ValueError("Another game instance already owns the bridge")
            if previous and not bootstrap and incoming.seq <= previous.seq:
                raise ValueError("Duplicate or reordered state")
            if isinstance(incoming, RealtimeState):
                state = self._merge_fast(incoming)
                if state is None:
                    self._observe_ack(request_full=True, instance=incoming.instance_id, peer=addr)
                    return
                full = False
            else:
                state = incoming
                full = True
                if state.protocol == 2 and state.full_seq != state.seq:
                    raise ValueError("Full snapshot sequence mismatch")
                self._full = state
            now = time.monotonic()
            if bootstrap:
                self.receipts.clear()
                self._observer_state = None
                self._intervals.clear()
                self._apply_latencies.clear()
                self._latency_seen.clear()
                self._owner_base = state.owner_epoch + 1
                if previous is not None:
                    self.authority.revoke()
            elif self.last_seen:
                ticks = state.input_tick - previous.input_tick
                if ticks > 0:
                    measured = (now-self.last_seen)*1000/ticks
                    if 5 <= measured <= 200:
                        self.input_tick_ms = .8*self.input_tick_ms + .2*measured
                self._intervals.append((now - self.last_seen) * 1000)
                self.dropped_samples += max(0, state.seq - previous.seq - 1)
            self.peer = addr
            events_changed = self._ingest_events(state, bootstrap)
            if state.protocol == 2:
                state = state.model_copy(update={"events": list(self._events)})
            for receipt in state.input_receipts:
                self.receipts[receipt.seq] = receipt
                self.receipts.move_to_end(receipt.seq)
                if receipt.first_tick and receipt.apply_latency_ms is not None and receipt.seq not in self._latency_seen:
                    self._latency_seen.append(receipt.seq)
                    self._apply_latencies.append(receipt.apply_latency_ms)
            while len(self.receipts) > 512:
                self.receipts.popitem(last=False)
            self.state, self.last_seen = state, now
            self.command_seq = max(self.command_seq, state.last_command_seq, state.last_received_seq)
            self.last_validation_error = None
            self.changed.set()
            # No persistence or large UI snapshot on each fast sample. Events bypass slow throttling.
            if full or events_changed:
                old = self._observer_state
                self._observer_state = state
                if self.on_state:
                    self.on_state(state, old)
        except (ValueError, TypeError, AttributeError, ValidationError, UnicodeError):
            self.rejected_packets += 1

    def status(self) -> dict:
        def percentile(values, fraction):
            ordered = sorted(values)
            if not ordered:
                return None
            index = min(len(ordered)-1, max(0, math.ceil(len(ordered)*fraction)-1))
            return round(ordered[index], 2)
        last_consumed = next((row for row in reversed(list(self.receipts.values())) if row.first_tick > 0), None)
        return {"connected": self.connected, "last_seen_age_s": round(time.monotonic()-self.last_seen, 3)
            if self.last_seen else None, "rejected_packets": self.rejected_packets,
            "last_validation_error": self.last_validation_error,
            "realtime": {"enabled": self.realtime, "owner": self.authority.label,
                "state_hz": round(1000 * len(self._intervals) / sum(self._intervals), 2)
                    if self._intervals and sum(self._intervals) else None,
                "state_interval_p95_ms": percentile(self._intervals, .95),
                "native_apply_p50_ms": percentile(self._apply_latencies, .5),
                "native_apply_p95_ms": percentile(self._apply_latencies, .95),
                "native_apply_p99_ms": percentile(self._apply_latencies, .99),
                "last_receipt": last_consumed.model_dump() if last_consumed else None,
                "dropped_samples": self.dropped_samples, "event_gaps": self.event_gaps,
                "fast_base_misses": self.fast_base_misses,
                "ack_semantics": "input_consumer_delivery" if self.realtime else "accepted_only_legacy"},
            "state": self.state.model_dump() if self.state else None}

    async def next_state(self, after_seq: int, *, timeout: float = .35) -> GameState:
        deadline = time.monotonic() + timeout
        while True:
            self.authority.check()
            self.changed.clear()
            if self.state and self.state.seq > after_seq:
                return self.state
            left = deadline - time.monotonic()
            if left <= 0:
                raise RuntimeError("state_feedback_timeout")
            try:
                await asyncio.wait_for(self.changed.wait(), left)
            except asyncio.TimeoutError:
                raise RuntimeError("state_feedback_timeout") from None

    def _command(self, kind: str, *, buttons=0, stick_x=0, stick_y=0, lease_ms=300, **extra) -> int:
        if not self.connected or not self.peer or not self.transport or not self.state:
            raise RuntimeError("SoH bridge is disconnected")
        if kind not in {"release", "cancel"} and self.realtime and time.monotonic()-self.last_seen > .3:
            raise RuntimeError("state_feedback_stale")
        if not 0 <= buttons <= 65535 or not -80 <= stick_x <= 80 or not -80 <= stick_y <= 80:
            raise ValueError("Invalid controller input")
        if not 0 <= lease_ms <= 500:
            raise ValueError("Input lease must be at most 500 ms")
        self.command_seq += 1
        command = {"protocol": self.state.protocol, "token": self.token,
            "instance_id": self.state.instance_id, "scene_epoch": self.state.scene_epoch,
            "base_seq": self.state.seq, "seq": self.command_seq,
            "buttons": buttons, "stick_x": stick_x, "stick_y": stick_y, "lease_ms": lease_ms}
        if self.realtime:
            command.update(kind=kind, owner_epoch=self._owner_base+self.authority.epoch,
                           context_epoch=self.state.context_epoch, **extra)
        else:
            command["active"] = kind not in {"release", "cancel"}
        self.transport.sendto(json.dumps(command, separators=(",", ":")).encode(), self.peer)
        return self.command_seq

    def send(self, *, buttons=0, stick_x=0, stick_y=0, lease_ms=300, active=True) -> int:
        self.authority.check()
        return self._command("setpoint" if active else "cancel", buttons=buttons,
            stick_x=stick_x, stick_y=stick_y, lease_ms=lease_ms if active else (200 if self.realtime else 0))

    async def pulse_receipt(self, *, buttons=0, stick_x=0, stick_y=0, hold_ticks=1,
                            baseline_buttons=0, edge_buttons=None, timeout=1.5) -> InputReceipt | None:
        """Return the exact native sequence receipt; consumption is delivery evidence, not gameplay success."""
        self.authority.check()
        if not self.realtime:
            raise RuntimeError("native_sequence_not_supported")
        if not 1 <= hold_ticks <= 8:
            raise ValueError("hold_ticks must be between 1 and 8")
        state = self.state
        context = (state.instance_id, state.scene_epoch, state.context_epoch)
        seq = self._command("sequence", buttons=baseline_buttons, lease_ms=300,
            edge_buttons=buttons if edge_buttons is None else edge_buttons, steps=[
                {"buttons": buttons, "stick_x": stick_x, "stick_y": stick_y, "ticks": hold_ticks},
                {"buttons": baseline_buttons, "stick_x": 0, "stick_y": 0, "ticks": 1}])
        deadline = time.monotonic()+timeout
        last_seq = state.seq
        try:
            while time.monotonic() < deadline:
                current = await self.next_state(last_seq, timeout=min(.35, max(.001, deadline-time.monotonic())))
                last_seq = current.seq
                row = self.receipts.get(seq)
                if (current.instance_id, current.scene_epoch, current.context_epoch) != context:
                    return row
                if row and row.status in {"rejected", "cancelled", "superseded", "completed"}:
                    return row
                self._command("renew", lease_ms=300)
            return self.receipts.get(seq)
        finally:
            current = self.state
            same_context = bool(current and (current.instance_id, current.scene_epoch, current.context_epoch) == context)
            if baseline_buttons and same_context and self.connected and self.authority.valid():
                self._command("cancel", buttons=baseline_buttons, lease_ms=200)
            else:
                self.release()

    async def pulse(self, *, buttons=0, stick_x=0, stick_y=0, hold_ticks=1, baseline_buttons=0,
                    edge_buttons=None, timeout=1.5) -> bool:
        """Return delivery evidence, NOT gameplay-effect confirmation. No automatic paid/model calls."""
        row = await self.pulse_receipt(buttons=buttons, stick_x=stick_x, stick_y=stick_y,
            hold_ticks=hold_ticks, baseline_buttons=baseline_buttons, edge_buttons=edge_buttons,
            timeout=timeout)
        return bool(row and row.first_tick > 0 and row.status not in {"rejected", "cancelled", "superseded"})

    def command_consumed(self, seq: int | None) -> bool:
        if seq is None:
            return False
        if self.realtime:
            row = self.receipts.get(seq)
            return bool(row and row.first_tick > 0)
        return bool(self.state and self.state.last_command_seq >= seq)

    @contextmanager
    def input_scope(self, label: str):
        with self.authority.scope(label):
            initial_seq = self.command_seq
            try:
                yield
            finally:
                if self.command_seq != initial_seq:
                    self.release()

    def enable_control(self):
        return self.authority.enable()

    def release(self):
        if self.connected and self.authority.valid():
            self._command("cancel", lease_ms=200 if self.realtime else 0)

    def revoke(self):
        generation = self.authority.revoke()
        if self.connected and self.transport and self.peer:
            self._command("release", lease_ms=0)
        self.changed.set()
        return generation

    def close(self):
        self.revoke()
        if self.transport:
            self.transport.close()


async def bind_bridge(bridge: Bridge, port: int):
    loop = asyncio.get_running_loop()
    await loop.create_datagram_endpoint(lambda: bridge, local_addr=("127.0.0.1", port))
