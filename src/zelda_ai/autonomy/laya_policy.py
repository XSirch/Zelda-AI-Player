"""Nonblocking walking candidate IPC with causal, context-bound output leases."""

from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from ..laya_data import bounded_state, digest
from .imitation import encode_surface


@dataclass
class DecisionLease:
    owner: tuple | None = None
    submitted_at: float = 0
    features: tuple = ()
    stick: tuple = (0, 0)

    def resolve(self, owner, features, *, now, budget_s):
        if self.owner != owner or not 0 <= now - self.submitted_at <= budget_s:
            return (0, 0)
        alignment = sum(a * b for a, b in zip(self.features[:2], features[:2]))
        if alignment < math.cos(math.radians(12)) or abs(self.features[3] - features[3]) > 0.08:
            return (0, 0)
        return self.stick


class LayaWalkingPolicy:
    """Raw analog candidate for LocalTask only; never a native input sender."""

    refresh_at_motor_cadence = True

    def __init__(self, process, *, budget_s=0.1):
        if not 0 < budget_s <= 0.5:
            raise ValueError("Decision age budget must be in (0, .5] seconds")
        self.process, self.budget_s = process, budget_s
        self.owner, self.pending, self.history = None, None, []
        self.last_seq, self.request_id = -1, 0
        self.closed = False
        self.failure = None
        self.lease = DecisionLease()
        self.metrics = {
            "responses": 0,
            "expired_responses": 0,
            "context_rejections": 0,
            "round_trip_ms": [],
            "inference_ms": [],
            "output_age_ms": [],
            "provider_calls": 0,
            "training_updates": 0,
        }
        self.event = asyncio.Event()
        self.runner = asyncio.create_task(self._run())

    @classmethod
    async def start(cls, python: Path, base: Path, candidate: Path, log: Path, *, budget_s=0.1):
        # Process ownership and stdout are exclusive to this candidate session.
        with log.open("wb") as stderr:
            process = await asyncio.create_subprocess_exec(
                str(python.resolve()),
                "-m",
                "zelda_ai.laya_worker",
                str(base.resolve()),
                str(candidate.resolve()),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
        try:
            ready = json.loads(await asyncio.wait_for(process.stdout.readline(), timeout=45))
            if (
                ready.get("ready") is not True
                or ready.get("provider_calls") != 0
                or ready.get("training_updates") != 0
                or ready.get("candidate_sha256") != digest(candidate / "heads.safetensors")
            ):
                raise ValueError("Local inference worker failed provenance handshake")
            return cls(process, budget_s=budget_s)
        except BaseException:
            if process.returncode is None:
                process.terminate()
            await process.wait()
            raise

    @staticmethod
    def key(game, task):
        return (
            game.instance_id,
            game.scene_epoch,
            game.context_epoch,
            game.scene,
            game.room,
            game.mirrored_world,
            game.player.age,
            task.identity,
        )

    def __call__(self, game, task):
        if (
            self.closed
            or self.failure
            or task.terminal
            or task.phase == "verify"
            or game.source != "soh"
            or not game.in_game
            or not game.player
            or game.player.health <= 0
            or game.game_over_state
            or game.paused
            or game.cutscene_active
            or game.dialogue.active
            or game.pause_menu.active
            or game.player.climbing_ladder
            or game.player.hanging_ledge
            or game.player.climbing_ledge
        ):
            self.invalidate()
            return (0, 0)
        owner, features = self.key(game, task), encode_surface(game, task)
        if owner != self.owner:
            self.invalidate()
            self.owner = owner
        now = time.monotonic()
        if game.seq > self.last_seq:
            self.history = (self.history + [features])[-4:]
            self.last_seq, self.request_id = game.seq, self.request_id + 1
            self.pending = (self.request_id, owner, now, tuple(features), bounded_state(self.history))
            self.event.set()
        # A reply that met the decision budget gets at most one 20 Hz tick to
        # reach the arbiter. This is distinct from a 100 ms full-reaction claim.
        stick = self.lease.resolve(owner, features, now=now, budget_s=self.budget_s + 0.05)
        if stick != (0, 0):
            self.metrics["output_age_ms"].append((now - self.lease.submitted_at) * 1000)
            self.metrics["output_age_ms"] = self.metrics["output_age_ms"][-4096:]
        return stick

    def invalidate(self):
        self.owner, self.pending, self.history, self.last_seq = None, None, [], -1
        self.lease = DecisionLease()

    async def _run(self):
        try:
            while not self.closed:
                await self.event.wait()
                self.event.clear()
                request, self.pending = self.pending, None
                if request is None:
                    continue
                identity, owner, submitted, features, state = request
                self.process.stdin.write(
                    (json.dumps({"id": identity, "state": state}) + "\n").encode("utf-8")
                )
                await self.process.stdin.drain()
                response = json.loads(await asyncio.wait_for(self.process.stdout.readline(), timeout=2))
                stick = response.get("stick", [])
                elapsed = time.monotonic() - submitted
                if (
                    response.get("id") != identity
                    or len(stick) != 2
                    or any(type(v) is not int or not -80 <= v <= 80 for v in stick)
                    or not math.isfinite(response.get("inference_ms", float("nan")))
                ):
                    raise ValueError("Invalid or mismatched inference response")
                self.metrics["responses"] += 1
                self.metrics["round_trip_ms"].append(elapsed * 1000)
                self.metrics["inference_ms"].append(response["inference_ms"])
                # Bound telemetry retained by a continuous session.
                for name in ("round_trip_ms", "inference_ms"):
                    self.metrics[name] = self.metrics[name][-4096:]
                if owner != self.owner:
                    self.metrics["context_rejections"] += 1
                elif elapsed > self.budget_s:
                    self.metrics["expired_responses"] += 1
                else:
                    self.lease = DecisionLease(owner, submitted, features, tuple(stick))
        except asyncio.CancelledError:
            raise
        except (OSError, ValueError, KeyError, TypeError, asyncio.TimeoutError) as exc:
            self.failure = f"{type(exc).__name__}: {str(exc)[:160]}"
            self.invalidate()

    async def close(self):
        self.closed = True
        self.invalidate()
        self.runner.cancel()
        await asyncio.gather(self.runner, return_exceptions=True)
        self.process.stdin.close()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=3)
        except asyncio.TimeoutError:
            self.process.terminate()
            await self.process.wait()
