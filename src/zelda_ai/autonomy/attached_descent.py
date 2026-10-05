"""Reference QA calibration of physical analog effects in an observed ladder mode.

Walking projection never runs underneath attachment. A raw vector is accepted
only after consumed input causes measured motion toward the observed lower floor.
This is reference preparation, not a trained Laya traversal policy.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .ladder_task import LadderDescentTask

# A numeric action palette, with no semantic context/button association.
PROBES = ((-60, 0), (0, -60), (0, 60), (60, 0))


@dataclass
class AttachedDescentTask(LadderDescentTask):
    VERSION = "observed-attached-descent-reference-v1"
    stable_position: tuple | None = None
    stable_since: float | None = None
    raw_probe: tuple | None = None
    probe_index: int = 0
    probe_seq: int = -1
    probe_position: tuple | None = None
    probe_at: float | None = None
    probe_consumed: bool = False
    learned_stick: tuple | None = None
    calibration: list = field(default_factory=list)
    input_frame: tuple | None = None

    def observe(self, game, *, consumed, now=None):
        now = time.monotonic() if now is None else now
        fresh = game.seq > self.last_seq and game.seq > self.origin_seq
        super().observe(game, consumed=consumed, now=now)
        if self.terminal or not fresh:
            return
        if not game.player.climbing_ladder:
            self.raw_probe = None
            return
        current_frame = (game.camera_input_yaw, game.player.yaw)
        if current_frame[0] is None:
            self.interrupt("missing_mode_input_frame")
            return
        if self.input_frame is not None and any(
            abs((a - b + 32768) % 65536 - 32768) > 2185 for a, b in zip(current_frame, self.input_frame)
        ):
            # A learned raw vector is bound to the actual input/body frame.
            self.learned_stick = self.raw_probe = self.stable_since = None
            self.probe_index = 0
        # Keep the calibration frame fixed while its physical action is active:
        # a slow orbit must not evade invalidation through small per-tick cuts.
        if self.learned_stick is None and self.raw_probe is None:
            self.input_frame = current_frame
        if self.learned_stick is not None:
            return
        position = game.player.position
        if self.raw_probe is not None:
            self.probe_consumed |= bool(consumed and game.seq > self.probe_seq)
            if now - self.probe_at < .35:
                return
            delta_y = position[1] - self.probe_position[1]
            self.calibration.append({"stick": self.raw_probe, "delta_y": delta_y,
                                     "consumed": self.probe_consumed})
            self.calibration = self.calibration[-16:]
            if self.probe_consumed and delta_y <= -4:
                self.learned_stick = self.raw_probe
            self.raw_probe = None
            self.stable_since, self.stable_position = now, position
            return
        if self.stable_position is None or math.dist(position, self.stable_position) > 1:
            self.stable_position, self.stable_since = position, now
            return
        if self.stable_since is None:
            self.stable_since = now
        if now - self.stable_since < .25 or not consumed:
            return
        if self.probe_index >= len(PROBES):
            self.interrupt("no_consumed_downward_mode_effect")
            return
        self.raw_probe = PROBES[self.probe_index]
        self.probe_index += 1
        self.probe_at, self.probe_seq, self.probe_position = now, game.seq, position
        self.probe_consumed = False

    def reference_stick(self, game):
        if self.terminal or self.phase == "verify" or not game.player or not game.player.climbing_ladder:
            return (0, 0)
        return self.learned_stick or self.raw_probe or (0, 0)
