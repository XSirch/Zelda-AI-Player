"""Raw attached-mode Laya output with an input/body-frame lease, never walking reprojection."""
from __future__ import annotations

import math
import time

from ..laya_data import LADDER_PROFILE, bounded_state
from .ladder_task import LadderDescentTask
from .laya_policy import DecisionLease, DecisionRequest, LayaWalkingPolicy


def encode_ladder(game, task):
    if not game.player or game.camera_input_yaw is None:
        raise ValueError("Attached telemetry requires current native input orientation")
    position, target = game.player.position, task.target
    body = game.player.yaw * math.pi / 32768
    camera = game.camera_input_yaw * math.pi / 32768
    return [max(-1., min(1., (target[1] - position[1]) / 280)),
            min(1., math.hypot(target[0] - position[0], target[2] - position[2]) / 280),
            math.sin(body), math.cos(body), math.sin(camera), math.cos(camera),
            float(game.mirrored_world), float(game.player.climbing_ladder), float(bool(game.player.bg_check_flags & 1))]


class AttachedDecisionLease(DecisionLease):
    def resolve(self, owner, features, *, now, budget_s, camera_yaw=0):
        if (self.owner != owner or not 0 <= now - self.submitted_at <= budget_s
                or abs(self.features[0] - features[0]) > .08):
            return (0, 0)
        for index in (2, 4):
            alignment = sum(a * b for a, b in zip(self.features[index:index + 2], features[index:index + 2]))
            if alignment < math.cos(math.radians(12)):
                return (0, 0)
        return self.stick


class LayaLadderPolicy(LayaWalkingPolicy):
    profile = LADDER_PROFILE
    lease_type = AttachedDecisionLease

    def __call__(self, game, task):
        if (self.closed or self.failure or task.terminal or task.phase == "verify"
                or not isinstance(task, LadderDescentTask) or task.version != LadderDescentTask.VERSION
                or task.kind != "ladder_down"
                or game.source != "soh" or not game.in_game or not game.player
                or game.player.health <= 0 or game.game_over_state or game.paused
                or game.cutscene_active or game.dialogue.active or game.pause_menu.active
                or not game.player.climbing_ladder or game.player.hanging_ledge or game.player.climbing_ledge
                or game.camera_input_yaw is None):
            self.invalidate()
            return (0, 0)
        owner, features = self.key(game, task), encode_ladder(game, task)
        if owner != self.owner:
            self.invalidate()
            self.owner = owner
        now = time.monotonic()
        if game.seq > self.last_seq:
            self.history = (self.history + [features])[-4:]
            self.last_seq, self.request_id = game.seq, self.request_id + 1
            self.pending = DecisionRequest(self.request_id, owner, now, tuple(features),
                bounded_state(self.history, profile=self.profile), game.camera_input_yaw * math.pi / 32768)
            self.event.set()
        stick = self.lease.resolve(owner, features, now=now, budget_s=self.budget_s + .05)
        if stick != (0, 0):
            self.metrics["output_age_ms"].append((now - self.lease.submitted_at) * 1000)
            self.metrics["output_age_ms"] = self.metrics["output_age_ms"][-4096:]
        return stick
