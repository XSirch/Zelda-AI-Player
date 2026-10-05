"""Bounded ascent of a ledge the native player currently observes at contact.

The reference approach uses analog only. The game's own normal movement may
step/jump/climb; no semantic jump button or walking connection is invented.
"""
from __future__ import annotations

import math

from .local_tasks import LocalTask
from .locomotion import FREEFALL, JUMPING, grounded


class ObservedLedgeAscentTask(LocalTask):
    @classmethod
    def create(cls, game, affordance, *, now=None, budget_s=8):
        if affordance not in game.traversal_affordances or affordance.kind != "ledge_up":
            raise ValueError("Ascent requires a currently observed native ledge_up")
        if not grounded(game.player) or game.camera_input_yaw is None:
            raise ValueError("Ledge ascent needs dry ground contact and native input yaw")
        origin, target = game.player.position, affordance.target_position
        if (affordance.direction != "up" or not 18 <= target[1] - origin[1] <= 120
                or abs(target[1] - game.player.floor_height - affordance.height_delta) > 4
                or math.hypot(target[0] - origin[0], target[2] - origin[2]) > 70
                or math.dist(affordance.approach_position, origin) > 4):
            raise ValueError("Ledge ascent must retain the current contact's observed local landing")
        task = cls._create(game, "ledge_up", target, origin, now=now, budget_s=budget_s)
        task.version = "observed-contact-ledge-ascent-reference-v1"
        return task

    def accepts_attached_mode(self, game):
        return bool(game.player.climbing_ledge and not game.player.climbing_ladder
                    and not game.player.hanging_ledge)

    def reference_stick(self, game):
        if (not game.player or game.player.climbing_ledge
                or game.player.state_flags_1 & (JUMPING | FREEFALL)):
            # Let the native animation/ballistic movement settle. The original
            # task budget remains in force; an airborne apex is never success.
            return (0, 0)
        return super().reference_stick(game)
