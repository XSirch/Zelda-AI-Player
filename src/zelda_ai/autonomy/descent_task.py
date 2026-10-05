"""Bounded QA descent from an observed ledge to an observed landing.

This is a separate reference preparation, not an extension of a trained walking
candidate. Attached ladder/ledge modes retain their existing stop guard.
"""
from __future__ import annotations

import math

from .local_tasks import LocalTask
from .navigation import observed_local_path


class ObservedDescentTask(LocalTask):
    @classmethod
    def create(cls, game, affordance, *, now=None, budget_s=20):
        if affordance not in game.traversal_affordances or affordance.kind != "ledge_down":
            raise ValueError("Descent needs a currently observed ledge_down affordance")
        if not game.player or not game.navmesh.available or not cls.grounded(game):
            raise ValueError("Descent needs current collision observations")
        drop = game.player.position[1] - affordance.target_position[1]
        observed_delta = affordance.target_position[1] - game.player.floor_height
        if not 8 <= drop <= 120 or abs(observed_delta - affordance.height_delta) > 4:
            raise ValueError("Descent landing must agree with the native observed floor delta")
        # Native ledge_down targets are actual floor-ray hits. Do not demand an
        # identical height from a different, 70-unit mesh sample on a slope.
        # The radial affordance's approach can already lie below the edge.
        # Approach only via directed cells observed on the current upper floor.
        path = observed_local_path(game, affordance.target_position, minimum_gain=0)
        if not path or not path["waypoints"]:
            raise ValueError("No observed upper-floor approach to the landing direction")
        approach = path["waypoints"][-1]
        if abs(approach[1] - game.player.position[1]) > 24:
            raise ValueError("Descent approach must stay on the currently observed upper floor")
        task = cls._create(game, "ledge_down", affordance.target_position, approach,
                           now=now, budget_s=budget_s)
        task.version = "observed-descent-reference-v1"
        return task

    @staticmethod
    def grounded(game):
        player = game.player
        # Pinned SoH BGCHECKFLAG_GROUND, plus actual floor contact height.
        return bool(player and player.bg_check_flags & 1
                    and abs(player.position[1] - player.floor_height) <= 4)

    def observe(self, game, *, consumed, now=None):
        grounded = self.grounded(game)
        if not grounded:
            self.verification_frames = 0
        super().observe(game, consumed=consumed, now=now)
        if not grounded:
            self.verification_frames = 0
        if self.phase == "succeeded" and not grounded:
            self.phase = "verify"

    def landing_verified(self, game):
        return bool(self.phase == "succeeded" and self.grounded(game)
                    and game.player.speed_xz < .1
                    and abs(game.player.position[1] - self.target[1]) <= 4
                    and math.hypot(game.player.position[0] - self.target[0],
                                   game.player.position[2] - self.target[2]) <= 35)
