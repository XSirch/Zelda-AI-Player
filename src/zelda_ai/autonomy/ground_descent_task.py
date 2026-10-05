"""Dry-ground approach to a current descent; attached output belongs elsewhere.

The projected point expresses a heading on the observed upper floor. It does
not claim a walking link across the drop or change the actual landing target.
"""
from __future__ import annotations

import math
import time

from .descent_task import ObservedDescentTask
from .navigation import observed_local_path


class GroundDescentApproachTask(ObservedDescentTask):
    VERSION = "observed-ground-descent-approach-v1"

    @classmethod
    def create(cls, game, affordance, *, now=None, budget_s=20):
        task = super().create(game, affordance, now=now, budget_s=budget_s)
        task.kind, task.version = "observed_descent_approach", cls.VERSION
        if math.dist(game.player.position, task.approach) > 18:
            path = observed_local_path(game, task.approach, minimum_gain=0)
            if not path or path["partial"] or path["target_gap"] > 4:
                raise ValueError("Descent preparation needs a complete observed upper-floor path")
            task.corridor, task.waypoint_index = path["waypoints"], 0
            task.phase = "prepare"
        task._track_waypoint(game)
        return task

    def steering_point(self, game):
        if self.phase == "prepare":
            return self.corridor[self.waypoint_index] if self.corridor else self.approach
        return (self.target[0], self.origin[1], self.target[2])

    def observe(self, game, *, consumed, now=None):
        if self.terminal:
            return
        # Current support may be a native landing ray or a neighboring mesh
        # floor; the exact landing need not be a body-clear walking node.
        if self.grounded(game):
            mesh = game.navmesh
            supported = (mesh.available and any(
                abs(y - self.target[1]) <= 4 and math.hypot(
                    mesh.origin[0] + x * mesh.step - self.target[0],
                    mesh.origin[2] + z * mesh.step - self.target[2]) <= mesh.step * 1.2
                for x, z, y, _ in mesh.cells))
            supported |= any(a.kind == "ledge_down" and math.dist(a.target_position, self.target) <= 35
                             for a in game.traversal_affordances)
            if not supported:
                self.interrupt("current_landing_support_lost")
                return
        preparing = self.phase == "prepare"
        if (game.player and game.seq > max(self.last_seq, self.origin_seq) and consumed
                and self.progress_point is not None):
            distance = math.dist(game.player.position, self.progress_point)
            if distance < self.best_waypoint_distance - 4:
                self.best_waypoint_distance = distance
                self.progress_at = time.monotonic() if now is None else now
        super().observe(game, consumed=consumed, now=now)
        if self.terminal:
            return  # Includes the actual ladder handoff; walking cannot continue underneath it.
        if preparing:
            while (self.waypoint_index + 1 < len(self.corridor)
                   and self._waypoint_reached(game.player.position, self.corridor[self.waypoint_index])):
                self.waypoint_index += 1
            self.phase = "execute" if self._waypoint_reached(game.player.position, self.approach) else "prepare"
        self._track_waypoint(game)

    def guidance(self, game):
        return {**super().guidance(game), "local_path_active": self.phase == "prepare" and bool(self.corridor)}
