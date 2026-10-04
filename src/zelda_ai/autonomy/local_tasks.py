"""Bounded physical tasks over current observations, independent of cognition.

These tasks own analog control exclusively. They neither change the strategic
objective nor create route-memory edges. Walking/slope control is deliberately
not reused while Link is attached to a ladder or hanging from a ledge.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass

from .execution import physical_context
from .features import _PROBE_YAW_OFFSETS, _collision_detour, camera_relative_stick
from .navigation import observed_local_path


def walking_surface_supported(game, affordance):
    """A nearby floor sample alone does not prove the raised cell is walkable."""
    if not game.navmesh.available:
        return True  # The explicit local affordance remains bounded evidence.
    ox, _, oz = game.navmesh.origin
    tx, ty, tz = affordance.target_position
    return any(abs(y - ty) <= 4 and math.hypot(ox + x * game.navmesh.step - tx,
                oz + z * game.navmesh.step - tz) <= game.navmesh.step * 1.2
               for x, z, y, _ in game.navmesh.cells)


@dataclass
class LocalTask:
    identity: str
    kind: str
    context: tuple
    scene_epoch: int
    origin_seq: int
    load_events: tuple
    origin: tuple
    target: tuple
    approach: tuple
    started: float
    deadline: float
    progress_at: float
    best_distance: float
    phase: str = "prepare"
    failure: str | None = None
    consumed: bool = False
    verification_frames: int = 0
    last_seq: int = -1
    detour: tuple | None = None
    version: str = "observed-analog-task-v1"

    @classmethod
    def traversal(cls, game, affordance, *, now=None, budget_s=8.0):
        if affordance not in game.traversal_affordances:
            raise ValueError("Traversal must come from a currently observed affordance")
        if affordance.kind not in {"stairs_or_slope_up", "stairs_or_slope_down"}:
            raise ValueError("This task supports observed walking surfaces only")
        if not walking_surface_supported(game, affordance):
            raise ValueError("Observed surface floor is not supported by current walkable cells")
        return cls._create(game, affordance.kind, affordance.target_position,
                           affordance.approach_position, now=now, budget_s=budget_s)

    @classmethod
    def observed_cell(cls, game, target, *, now=None, budget_s=8.0):
        path = observed_local_path(game, target, minimum_gain=0)
        if not path or path["partial"] or path["target_gap"] > 4:
            raise ValueError("Cell must be reachable through currently observed collision links")
        return cls._create(game, "observed_cell", target, game.player.position,
                           now=now, budget_s=budget_s)

    @classmethod
    def _create(cls, game, kind, target, approach, *, now, budget_s):
        if (game.source != "soh" or not game.in_game or not game.player
                or game.player.health <= 0 or game.game_over_state
                or game.cutscene_active or game.paused
                or game.dialogue.active or game.pause_menu.active):
            raise ValueError("A physical task requires a playable observed state")
        if not 0 < budget_s <= 30:
            raise ValueError("Task budget must be in (0, 30] seconds")
        now = time.monotonic() if now is None else now
        origin, target, approach = tuple(game.player.position), tuple(target), tuple(approach)
        return cls(f"{kind}:{game.seq}:{target}", kind, physical_context(game),
                   game.scene_epoch, game.seq, cls._loads(game), origin, target, approach,
                   now, now + budget_s, now, math.dist(origin, target),
                   phase="execute" if math.dist(origin, approach) <= 35 else "prepare")

    @staticmethod
    def _loads(game):
        return tuple(event.id for event in game.events if event.kind == "save_loaded")

    @property
    def terminal(self):
        return self.phase in {"succeeded", "failed", "interrupted"}

    def interrupt(self, failure):
        if not self.terminal:
            self.phase, self.failure = "interrupted", failure

    def observe(self, game, *, consumed, now=None):
        if self.terminal:
            return
        now = time.monotonic() if now is None else now
        if (physical_context(game) != self.context or game.scene_epoch != self.scene_epoch
                or game.seq < self.origin_seq
                or any(event not in self.load_events for event in self._loads(game))):
            self.interrupt("context_changed")
            return
        if not game.player or game.player.health <= 0 or game.game_over_state or not game.in_game:
            self.interrupt("game_not_playable")
            return
        if game.dialogue.active or game.pause_menu.active or game.paused or game.cutscene_active:
            self.interrupt("modal_owns_control")
            return
        if game.player.climbing_ladder or game.player.hanging_ledge or game.player.climbing_ledge:
            self.interrupt("unsupported_locomotor_mode")
            return
        if self.kind == "observed_cell":
            if self.detour and math.dist(game.player.position, self.detour) <= 25:
                self.detour = None
            if self.detour is None:
                dx, dz = self.target[0] - game.player.position[0], self.target[2] - game.player.position[2]
                _, evidence = _collision_detour(
                    game, game.player, math.atan2(dx, dz), use_body_contact=True,
                )
                direction = evidence.get("detour")
                if direction:
                    probe = min((p for p in game.navigation_probes if p.direction == direction),
                                key=lambda p: abs(p.distance - 70))
                    yaw = game.player.yaw * math.pi / 32768 + _PROBE_YAW_OFFSETS[direction]
                    x, y, z = game.player.position
                    self.detour = (x + math.sin(yaw) * probe.distance,
                                   y + (probe.delta_y or 0), z + math.cos(yaw) * probe.distance)
        self.consumed = self.consumed or consumed
        fresh = game.seq > self.last_seq and game.seq > self.origin_seq
        self.last_seq = max(self.last_seq, game.seq)
        position = game.player.position
        distance = math.dist(position, self.target)
        if distance < self.best_distance - 4:
            self.best_distance, self.progress_at = distance, now
        horizontal = math.hypot(position[0] - self.target[0], position[2] - self.target[2])
        reached = horizontal <= 35 and abs(position[1] - self.target[1]) <= 4
        # Actual gain is necessary: approaching the base of a raised surface
        # must not count as a climb. Verify stopped, fresh frames, not a jump apex.
        delta = self.target[1] - self.origin[1]
        reached = reached and (abs(delta) <= 4 or (position[1] - self.origin[1]) * delta > 0)
        reached = reached and game.player.speed_xz < .1
        if fresh:
            self.verification_frames = self.verification_frames + 1 if reached and self.consumed else 0
        if self.verification_frames >= 3:
            self.phase = "succeeded"
        elif now >= self.deadline:
            self.phase, self.failure = "failed", "attempt_timeout"
        elif now - self.progress_at >= 2.5:
            self.phase, self.failure = "failed", "no_geometric_progress"
        elif (horizontal <= 35 and abs(position[1] - self.target[1]) <= 4
              or self.phase == "verify" and game.player.speed_xz >= .1):
            self.phase = "verify"
        elif math.dist(position, self.approach) > 35 and self.phase == "prepare":
            self.phase = "prepare"
        else:
            self.phase = "execute"

    def steering_point(self, game):
        if self.detour:
            return self.detour
        if self.phase == "prepare":
            path = observed_local_path(game, self.approach, minimum_gain=0)
            if path and path["path_nodes"] > 2:
                return path["waypoint"]
            return self.approach
        if self.kind == "observed_cell":
            path = observed_local_path(game, self.target, minimum_gain=0)
            if path and path["path_nodes"] > 2:
                return path["waypoint"]
        return self.target

    def reference_stick(self, game):
        if self.terminal or self.phase == "verify" or not game.player:
            return (0, 0)
        point = self.steering_point(game)
        dx, dz = point[0] - game.player.position[0], point[2] - game.player.position[2]
        if math.hypot(dx, dz) < 1e-4:
            return (0, 0)
        direction = camera_relative_stick(game, game.player, math.atan2(dx, dz))
        # Approach slowly before entering verification. A full-speed command
        # followed by an immediate reversal can orbit a small success radius.
        # Maintain enough physical stick to traverse an observed step. The
        # neutral verification phase handles braking after reaching its floor.
        magnitude = min(60, max(40, math.hypot(dx, dz)))
        return tuple(round(value * magnitude) for value in direction)

    def guidance(self, game):
        point = self.steering_point(game)
        stick = self.reference_stick(game)
        return {"active": not self.terminal, "source": "local_physical_task",
                "target": point, "distance": math.dist(game.player.position, self.target),
                "stick": tuple(v / 80 for v in stick), "strength": 1.0,
                "button_quiet": 1.0, "local_task": self.identity, "blocked": False}

    def snapshot(self):
        return asdict(self)
