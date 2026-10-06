"""Walk a current native exit; completion is a settled physical transition.

Partial approach corridors use only observed directed collision links. They
are reobserved at their endpoint and never persisted as traversed routes.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .execution import physical_context
from .local_tasks import LocalTask
from .locomotion import grounded
from .navigation import observed_local_path


@dataclass
class ObservedPortalTask(LocalTask):
    exit_index: int = 0
    entrance_index: int = 0
    initial_events: tuple = ()
    last_source_position: tuple | None = None
    crossing_context: tuple | None = None
    crossing_seq: int | None = None
    corridor_replans: int = 0
    local_detours: tuple = ()
    visited_positions: tuple = ()
    collision_refinement_count: int = 0
    collision_refinement_started: float | None = None
    collision_refinement_full_seq: int = 0

    @classmethod
    def create(cls, game, exit_observation, *, now=None, budget_s=30):
        if exit_observation not in game.scene_exits:
            raise ValueError("A portal must be a currently observed native exit")
        if not grounded(game.player) or game.camera_input_yaw is None:
            raise ValueError("Walking portal traversal requires dry ground and native input yaw")
        if abs(exit_observation.position[1] - game.player.floor_height) > 25:
            raise ValueError("This portal needs a separate vertical traversal")
        task = cls._create(game, "observed_portal", exit_observation.position,
                           game.player.position, now=now, budget_s=budget_s)
        task.exit_index, task.entrance_index = exit_observation.exit_index, exit_observation.entrance_index
        task.initial_events = tuple(e.id for e in game.events)
        task.last_source_position = tuple(game.player.position)
        task.visited_positions = (tuple(game.player.position),)
        task.version = "observed-walking-portal-v3"
        task._plan(game, exit_observation)
        task._track_waypoint(game)
        return task

    def _exit(self, game):
        return next((e for e in game.scene_exits if e.exit_index == self.exit_index
                     and e.entrance_index == self.entrance_index), None)

    def _plan(self, game, observation):
        self.target = tuple(observation.position)
        if observation.direct_reachable:
            self.corridor, self.waypoint_index = (), 0
        else:
            path = observed_local_path(game, self.target, minimum_gain=4)
            if path is None:
                path = self._observed_detour(game)
                if path is None:
                    raise ValueError("Observed portal approach has no further collision-proven progress")
            self.corridor, self.waypoint_index = path["waypoints"], 0
        self.corridor_replans += 1

    def _observed_detour(self, game):
        """Bounded recovery over actual mesh links, even with initial negative gain."""
        if len(self.local_detours) >= 3 or not game.navmesh.available:
            return None
        mesh, position = game.navmesh, game.player.position
        candidates = []
        for x, z, y, _ in mesh.cells:
            point = (mesh.origin[0] + x * mesh.step, y, mesh.origin[2] + z * mesh.step)
            if (abs(y - game.player.floor_height) > 4 or not 40 <= math.dist(position, point) <= 280
                    or any(math.dist(point, p) < 35 for p in self.visited_positions + self.local_detours)):
                continue
            path = observed_local_path(game, point, minimum_gain=0)
            if path and not path["partial"] and path["target_gap"] <= 4 and path["path_length"] <= 560:
                candidates.append((math.dist(point, self.target), path["path_length"], point, path))
        if not candidates:
            return None
        _, _, point, path = min(candidates, key=lambda row: row[:3])
        self.local_detours += (point,)
        return path

    def steering_point(self, game):
        return self._corridor_point()

    def _exhausted_approach(self, game, now):
        if (game.protocol == 3 and 'navmesh_refinement' in game.capabilities
                and game.navmesh.step > 35 and not self.collision_refinement_count):
            self.collision_refinement_count = 1
            self.collision_refinement_started = now
            self.collision_refinement_full_seq = game.full_seq
            self.phase = 'collision_refinement'
        else:
            self.phase, self.failure = 'failed', 'observed_approach_exhausted'

    def observe(self, game, *, consumed, now=None):
        if self.terminal:
            return
        now = time.monotonic() if now is None else now
        if (game.source != "soh" or game.instance_id != self.context[0]
                or game.mirrored_world != self.context[3] or game.seq < self.origin_seq
                or any(e.id not in self.initial_events and e.kind in {"save_loaded", "player_died", "game_over"}
                       for e in game.events)):
            self.interrupt("portal_episode_changed")
            return
        if now >= self.deadline:
            self.phase, self.failure = "failed", "attempt_timeout"
            return
        fresh = game.seq > self.last_seq and game.seq > self.origin_seq
        self.last_seq = max(self.last_seq, game.seq)
        self.consumed = self.consumed or consumed
        if game.player and (game.player.health <= 0 or game.game_over_state or game.player.age != self.context[4]):
            self.interrupt("portal_player_context_changed")
            return
        if not game.in_game or not game.player:
            self.phase = "transition"
            return  # No input during loading; the original deadline still applies.
        current = physical_context(game)
        if current[1:3] != self.context[1:3]:
            near = math.dist(self.last_source_position, self.target) <= 80
            if not self.consumed or not near or game.scene_epoch <= self.scene_epoch:
                self.interrupt("unverified_portal_transition")
                return
            if self.crossing_context is None:
                self.crossing_context, self.crossing_seq = current, game.seq
            elif current != self.crossing_context:
                self.interrupt("another_transition_before_verification")
                return
            settled = (grounded(game.player) and game.player.speed_xz < .1
                       and not game.paused and not game.cutscene_active
                       and not game.dialogue.active and not game.pause_menu.active)
            if fresh:
                self.verification_frames = self.verification_frames + 1 if settled else 0
            self.phase = "succeeded" if self.verification_frames >= 3 else "transition"
            return
        if self.crossing_context is not None or game.scene_epoch != self.scene_epoch:
            self.interrupt("returned_before_portal_verification")
            return
        if (game.paused or game.cutscene_active or game.dialogue.active or game.pause_menu.active):
            self.interrupt("modal_owns_control")
            return
        if not grounded(game.player):
            self.interrupt("unsupported_locomotor_mode")
            return
        self.last_source_position = tuple(game.player.position)
        if fresh and all(math.dist(game.player.position, p) >= 20 for p in self.visited_positions):
            self.visited_positions = (self.visited_positions + (tuple(game.player.position),))[-96:]
        if fresh and consumed and self.progress_point is not None:
            distance = math.dist(game.player.position, self.progress_point)
            if distance < self.best_waypoint_distance - 4:
                self.best_waypoint_distance, self.progress_at = distance, now
        observation = self._exit(game)
        if observation is None:
            self.interrupt("current_native_exit_lost")
            return
        if self.phase == 'collision_refinement':
            if now-self.progress_at >= 2.5:
                self.phase, self.failure = 'failed','no_geometric_progress'
                return
            owned_fine = (game.full_seq > self.collision_refinement_full_seq
                and 0 < game.navmesh.step <= 35
                and game.navmesh.refinement_request_id == self.origin_seq)
            if not owned_fine:
                if now-self.collision_refinement_started >= 1:
                    self.phase, self.failure = 'failed','collision_refinement_timeout'
                return
            try:
                self._plan(game,observation)
            except ValueError:
                self.phase, self.failure = 'failed','observed_approach_exhausted'
                return
            self.phase = 'execute'
        while (self.waypoint_index + 1 < len(self.corridor)
               and self._waypoint_reached(game.player.position, self._corridor_point())):
            self.waypoint_index += 1
        if self.corridor and self._waypoint_reached(game.player.position, self._corridor_point()):
            try:
                self._plan(game, observation)
            except ValueError:
                self._exhausted_approach(game,now)
                return
        elif not self.corridor:
            # A direct exit floor query is current evidence, not a permanent
            # attractor. Losing it requires another observed approach corridor.
            try:
                self._plan(game, observation)
            except ValueError:
                self._exhausted_approach(game,now)
                return
        self._track_waypoint(game)
        self.best_distance = min(self.best_distance, math.dist(game.player.position, self.target))
        if now - self.progress_at >= 2.5:
            self.phase, self.failure = "failed", "no_geometric_progress"
        else:
            self.phase = "execute"  # Proximity never completes or brakes a portal.

    def reference_stick(self, game):
        if self.phase in {"transition",'collision_refinement'}:
            return (0, 0)
        return super().reference_stick(game)

    def guidance(self, game):
        if self.phase in {"transition",'collision_refinement'} or not game.player:
            return {"active": False, "source": "observed_portal_transition", "target": self.target,
                    "stick": (0, 0), "strength": 0., "button_quiet": 1., "blocked": False}
        return {**super().guidance(game), "local_path_active": bool(self.corridor)}
