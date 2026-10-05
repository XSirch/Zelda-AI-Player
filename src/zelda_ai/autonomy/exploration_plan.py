"""Bounded goal-local planning from current collision, with actual footprints.

Working memory stores occupied regions and attempted targets, never proposed
route edges. It is ephemeral in frozen evaluation and isolated by context.
"""
from __future__ import annotations

import math
import time
from collections import Counter

from .execution import physical_context
from .ground_descent_task import GroundDescentApproachTask
from .local_tasks import LocalTask
from .locomotion import grounded, locomotor_mode
from .navigation import observed_local_path
from .portal_task import ObservedPortalTask


class ObservedExplorationPlan:
    VERSION = "current-collision-exploration-plan-v3"

    def __init__(self):
        self.visits, self.failures, self.retry_after = {}, {}, {}
        self.last_seq, self.last_context, self.last_region = -1, None, None
        self.entry_position, self.entry_at = None, 0
        self.origins, self.max_radius = {}, {}
        self.context_actions = Counter()
        self.event_ids = set()
        self.replans = self.actual_transitions = 0
        self.selection = {}

    @staticmethod
    def region(context, point):
        return (*context, math.floor(point[0] / 160), math.floor(point[1] / 50), math.floor(point[2] / 160))

    @staticmethod
    def bounded_put(mapping, key, value, *, limit=4096):
        if key not in mapping and len(mapping) >= limit:
            del mapping[next(iter(mapping))]
        mapping[key] = value

    def observe(self, game, *, now=None):
        if (not game.player or not game.in_game or game.player.health <= 0
                or game.game_over_state or game.seq <= self.last_seq):
            return
        now = time.monotonic() if now is None else now
        self.last_seq = game.seq
        context, position = physical_context(game), tuple(game.player.position)
        if self.last_context != context:
            same_lifetime = (self.last_context is not None and context[0] == self.last_context[0]
                and context[3:] == self.last_context[3:] and not any(e.id not in self.event_ids
                and e.kind in {"save_loaded", "player_died", "game_over"} for e in game.events))
            if same_lifetime and context[1:3] != self.last_context[1:3]:
                self.actual_transitions += 1
            self.last_context, self.entry_position, self.entry_at = context, position, now
        if context not in self.origins:
            self.bounded_put(self.origins, context, position, limit=128)
        self.bounded_put(self.max_radius, context,
                         max(self.max_radius.get(context, 0), math.dist(position, self.origins[context])), limit=128)
        region = self.region(context, position)
        if region != self.last_region:
            self.bounded_put(self.visits, region, self.visits.get(region, 0) + 1)
            self.last_region = region
        label = game.context_action.label
        if label != "none" and (label in self.context_actions or len(self.context_actions) < 32):
            self.context_actions[label] += 1
        self.event_ids = {e.id for e in game.events}

    def task_key(self, task):
        if isinstance(task, ObservedPortalTask):
            return (*task.context, task.kind, task.exit_index, task.entrance_index)
        return (*self.region(task.context, task.target), task.kind)

    def outcome(self, task, *, success, now=None):
        now = time.monotonic() if now is None else now
        key = self.task_key(task)
        failures = max(0, self.failures.get(key, 0) - 1) if success else self.failures.get(key, 0) + 1
        # Modal/context interrupts are not physical geometry failures.
        if (not success and task.phase == "interrupted" and task.failure in {
                "context_changed", "modal_owns_control", "game_not_playable",
                "portal_episode_changed", "portal_player_context_changed"}):
            failures = self.failures.get(key, 0)
        self.bounded_put(self.failures, key, failures, limit=512)
        self.bounded_put(self.retry_after, key, now + (30 if isinstance(task, ObservedPortalTask) else 20), limit=512)

    def choose(self, game, *, budget_s, now=None):
        now = time.monotonic() if now is None else now
        self.observe(game, now=now)
        self.selection = {"observation_seq": game.seq, "full_seq": game.full_seq, "monotonic_s": now,
                          "mesh_cells": len(game.navmesh.cells), "reachable_floor_candidates": 0,
                          "cooled_floor_candidates": 0, "eligible_floor_candidates": 0}
        if (game.dialogue.active or game.pause_menu.active or game.paused or game.cutscene_active):
            raise ValueError("modal_requires_separate_controller")
        if not grounded(game.player):
            raise ValueError(f"unsupported_planning_mode:{locomotor_mode(game)}")
        if not game.navmesh.available or game.camera_input_yaw is None:
            raise ValueError("current_ground_collision_unavailable")
        if not 1 <= budget_s <= 300:
            raise ValueError("planning_budget_exhausted")
        context, position = physical_context(game), game.player.position
        candidates = []

        def offer(task, priority, distance, *, approach_distance=0):
            key = self.task_key(task)
            if self.retry_after.get(key, 0) > now:
                return
            visits = self.visits.get(self.region(context, task.target), 0)
            # Horizontal frontiers seek expansion; vertical traversals first
            # prefer an observed nearby upper approach, then a shorter landing.
            rank = (priority, self.failures.get(key, 0), visits, approach_distance,
                    -distance if priority == 2 else distance, tuple(task.target))
            candidates.append((rank, task))

        for row in game.scene_exits:
            # A new scene's entry doorway is held back briefly to avoid a
            # reflexive return. This does not infer its unseen destination.
            if (self.actual_transitions and now - self.entry_at < 15
                    and math.dist(row.position, self.entry_position) <= 130):
                continue
            try:
                task = ObservedPortalTask.create(game, row, now=now, budget_s=min(30, budget_s))
            except ValueError:
                continue
            offer(task, 0, math.dist(position, row.position))

        for row in game.traversal_affordances:
            if row.kind != "ledge_down" or self.region(context, row.target_position) in self.visits:
                continue
            try:
                task = GroundDescentApproachTask.create(game, row, now=now, budget_s=min(20, budget_s))
            except ValueError:
                continue
            offer(task, 1, math.dist(position, row.target_position), approach_distance=row.distance)

        mesh, floor_candidates = game.navmesh, []
        for x, z, y, _ in mesh.cells:
            point = (mesh.origin[0] + x * mesh.step, y, mesh.origin[2] + z * mesh.step)
            distance = math.dist(position, point)
            if not 55 <= distance <= 280 or abs(y - game.player.floor_height) > 24:
                continue
            path = observed_local_path(game, point, minimum_gain=0)
            if not path or path["partial"] or path["target_gap"] > 4:
                continue
            self.selection["reachable_floor_candidates"] += 1
            heading = math.atan2(point[0] - position[0], point[2] - position[2])
            forward = math.cos(heading - game.player.yaw * math.pi / 32768) >= -.25
            task = LocalTask.observed_cell(game, point, now=now, budget_s=min(12, budget_s))
            if self.retry_after.get(self.task_key(task), 0) <= now:
                floor_candidates.append((forward, task, distance))
            else:
                self.selection["cooled_floor_candidates"] += 1
        # A rearward frontier is eligible only without a current forward/side
        # path. Every candidate still needs actual directed collision links.
        if any(row[0] for row in floor_candidates):
            floor_candidates = [row for row in floor_candidates if row[0]]
        for _, task, distance in floor_candidates:
            offer(task, 2, distance)
        self.selection["eligible_floor_candidates"] = len(floor_candidates)
        if not candidates:
            raise ValueError("no_eligible_current_collision_task")
        self.replans += 1
        return min(candidates, key=lambda row: row[0])[1]

    def snapshot(self, *, include_details=False):
        result = {"version": self.VERSION, "replans": self.replans,
                "actual_scene_room_transitions": self.actual_transitions,
                "occupied_macro_regions": len(self.visits),
                "failed_target_regions": sum(value > 0 for value in self.failures.values()),
                "max_outward_radius": max(self.max_radius.values(), default=0),
                "observed_context_actions": dict(self.context_actions),
                "memory": "ephemeral_actual_positions_and_attempts_only", "last_selection": dict(self.selection)}
        if include_details:
            result.update(last_seq=self.last_seq, last_context=self.last_context, last_region=self.last_region,
                entry_position=self.entry_position, entry_at=self.entry_at,
                visited_regions=[{"key": key, "visits": value} for key, value in self.visits.items()],
                attempted_targets=[{"key": key, "failures": value, "retry_after": self.retry_after.get(key, 0)}
                                   for key, value in self.failures.items()])
        return result
