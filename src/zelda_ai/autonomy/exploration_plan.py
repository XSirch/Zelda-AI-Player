"""Bounded goal-local planning from current collision, with actual footprints.

Working memory stores occupied regions and failed local attempts, never proposed
traversed route edges. It is ephemeral in frozen evaluation and isolated by context.
"""
from __future__ import annotations

import math
import time
from collections import Counter

from .camera_task import ObservedCameraReturnTask, eligible_camera_return
from .collectible_task import ObservedCollectibleApproachTask
from .container_task import ObservedContainerApproachTask
from .context_tasks import LinearDialogueTask, NativeModalWaitTask, ObservedContextTask, eligible_context
from .execution import physical_context
from .ground_descent_task import GroundDescentApproachTask
from .local_tasks import LocalTask
from .locomotion import camera_modal_active, grounded, locomotor_mode
from .navigation import observed_local_path
from .portal_task import ObservedPortalTask


class ObservedExplorationPlan:
    VERSION = "current-collision-exploration-plan-v13"

    def __init__(self, *, contextual_interactions=False):
        self.contextual_interactions = contextual_interactions
        self.visits, self.failures, self.retry_after = {}, {}, {}
        self.occupied_floors = {}
        self.last_seq, self.last_context, self.last_region = -1, None, None
        self.last_position = None
        self.failed_departures, self.departure_deferrals = {}, 0
        self.entry_position, self.entry_at = None, 0
        self.origins, self.max_radius = {}, {}
        self.context_actions = Counter()
        self.opened_containers = {}
        self.read_actor_contexts = {}
        self.progress_tokens, self.progress_revision, self.progress_initialized = set(), 0, False
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
        tokens = ({("item", value) for value in game.inventory if 0 <= value < 255}
            | {("equipment", name) for name in game.progress.owned_equipment}
            | {("quest", name) for name in game.progress.quest_items}
            | {("story", name) for name, value in game.progress.story_flags.items() if value})
        novel = sorted(tokens - self.progress_tokens)[:max(0, 512-len(self.progress_tokens))]
        if novel and self.progress_initialized:
            self.progress_revision += 1
        self.progress_tokens.update(novel)
        self.progress_initialized = True
        context, position = physical_context(game), tuple(game.player.position)
        self.last_position = position
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
        if grounded(game.player) and abs(position[1] - game.player.floor_height) <= 4:
            point = (position[0], game.player.floor_height, position[2])
            key = (*context, math.floor(point[0] / 20), math.floor(point[1] / 8), math.floor(point[2] / 20))
            self.bounded_put(self.occupied_floors, key, (context, point))
        label = game.context_action.label
        if label != "none" and (label in self.context_actions or len(self.context_actions) < 32):
            self.context_actions[label] += 1
        self.event_ids = {e.id for e in game.events}

    def task_key(self, task):
        if isinstance(task, ObservedPortalTask):
            return (*task.context, task.kind, task.exit_index, task.entrance_index)
        if isinstance(task, ObservedContextTask):
            if task.context_actor_uid:
                return (*task.context, task.kind, task.context_actor_uid, task.context_action_key)
            return (*self.region(task.context, task.origin), task.kind, task.context_action_key)
        if isinstance(task, (ObservedContainerApproachTask, ObservedCollectibleApproachTask)):
            return (*task.context, task.kind, task.actor_uid)
        return (*self.region(task.context, task.target), task.kind)

    def outcome(self, task, *, success, now=None):
        now = time.monotonic() if now is None else now
        key = self.task_key(task)
        self._departure_outcome(task, success=success, now=now)
        if (success and isinstance(task, LinearDialogueTask) and task.phase == "succeeded"
                and task.observed_effect == "dialogue_closed" and task.consumed
                and task.verification_frames >= 3):
            for speaker_context in task.read_speaker_contexts:
                self.bounded_put(self.read_actor_contexts, speaker_context,
                    {"progress_revision": self.progress_revision, "retry_after": now+120}, limit=512)
        if (success and isinstance(task, ObservedContextTask) and task.observed_effect == "chest_opened"
                and task.context_actor_uid and task.effect_event_id):
            self.bounded_put(self.opened_containers, (*task.context, task.context_actor_uid),
                task.effect_event_id, limit=512)
        failures = max(0, self.failures.get(key, 0) - 1) if success else self.failures.get(key, 0) + 1
        # Modal/context interrupts are not physical geometry failures.
        if (not success and task.phase == "interrupted" and task.failure in {
                "context_changed", "modal_owns_control", "game_not_playable",
                "portal_episode_changed", "portal_player_context_changed", "observed_context_interaction",
                "camera_modal_owns_control"}):
            failures = self.failures.get(key, 0)
        self.bounded_put(self.failures, key, failures, limit=512)
        self.bounded_put(self.retry_after, key, now + (30 if isinstance(task, ObservedPortalTask) else 20), limit=512)

    def _departure_outcome(self, task, *, success, now):
        # A stationary first segment can fail under several different distant
        # targets. Remember only that actual consumed attempt, not a blocked
        # collision edge or a traversed route. Explicit interactions/traversals
        # retain their own contracts and are never filtered by this memory.
        if (task.kind != "observed_cell" or self.last_position is None
                or self.last_context != task.context or self.last_seq <= task.origin_seq):
            return
        if success and task.phase == "succeeded" and task.consumed and task.verification_frames >= 3:
            for key, row in list(self.failed_departures.items()):
                if (row["context"] == task.context and math.dist(task.origin, row["origin"]) <= 18
                        and math.dist(self.last_position, row["point"]) <= 18):
                    del self.failed_departures[key]
            return
        if (success or task.phase != "failed" or task.failure != "no_geometric_progress"
                or not task.consumed or not task.corridor or task.waypoint_index != 0 or task.detour
                or task.progress_point != task.corridor[0]
                or math.dist(self.last_position, task.origin) > 18):
            return
        point, origin = tuple(task.progress_point), tuple(task.origin)
        key = (*task.context, *(round(v, 1) for v in origin), *(round(v, 1) for v in point))
        failures = self.failed_departures.get(key, {}).get("failures", 0) + 1
        self.bounded_put(self.failed_departures, key,
            {"context": task.context, "origin": origin, "point": point, "failures": failures,
             "retry_after": now + min(120, 20 * failures)}, limit=512)

    def departure_deferred(self, task, *, now):
        if task.kind != "observed_cell" or not task.corridor:
            return False
        point = task.corridor[0]
        return any(row["context"] == task.context and row["retry_after"] > now
            and math.dist(task.origin, row["origin"]) <= 18 and math.dist(point, row["point"]) <= 18
            and abs(task.origin[1] - row["origin"][1]) <= 4 and abs(point[1] - row["point"][1]) <= 4
            for row in self.failed_departures.values())

    def context_available(self, game, *, now=None):
        if not self.contextual_interactions or not eligible_context(game):
            return False
        now = time.monotonic() if now is None else now
        if self.read_context_deferred(game, now=now):
            return False
        task = ObservedContextTask.create(game, now=now)
        return self.retry_after.get(self.task_key(task), 0) <= now

    def read_context_deferred(self, game, *, now):
        actor = game.context_actor
        if game.context_action.label.lower() not in {"check", "speak"} or not actor or not actor.actor_uid:
            return False
        read = self.read_actor_contexts.get((*physical_context(game), actor.actor_uid))
        return bool(read and read["progress_revision"] == self.progress_revision and read["retry_after"] > now)

    def upward_floor_visited(self, context, target):
        # Actual grounded occupation is finer than macro dwell accounting.
        # A small rise in an already occupied macro-region is still new floor.
        return any(owner == context and abs(point[1] - target[1]) <= 4
            and math.hypot(point[0] - target[0], point[2] - target[2]) <= 35
            for owner, point in self.occupied_floors.values())

    def upward_floor_task(self, game, affordance, *, now, budget_s):
        # The affordance nominates a currently observed upper surface. Only a
        # matching CURRENT cell reached by directed collision links supplies
        # the held walking target and its corridor; no thin probe or floor
        # sample alone creates a climb route. The usual stopped/height/consumed
        # postcondition of observed_cell proves actual arrival upstairs.
        mesh, target, context = game.navmesh, affordance.target_position, physical_context(game)
        cells = []
        for x, z, y, _ in mesh.cells:
            point = (mesh.origin[0] + x * mesh.step, y, mesh.origin[2] + z * mesh.step)
            gap = math.dist(point, target)
            if (y - game.player.floor_height <= 8 or abs(y - target[1]) > 4
                    or gap > mesh.step * .8 or self.upward_floor_visited(context, point)):
                continue
            cells.append((gap, point))
        for _, point in sorted(cells):
            try:
                return LocalTask.observed_cell(game, point, now=now, budget_s=min(20, budget_s))
            except ValueError:
                continue
        return None

    def choose(self, game, *, budget_s, now=None, suppress_context=False):
        now = time.monotonic() if now is None else now
        self.observe(game, now=now)
        self.selection = {"observation_seq": game.seq, "full_seq": game.full_seq, "monotonic_s": now,
                          "mesh_cells": len(game.navmesh.cells), "reachable_floor_candidates": 0,
                          "cooled_floor_candidates": 0, "eligible_floor_candidates": 0,
                          "reachable_short_floor_candidates": 0, "cooled_short_floor_candidates": 0,
                          "eligible_short_floor_candidates": 0,
                          "cooled_departure_candidates": 0,
                          "container_recovery_candidates": 0,
                          "visibly_open_containers_skipped": 0,
                          "eligible_collectible_candidates": 0,
                          "eligible_upward_candidates": 0,
                          "read_context_deferred": self.read_context_deferred(game, now=now)}
        if not 1 <= budget_s <= 300:
            raise ValueError("planning_budget_exhausted")
        if self.contextual_interactions and game.dialogue.active:
            self.replans += 1
            return LinearDialogueTask.create(game, now=now, budget_s=min(20, budget_s))
        if self.contextual_interactions and game.cutscene_active and not game.dialogue.active:
            self.replans += 1
            return NativeModalWaitTask.create(game, now=now, budget_s=min(20, budget_s))
        if (game.dialogue.active or game.pause_menu.active or game.paused or game.cutscene_active):
            raise ValueError("modal_requires_separate_controller")
        if camera_modal_active(game.player):
            if self.contextual_interactions and eligible_camera_return(game):
                self.replans += 1
                return ObservedCameraReturnTask.create(game, now=now, budget_s=min(20, budget_s))
            raise ValueError("first_person_requires_separate_controller")
        if not grounded(game.player):
            raise ValueError(f"unsupported_planning_mode:{locomotor_mode(game)}")
        if not game.navmesh.available or game.camera_input_yaw is None:
            raise ValueError("current_ground_collision_unavailable")
        if not suppress_context and self.context_available(game, now=now):
            self.replans += 1
            return ObservedContextTask.create(game, now=now, budget_s=min(20, budget_s))
        context, position = physical_context(game), game.player.position
        candidates = []

        def offer(task, priority, distance, *, approach_distance=0):
            key = self.task_key(task)
            if self.retry_after.get(key, 0) > now:
                return False
            visits = self.visits.get(self.region(context, task.target), 0)
            # Horizontal frontiers seek expansion; vertical traversals first
            # prefer an observed nearby upper approach, then a shorter landing.
            rank = (priority, self.failures.get(key, 0), visits, approach_distance,
                    -distance if priority == 2 else distance, tuple(task.target))
            candidates.append((rank, task))
            return True

        if self.contextual_interactions and not suppress_context:
            for actor in game.nearby_actors:
                if ObservedCollectibleApproachTask.is_collectible(actor):
                    try:
                        task = ObservedCollectibleApproachTask.create(game, actor, now=now, budget_s=min(20, budget_s))
                    except ValueError:
                        continue
                    priority = 1 if self.failures.get(self.task_key(task), 0) >= 2 else -1
                    if offer(task, priority, math.dist(position, actor.position)):
                        self.selection["eligible_collectible_candidates"] += 1
                    continue
                if (*context, actor.actor_uid) in self.opened_containers:
                    continue
                if (actor.drawn and actor.category_name.lower() == "chest"
                        and actor.container_lid_pose == "open"
                        and not ObservedContainerApproachTask.matching_prompt(game, actor.actor_uid)):
                    self.selection["visibly_open_containers_skipped"] += 1
                    continue
                try:
                    task = ObservedContainerApproachTask.create(game, actor, now=now, budget_s=min(15, budget_s))
                except ValueError:
                    continue
                # Repeated local failure can make a CURRENT native exit the
                # next recovery attempt. This does not classify the container
                # as open/inert or infer a destination beyond the observed exit.
                priority = 1 if self.failures.get(self.task_key(task), 0) >= 2 else -1
                if offer(task, priority, math.dist(position, actor.position)) and priority == 1:
                    self.selection["container_recovery_candidates"] += 1

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
            if row.kind == "stairs_or_slope_up":
                task = self.upward_floor_task(game, row, now=now, budget_s=budget_s)
                if task and not self.departure_deferred(task, now=now):
                    if offer(task, 1, math.dist(position, task.target), approach_distance=row.distance):
                        self.selection["eligible_upward_candidates"] += 1
                continue
            if row.kind != "ledge_down" or self.region(context, row.target_position) in self.visits:
                continue
            try:
                task = GroundDescentApproachTask.create(game, row, now=now, budget_s=min(20, budget_s))
            except ValueError:
                continue
            offer(task, 1, math.dist(position, row.target_position), approach_distance=row.distance)

        mesh, floor_candidates, short_candidates = game.navmesh, [], []
        for x, z, y, _ in mesh.cells:
            point = (mesh.origin[0] + x * mesh.step, y, mesh.origin[2] + z * mesh.step)
            distance = math.dist(position, point)
            short = 20 <= distance < 55 and mesh.step < 55
            if (not (short or 55 <= distance <= 280)
                    or abs(y - game.player.floor_height) > 24):
                continue
            path = observed_local_path(game, point, minimum_gain=0)
            if not path or path["partial"] or path["target_gap"] > 4:
                continue
            self.selection["reachable_short_floor_candidates" if short else "reachable_floor_candidates"] += 1
            heading = math.atan2(point[0] - position[0], point[2] - position[2])
            forward = math.cos(heading - game.player.yaw * math.pi / 32768) >= -.25
            task = LocalTask.observed_cell(game, point, now=now, budget_s=min(12, budget_s))
            if self.retry_after.get(self.task_key(task), 0) > now:
                self.selection["cooled_short_floor_candidates" if short else "cooled_floor_candidates"] += 1
            elif self.departure_deferred(task, now=now):
                self.selection["cooled_departure_candidates"] += 1
                self.departure_deferrals += 1
            else:
                (short_candidates if short else floor_candidates).append((forward, task, distance))
        # A refined graph can expose only one short supported escape step.
        # Reuse the same directed-path, cooldown and task completion contracts;
        # never let this fallback replace an eligible longer frontier or reset
        # the macro dwell clock. No route edge is created by selecting a step.
        if not floor_candidates:
            floor_candidates = short_candidates
            self.selection["eligible_short_floor_candidates"] = len(short_candidates)
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
                "occupied_ground_floor_samples": len(self.occupied_floors),
                "failed_target_regions": sum(value > 0 for value in self.failures.values()),
                "max_outward_radius": max(self.max_radius.values(), default=0),
                "observed_context_actions": dict(self.context_actions),
                "verified_opened_containers": len(self.opened_containers),
                "verified_read_actor_contexts": len(self.read_actor_contexts),
                "failed_departures": len(self.failed_departures),
                "departure_deferrals": self.departure_deferrals,
                "durable_progress_revision": self.progress_revision,
                "memory": "ephemeral_actual_positions_and_attempts_only", "last_selection": dict(self.selection)}
        if include_details:
            result.update(last_seq=self.last_seq, last_context=self.last_context, last_region=self.last_region,
                entry_position=self.entry_position, entry_at=self.entry_at,
                visited_regions=[{"key": key, "visits": value} for key, value in self.visits.items()],
                attempted_targets=[{"key": key, "failures": value, "retry_after": self.retry_after.get(key, 0)}
                                   for key, value in self.failures.items()],
                opened_containers=[{"key": key, "event_id": value} for key, value in self.opened_containers.items()])
            result.update(read_actor_contexts=[{"key": key, **value} for key, value in self.read_actor_contexts.items()],
                progress_tokens=sorted(self.progress_tokens), progress_initialized=self.progress_initialized)
            result.update(last_position=self.last_position,
                failed_departure_attempts=[{"key": key, **value} for key, value in self.failed_departures.items()])
            result.update(occupied_floor_samples=[{"context": owner, "position": point}
                for owner, point in self.occupied_floors.values()])
        return result
