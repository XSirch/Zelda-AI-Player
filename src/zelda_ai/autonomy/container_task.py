"""Approach a currently drawn container; never inspect its hidden contents."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .execution import physical_context
from .local_tasks import LocalTask
from .locomotion import grounded
from .navigation import observed_local_path


@dataclass
class ObservedContainerApproachTask(LocalTask):
    VERSION = "observed-container-approach-v3"
    actor_uid: str = ""
    floor_task: LocalTask | None = None
    version: str = VERSION

    @classmethod
    def create(cls, game, actor, *, now=None, budget_s=15):
        if actor.container_lid_pose == "open" and not cls.matching_prompt(game, actor.actor_uid):
            raise ValueError("Container is currently visibly open")
        if (actor not in game.nearby_actors or not actor.drawn or not actor.actor_uid
                or actor.category_name.lower() != "chest" or actor.room not in {-1, game.room}
                or not grounded(game.player) or not game.navmesh.available
                or abs(actor.position[1] - game.player.floor_height) > 8):
            raise ValueError("Container needs current drawn actor and compatible dry floor")
        task = cls._create(game, "observed_container_approach", actor.position,
            game.player.position, now=now, budget_s=budget_s)
        task.actor_uid = actor.actor_uid
        # A chest terminates at collision. Stop the observed floor corridor
        # nearby, then use its explicit interaction target for a bounded final
        # approach. This final segment is NOT a new walkable/traversed edge.
        path = observed_local_path(game, actor.position, minimum_gain=4)
        if path and path["target_gap"] <= 90 and abs(path["waypoints"][-1][1] - actor.position[1]) <= 4:
            task.floor_task = LocalTask.observed_cell(game, path["waypoints"][-1], now=now, budget_s=budget_s)
        elif math.dist(game.player.position, actor.position) > 90:
            raise ValueError("No currently linked floor approach to the observed container")
        return task

    def current_actor(self, game):
        # Fast packets intentionally omit the optional nearby list. Their
        # current room list owns UID, draw state and pose; never retain the
        # previous full packet's actor as a substitute for a missing one.
        actors = (game.room_actors if game.protocol == 3 and game.seq != game.full_seq
                  else game.nearby_actors)
        return next((a for a in actors if a.actor_uid == self.actor_uid and a.drawn
            and a.category_name.lower() == "chest" and a.room in {-1, game.room}
            and math.dist(a.position, self.target) <= 8), None)

    def actionable(self, game):
        return self.matching_prompt(game, self.actor_uid)

    @staticmethod
    def matching_prompt(game, actor_uid):
        actor = game.context_actor
        return bool(actor_uid and actor and actor.actor_uid == actor_uid
            and game.context_action.label.lower() == "open")

    def steering_point(self, game):
        if self.floor_task and not self.floor_task.terminal:
            return self.floor_task.steering_point(game)
        return self.target

    def observe(self, game, *, consumed, now=None):
        if self.terminal:
            return
        now = time.monotonic() if now is None else now
        if (physical_context(game) != self.context or game.scene_epoch != self.scene_epoch
                or any(e not in self.load_events for e in self._loads(game))):
            self.interrupt("context_changed")
            return
        if not game.player or game.player.health <= 0 or game.game_over_state or not game.in_game:
            self.interrupt("game_not_playable")
            return
        if game.dialogue.active or game.cutscene_active or game.pause_menu.active or game.paused:
            self.interrupt("modal_owns_control")
            return
        if not grounded(game.player):
            self.interrupt("unsupported_locomotor_mode")
            return
        actor = self.current_actor(game)
        if not actor:
            self.interrupt("container_target_lost")
            return
        if game.seq <= self.origin_seq or game.seq <= self.last_seq:
            return
        self.last_seq, self.consumed = game.seq, self.consumed or consumed
        if actor.container_lid_pose == "open" and not self.actionable(game):
            self.interrupt("container_visibly_open")
            return
        if self.actionable(game):
            self.phase = "verify"
            self.verification_frames = (self.verification_frames + 1
                if self.consumed and game.player.speed_xz < .1 else 0)
            if self.verification_frames >= 3:
                self.phase = "succeeded"
            elif now >= self.deadline:
                self.phase, self.failure = "failed", "container_prompt_not_stable"
            return
        self.verification_frames = 0
        distance = math.dist(game.player.position, self.target)
        if consumed and distance < self.best_distance - 4:
            self.best_distance, self.progress_at = distance, now
        if now >= self.deadline:
            self.phase, self.failure = "failed", "container_approach_timeout"
            return
        if self.floor_task and not self.floor_task.terminal:
            self.floor_task.observe(game, consumed=consumed, now=now)
            self.phase = self.floor_task.phase
            if self.floor_task.phase == "succeeded":
                # Floor arrival alone is not an interaction postcondition.
                self.phase, self.progress_at = "execute", now
            elif self.floor_task.terminal:
                self.phase, self.failure = "failed", self.floor_task.failure
            return
        if distance > 90:
            self.interrupt("container_approach_evidence_lost")
        elif now - self.progress_at >= 2.5:
            self.phase, self.failure = "failed", "container_no_observed_prompt"
        else:
            self.phase = "execute"
