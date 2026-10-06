"""Bounded contact with a drawn collectible, verified by actual state deltas."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .execution import physical_context
from .local_tasks import LocalTask
from .locomotion import grounded
from .navigation import observed_local_path


def resource_values(game):
    values = {"rupees": game.player.rupees, "health": game.player.health,
              "magic": game.player.magic, "max_health": game.player.max_health,
              "heart_pieces": game.progress.heart_pieces, "skull_tokens": game.progress.skull_tokens,
              f"small_keys:{game.progress.map_index}": game.progress.small_keys}
    values.update({f"ammo:{slot}": 0 for slot, item in enumerate(game.inventory)
                   if item in {254, 255}})
    values.update({f"ammo:{item.slot}": max(0, item.ammo) for item in game.inventory_named
                   if item.ammo is not None})
    return values


@dataclass
class ObservedCollectibleApproachTask(LocalTask):
    VERSION = "observed-collectible-approach-v2"
    actor_uid: str = ""
    floor_task: LocalTask | None = None
    initial_resources: dict = field(default_factory=dict)
    gain_values: dict = field(default_factory=dict)
    contact_seen: bool = False
    gain_seq: int | None = None
    corridor_replans: int = 0
    lost_at: float | None = None
    version: str = VERSION

    @staticmethod
    def is_collectible(actor):
        # The public actor class identifies the visible affordance, not its
        # contents. Actor params and item/drop/treasure flags are never read.
        return actor.name == "En_Item00"

    @classmethod
    def create(cls, game, actor, *, now=None, budget_s=20):
        if (actor not in game.nearby_actors or not actor.drawn or not actor.actor_uid
                or not cls.is_collectible(actor) or actor.room not in {-1, game.room}
                or not grounded(game.player) or not game.navmesh.available
                or abs(actor.position[1] - game.player.floor_height) > 8):
            raise ValueError("Collectible needs a current drawn actor on compatible dry floor")
        target = (actor.position[0], game.player.floor_height, actor.position[2])
        task = cls._create(game, "observed_collectible_approach", target,
                           game.player.position, now=now, budget_s=budget_s)
        task.actor_uid, task.initial_resources = actor.actor_uid, resource_values(game)
        if game.protocol == 3 and not task.current_actor(game):
            raise ValueError("Collectible needs a current authoritative room actor")
        task._plan_floor(game, now=task.started)
        return task

    def actors(self, game):
        # Protocol 3 owns actor lifetimes through the bounded room list in
        # BOTH full and fast packets. The separate nearby list can expose a
        # distant actor omitted from that scope, even on a complete packet.
        return game.room_actors if game.protocol == 3 else game.nearby_actors

    def current_actor(self, game):
        return next((a for a in self.actors(game) if a.actor_uid == self.actor_uid and a.drawn
            and self.is_collectible(a) and a.room in {-1, game.room}
            and math.hypot(a.position[0] - self.target[0], a.position[2] - self.target[2]) <= 16
            and abs(a.position[1] - self.target[1]) <= 12), None)

    def _plan_floor(self, game, *, now):
        if math.dist(game.player.position, self.target) <= 60:
            self.floor_task = None
            return
        path = observed_local_path(game, self.target, minimum_gain=4)
        if (not path or abs(path["waypoints"][-1][1] - self.target[1]) > 4
                or self.corridor_replans >= 3 or self.deadline - now <= .5):
            raise ValueError("No further currently linked floor approach to collectible")
        self.floor_task = LocalTask.observed_cell(game, path["waypoints"][-1],
            now=now, budget_s=min(30, self.deadline - now))
        self.corridor_replans += 1

    def steering_point(self, game):
        if self.floor_task and not self.floor_task.terminal:
            return self.floor_task.steering_point(game)
        # The final contact is an explicit currently observed interaction
        # target, not an invented walkable edge or permanent frontier magnet.
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
        if game.seq <= self.origin_seq or game.seq <= self.last_seq:
            return
        self.last_seq, self.consumed = game.seq, self.consumed or consumed
        near = math.dist(game.player.position, self.target) <= 36
        self.contact_seen = self.contact_seen or bool(consumed and near)
        gains = {key: value - self.initial_resources.get(key, value)
                 for key, value in resource_values(game).items()
                 if value > self.initial_resources.get(key, value)}
        # An unrelated remote pickup cannot pay this contact task. During an
        # observed gain, brake and wait for this UID to disappear; culling,
        # actor disappearance or item_received alone never prove acquisition.
        if self.gain_seq is None and self.contact_seen and near and gains:
            self.gain_seq, self.gain_values = game.seq, gains
        present = any(a.actor_uid == self.actor_uid for a in self.actors(game))
        if self.gain_seq is not None:
            self.phase = "verify"
            complete_scope = game.protocol != 3 or not game.room_actors_truncated
            self.verification_frames = (self.verification_frames + 1
                if complete_scope and not present and near and gains and game.player.speed_xz < .1 else 0)
            if self.verification_frames >= 3:
                self.phase = "succeeded"
            elif now >= self.deadline:
                self.phase, self.failure = "failed", "collectible_gain_not_verified"
            return
        if not self.current_actor(game):
            if self.contact_seen and near:
                # Fast packets may lose the picked-up actor before the next
                # complete inventory observation. Release input during a short
                # wait; no delta still fails, including full-capacity pickups.
                self.lost_at = now if self.lost_at is None else self.lost_at
                self.phase = "verify"
                if now - self.lost_at >= 1 or now >= self.deadline:
                    self.phase, self.failure = "failed", "collectible_no_resource_gain"
                return
            self.interrupt("collectible_target_lost")
            return
        if now >= self.deadline:
            self.phase, self.failure = "failed", "collectible_approach_timeout"
            return
        if self.floor_task and not self.floor_task.terminal:
            self.floor_task.observe(game, consumed=consumed, now=now)
            self.phase = self.floor_task.phase
            if self.floor_task.phase == "succeeded":
                self.phase = "execute"
                try:
                    self._plan_floor(game, now=now)
                except ValueError:
                    self.phase, self.failure = "failed", "collectible_observed_approach_exhausted"
            elif self.floor_task.terminal:
                self.phase, self.failure = "failed", self.floor_task.failure
            return
        distance = math.dist(game.player.position, self.target)
        if consumed and distance < self.best_distance - 4:
            self.best_distance, self.progress_at = distance, now
        if distance > 60:
            self.interrupt("collectible_contact_evidence_lost")
        elif now - self.progress_at >= 2.5:
            self.phase, self.failure = "failed", "collectible_no_observed_gain"
        else:
            self.phase = "execute"
