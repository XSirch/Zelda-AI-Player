from __future__ import annotations

import math
from dataclasses import dataclass

from ..models import GameState
from .features import target_point
from .models import AgentIntent


@dataclass
class RewardResult:
    reward: float
    done: bool
    breakdown: dict[str, float]


class RewardTracker:
    """Game-generic reward shaping plus ML curiosity.

    There is no scripted Zelda route here. Rewards come from observable novelty,
    durable progress, target progress supplied by cognition, and survival.
    """

    def __init__(self):
        self.visited_cells: set[tuple[int, int, int, int]] = set()
        self.seen_actors: set[tuple[int, int, str | None, int, int]] = set()
        self.seen_events: set[str] = set()
        self.previous: dict | None = None

    @staticmethod
    def _progress_fingerprint(game: GameState):
        return (
            tuple(game.inventory),
            tuple(game.progress.quest_items),
            tuple(game.progress.owned_equipment),
            tuple(sorted(game.progress.upgrade_levels.items())),
            game.progress.heart_pieces,
            game.progress.skull_tokens,
            game.progress.magic_acquired,
            game.progress.double_magic,
            game.progress.double_defense,
            game.progress.small_keys,
        )

    @staticmethod
    def _enemy_health(game: GameState):
        result = {}
        for actor in game.room_actors:
            if actor.collision_health_hint is None:
                continue
            key = actor.actor_uid or f"{actor.actor_id}:{actor.params}:{actor.room}"
            result[key] = actor.collision_health_hint
        return result

    @staticmethod
    def _target_distance(game: GameState, intent: AgentIntent) -> float | None:
        if not game.player:
            return None
        point = target_point(game, intent)
        if point is None:
            return None
        return math.dist(game.player.position, point)

    def _snapshot(self, game: GameState, intent: AgentIntent):
        player = game.player
        return {
            "instance": game.instance_id,
            "scene": game.scene,
            "room": game.room,
            "scene_epoch": game.scene_epoch,
            "position": tuple(player.position) if player else None,
            "health": player.health if player else 0,
            "dialogue": (game.dialogue.active, game.dialogue.text_id, game.dialogue.state_code),
            "context": (game.context_action.code, game.context_action.label),
            "progress": self._progress_fingerprint(game),
            "target_distance": self._target_distance(game, intent),
            "enemy_health": self._enemy_health(game),
        }

    def step(
        self,
        game: GameState,
        intent: AgentIntent,
        *,
        intrinsic: float,
        pressed_buttons: int,
    ) -> RewardResult:
        current = self._snapshot(game, intent)
        b: dict[str, float] = {}

        # RND curiosity is the core exploration signal.
        b["curiosity"] = max(0.0, min(1.0, intrinsic)) * 0.35

        player = game.player
        if player:
            cell = (game.scene, game.room,
                    round(player.position[0] / 100.0),
                    round(player.position[2] / 100.0))
            if cell not in self.visited_cells:
                self.visited_cells.add(cell)
                b["new_space"] = 0.18

        for actor in game.room_actors:
            key = (game.scene, game.room, actor.actor_uid, actor.actor_id, actor.params)
            if key not in self.seen_actors:
                self.seen_actors.add(key)
                b["new_actor"] = b.get("new_actor", 0.0) + 0.05

        for event in game.events:
            key = f"{game.instance_id}:{event.id}"
            if key not in self.seen_events:
                self.seen_events.add(key)
                b["native_event"] = b.get("native_event", 0.0) + 0.15

        done = False
        previous = self.previous
        if previous and previous["instance"] == current["instance"]:
            if (previous["scene"], previous["room"], previous["scene_epoch"]) != (
                current["scene"], current["room"], current["scene_epoch"]
            ):
                b["world_transition"] = 0.8

            if previous["dialogue"] != current["dialogue"]:
                b["dialogue_change"] = 0.25
            if previous["context"] != current["context"]:
                b["context_change"] = 0.08
            if previous["progress"] != current["progress"]:
                b["durable_progress"] = 2.0

            if previous["position"] and current["position"]:
                moved = math.dist(previous["position"], current["position"])
                if moved >= 4.0:
                    b["movement"] = min(0.008, moved / 6000.0)

            if current["health"] < previous["health"]:
                b["damage_taken"] = -min(1.2, (previous["health"] - current["health"]) / 16.0 * 0.25)
            if previous["health"] > 0 and current["health"] <= 0:
                b["death"] = -2.5
                done = True

            for key, old_health in previous["enemy_health"].items():
                new_health = current["enemy_health"].get(key)
                if new_health is not None and new_health < old_health:
                    b["enemy_damage"] = b.get("enemy_damage", 0.0) + min(
                        0.8, (old_health - new_health) * 0.08)

            old_dist, new_dist = previous["target_distance"], current["target_distance"]
            if old_dist is not None and new_dist is not None:
                improvement = max(-100.0, min(100.0, old_dist - new_dist))
                b["intent_progress"] = improvement / 500.0

        # Small efficiency cost discourages random button mashing after the policy
        # has learned that it does not create useful state changes.
        if pressed_buttons:
            b["button_cost"] = -0.003 * pressed_buttons.bit_count()

        self.previous = current
        total = max(-5.0, min(5.0, sum(b.values())))
        return RewardResult(round(total, 6), done, {k: round(v, 6) for k, v in b.items()})
