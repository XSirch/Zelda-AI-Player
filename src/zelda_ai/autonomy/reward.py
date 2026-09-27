from __future__ import annotations

import math
from dataclasses import dataclass

from ..models import GameState
from .features import target_point
from .models import AgentIntent


EQUIPMENT_OBJECTIVES: dict[tuple[str, int], tuple[str, int, float]] = {
    ("sword", 1): ("Kokiri Sword", 100, 3.0),
    ("sword", 2): ("Master Sword", 300, 4.0),
    ("sword", 3): ("Biggoron Sword", 400, 4.5),
    ("shield", 1): ("Deku Shield", 60, 1.8),
    ("shield", 2): ("Hylian Shield", 100, 2.2),
    ("shield", 3): ("Mirror Shield", 300, 4.0),
    ("tunic", 2): ("Goron Tunic", 150, 2.5),
    ("tunic", 3): ("Zora Tunic", 150, 2.5),
    ("boots", 2): ("Iron Boots", 150, 2.5),
    ("boots", 3): ("Hover Boots", 150, 2.5),
}

STORY_OBJECTIVES: dict[str, tuple[str, int, float]] = {
    "greeted_by_saria": ("Falou com Saria", 20, 0.4),
    "first_spoke_to_mido": ("Falou com Mido", 25, 0.5),
    "showed_mido_sword_shield": ("Provou equipamento a Mido", 150, 3.0),
    "deku_tree_opened_mouth": ("Abriu o caminho da Great Deku Tree", 200, 3.5),
    "met_deku_tree": ("Encontrou a Great Deku Tree", 100, 2.0),
    "obtained_kokiri_emerald": ("Kokiri Emerald", 500, 5.0),
}


@dataclass
class RewardResult:
    reward: float
    done: bool
    breakdown: dict[str, float]
    achievements: list[dict]


class RewardTracker:
    """Game-generic shaping plus explicit observed Zelda milestones.

    PPO receives bounded rewards. Human-facing achievement points are tracked
    separately so a milestone such as Kokiri Sword can be worth +100 points
    without destabilizing policy optimization.
    """

    def __init__(self):
        self.visited_cells: set[tuple[int, int, int, int]] = set()
        self.seen_actors: set[tuple[int, int, str | None, int, int]] = set()
        self.seen_events: set[str] = set()
        self.seen_dialogue: set[tuple[int, int | None, str]] = set()
        self.seen_contexts: set[tuple[int, int, int, str | None]] = set()
        self.seen_transitions: set[tuple[int, int, int, int]] = set()
        self.objective_score = 0
        self.combat_contact = False
        self.seen_inventory_items: set[int] = set()
        self.stagnation_steps = 0
        self.previous: dict | None = None

    def break_causal_chain(self):
        """Drop only the previous transition; keep novelty history for the run."""
        self.previous = None

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
    def _intent_key(intent: AgentIntent):
        return (
            intent.mode,
            intent.target_actor_id,
            intent.target_actor_params,
            intent.target_actor_uid,
            tuple(intent.target_position) if intent.target_position is not None else None,
            intent.target_item_id,
            intent.direction,
            intent.choice_index,
        )

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
            "intent_key": self._intent_key(intent),
            "target_distance": self._target_distance(game, intent),
            "enemy_health": self._enemy_health(game),
            "inventory_items": {row.item_id: row.name for row in game.inventory_named},
            "equipment": {
                row.item_id: (row.name, row.equipment_type, row.value)
                for row in game.progress.equipment
            },
            "quest_items": set(game.progress.quest_items),
            "story_flags": dict(game.progress.story_flags),
            "upgrades": dict(game.progress.upgrade_levels),
            "heart_pieces": game.progress.heart_pieces,
            "skull_tokens": game.progress.skull_tokens,
            "magic_acquired": game.progress.magic_acquired,
            "double_magic": game.progress.double_magic,
            "double_defense": game.progress.double_defense,
        }

    def _achievement(
        self,
        *,
        key: str,
        kind: str,
        title: str,
        detail: str,
        points: int,
        training_reward: float,
    ) -> dict:
        self.objective_score += points
        return {
            "id": key,
            "kind": kind,
            "title": title,
            "detail": detail,
            "points": points,
            "training_reward": round(training_reward, 3),
            "score_after": self.objective_score,
        }

    @staticmethod
    def _exploration_milestone(previous: int, current: int) -> int | None:
        crossed = [
            value
            for value in (10, 25, 50, 100, 250, 500, 1000)
            if previous < value <= current
        ]
        return max(crossed) if crossed else None

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
        achievements: list[dict] = []
        done = False

        # RND is only a weak tie-breaker for genuinely surprising states.
        # Useful game progress must dominate intrinsic exploration.
        curiosity = max(0.0, min(1.0, intrinsic)) * 0.08
        if curiosity > 0:
            b["curiosity"] = curiosity

        player = game.player
        if player:
            cell = (
                game.scene,
                game.room,
                round(player.position[0] / 100.0),
                round(player.position[2] / 100.0),
            )
            if cell not in self.visited_cells:
                before = len(self.visited_cells)
                self.visited_cells.add(cell)
                b["new_space"] = 0.18
                milestone = self._exploration_milestone(before, len(self.visited_cells))
                if milestone is not None:
                    achievements.append(self._achievement(
                        key=f"space:{milestone}",
                        kind="exploration",
                        title=f"Explorou {milestone} regiões",
                        detail="Células espaciais únicas visitadas nesta run.",
                        points=max(10, milestone // 2),
                        training_reward=0.0,
                    ))

        for actor in game.room_actors:
            key = (game.scene, game.room, actor.actor_uid, actor.actor_id, actor.params)
            if key not in self.seen_actors:
                self.seen_actors.add(key)
                b["new_actor"] = b.get("new_actor", 0.0) + 0.05

        if game.dialogue.active:
            dialogue_key = (
                game.scene,
                game.dialogue.text_id,
                game.dialogue.text.strip()[:160],
            )
            if dialogue_key not in self.seen_dialogue:
                self.seen_dialogue.add(dialogue_key)
                b["new_dialogue"] = 0.3

        if game.context_action.label != "none":
            actor_uid = game.context_actor.actor_uid if game.context_actor else None
            context_key = (
                game.scene,
                game.room,
                game.context_action.code,
                actor_uid,
            )
            if context_key not in self.seen_contexts:
                self.seen_contexts.add(context_key)
                b["new_context"] = 0.06

        for event in game.events:
            key = f"{game.instance_id}:{event.id}"
            if key in self.seen_events:
                continue
            self.seen_events.add(key)
            event_reward = {
                "enemy_defeated": 1.0,
                "boss_defeated": 2.5,
                "game_completed": 5.0,
            }.get(event.kind, 0.0)
            if event_reward > 0:
                b["native_event"] = b.get("native_event", 0.0) + event_reward
            if event.kind == "enemy_defeated":
                achievements.append(self._achievement(
                    key=f"enemy:{event.id}",
                    kind="combat",
                    title="Inimigo derrotado",
                    detail=event.detail or "Evento nativo de derrota de inimigo.",
                    points=25,
                    training_reward=event_reward,
                ))
            elif event.kind == "boss_defeated":
                achievements.append(self._achievement(
                    key=f"boss:{event.id}",
                    kind="combat",
                    title="Boss derrotado",
                    detail=event.detail or "Evento nativo de derrota de boss.",
                    points=500,
                    training_reward=event_reward,
                ))
            elif event.kind == "game_completed":
                achievements.append(self._achievement(
                    key=f"game:{event.id}",
                    kind="story",
                    title="Ocarina of Time concluído",
                    detail="O bridge observou o evento nativo de conclusão.",
                    points=5000,
                    training_reward=event_reward,
                ))
                done = True

        previous = self.previous
        if previous is None:
            # Progress already present when the run starts is baseline, not an
            # autonomous achievement. Remember inventory IDs so depletion and
            # later reacquisition cannot farm the same objective.
            self.seen_inventory_items.update(current["inventory_items"])
        if previous and previous["instance"] == current["instance"]:
            if (previous["scene"], previous["room"], previous["scene_epoch"]) != (
                current["scene"], current["room"], current["scene_epoch"]
            ):
                edge = (
                    previous["scene"],
                    previous["room"],
                    current["scene"],
                    current["room"],
                )
                if edge not in self.seen_transitions:
                    self.seen_transitions.add(edge)
                    b["new_world_transition"] = 0.8
                    achievements.append(self._achievement(
                        key=f"edge:{edge}",
                        kind="navigation",
                        title="Nova transição descoberta",
                        detail=(
                            f"Scene/room {previous['scene']}/{previous['room']} → "
                            f"{current['scene']}/{current['room']}."
                        ),
                        points=20,
                        training_reward=0.8,
                    ))
                else:
                    # Backtracking may be necessary, but a known edge must not
                    # become a repeatable reward farm.
                    b["repeated_transition"] = 0.0

            if previous["progress"] != current["progress"]:
                b["durable_progress"] = 2.0

            new_equipment = set(current["equipment"]) - set(previous["equipment"])
            for item_id in sorted(new_equipment):
                name, equipment_type, value = current["equipment"][item_id]
                canonical, points, training_reward = EQUIPMENT_OBJECTIVES.get(
                    (equipment_type, value),
                    (name, 100, 2.0),
                )
                b["objective_milestone"] = b.get("objective_milestone", 0.0) + training_reward
                achievements.append(self._achievement(
                    key=f"equipment:{item_id}",
                    kind="equipment",
                    title=canonical,
                    detail=f"Equipamento adquirido: {name}.",
                    points=points,
                    training_reward=training_reward,
                ))

            new_inventory = set(current["inventory_items"]) - set(previous["inventory_items"])
            for item_id in sorted(new_inventory):
                if item_id in self.seen_inventory_items:
                    continue
                self.seen_inventory_items.add(item_id)
                name = current["inventory_items"][item_id]
                training_reward = 1.2
                b["objective_milestone"] = b.get("objective_milestone", 0.0) + training_reward
                achievements.append(self._achievement(
                    key=f"item:{item_id}",
                    kind="item",
                    title=name,
                    detail="Novo item persistente observado no inventário.",
                    points=50,
                    training_reward=training_reward,
                ))

            for quest_name in sorted(current["quest_items"] - previous["quest_items"]):
                training_reward = 3.5
                b["objective_milestone"] = b.get("objective_milestone", 0.0) + training_reward
                achievements.append(self._achievement(
                    key=f"quest:{quest_name}",
                    kind="quest",
                    title=quest_name,
                    detail="Novo quest item observado.",
                    points=250,
                    training_reward=training_reward,
                ))

            for flag, enabled in current["story_flags"].items():
                if not enabled or previous["story_flags"].get(flag):
                    continue
                title, points, training_reward = STORY_OBJECTIVES.get(
                    flag,
                    (flag.replace("_", " ").title(), 100, 2.0),
                )
                b["objective_milestone"] = b.get("objective_milestone", 0.0) + training_reward
                achievements.append(self._achievement(
                    key=f"story:{flag}",
                    kind="story",
                    title=title,
                    detail=f"Flag de progresso observada: {flag}.",
                    points=points,
                    training_reward=training_reward,
                ))

            for upgrade, level in current["upgrades"].items():
                previous_level = previous["upgrades"].get(upgrade, 0)
                if level <= previous_level:
                    continue
                delta = level - previous_level
                points = 100 * delta
                training_reward = min(3.0, 1.5 * delta)
                b["objective_milestone"] = b.get("objective_milestone", 0.0) + training_reward
                achievements.append(self._achievement(
                    key=f"upgrade:{upgrade}:{level}",
                    kind="upgrade",
                    title=f"Upgrade: {upgrade}",
                    detail=f"Nível {previous_level} → {level}.",
                    points=points,
                    training_reward=training_reward,
                ))

            heart_delta = current["heart_pieces"] - previous["heart_pieces"]
            if heart_delta > 0:
                training_reward = min(2.0, 0.8 * heart_delta)
                b["objective_milestone"] = b.get("objective_milestone", 0.0) + training_reward
                achievements.append(self._achievement(
                    key=f"heart:{current['heart_pieces']}",
                    kind="progress",
                    title="Heart Piece",
                    detail=f"+{heart_delta} peça(s) de coração.",
                    points=30 * heart_delta,
                    training_reward=training_reward,
                ))

            skull_delta = current["skull_tokens"] - previous["skull_tokens"]
            if skull_delta > 0:
                training_reward = min(1.5, 0.3 * skull_delta)
                b["objective_milestone"] = b.get("objective_milestone", 0.0) + training_reward
                achievements.append(self._achievement(
                    key=f"skull:{current['skull_tokens']}",
                    kind="progress",
                    title="Gold Skulltula Token",
                    detail=f"+{skull_delta} token(s).",
                    points=10 * skull_delta,
                    training_reward=training_reward,
                ))

            special_progress = (
                ("magic_acquired", "Magic Meter", 200, 3.0),
                ("double_magic", "Double Magic", 300, 4.0),
                ("double_defense", "Double Defense", 300, 4.0),
            )
            for field, title, points, training_reward in special_progress:
                if current[field] and not previous[field]:
                    b["objective_milestone"] = b.get("objective_milestone", 0.0) + training_reward
                    achievements.append(self._achievement(
                        key=f"progress:{field}",
                        kind="progress",
                        title=title,
                        detail="Progresso permanente observado pelo bridge.",
                        points=points,
                        training_reward=training_reward,
                    ))

            if current["health"] < previous["health"]:
                b["damage_taken"] = -min(
                    1.2,
                    (previous["health"] - current["health"]) / 16.0 * 0.25,
                )
            if previous["health"] > 0 and current["health"] <= 0:
                b["death"] = -2.5
                done = True

            for key, old_health in previous["enemy_health"].items():
                new_health = current["enemy_health"].get(key)
                if new_health is not None and new_health < old_health:
                    dealt = min(0.8, (old_health - new_health) * 0.08)
                    b["enemy_damage"] = b.get("enemy_damage", 0.0) + dealt
                    if not self.combat_contact:
                        self.combat_contact = True
                        achievements.append(self._achievement(
                            key="combat:first_damage",
                            kind="combat",
                            title="Primeiro dano causado",
                            detail="A política reduziu o HP observado de um inimigo.",
                            points=15,
                            training_reward=dealt,
                        ))

            old_dist, new_dist = previous["target_distance"], current["target_distance"]
            if (
                previous["intent_key"] == current["intent_key"]
                and old_dist is not None
                and new_dist is not None
            ):
                improvement = max(-100.0, min(100.0, old_dist - new_dist))
                b["intent_progress"] = improvement / 500.0

        if previous and previous["position"] is not None and current["position"] is not None:
            dx = current["position"][0] - previous["position"][0]
            dz = current["position"][2] - previous["position"][2]
            moved = math.hypot(dx, dz) >= 4.0
            useful_keys = {
                "new_space",
                "new_dialogue",
                "new_context",
                "new_world_transition",
                "durable_progress",
                "objective_milestone",
                "enemy_damage",
                "native_event",
            }
            useful = any(b.get(key, 0.0) > 0 for key in useful_keys)
            useful = useful or b.get("intent_progress", 0.0) > 0.01
            if moved or useful:
                self.stagnation_steps = 0
            else:
                self.stagnation_steps += 1
                if self.stagnation_steps >= 15:
                    b["stagnation"] = -min(
                        0.08,
                        0.005 * (self.stagnation_steps - 14),
                    )
        else:
            self.stagnation_steps = 0

        # Small efficiency cost discourages random button mashing. With the RND
        # baseline removed, ineffective button combinations become net-negative.
        if pressed_buttons:
            b["button_cost"] = -0.004 * pressed_buttons.bit_count()

        self.previous = current
        total = max(-5.0, min(5.0, sum(b.values())))
        return RewardResult(
            round(total, 6),
            done,
            {k: round(v, 6) for k, v in b.items()},
            achievements,
        )
