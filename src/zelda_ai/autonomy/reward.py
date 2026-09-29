from __future__ import annotations

import math
import time
from dataclasses import dataclass

from ..models import GameState
from .features import target_point
from .models import AgentIntent
from .routes import ROUTE_WAYPOINT_MIN_DISTANCE


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

LOCAL_REGION_XZ = 500.0
LOCAL_REGION_Y = 160.0
LOCAL_DWELL_GRACE_S = 60.0
LOCAL_DWELL_RAMP_S = 180.0
LOCAL_DWELL_MAX_PENALTY = 0.35
NEW_MACRO_REGION_REWARD = 0.8
FRONTIER_PROGRESS_MIN_DELTA = 6.0
FRONTIER_PROGRESS_SCALE = 160.0
FRONTIER_PROGRESS_MAX_REWARD = 0.18

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
        self.visited_cells: set[tuple[int, int, int, int, int]] = set()
        self.seen_actors: set[tuple[int, int, str | None, int, int]] = set()
        self.seen_events: set[str] = set()
        self.seen_dialogue: set[tuple[int, int | None, str]] = set()
        self.seen_contexts: set[tuple[int, int, int, str | None]] = set()
        self.seen_transitions: set[tuple[int, int, int, int]] = set()
        self.seen_route_waypoints: set[str] = set()
        self.last_route_waypoint_reached: str | None = None
        self.route_waypoint_advanced = False
        self.route_waypoints_advanced = 0
        self.objective_score = 0
        self.combat_contact = False
        self.seen_inventory_items: set[int] = set()
        self.stagnation_steps = 0
        self.seen_macro_regions: set[tuple[int, int, int, int, int]] = set()
        self.local_anchor_position: tuple[float, float, float] | None = None
        self.local_progress_at: float | None = None
        self.local_dwell_seconds = 0.0
        self.local_anchor_distance = 0.0
        self.local_frontier_radius = 0.0
        self.local_dwell_penalty = 0.0
        self.last_route_waypoint_reached = None
        self.route_waypoint_advanced = False
        self.rupees_collected = 0
        self.ammo_collected = 0
        self.health_recovered = 0
        self.magic_recovered = 0
        self.chests_opened = 0
        self.previous: dict | None = None

    def break_causal_chain(self):
        """Drop causal/local residence state; keep run-wide novelty history."""
        self.previous = None
        self.local_anchor_position = None
        self.local_progress_at = None
        self.local_dwell_seconds = 0.0
        self.local_anchor_distance = 0.0
        self.local_frontier_radius = 0.0
        self.local_dwell_penalty = 0.0

    @staticmethod
    def _macro_region(current: dict) -> tuple[int, int, int, int, int] | None:
        position = current.get("position")
        if position is None:
            return None
        return (
            current["scene"],
            current["room"],
            math.floor((position[0] + LOCAL_REGION_XZ / 2.0) / LOCAL_REGION_XZ),
            math.floor((position[1] + LOCAL_REGION_Y / 2.0) / LOCAL_REGION_Y),
            math.floor((position[2] + LOCAL_REGION_XZ / 2.0) / LOCAL_REGION_XZ),
        )

    def _reset_local_pressure(self, current: dict, now_s: float):
        position = current.get("position")
        self.local_anchor_position = (
            tuple(position) if position is not None else None
        )
        self.local_progress_at = now_s if position is not None else None
        self.local_dwell_seconds = 0.0
        self.local_anchor_distance = 0.0
        self.local_frontier_radius = 0.0
        self.local_dwell_penalty = 0.0

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
    def _durable_progress_gain(previous: dict, current: dict) -> bool:
        if set(current["equipment"]) - set(previous["equipment"]):
            return True
        if current["quest_items"] - previous["quest_items"]:
            return True
        if current["dungeon_items"] - previous["dungeon_items"]:
            return True
        if current["max_health"] > previous["max_health"]:
            return True
        if current["heart_pieces"] > previous["heart_pieces"]:
            return True
        if current["skull_tokens"] > previous["skull_tokens"]:
            return True
        if current["small_keys"] > previous["small_keys"]:
            return True
        if current["magic_acquired"] and not previous["magic_acquired"]:
            return True
        if current["double_magic"] and not previous["double_magic"]:
            return True
        if current["double_defense"] and not previous["double_defense"]:
            return True
        for name, level in current["upgrades"].items():
            if level > previous["upgrades"].get(name, 0):
                return True
        for name, enabled in current["story_flags"].items():
            if enabled and not previous["story_flags"].get(name, False):
                return True
        return False

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
            "max_health": player.max_health if player else 0,
            "rupees": player.rupees if player else 0,
            "magic": player.magic if player else 0,
            "dialogue": (game.dialogue.active, game.dialogue.text_id, game.dialogue.state_code),
            "context": (game.context_action.code, game.context_action.label),
            "intent_key": self._intent_key(intent),
            "target_distance": self._target_distance(game, intent),
            "enemy_health": self._enemy_health(game),
            "inventory_items": {row.item_id: row.name for row in game.inventory_named},
            "inventory_ammo": {
                row.item_id: (row.name, row.ammo)
                for row in game.inventory_named
                if row.ammo is not None
            },
            "equipment": {
                row.item_id: (row.name, row.equipment_type, row.value)
                for row in game.progress.equipment
            },
            "quest_items": set(game.progress.quest_items),
            "dungeon_items": set(game.progress.dungeon_items),
            "story_flags": dict(game.progress.story_flags),
            "upgrades": dict(game.progress.upgrade_levels),
            "heart_pieces": game.progress.heart_pieces,
            "skull_tokens": game.progress.skull_tokens,
            "small_keys": game.progress.small_keys,
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
        guidance: dict | None = None,
        now_s: float | None = None,
    ) -> RewardResult:
        now_s = time.monotonic() if now_s is None else float(now_s)
        current = self._snapshot(game, intent)
        self.route_waypoint_advanced = False
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
                round(player.position[1] / 80.0),
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

        if self.previous is None:
            # Events already present at the first observation predate this run
            # and must not become free achievements.
            self.seen_events.update(
                f"{game.instance_id}:{event.id}" for event in game.events
            )

        for event in game.events:
            key = f"{game.instance_id}:{event.id}"
            if key in self.seen_events:
                continue
            self.seen_events.add(key)
            event_reward = {
                "enemy_defeated": 1.0,
                "boss_defeated": 2.5,
                "game_completed": 5.0,
                "chest_opened": 0.6,
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
            elif event.kind == "chest_opened":
                self.chests_opened += 1
                achievements.append(self._achievement(
                    key=f"chest:{event.id}",
                    kind="exploration",
                    title="Baú aberto",
                    detail=event.detail or "Treasure flag observado pela bridge.",
                    points=10,
                    training_reward=event_reward,
                ))

        previous = self.previous
        if previous is None:
            # Progress already present when the run starts is baseline, not an
            # autonomous achievement. Remember inventory IDs so depletion and
            # later reacquisition cannot farm the same objective.
            self.seen_inventory_items.update(current["inventory_items"])
        if previous and previous["instance"] == current["instance"]:
            if (previous["scene"], previous["room"]) != (
                current["scene"], current["room"]
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

            if self._durable_progress_gain(previous, current):
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

            for dungeon_name in sorted(
                current["dungeon_items"] - previous["dungeon_items"]
            ):
                training_reward = 2.0
                b["objective_milestone"] = (
                    b.get("objective_milestone", 0.0) + training_reward
                )
                achievements.append(self._achievement(
                    key=f"dungeon:{dungeon_name}",
                    kind="item",
                    title=dungeon_name,
                    detail="Novo dungeon item persistente observado.",
                    points=100,
                    training_reward=training_reward,
                ))

            max_health_gain = current["max_health"] - previous["max_health"]
            if max_health_gain > 0:
                training_reward = min(
                    4.0,
                    (max_health_gain / 16.0) * 2.0,
                )
                b["objective_milestone"] = (
                    b.get("objective_milestone", 0.0) + training_reward
                )
                achievements.append(self._achievement(
                    key=f"max_health:{current['max_health']}",
                    kind="progress",
                    title="Capacidade de vida aumentada",
                    detail=f"+{max_health_gain / 16.0:g} coração(ões) de capacidade.",
                    points=max(50, round((max_health_gain / 16.0) * 150)),
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

            rupee_gain = current["rupees"] - previous["rupees"]
            if rupee_gain > 0:
                self.rupees_collected += rupee_gain
                b["resource_rupees"] = min(
                    0.30,
                    0.03 + 0.01 * rupee_gain,
                )

            ammo_reward = 0.0
            for item_id, (name, ammo) in current["inventory_ammo"].items():
                previous_row = previous["inventory_ammo"].get(item_id)
                if previous_row is None:
                    # First acquisition is already handled by the stronger
                    # persistent-item milestone; do not double-count its ammo.
                    continue
                old_ammo = previous_row[1]
                if ammo is None or old_ammo is None:
                    continue
                gained = ammo - old_ammo
                if gained <= 0:
                    continue
                self.ammo_collected += gained
                ammo_reward += min(0.18, 0.025 + 0.0125 * gained)
            if ammo_reward > 0:
                b["resource_ammo"] = min(0.35, ammo_reward)

            health_gain = current["health"] - previous["health"]
            if health_gain > 0:
                self.health_recovered += health_gain
                b["resource_health"] = min(
                    0.24,
                    (health_gain / 16.0) * 0.08,
                )

            magic_gain = current["magic"] - previous["magic"]
            if magic_gain > 0:
                self.magic_recovered += magic_gain
                b["resource_magic"] = min(
                    0.18,
                    magic_gain * 0.004,
                )

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
            straight_line_shaping_suppressed = bool(
                guidance
                and (
                    guidance.get("blocked")
                    or guidance.get("route_active")
                )
            )
            if (
                previous["intent_key"] == current["intent_key"]
                and old_dist is not None
                and new_dist is not None
                and not straight_line_shaping_suppressed
            ):
                improvement = max(-60.0, min(60.0, old_dist - new_dist))
                raw_intent_progress = max(
                    -0.4,
                    min(0.4, improvement / 150.0),
                )
                # A stale waypoint must not pay forever.  Otherwise wandering
                # away and returning to an unreachable target becomes a
                # repeatable reward loop.  Fade both positive and negative
                # target shaping after prolonged residence so exploration can
                # break the attractor; real macro expansion/progress resets it.
                dwell_age = (
                    max(0.0, now_s - self.local_progress_at)
                    if self.local_progress_at is not None
                    else 0.0
                )
                if dwell_age <= LOCAL_DWELL_GRACE_S:
                    intent_progress_scale = 1.0
                else:
                    intent_progress_scale = max(
                        0.0,
                        1.0 - (dwell_age - LOCAL_DWELL_GRACE_S) / 120.0,
                    )
                shaped_progress = raw_intent_progress * intent_progress_scale
                if abs(shaped_progress) > 1e-9:
                    b["intent_progress"] = shaped_progress

        if previous is not None and previous["intent_key"] != current["intent_key"]:
            # A route waypoint belongs to the intent that selected it. Do not
            # credit a late arrival after cognition has already changed goals.
            self.last_route_waypoint_reached = None

        if (
            previous is not None
            and previous["intent_key"] == current["intent_key"]
            and guidance
            and guidance.get("route_active")
            and current.get("position") is not None
        ):
            waypoint_id = guidance.get("route_waypoint_id")
            waypoint = guidance.get("route_waypoint")
            if (
                isinstance(waypoint_id, str)
                and waypoint_id
                and isinstance(waypoint, (list, tuple))
                and len(waypoint) == 3
            ):
                try:
                    route_distance = math.dist(
                        current["position"],
                        tuple(float(value) for value in waypoint),
                    )
                except (TypeError, ValueError):
                    route_distance = float("inf")
                if (
                    route_distance <= ROUTE_WAYPOINT_MIN_DISTANCE
                    and waypoint_id != self.last_route_waypoint_reached
                ):
                    self.last_route_waypoint_reached = waypoint_id
                    self.route_waypoint_advanced = True
                    self.route_waypoints_advanced += 1
                    # Reward a route node only once per run so replaying a known
                    # path cannot farm PPO reward. The progress flag still
                    # resets dwell/stuck timers on later legitimate replays.
                    if waypoint_id not in self.seen_route_waypoints:
                        self.seen_route_waypoints.add(waypoint_id)
                        b["route_waypoint"] = 0.12

        major_progress_keys = {
            "new_world_transition",
            "durable_progress",
            "objective_milestone",
            "native_event",
        }
        major_progress = any(
            b.get(key, 0.0) > 0 for key in major_progress_keys
        )
        current_position = current["position"]
        macro_region = self._macro_region(current)
        macro_expansion = (
            macro_region is not None
            and macro_region not in self.seen_macro_regions
        )
        if macro_expansion:
            self.seen_macro_regions.add(macro_region)
            if previous is not None:
                b["new_macro_region"] = NEW_MACRO_REGION_REWARD

        frontier_mode = intent.mode in {
            "explore",
            "navigate",
            "interact",
            "observe",
        }
        if (
            current_position is not None
            and self.local_anchor_position is not None
            and not macro_expansion
            and frontier_mode
            and not game.dialogue.active
            and not game.pause_menu.active
            and not game.cutscene_active
        ):
            radius = math.dist(current_position, self.local_anchor_position)
            frontier_delta = radius - self.local_frontier_radius
            if frontier_delta >= FRONTIER_PROGRESS_MIN_DELTA:
                b["frontier_progress"] = min(
                    FRONTIER_PROGRESS_MAX_REWARD,
                    frontier_delta / FRONTIER_PROGRESS_SCALE,
                )
                self.local_frontier_radius = radius

        if current_position is None:
            self._reset_local_pressure(current, now_s)
        elif (
            macro_expansion
            or self.local_anchor_position is None
            or self.local_progress_at is None
            or major_progress
            or game.cutscene_active
        ):
            # The full frontier anchor resets only for real coarse spatial
            # expansion or useful game progress. Circling through already-known
            # nearby regions does not buy another frontier reward cycle.
            self._reset_local_pressure(current, now_s)
        else:
            self.local_anchor_distance = math.dist(
                current_position,
                self.local_anchor_position,
            )
            self.local_dwell_seconds = max(
                0.0,
                now_s - self.local_progress_at,
            )
            if self.local_dwell_seconds > LOCAL_DWELL_GRACE_S:
                ramp = min(
                    1.0,
                    (self.local_dwell_seconds - LOCAL_DWELL_GRACE_S)
                    / LOCAL_DWELL_RAMP_S,
                )
                self.local_dwell_penalty = -(
                    0.02
                    + (LOCAL_DWELL_MAX_PENALTY - 0.02) * ramp
                )
                b["local_dwell"] = self.local_dwell_penalty
            else:
                self.local_dwell_penalty = 0.0

        if previous and previous["position"] is not None and current["position"] is not None:
            dx = current["position"][0] - previous["position"][0]
            dy = current["position"][1] - previous["position"][1]
            dz = current["position"][2] - previous["position"][2]
            moved = math.sqrt(dx * dx + dy * dy + dz * dz) >= 4.0
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
