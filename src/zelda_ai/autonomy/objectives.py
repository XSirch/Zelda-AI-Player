from __future__ import annotations

from dataclasses import dataclass

from ..models import GameState
from .models import AgentIntent, ObjectiveCompletion


def _norm(value: str | None) -> str:
    return " ".join((value or "").strip().casefold().split())


@dataclass
class ObjectiveStatus:
    completed: bool
    reason: str = ""
    evidence: dict | None = None


class ObjectiveTracker:
    """Locks one strategic objective until structured telemetry proves completion."""

    def __init__(self):
        self.intent: AgentIntent | None = None
        self.accepted_seq = 0
        self.accepted_scene_room: tuple[int, int] | None = None
        self.baseline_event_ids: set[str] = set()
        self.completed_count = 0
        self.last_completion: dict | None = None

    @property
    def active(self) -> bool:
        return self.intent is not None

    @property
    def trackable(self) -> bool:
        return bool(self.intent and self.intent.completion.kind != "manual")

    def clear(self):
        self.intent = None
        self.accepted_seq = 0
        self.accepted_scene_room = None
        self.baseline_event_ids.clear()

    def assign(self, intent: AgentIntent, game: GameState):
        self.intent = intent
        self.accepted_seq = int(game.seq)
        self.accepted_scene_room = (int(game.scene), int(game.room))
        self.baseline_event_ids = {str(row.id) for row in game.events}

    def _has_named(self, values, name: str | None) -> bool:
        wanted = _norm(name)
        return bool(wanted and any(_norm(value) == wanted for value in values))

    def evaluate(self, game: GameState) -> ObjectiveStatus:
        if self.intent is None:
            return ObjectiveStatus(False)
        completion = self.intent.completion
        kind = completion.kind

        if kind == "manual":
            return ObjectiveStatus(False)

        if kind == "equipment":
            if completion.item_id is not None and any(
                row.item_id == completion.item_id
                for row in game.progress.equipment
            ):
                return ObjectiveStatus(
                    True,
                    "equipment_item_id",
                    {"item_id": completion.item_id},
                )
            names = [
                *game.progress.owned_equipment,
                *(row.name for row in game.progress.equipment),
            ]
            if self._has_named(names, completion.name):
                return ObjectiveStatus(
                    True,
                    "equipment_name",
                    {"name": completion.name},
                )
            return ObjectiveStatus(False)

        if kind == "inventory_item":
            for row in game.inventory_named:
                if (
                    completion.item_id is not None
                    and row.item_id == completion.item_id
                ) or (
                    completion.name
                    and _norm(row.name) == _norm(completion.name)
                ):
                    return ObjectiveStatus(
                        True,
                        "inventory_item",
                        {"item_id": row.item_id, "name": row.name},
                    )
            return ObjectiveStatus(False)

        if kind == "quest_item":
            if self._has_named(game.progress.quest_items, completion.name):
                return ObjectiveStatus(
                    True,
                    "quest_item",
                    {"name": completion.name},
                )
            return ObjectiveStatus(False)

        if kind == "story_flag":
            if completion.flag and game.progress.story_flags.get(completion.flag):
                return ObjectiveStatus(
                    True,
                    "story_flag",
                    {"flag": completion.flag},
                )
            return ObjectiveStatus(False)

        if kind == "scene":
            scene_match = (
                completion.scene is not None
                and game.scene == completion.scene
            )
            name_match = bool(
                completion.name
                and _norm(game.scene_name) == _norm(completion.name)
            )
            if scene_match or name_match:
                return ObjectiveStatus(
                    True,
                    "scene",
                    {
                        "scene": game.scene,
                        "scene_name": game.scene_name,
                        "room": game.room,
                    },
                )
            return ObjectiveStatus(False)

        if kind == "scene_room":
            scene_match = (
                completion.scene is None
                or game.scene == completion.scene
            )
            name_match = (
                not completion.name
                or _norm(game.scene_name) == _norm(completion.name)
            )
            if (
                scene_match
                and name_match
                and game.room == completion.room
            ):
                return ObjectiveStatus(
                    True,
                    "scene_room",
                    {
                        "scene": game.scene,
                        "scene_name": game.scene_name,
                        "room": game.room,
                    },
                )
            return ObjectiveStatus(False)

        if kind == "leave_scene_room":
            origin_scene = (
                completion.scene
                if completion.scene is not None
                else self.accepted_scene_room[0]
                if self.accepted_scene_room
                else None
            )
            origin_room = (
                completion.room
                if completion.room is not None
                else self.accepted_scene_room[1]
                if self.accepted_scene_room
                else None
            )
            if (
                origin_scene is not None
                and origin_room is not None
                and (game.scene, game.room) != (origin_scene, origin_room)
            ):
                return ObjectiveStatus(
                    True,
                    "left_scene_room",
                    {
                        "from_scene": origin_scene,
                        "from_room": origin_room,
                        "to_scene": game.scene,
                        "to_room": game.room,
                    },
                )
            return ObjectiveStatus(False)

        if kind == "rupees_at_least":
            if game.player and game.player.rupees >= int(completion.threshold or 0):
                return ObjectiveStatus(
                    True,
                    "rupees_at_least",
                    {"rupees": game.player.rupees},
                )
            return ObjectiveStatus(False)

        if kind == "heart_pieces_at_least":
            if game.progress.heart_pieces >= int(completion.threshold or 0):
                return ObjectiveStatus(
                    True,
                    "heart_pieces_at_least",
                    {"heart_pieces": game.progress.heart_pieces},
                )
            return ObjectiveStatus(False)

        if kind == "skull_tokens_at_least":
            if game.progress.skull_tokens >= int(completion.threshold or 0):
                return ObjectiveStatus(
                    True,
                    "skull_tokens_at_least",
                    {"skull_tokens": game.progress.skull_tokens},
                )
            return ObjectiveStatus(False)

        if kind == "small_keys_at_least":
            if game.progress.small_keys >= int(completion.threshold or 0):
                return ObjectiveStatus(
                    True,
                    "small_keys_at_least",
                    {"small_keys": game.progress.small_keys},
                )
            return ObjectiveStatus(False)

        if kind == "magic_acquired":
            if game.progress.magic_acquired:
                return ObjectiveStatus(
                    True,
                    "magic_acquired",
                    {"magic_acquired": True},
                )
            return ObjectiveStatus(False)

        if kind == "dialogue_actor":
            speaker = game.dialogue.speaker
            if game.dialogue.active and speaker is not None:
                id_match = (
                    completion.actor_id is not None
                    and speaker.actor_id == completion.actor_id
                )
                name_match = bool(
                    completion.actor_name
                    and _norm(
                        speaker.description or speaker.name
                    ) == _norm(completion.actor_name)
                )
                if id_match or name_match:
                    return ObjectiveStatus(
                        True,
                        "dialogue_actor",
                        {
                            "actor_id": speaker.actor_id,
                            "actor_name": speaker.description or speaker.name,
                        },
                    )
            return ObjectiveStatus(False)

        if kind in {"event_kind", "game_completed"}:
            wanted = (
                "game_completed"
                if kind == "game_completed"
                else completion.event_kind
            )
            for event in game.events:
                if (
                    str(event.id) not in self.baseline_event_ids
                    and event.kind == wanted
                ):
                    return ObjectiveStatus(
                        True,
                        "event_kind",
                        {"event_kind": event.kind, "event_id": event.id},
                    )
            return ObjectiveStatus(False)

        return ObjectiveStatus(False)

    def complete(self, status: ObjectiveStatus):
        if not status.completed or self.intent is None:
            return
        self.completed_count += 1
        self.last_completion = {
            "objective": self.intent.objective,
            "completion": self.intent.completion.model_dump(),
            "reason": status.reason,
            "evidence": status.evidence or {},
        }
        self.clear()
