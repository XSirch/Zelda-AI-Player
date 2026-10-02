from __future__ import annotations

import json
import math
import os
import time
import uuid
from pathlib import Path

from ..models import GameState


ROOM_MAP_VERSION = 1
ROOM_MAP_CELL_XZ = 80.0
ROOM_MAP_CELL_Y = 50.0
ROOM_MAP_MAX_ROOMS = 4096
ROOM_MAP_MAX_CELLS_PER_ROOM = 8192
ROOM_MAP_MAX_BLOCKED_PER_ROOM = 4096
ROOM_MAP_MAX_ENTRIES = 32
ROOM_MAP_MAX_TRANSITIONS = 64
ROOM_MAP_MAX_EXITS = 64
ROOM_MAP_MAX_DOORS = 64
ROOM_MAP_MAX_AFFORDANCES = 128

_PROBE_YAW_OFFSETS = {
    "forward": 0.0,
    "forward_left": math.pi / 4.0,
    "left": math.pi / 2.0,
    "back_left": 3.0 * math.pi / 4.0,
    "back": math.pi,
    "back_right": -3.0 * math.pi / 4.0,
    "right": -math.pi / 2.0,
    "forward_right": -math.pi / 4.0,
}


def _distance(a, b) -> float:
    return math.dist(tuple(a), tuple(b))


def _context(game: GameState) -> tuple[bool, str, int, int]:
    age = "adult" if game.player and game.player.age == "adult" else "child"
    return (
        bool(game.mirrored_world),
        age,
        int(game.scene),
        int(game.room),
    )


def _context_key(context: tuple[bool, str, int, int]) -> str:
    mirrored, age, scene, room = context
    return f"{int(mirrored)}:{age}:{scene}:{room}"


def _cell_key(position) -> str:
    x, y, z = position
    gx = round(float(x) / ROOM_MAP_CELL_XZ)
    gy = round(float(y) / ROOM_MAP_CELL_Y)
    gz = round(float(z) / ROOM_MAP_CELL_XZ)
    return f"{gx}:{gy}:{gz}"


def _finite_position(value) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in value):
        return None
    return tuple(float(v) for v in value)


class RoomMapMemory:
    """Persistent empirical geometry and escape memory for visited rooms.

    The room map never invents hidden Zelda geometry. It stores only observations
    already exposed by the bridge (navmesh/probes/actors/exits/affordances) plus
    actual room transitions made by Link. Learned directed movement remains in
    LearnedRouteGraph; this file supplies stable room-local landmarks and escape
    targets when the current frame no longer exposes them.
    """

    def __init__(self, path: Path, *, writable: bool = True):
        self.path = Path(path)
        self.writable = writable
        self.rooms: dict[str, dict] = {}
        self.last_context: str | None = None
        self.last_position: tuple[float, float, float] | None = None
        self.last_full_seq: dict[str, int] = {}
        self.dirty = False
        self.persistence_revision = 0
        self.load_error = ""
        self._load()

    @staticmethod
    def _empty_room(
        context: tuple[bool, str, int, int],
        now_s: float,
    ) -> dict:
        mirrored, age, scene, room = context
        return {
            "scene": scene,
            "room": room,
            "mirrored": mirrored,
            "age": age,
            "visits": 0,
            "first_seen_at": now_s,
            "last_seen_at": now_s,
            "entries": {},
            "transitions": {},
            "exits": {},
            "doors": {},
            "walkable": {},
            "blocked": {},
            "affordances": {},
        }

    @staticmethod
    def _touch_position(
        bucket: dict,
        key: str,
        position,
        now_s: float,
        *,
        limit: int,
        extras: dict | None = None,
    ) -> bool:
        position = tuple(float(v) for v in position)
        current = bucket.get(key)
        if current is None:
            if len(bucket) >= limit:
                return False
            row = {
                "position": list(position),
                "observations": 1,
                "last_seen_at": now_s,
            }
            if extras:
                row.update(extras)
            bucket[key] = row
            return True

        count = max(1, int(current.get("observations") or 1))
        alpha = min(0.20, 1.0 / (count + 1))
        old_position = _finite_position(current.get("position")) or position
        current["position"] = [
            old * (1.0 - alpha) + new * alpha
            for old, new in zip(old_position, position)
        ]
        current["observations"] = count + 1
        current["last_seen_at"] = now_s
        if extras:
            for name, value in extras.items():
                if isinstance(value, bool):
                    current[name] = bool(current.get(name, False) or value)
                elif name in {"samples", "links", "wall_flags"} and isinstance(
                    value, (int, float)
                ):
                    current[name] = max(
                        int(current.get(name) or 0),
                        int(value),
                    )
                else:
                    current[name] = value
        return True

    def _load(self):
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != ROOM_MAP_VERSION:
                raise ValueError("unsupported room map version")
            rooms = data.get("rooms")
            if not isinstance(rooms, dict):
                raise ValueError("invalid room map payload")

            normalized_rooms: dict[str, dict] = {}
            for key, row in rooms.items():
                if not isinstance(key, str) or not isinstance(row, dict):
                    continue
                scene = row.get("scene")
                room = row.get("room")
                if not isinstance(scene, int) or not isinstance(room, int):
                    continue
                normalized = {
                    "scene": scene,
                    "room": room,
                    "mirrored": bool(row.get("mirrored", False)),
                    "age": "adult" if row.get("age") == "adult" else "child",
                    "visits": max(0, int(row.get("visits") or 0)),
                    "first_seen_at": float(row.get("first_seen_at") or 0.0),
                    "last_seen_at": float(row.get("last_seen_at") or 0.0),
                    "entries": {},
                    "transitions": {},
                    "exits": {},
                    "doors": {},
                    "walkable": {},
                    "blocked": {},
                    "affordances": {},
                }
                for bucket_name in (
                    "entries",
                    "transitions",
                    "exits",
                    "doors",
                    "walkable",
                    "blocked",
                    "affordances",
                ):
                    source = row.get(bucket_name) or {}
                    if not isinstance(source, dict):
                        continue
                    for item_key, item in source.items():
                        if not isinstance(item_key, str) or not isinstance(item, dict):
                            continue
                        position = _finite_position(item.get("position"))
                        if position is None:
                            continue
                        copied = dict(item)
                        copied["position"] = list(position)
                        copied["observations"] = max(
                            1, int(item.get("observations") or 1)
                        )
                        copied["last_seen_at"] = float(
                            item.get("last_seen_at") or 0.0
                        )
                        normalized[bucket_name][item_key[:220]] = copied
                normalized_rooms[key[:160]] = normalized
            self.rooms = normalized_rooms
        except (OSError, ValueError, TypeError) as exc:
            self.rooms = {}
            self.load_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    def reset_trace(self):
        self.last_context = None
        self.last_position = None
        self.last_full_seq.clear()

    def _ensure_room(
        self,
        game: GameState,
        now_s: float,
    ) -> tuple[str, dict, bool]:
        context = _context(game)
        key = _context_key(context)
        room = self.rooms.get(key)
        created = False
        if room is None:
            if len(self.rooms) >= ROOM_MAP_MAX_ROOMS:
                return key, self._empty_room(context, now_s), False
            room = self._empty_room(context, now_s)
            self.rooms[key] = room
            created = True
        room["last_seen_at"] = now_s
        return key, room, created

    def _record_transition(
        self,
        previous_key: str,
        destination_game: GameState,
        now_s: float,
    ) -> bool:
        previous = self.rooms.get(previous_key)
        if previous is None or self.last_position is None:
            return False
        destination = _context(destination_game)
        transition_key = (
            f"{int(destination[0])}:{destination[1]}:"
            f"{destination[2]}:{destination[3]}:"
            f"{_cell_key(self.last_position)}"
        )
        return self._touch_position(
            previous["transitions"],
            transition_key,
            self.last_position,
            now_s,
            limit=ROOM_MAP_MAX_TRANSITIONS,
            extras={
                "destination_scene": destination[2],
                "destination_room": destination[3],
                "destination_mirrored": destination[0],
                "destination_age": destination[1],
            },
        )

    def _record_navmesh(self, game: GameState, room: dict, now_s: float) -> bool:
        navmesh = game.navmesh
        if not navmesh.available:
            return False
        changed = False
        ox, _, oz = navmesh.origin
        for gx, gz, floor_y, links in navmesh.cells:
            position = (
                float(ox) + float(gx) * float(navmesh.step),
                float(floor_y),
                float(oz) + float(gz) * float(navmesh.step),
            )
            changed = self._touch_position(
                room["walkable"],
                _cell_key(position),
                position,
                now_s,
                limit=ROOM_MAP_MAX_CELLS_PER_ROOM,
                extras={"links": int(links), "source": "navmesh"},
            ) or changed
        return changed

    def _record_probes(self, game: GameState, room: dict, now_s: float) -> bool:
        player = game.player
        if player is None:
            return False
        changed = False
        player_yaw = player.yaw * math.pi / 32768.0
        px, py, pz = (float(v) for v in player.position)
        for probe in game.navigation_probes:
            offset = _PROBE_YAW_OFFSETS.get(probe.direction)
            if offset is None:
                continue
            yaw = player_yaw + offset
            if probe.wall_hit and isinstance(probe.wall_distance, (int, float)):
                distance = max(1.0, float(probe.wall_distance))
            else:
                distance = float(probe.distance)
            position = (
                px + math.sin(yaw) * distance,
                float(probe.floor_y) if probe.floor_y is not None else py,
                pz + math.cos(yaw) * distance,
            )
            if probe.floor_found and not probe.wall_hit:
                changed = self._touch_position(
                    room["walkable"],
                    _cell_key(position),
                    position,
                    now_s,
                    limit=ROOM_MAP_MAX_CELLS_PER_ROOM,
                    extras={
                        "source": "probe",
                        "floor_type": probe.floor_type,
                    },
                ) or changed
            elif probe.wall_hit or not probe.floor_found:
                changed = self._touch_position(
                    room["blocked"],
                    _cell_key(position),
                    position,
                    now_s,
                    limit=ROOM_MAP_MAX_BLOCKED_PER_ROOM,
                    extras={
                        "source": "probe",
                        "wall_hit": bool(probe.wall_hit),
                        "floor_missing": not bool(probe.floor_found),
                        "wall_flags": int(probe.wall_flags or 0),
                    },
                ) or changed
        return changed

    def _record_exits(self, game: GameState, room: dict, now_s: float) -> bool:
        changed = False
        for exit_row in game.scene_exits:
            key = f"{exit_row.exit_index}:{exit_row.entrance_index}"
            changed = self._touch_position(
                room["exits"],
                key,
                exit_row.position,
                now_s,
                limit=ROOM_MAP_MAX_EXITS,
                extras={
                    "exit_index": int(exit_row.exit_index),
                    "entrance_index": int(exit_row.entrance_index),
                    "samples": int(exit_row.samples),
                    "direct_reachable": bool(exit_row.direct_reachable),
                },
            ) or changed
        return changed

    def _record_doors(self, game: GameState, room: dict, now_s: float) -> bool:
        changed = False
        seen = set()
        for actor in [game.context_actor, *list(game.room_actors)]:
            if actor is None:
                continue
            if (actor.category_name or "").strip().lower() != "door":
                continue
            quantized = _cell_key(actor.position)
            key = actor.actor_uid or f"{actor.actor_id}:{actor.params}:{quantized}"
            if key in seen:
                continue
            seen.add(key)
            changed = self._touch_position(
                room["doors"],
                str(key)[:220],
                actor.position,
                now_s,
                limit=ROOM_MAP_MAX_DOORS,
                extras={
                    "actor_uid": actor.actor_uid,
                    "actor_id": int(actor.actor_id),
                    "params": int(actor.params),
                },
            ) or changed
        return changed

    def _record_affordances(
        self,
        game: GameState,
        room: dict,
        now_s: float,
    ) -> bool:
        changed = False
        for affordance in game.traversal_affordances:
            key = (
                f"{affordance.kind}:{affordance.direction}:"
                f"{_cell_key(affordance.approach_position)}:"
                f"{_cell_key(affordance.target_position)}"
            )
            changed = self._touch_position(
                room["affordances"],
                key[:220],
                affordance.approach_position,
                now_s,
                limit=ROOM_MAP_MAX_AFFORDANCES,
                extras={
                    "kind": affordance.kind,
                    "direction": affordance.direction,
                    "target_position": list(affordance.target_position),
                    "height_delta": float(affordance.height_delta),
                    "wall_flags": int(affordance.wall_flags or 0),
                },
            ) or changed
        return changed

    def observe(self, game: GameState, *, now_s: float | None = None) -> bool:
        if (
            not self.writable
            or not game.in_game
            or game.player is None
            or game.cutscene_active
        ):
            return False
        now_s = time.time() if now_s is None else float(now_s)
        key, room, created = self._ensure_room(game, now_s)
        changed = created

        if self.last_context != key:
            if self.last_context is not None:
                changed = self._record_transition(
                    self.last_context,
                    game,
                    now_s,
                ) or changed
            room["visits"] = max(0, int(room.get("visits") or 0)) + 1
            changed = self._touch_position(
                room["entries"],
                _cell_key(game.player.position),
                game.player.position,
                now_s,
                limit=ROOM_MAP_MAX_ENTRIES,
            ) or changed

        changed = self._touch_position(
            room["walkable"],
            _cell_key(game.player.position),
            game.player.position,
            now_s,
            limit=ROOM_MAP_MAX_CELLS_PER_ROOM,
            extras={"source": "traversed"},
        ) or changed
        slow_refresh = (
            game.protocol == 1
            or self.last_full_seq.get(key) != int(game.full_seq)
        )
        if slow_refresh:
            changed = self._record_navmesh(game, room, now_s) or changed
            changed = self._record_exits(game, room, now_s) or changed
            changed = self._record_affordances(game, room, now_s) or changed
            self.last_full_seq[key] = int(game.full_seq)
        changed = self._record_probes(game, room, now_s) or changed
        changed = self._record_doors(game, room, now_s) or changed

        self.last_context = key
        self.last_position = tuple(float(v) for v in game.player.position)
        if changed:
            self.persistence_revision += 1
            self.dirty = True
        return changed

    def save(self, *, force: bool = False):
        if not self.writable:
            return
        if not self.dirty and (not force or self.path.is_file()):
            return
        save_revision = self.persistence_revision
        temporary = self.path.with_name(
            self.path.name + ".tmp-" + uuid.uuid4().hex
        )
        try:
            rooms = {
                key: {
                    **row,
                    "entries": {
                        item_key: dict(item)
                        for item_key, item in row["entries"].items()
                    },
                    "transitions": {
                        item_key: dict(item)
                        for item_key, item in row["transitions"].items()
                    },
                    "exits": {
                        item_key: dict(item)
                        for item_key, item in row["exits"].items()
                    },
                    "doors": {
                        item_key: dict(item)
                        for item_key, item in row["doors"].items()
                    },
                    "walkable": {
                        item_key: dict(item)
                        for item_key, item in row["walkable"].items()
                    },
                    "blocked": {
                        item_key: dict(item)
                        for item_key, item in row["blocked"].items()
                    },
                    "affordances": {
                        item_key: dict(item)
                        for item_key, item in row["affordances"].items()
                    },
                }
                for key, row in self.rooms.items()
            }
            payload = {
                "version": ROOM_MAP_VERSION,
                "rooms": rooms,
                "saved_at": time.time(),
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, separators=(",", ":"), sort_keys=True),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
        except Exception:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise
        if self.persistence_revision == save_revision:
            self.dirty = False

    def current_room(self, game: GameState) -> dict | None:
        return self.rooms.get(_context_key(_context(game)))

    def has_known_escape(self, game: GameState) -> bool:
        room = self.current_room(game)
        if room is None:
            return False
        return bool(room["transitions"] or room["exits"] or room["doors"])

    def known_escape_candidates(self, game: GameState) -> list[dict]:
        room = self.current_room(game)
        if room is None or game.player is None:
            return []
        player_position = game.player.position
        candidates = []

        for key, row in room["transitions"].items():
            position = _finite_position(row.get("position"))
            if position is None:
                continue
            candidates.append({
                "kind": "transition",
                "memory_key": key,
                "position": position,
                "observations": max(1, int(row.get("observations") or 1)),
                "confidence": 1.0,
                "destination_scene": row.get("destination_scene"),
                "destination_room": row.get("destination_room"),
            })

        for key, row in room["exits"].items():
            position = _finite_position(row.get("position"))
            if position is None:
                continue
            observations = max(1, int(row.get("observations") or 1))
            candidates.append({
                "kind": "exit",
                "memory_key": key,
                "position": position,
                "observations": observations,
                "confidence": min(
                    0.98,
                    0.82
                    + (0.08 if row.get("direct_reachable") else 0.0)
                    + min(0.08, observations * 0.01),
                ),
                "exit_index": row.get("exit_index"),
                "entrance_index": row.get("entrance_index"),
            })

        for key, row in room["doors"].items():
            position = _finite_position(row.get("position"))
            if position is None:
                continue
            observations = max(1, int(row.get("observations") or 1))
            candidates.append({
                "kind": "door",
                "memory_key": key,
                "position": position,
                "observations": observations,
                "confidence": min(0.92, 0.72 + min(0.20, observations * 0.01)),
                "actor_uid": row.get("actor_uid"),
                "actor_id": row.get("actor_id"),
                "params": row.get("params"),
            })

        priority = {"transition": 0, "exit": 1, "door": 2}
        candidates.sort(
            key=lambda row: (
                priority.get(row["kind"], 9),
                -float(row["confidence"]),
                -int(row["observations"]),
                _distance(player_position, row["position"]),
                row["memory_key"],
            )
        )
        return candidates

    def remembered_escape_waypoint(self, game: GameState, route_graph) -> dict | None:
        candidates = self.known_escape_candidates(game)
        if not candidates or game.player is None:
            return None
        candidate = candidates[0]
        position = candidate["position"]
        routed = route_graph.next_waypoint(game, position)
        common = {
            "exit": True,
            "door": candidate["kind"] == "door",
            "remembered": True,
            "memory_kind": candidate["kind"],
            "memory_key": candidate["memory_key"],
            "exit_position": tuple(position),
            "forced_escape": True,
        }
        if routed is not None:
            return {
                **routed,
                **common,
                "direct_reachable": False,
            }

        distance = _distance(game.player.position, position)
        return {
            "waypoint": tuple(position),
            "waypoint_id": (
                f"room-map:{_context_key(_context(game))}:"
                f"{candidate['kind']}:{candidate['memory_key']}"
            )[:220],
            "path_nodes": 1,
            "waypoint_index": 0,
            "target_gap": 0.0,
            "confidence": float(candidate["confidence"]),
            "partial": distance > 95.0,
            "direct_reachable": False,
            **common,
        }

    def stats(self, game: GameState | None = None) -> dict:
        rows = list(self.rooms.values())
        result = {
            "rooms": len(rows),
            "entries": sum(len(row["entries"]) for row in rows),
            "transitions": sum(len(row["transitions"]) for row in rows),
            "exits": sum(len(row["exits"]) for row in rows),
            "doors": sum(len(row["doors"]) for row in rows),
            "walkable_cells": sum(len(row["walkable"]) for row in rows),
            "blocked_cells": sum(len(row["blocked"]) for row in rows),
            "affordances": sum(len(row["affordances"]) for row in rows),
            "writable": self.writable,
            "dirty": self.dirty,
            "load_error": self.load_error or None,
        }
        if game is not None:
            current = self.current_room(game)
            result["current_room"] = (
                {
                    "scene": current["scene"],
                    "room": current["room"],
                    "visits": current["visits"],
                    "entries": len(current["entries"]),
                    "transitions": len(current["transitions"]),
                    "exits": len(current["exits"]),
                    "doors": len(current["doors"]),
                    "walkable_cells": len(current["walkable"]),
                    "blocked_cells": len(current["blocked"]),
                    "affordances": len(current["affordances"]),
                }
                if current is not None
                else None
            )
        return result
