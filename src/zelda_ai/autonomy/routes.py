from __future__ import annotations

import heapq
import json
import math
import os
import time
import uuid
from pathlib import Path

from ..models import GameState


ROUTE_GRAPH_VERSION = 1
ROUTE_CELL_XZ = 80.0
ROUTE_CELL_Y = 50.0
ROUTE_MAX_EDGE_DISTANCE = 190.0
ROUTE_START_RADIUS = 170.0
ROUTE_TARGET_RADIUS = 240.0
ROUTE_PARTIAL_MIN_GAIN = 80.0
ROUTE_WAYPOINT_MIN_DISTANCE = 95.0
ROUTE_MAX_NODES = 50_000


def _distance(a, b) -> float:
    return math.dist(tuple(a), tuple(b))


def _node_id(
    scene: int,
    room: int,
    position,
    *,
    mirrored: bool = False,
    age: str = "child",
) -> str:
    x, y, z = position
    gx = round(float(x) / ROUTE_CELL_XZ)
    gy = round(float(y) / ROUTE_CELL_Y)
    gz = round(float(z) / ROUTE_CELL_XZ)
    age_key = "adult" if age == "adult" else "child"
    return f"{int(bool(mirrored))}:{age_key}:{scene}:{room}:{gx}:{gy}:{gz}"


class LearnedRouteGraph:
    """Persistent topological memory learned only from Link's observed traversal.

    Nodes are coarse scene-local positions actually occupied by Link. Directed
    edges are added only when Link physically moves between nodes. Path reuse
    therefore contains no hidden map knowledge and cannot invent an unobserved
    corridor.
    """

    def __init__(self, path: Path, *, writable: bool = True):
        self.path = Path(path)
        self.writable = writable
        self.nodes: dict[str, dict] = {}
        self.edges: dict[str, dict[str, dict]] = {}
        self.by_scene: dict[tuple[int, int, bool, str], set[str]] = {}
        self.last_node_id: str | None = None
        self.last_position: tuple[float, float, float] | None = None
        self.last_scene_room: tuple[int, int] | None = None
        self.dirty = False
        self.revision = 0
        self.last_failed_search: tuple[str, str, int] | None = None
        self.routes_reused = 0
        self.active_target_signature: str | None = None
        self.active_target_node_id: str | None = None
        self.active_path: list[str] = []
        self.counted_route_target_signature: str | None = None
        self.last_path_nodes = 0
        self.last_target_gap: float | None = None
        self.load_error = ""
        self._load()

    def _index_node(self, node_id: str, row: dict):
        key = (
            int(row["scene"]),
            int(row["room"]),
            bool(row.get("mirrored", False)),
            "adult" if row.get("age") == "adult" else "child",
        )
        self.by_scene.setdefault(key, set()).add(node_id)

    def _load(self):
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if data.get("version") != ROUTE_GRAPH_VERSION:
                raise ValueError("unsupported route graph version")
            nodes = data.get("nodes")
            edges = data.get("edges")
            if not isinstance(nodes, dict) or not isinstance(edges, dict):
                raise ValueError("invalid route graph payload")
            for node_id, row in nodes.items():
                if not isinstance(row, dict):
                    continue
                position = row.get("position")
                if (
                    not isinstance(position, list)
                    or len(position) != 3
                    or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in position)
                ):
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
                    "position": [float(v) for v in position],
                    "visits": max(1, int(row.get("visits") or 1)),
                    "updated_at": float(row.get("updated_at") or 0.0),
                }
                self.nodes[str(node_id)] = normalized
                self._index_node(str(node_id), normalized)
            for source, destinations in edges.items():
                if source not in self.nodes or not isinstance(destinations, dict):
                    continue
                for target, edge in destinations.items():
                    if target not in self.nodes or not isinstance(edge, dict):
                        continue
                    distance = edge.get("distance")
                    if not isinstance(distance, (int, float)) or not math.isfinite(distance) or distance <= 0:
                        continue
                    self.edges.setdefault(source, {})[target] = {
                        "distance": float(distance),
                        "traversals": max(1, int(edge.get("traversals") or 1)),
                        "updated_at": float(edge.get("updated_at") or 0.0),
                    }
        except (OSError, ValueError, TypeError) as exc:
            self.nodes = {}
            self.edges = {}
            self.by_scene = {}
            self.load_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    def reset_trace(self):
        self.last_node_id = None
        self.last_position = None
        self.last_scene_room = None
        self.active_target_signature = None
        self.active_target_node_id = None
        self.active_path = []
        self.counted_route_target_signature = None
        self.last_failed_search = None

    def _touch_node(
        self,
        scene: int,
        room: int,
        position,
        now_s: float,
        *,
        mirrored: bool,
        age: str,
    ) -> str:
        node_id = _node_id(
            scene,
            room,
            position,
            mirrored=mirrored,
            age=age,
        )
        existing = self.nodes.get(node_id)
        if existing is None:
            if len(self.nodes) >= ROUTE_MAX_NODES:
                return node_id
            row = {
                "scene": int(scene),
                "room": int(room),
                "mirrored": bool(mirrored),
                "age": "adult" if age == "adult" else "child",
                "position": [float(v) for v in position],
                "visits": 1,
                "updated_at": now_s,
            }
            self.nodes[node_id] = row
            self._index_node(node_id, row)
            self.revision += 1
            self.last_failed_search = None
            self.dirty = True
        else:
            visits = max(1, int(existing.get("visits") or 1))
            # Slow running average keeps a stable representative point for the
            # coarse cell instead of chasing every frame's jitter.
            alpha = min(0.15, 1.0 / (visits + 1))
            existing["position"] = [
                float(old) * (1.0 - alpha) + float(new) * alpha
                for old, new in zip(existing["position"], position)
            ]
            existing["visits"] = visits + 1
            existing["updated_at"] = now_s
            self.dirty = True
        return node_id

    def observe(self, game: GameState, *, now_s: float | None = None) -> bool:
        if not self.writable or not game.player or not game.in_game or game.cutscene_active:
            return False
        now_s = time.time() if now_s is None else float(now_s)
        position = tuple(float(v) for v in game.player.position)
        scene_room = (int(game.scene), int(game.room))
        node_id = _node_id(
            game.scene,
            game.room,
            position,
            mirrored=game.mirrored_world,
            age=game.player.age,
        )

        if node_id == self.last_node_id and scene_room == self.last_scene_room:
            self.last_position = position
            return False

        node_id = self._touch_node(
            game.scene,
            game.room,
            position,
            now_s,
            mirrored=game.mirrored_world,
            age=game.player.age,
        )
        edge_added = False
        if (
            self.last_node_id is not None
            and self.last_position is not None
            and self.last_scene_room == scene_room
            and self.last_node_id in self.nodes
            and node_id in self.nodes
            and self.last_node_id != node_id
        ):
            distance = _distance(
                self.nodes[self.last_node_id]["position"],
                self.nodes[node_id]["position"],
            )
            if 1.0 <= distance <= ROUTE_MAX_EDGE_DISTANCE:
                bucket = self.edges.setdefault(self.last_node_id, {})
                edge = bucket.get(node_id)
                if edge is None:
                    bucket[node_id] = {
                        "distance": distance,
                        "traversals": 1,
                        "updated_at": now_s,
                    }
                    self.revision += 1
                    self.last_failed_search = None
                    edge_added = True
                else:
                    traversals = max(1, int(edge.get("traversals") or 1))
                    edge["distance"] = (
                        float(edge["distance"]) * traversals + distance
                    ) / (traversals + 1)
                    edge["traversals"] = traversals + 1
                    edge["updated_at"] = now_s
                self.dirty = True

        self.last_node_id = node_id
        self.last_position = position
        self.last_scene_room = scene_room
        return edge_added

    def save(self, *, force: bool = False):
        if not self.writable:
            return
        if not self.dirty and (not force or self.path.is_file()):
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": ROUTE_GRAPH_VERSION,
            "nodes": self.nodes,
            "edges": self.edges,
            "saved_at": time.time(),
        }
        temporary = self.path.with_name(
            self.path.name + ".tmp-" + uuid.uuid4().hex
        )
        temporary.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
        self.dirty = False

    def _nearest_node(
        self,
        scene: int,
        room: int,
        position,
        radius: float,
        *,
        mirrored: bool,
        age: str,
    ):
        candidates = self.by_scene.get((
            int(scene),
            int(room),
            bool(mirrored),
            "adult" if age == "adult" else "child",
        )) or ()
        best_id = None
        best_distance = float("inf")
        for node_id in candidates:
            node = self.nodes[node_id]
            distance = _distance(node["position"], position)
            if distance < best_distance:
                best_id = node_id
                best_distance = distance
        if best_id is None or best_distance > radius:
            return None, None
        return best_id, best_distance

    @staticmethod
    def _edge_cost(edge: dict) -> float:
        distance = max(1.0, float(edge.get("distance") or 1.0))
        traversals = max(1, int(edge.get("traversals") or 1))
        confidence_discount = min(0.30, math.log1p(traversals) * 0.07)
        return distance * (1.0 - confidence_discount)

    def _best_reachable_path(
        self,
        start_id: str,
        target_position,
    ) -> tuple[list[str] | None, float | None]:
        if start_id not in self.nodes:
            return None, None

        queue: list[tuple[float, str]] = [(0.0, start_id)]
        costs = {start_id: 0.0}
        previous: dict[str, str] = {}
        best_id = start_id
        start_gap = _distance(self.nodes[start_id]["position"], target_position)
        best_gap = start_gap
        best_cost = 0.0
        best_score = start_gap

        while queue:
            current_cost, current = heapq.heappop(queue)
            if current_cost != costs.get(current):
                continue

            gap = _distance(self.nodes[current]["position"], target_position)
            # A learned route should make meaningful geometric progress without
            # preferring an enormous historical loop just to end a few units
            # closer to the target.
            score = gap + current_cost * 0.25
            if (
                score < best_score - 1e-6
                or (
                    abs(score - best_score) <= 1e-6
                    and gap < best_gap
                )
            ):
                best_id = current
                best_gap = gap
                best_cost = current_cost
                best_score = score

            for neighbor, edge in self.edges.get(current, {}).items():
                if neighbor not in self.nodes:
                    continue
                new_cost = current_cost + self._edge_cost(edge)
                if new_cost >= costs.get(neighbor, float("inf")):
                    continue
                costs[neighbor] = new_cost
                previous[neighbor] = current
                heapq.heappush(queue, (new_cost, neighbor))

        gain = start_gap - best_gap
        if (
            best_id == start_id
            or (
                best_gap > ROUTE_TARGET_RADIUS
                and gain < ROUTE_PARTIAL_MIN_GAIN
            )
        ):
            return None, best_gap

        path = [best_id]
        while path[-1] != start_id:
            parent = previous.get(path[-1])
            if parent is None:
                return None, best_gap
            path.append(parent)
        path.reverse()
        return path, best_gap


    def next_waypoint(self, game: GameState, target_position) -> dict | None:
        if not game.player or target_position is None:
            self.active_target_signature = None
            self.active_target_node_id = None
            self.active_path = []
            self.counted_route_target_signature = None
            self.last_failed_search = None
            self.last_path_nodes = 0
            self.last_target_gap = None
            return None

        start_id = _node_id(
            game.scene,
            game.room,
            game.player.position,
            mirrored=game.mirrored_world,
            age=game.player.age,
        )
        if start_id not in self.nodes:
            start_id, _ = self._nearest_node(
                game.scene,
                game.room,
                game.player.position,
                ROUTE_START_RADIUS,
                mirrored=game.mirrored_world,
                age=game.player.age,
            )
        if start_id is None:
            self.active_path = []
            self.last_path_nodes = 0
            self.last_target_gap = None
            return None

        target_signature = _node_id(
            game.scene,
            game.room,
            target_position,
            mirrored=game.mirrored_world,
            age=game.player.age,
        )
        if self.active_target_signature != target_signature:
            self.active_target_signature = target_signature
            self.active_target_node_id = None
            self.active_path = []
            self.counted_route_target_signature = None
            self.last_failed_search = None

        path = None
        target_gap = None
        if self.active_path:
            try:
                start_index = self.active_path.index(start_id)
            except ValueError:
                start_index = -1
            if 0 <= start_index < len(self.active_path) - 1:
                path = self.active_path[start_index:]
                endpoint = self.nodes.get(path[-1])
                if endpoint is not None:
                    target_gap = _distance(endpoint["position"], target_position)

        if path is None:
            search_key = (start_id, target_signature, self.revision)
            if self.last_failed_search == search_key:
                self.last_path_nodes = 0
                return None
            path, target_gap = self._best_reachable_path(
                start_id,
                target_position,
            )
            self.active_path = list(path or ())
            self.active_target_node_id = path[-1] if path else None
            self.last_failed_search = None if path else search_key

        if not path or len(path) < 2:
            self.active_path = list(path or ())
            self.last_path_nodes = len(path or ())
            self.last_target_gap = target_gap
            return None

        player_position = game.player.position
        waypoint_index = 1
        while waypoint_index < len(path) - 1:
            candidate = self.nodes[path[waypoint_index]]["position"]
            if _distance(player_position, candidate) >= ROUTE_WAYPOINT_MIN_DISTANCE:
                break
            waypoint_index += 1

        waypoint_id = path[waypoint_index]
        waypoint = tuple(self.nodes[waypoint_id]["position"])
        traversals = []
        for source, target in zip(path[:-1], path[1:]):
            edge = self.edges.get(source, {}).get(target)
            if edge:
                traversals.append(max(1, int(edge.get("traversals") or 1)))
        confidence = min(1.0, (min(traversals) if traversals else 1) / 4.0)

        if self.counted_route_target_signature != target_signature:
            self.routes_reused += 1
            self.counted_route_target_signature = target_signature
        self.last_path_nodes = len(path)
        self.last_target_gap = target_gap
        return {
            "waypoint": waypoint,
            "waypoint_id": waypoint_id,
            "path_nodes": len(path),
            "waypoint_index": waypoint_index,
            "target_gap": float(target_gap or 0.0),
            "confidence": confidence,
            "partial": bool(
                isinstance(target_gap, (int, float))
                and target_gap > ROUTE_TARGET_RADIUS
            ),
        }

    def stats(self) -> dict:
        edge_count = sum(len(rows) for rows in self.edges.values())
        return {
            "nodes": len(self.nodes),
            "edges": edge_count,
            "routes_reused": self.routes_reused,
            "last_path_nodes": self.last_path_nodes,
            "cached_path_nodes": len(self.active_path),
            "revision": self.revision,
            "last_target_gap": (
                round(self.last_target_gap, 1)
                if isinstance(self.last_target_gap, (int, float))
                else None
            ),
            "writable": self.writable,
            "load_error": self.load_error or None,
        }
