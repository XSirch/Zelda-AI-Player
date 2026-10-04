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
ROUTE_TARGET_REACHED_DISTANCE = 95.0
ROUTE_PARTIAL_MIN_GAIN = 80.0
ROUTE_WAYPOINT_MIN_DISTANCE = 45.0
ROUTE_MAX_NODES = 50_000
FRONTIER_REACHED_DISTANCE = 45.0
FRONTIER_PROGRESS_DELTA = 8.0
FRONTIER_STALL_S = 2.5
FRONTIER_MAX_AGE_S = 7.0
FRONTIER_RETRY_COOLDOWN_S = 12.0
FRONTIER_AVOID_FAILURES = 3
ROUTE_EDGE_PROGRESS_DELTA = 8.0
ROUTE_EDGE_STALL_S = 2.5
ROUTE_EDGE_MAX_AGE_S = 8.0
ROUTE_EDGE_RETRY_COOLDOWN_S = 20.0

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
        self.interactions: dict[str, dict] = {}
        self.by_scene: dict[tuple[int, int, bool, str], set[str]] = {}
        self.last_node_id: str | None = None
        self.last_position: tuple[float, float, float] | None = None
        self.last_context: tuple[int, int, bool, str] | None = None
        self.dirty = False
        self.persistence_revision = 0
        self.revision = 0
        self.last_failed_search: tuple[str, str, int] | None = None
        self.routes_reused = 0
        self.active_target_signature: str | None = None
        self.active_target_node_id: str | None = None
        self.active_path: list[str] = []
        self.counted_route_target_signature: str | None = None
        self.exhausted_partial_nodes: set[str] = set()
        self.exhaustion_revision = 0
        self.last_path_nodes = 0
        self.last_target_gap: float | None = None
        self.active_frontier: dict | None = None
        self.frontier_retry_after: dict[str, float] = {}
        self.frontier_failures: dict[str, int] = {}
        self.frontier_completed = 0
        self.frontier_abandoned = 0
        self.active_route_edge: dict | None = None
        self.route_edge_retry_after: dict[str, float] = {}
        self.route_edge_completed = 0
        self.route_edge_abandoned = 0
        self.last_route_failure_at: float | None = None
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
            interactions = data.get("interactions") or {}
            frontier_failures = data.get("frontier_failures") or {}
            if (
                not isinstance(nodes, dict)
                or not isinstance(edges, dict)
                or not isinstance(interactions, dict)
                or not isinstance(frontier_failures, dict)
            ):
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
                    entry_position = edge.get("entry_position")
                    if (
                        isinstance(entry_position, list)
                        and len(entry_position) == 3
                        and all(
                            isinstance(v, (int, float)) and math.isfinite(v)
                            for v in entry_position
                        )
                    ):
                        normalized_entry = [
                            float(v) for v in entry_position
                        ]
                    else:
                        normalized_entry = list(
                            self.nodes[target]["position"]
                        )
                    self.edges.setdefault(source, {})[target] = {
                        "distance": float(distance),
                        "traversals": max(1, int(edge.get("traversals") or 1)),
                        "failures": max(0, int(edge.get("failures") or 0)),
                        "entry_position": normalized_entry,
                        "updated_at": float(edge.get("updated_at") or 0.0),
                    }
            for key, row in interactions.items():
                if not isinstance(key, str) or not isinstance(row, dict):
                    continue
                button = row.get("button")
                if not isinstance(button, str) or not button:
                    continue
                self.interactions[key[:200]] = {
                    "button": button[:32],
                    "successes": max(1, int(row.get("successes") or 1)),
                    "failures": max(0, int(row.get("failures") or 0)),
                    "updated_at": float(row.get("updated_at") or 0.0),
                }
            for key, value in frontier_failures.items():
                if not isinstance(key, str):
                    continue
                try:
                    failures = max(0, int(value))
                except (TypeError, ValueError):
                    continue
                if failures > 0:
                    self.frontier_failures[key[:200]] = failures
        except (OSError, ValueError, TypeError) as exc:
            self.nodes = {}
            self.edges = {}
            self.interactions = {}
            self.frontier_failures = {}
            self.by_scene = {}
            self.load_error = f"{type(exc).__name__}: {str(exc)[:160]}"

    def reset_trace(self):
        self.last_node_id = None
        self.last_position = None
        self.last_context = None
        self.active_target_signature = None
        self.active_target_node_id = None
        self.active_path = []
        self.counted_route_target_signature = None
        self.exhausted_partial_nodes.clear()
        self.exhaustion_revision = self.revision
        self.last_failed_search = None
        self.active_frontier = None
        self.frontier_retry_after.clear()
        self.active_route_edge = None
        self.route_edge_retry_after.clear()
        self.last_route_failure_at = None

    def clear_navigation_commitment(self):
        """Drop the current target/route/frontier without forgetting trace history."""
        self.active_target_signature = None
        self.active_target_node_id = None
        self.active_path = []
        self.counted_route_target_signature = None
        self.exhausted_partial_nodes.clear()
        self.exhaustion_revision = self.revision
        self.last_failed_search = None
        self.active_frontier = None
        self.active_route_edge = None
        self.last_route_failure_at = None

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
            self.persistence_revision += 1
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
            self.persistence_revision += 1
            self.dirty = True
        return node_id

    def observe(self, game: GameState, *, now_s: float | None = None) -> bool:
        if not self.writable or not game.player or not game.in_game or game.cutscene_active:
            return False
        now_s = time.time() if now_s is None else float(now_s)
        position = tuple(float(v) for v in game.player.position)
        context = (
            int(game.scene),
            int(game.room),
            bool(game.mirrored_world),
            "adult" if game.player.age == "adult" else "child",
        )
        node_id = _node_id(
            game.scene,
            game.room,
            position,
            mirrored=game.mirrored_world,
            age=game.player.age,
        )

        if node_id == self.last_node_id and context == self.last_context:
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
            and self.last_context == context
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
                        "failures": 0,
                        # Store the actual observed crossing point. A coarse
                        # cell centroid/average can lie on the wrong side of a
                        # corner even though the directed transition itself was
                        # genuinely traversed.
                        "entry_position": [float(v) for v in position],
                        "updated_at": now_s,
                    }
                    self.revision += 1
                    self.persistence_revision += 1
                    self.last_failed_search = None
                    edge_added = True
                else:
                    traversals = max(1, int(edge.get("traversals") or 1))
                    edge["distance"] = (
                        float(edge["distance"]) * traversals + distance
                    ) / (traversals + 1)
                    edge["traversals"] = traversals + 1
                    edge["failures"] = max(
                        0,
                        int(edge.get("failures") or 0) - 1,
                    )
                    # Most recent successful crossing is safer than averaging
                    # gateway positions from potentially disconnected parts of
                    # the same coarse cell.
                    edge["entry_position"] = [
                        float(v) for v in position
                    ]
                    edge["updated_at"] = now_s
                    self.persistence_revision += 1
                self.dirty = True

        self.last_node_id = node_id
        self.last_position = position
        self.last_context = context
        return edge_added

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
            # The saver runs in a worker thread while the realtime actor keeps
            # observing. Copy into an immutable snapshot first. If a structural
            # mutation races the copy, Python raises RuntimeError; the caller
            # records it and retries on the next interval. Atomic replacement
            # means a failed snapshot never damages the last durable graph.
            nodes = {
                node_id: {
                    **row,
                    "position": list(row.get("position") or []),
                }
                for node_id, row in self.nodes.items()
            }
            edges = {
                source: {
                    target: dict(edge)
                    for target, edge in destinations.items()
                }
                for source, destinations in self.edges.items()
            }
            payload = {
                "version": ROUTE_GRAPH_VERSION,
                "nodes": nodes,
                "edges": edges,
                "interactions": {
                    key: dict(row)
                    for key, row in self.interactions.items()
                },
                "frontier_failures": dict(self.frontier_failures),
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

        # Mutations that happened while the worker was serializing were not
        # necessarily captured. Keep dirty=True so the next background pass
        # writes them instead of silently losing route memory.
        if self.persistence_revision == save_revision:
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
        failures = max(0, int(edge.get("failures") or 0))
        confidence_discount = min(0.30, math.log1p(traversals) * 0.07)
        failure_penalty = min(1.5, failures * 0.25)
        return distance * (1.0 - confidence_discount) * (1.0 + failure_penalty)

    @staticmethod
    def _edge_key(source: str, target: str) -> str:
        return f"{source}->{target}"

    @staticmethod
    def _edge_waypoint(edge: dict, target_node: dict):
        value = edge.get("entry_position")
        if (
            isinstance(value, (list, tuple))
            and len(value) == 3
            and all(
                isinstance(v, (int, float)) and math.isfinite(v)
                for v in value
            )
        ):
            return tuple(float(v) for v in value)
        return tuple(float(v) for v in target_node["position"])

    def _best_reachable_path(
        self,
        start_id: str,
        target_position,
        *,
        excluded_endpoints: set[str] | None = None,
        now_s: float | None = None,
    ) -> tuple[list[str] | None, float | None]:
        if start_id not in self.nodes:
            return None, None

        excluded_endpoints = excluded_endpoints or set()
        now_s = time.monotonic() if now_s is None else float(now_s)
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
            if current not in excluded_endpoints and (
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
                edge_key = self._edge_key(current, neighbor)
                if self.route_edge_retry_after.get(edge_key, 0.0) > now_s:
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


    def _record_route_edge_success(self):
        if self.active_route_edge is None:
            return
        self.route_edge_completed += 1
        self.active_route_edge = None

    def _record_route_edge_failure(
        self,
        *,
        now_s: float,
    ):
        active = self.active_route_edge
        if active is None:
            return
        source = str(active["source"])
        target = str(active["target"])
        edge_key = self._edge_key(source, target)
        self.route_edge_retry_after[edge_key] = (
            now_s + ROUTE_EDGE_RETRY_COOLDOWN_S
        )
        edge = self.edges.get(source, {}).get(target)
        if edge is not None and self.writable:
            edge["failures"] = max(
                0,
                int(edge.get("failures") or 0),
            ) + 1
            edge["updated_at"] = time.time()
            self.persistence_revision += 1
            self.dirty = True
        self.route_edge_abandoned += 1
        self.last_route_failure_at = now_s
        self.active_route_edge = None
        self.active_path = []
        self.last_failed_search = None

    def _update_route_edge_attempt(
        self,
        game: GameState,
        start_id: str,
        *,
        now_s: float,
    ) -> bool:
        """Return True when the current directed edge was just abandoned."""
        active = self.active_route_edge
        if active is None or game.player is None:
            return False

        target_id = active.get("target")
        passed_target = False
        if self.active_path and target_id in self.active_path and start_id in self.active_path:
            passed_target = (
                self.active_path.index(start_id)
                >= self.active_path.index(target_id)
            )
        if start_id == target_id or passed_target:
            self._record_route_edge_success()
            return False

        waypoint = tuple(active["waypoint"])
        distance = _distance(game.player.position, waypoint)
        # entry_position was sampled after Link actually crossed into the
        # target cell. Reaching "near" it is not sufficient; keep steering
        # until the current coarse node really changes to the edge target.
        best_distance = float(active.get("best_distance", distance))
        if distance <= best_distance - ROUTE_EDGE_PROGRESS_DELTA:
            active["best_distance"] = distance
            active["last_progress_at"] = now_s

        stalled = (
            now_s - float(active.get("last_progress_at", now_s))
            >= ROUTE_EDGE_STALL_S
        )
        expired = (
            now_s - float(active.get("started_at", now_s))
            >= ROUTE_EDGE_MAX_AGE_S
        )
        if stalled or expired:
            self._record_route_edge_failure(now_s=now_s)
            return True
        return False

    def route_recovery_needed(
        self,
        *,
        now_s: float | None = None,
        window_s: float = 5.0,
    ) -> bool:
        if self.last_route_failure_at is None:
            return False
        now_s = time.monotonic() if now_s is None else float(now_s)
        return now_s - self.last_route_failure_at <= window_s

    def next_waypoint(
        self,
        game: GameState,
        target_position,
        *,
        now_s: float | None = None,
    ) -> dict | None:
        now_s = time.monotonic() if now_s is None else float(now_s)
        expired_keys = [
            key for key, until in self.route_edge_retry_after.items()
            if until <= now_s
        ]
        if expired_keys:
            for key in expired_keys:
                self.route_edge_retry_after.pop(key, None)
            self.last_failed_search = None

        if not game.player or target_position is None:
            self.active_target_signature = None
            self.active_route_edge = None
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
            self.active_route_edge = None
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
        target_changed = self.active_target_signature != target_signature
        if target_changed:
            self.active_target_signature = target_signature
            self.active_target_node_id = None
            self.active_path = []
            self.active_route_edge = None
            self.counted_route_target_signature = None
            self.exhausted_partial_nodes.clear()
            self.exhaustion_revision = self.revision
            self.last_failed_search = None
        else:
            self._update_route_edge_attempt(
                game,
                start_id,
                now_s=now_s,
            )

        if not target_changed and self.exhaustion_revision != self.revision:
            # New observed nodes/edges may extend a formerly dead-end branch.
            self.exhausted_partial_nodes.clear()
            self.exhaustion_revision = self.revision
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
            elif start_index == len(self.active_path) - 1:
                endpoint = self.nodes.get(start_id)
                endpoint_gap = (
                    _distance(endpoint["position"], target_position)
                    if endpoint is not None
                    else None
                )
                if (
                    isinstance(endpoint_gap, (int, float))
                    and endpoint_gap > ROUTE_TARGET_REACHED_DISTANCE
                ):
                    # This partial branch has delivered everything currently
                    # known for this target. Do not repeatedly pull Link back
                    # here until newly observed graph structure changes it.
                    self.exhausted_partial_nodes.update(self.active_path)
                    self.active_path = []
                    self.active_route_edge = None
                    self.last_failed_search = None

        if path is None:
            search_key = (start_id, target_signature, self.revision)
            if self.last_failed_search == search_key:
                self.last_path_nodes = 0
                return None
            path, target_gap = self._best_reachable_path(
                start_id,
                target_position,
                excluded_endpoints=self.exhausted_partial_nodes,
                now_s=now_s,
            )
            self.active_path = list(path or ())
            self.active_target_node_id = path[-1] if path else None
            self.last_failed_search = None if path else search_key

        if not path or len(path) < 2:
            self.active_path = list(path or ())
            self.active_route_edge = None
            self.last_path_nodes = len(path or ())
            self.last_target_gap = target_gap
            return None

        player_position = game.player.position
        # Follow one observed directed edge at a time. Its crossing point is a
        # stronger waypoint than the coarse target-cell centroid.
        waypoint_index = 1
        source_id = path[0]
        waypoint_id = path[1]
        edge = self.edges.get(source_id, {}).get(waypoint_id)
        if edge is None:
            self.active_path = []
            self.active_route_edge = None
            return None
        waypoint = self._edge_waypoint(
            edge,
            self.nodes[waypoint_id],
        )
        edge_key = self._edge_key(source_id, waypoint_id)
        if (
            self.active_route_edge is None
            or self.active_route_edge.get("key") != edge_key
        ):
            distance_to_waypoint = _distance(player_position, waypoint)
            self.active_route_edge = {
                "key": edge_key,
                "source": source_id,
                "target": waypoint_id,
                "waypoint": waypoint,
                "started_at": now_s,
                "last_progress_at": now_s,
                "best_distance": distance_to_waypoint,
            }
        traversals = []
        for source, target in zip(path[:-1], path[1:]):
            route_edge = self.edges.get(source, {}).get(target)
            if route_edge:
                traversals.append(
                    max(1, int(route_edge.get("traversals") or 1))
                )
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
                and target_gap > ROUTE_TARGET_REACHED_DISTANCE
            ),
            "edge_key": edge_key,
            "edge_source": source_id,
            "edge_target": waypoint_id,
            "edge_failures": max(0, int(edge.get("failures") or 0)),
            "edge_entry_position": tuple(waypoint),
        }

    def clear_frontier(self):
        """Drop ephemeral frontier commitment without marking it failed."""
        self.active_frontier = None

    def interaction_button(self, key: str) -> str | None:
        row = self.interactions.get(str(key)[:200])
        if not row:
            return None
        button = row.get("button")
        return str(button) if button else None

    def record_interaction_success(
        self,
        key: str,
        button: str,
        *,
        now_s: float | None = None,
    ):
        if not self.writable:
            return
        key = str(key)[:200]
        button = str(button)[:32]
        if not key or not button:
            return
        now_s = time.time() if now_s is None else float(now_s)
        current = self.interactions.get(key)
        if current and current.get("button") == button:
            current["successes"] = max(
                1,
                int(current.get("successes") or 1),
            ) + 1
            current["failures"] = 0
            current["updated_at"] = now_s
        else:
            self.interactions[key] = {
                "button": button,
                "successes": 1,
                "failures": 0,
                "updated_at": now_s,
            }
        self.persistence_revision += 1
        self.dirty = True

    def record_interaction_failure(
        self,
        key: str,
        button: str,
        *,
        now_s: float | None = None,
    ):
        if not self.writable:
            return
        key = str(key)[:200]
        button = str(button)[:32]
        current = self.interactions.get(key)
        if not current or current.get("button") != button:
            return
        failures = max(0, int(current.get("failures") or 0)) + 1
        if failures >= 3:
            self.interactions.pop(key, None)
        else:
            current["failures"] = failures
            current["updated_at"] = (
                time.time() if now_s is None else float(now_s)
            )
        self.persistence_revision += 1
        self.dirty = True

    @staticmethod
    def _observed_doors(game: GameState) -> list:
        """Return unique door actors already exposed by the structured observer."""
        candidates = []
        seen = set()
        for actor in [game.context_actor, *list(game.room_actors)]:
            if actor is None:
                continue
            if (actor.category_name or "").strip().lower() != "door":
                continue
            key = actor.actor_uid or (
                int(actor.actor_id),
                int(actor.params),
                tuple(round(float(value), 1) for value in actor.position),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(actor)
        if game.player is None:
            return candidates
        player_position = game.player.position
        context_uid = (
            game.context_actor.actor_uid
            if game.context_actor is not None
            and (game.context_actor.category_name or "").strip().lower() == "door"
            else None
        )
        candidates.sort(
            key=lambda actor: (
                0 if context_uid and actor.actor_uid == context_uid else 1,
                _distance(player_position, actor.position),
                actor.actor_id,
                actor.params,
            )
        )
        return candidates

    def has_observed_escape(self, game: GameState) -> bool:
        return bool(game.scene_exits or self._observed_doors(game))

    def door_waypoint(self, game: GameState) -> dict | None:
        """Use an observed door as a room-escape target when no exit surface works.

        This is observation-driven rather than Zelda-specific route knowledge:
        ACTORCAT_DOOR is already part of the bridge's generic actor telemetry.
        A learned route is reused when available; otherwise collision-aware
        guidance approaches the observed actor directly until context interaction
        discovery can take over.
        """
        if not game.player:
            return None
        doors = self._observed_doors(game)
        if not doors:
            return None

        door = doors[0]
        door_position = tuple(float(value) for value in door.position)
        routed = self.next_waypoint(game, door_position)
        door_id = door.actor_uid or f"{door.actor_id}:{door.params}"
        if routed is not None:
            return {
                **routed,
                "exit": True,
                "door": True,
                "door_uid": door.actor_uid,
                "door_actor_id": door.actor_id,
                "door_params": door.params,
                "exit_position": door_position,
                "direct_reachable": False,
            }

        distance = _distance(game.player.position, door_position)
        return {
            "waypoint": door_position,
            "waypoint_id": (
                f"door:{game.scene}:{game.room}:{door_id}"
            ),
            "path_nodes": 1,
            "waypoint_index": 0,
            "target_gap": 0.0,
            "confidence": 1.0 if distance <= 140.0 else 0.85,
            "partial": distance > ROUTE_TARGET_REACHED_DISTANCE,
            "exit": True,
            "door": True,
            "door_uid": door.actor_uid,
            "door_actor_id": door.actor_id,
            "door_params": door.params,
            "exit_position": door_position,
            "direct_reachable": distance <= 140.0,
        }

    def escape_waypoint(self, game: GameState) -> dict | None:
        """Prefer native scene exits, then fall back to observed room doors."""
        routed_exit = self.exit_waypoint(game)
        if routed_exit is not None:
            return {
                **routed_exit,
                "forced_escape": True,
            }
        door = self.door_waypoint(game)
        if door is not None:
            return {
                **door,
                "forced_escape": True,
            }
        return None

    def exit_waypoint(
        self,
        game: GameState,
        *,
        allow_unreachable: bool = False,
    ) -> dict | None:
        """Choose an observed scene-exit surface, optionally via learned route.

        Normal callers reject an unreachable raw exit without route/door evidence.
        Forced room recovery may request that raw target so local traversal/collision
        guidance can still try to discover a route instead of ignoring the exit.
        """
        if not game.player or not game.scene_exits:
            return None

        player_position = game.player.position
        exits = sorted(
            game.scene_exits,
            key=lambda row: (
                0 if row.direct_reachable else 1,
                _distance(player_position, row.position),
                -row.samples,
                row.exit_index,
            ),
        )
        exit_row = exits[0]

        if not exit_row.direct_reachable:
            routed = self.next_waypoint(game, exit_row.position)
            if routed is not None:
                return {
                    **routed,
                    "exit": True,
                    "exit_index": exit_row.exit_index,
                    "entrance_index": exit_row.entrance_index,
                    "exit_position": tuple(exit_row.position),
                    "direct_reachable": False,
                }

            context_door = bool(
                game.context_actor is not None
                and (
                    game.context_actor.category_name or ""
                ).strip().lower() == "door"
            )
            nearby_door = any(
                (actor.category_name or "").strip().lower() == "door"
                and _distance(actor.position, exit_row.position) <= 220.0
                for actor in game.room_actors
            )
            if not context_door and not nearby_door and not allow_unreachable:
                # An exit surface behind unrelated collision is evidence of a
                # destination, not of a currently traversable straight line.
                # Keep exploring local frontiers until a real route is observed.
                return None

        return {
            "waypoint": tuple(exit_row.position),
            "waypoint_id": (
                f"exit:{game.scene}:{game.room}:"
                f"{exit_row.exit_index}:{exit_row.entrance_index}"
            ),
            "path_nodes": 1,
            "waypoint_index": 0,
            "target_gap": 0.0,
            "confidence": 1.0 if exit_row.direct_reachable else 0.65,
            "partial": not exit_row.direct_reachable,
            "exit": True,
            "exit_index": exit_row.exit_index,
            "entrance_index": exit_row.entrance_index,
            "exit_position": tuple(exit_row.position),
            "direct_reachable": bool(exit_row.direct_reachable),
        }

    def traversal_waypoint(
        self,
        game: GameState,
        *,
        include_visited: bool = False,
    ) -> dict | None:
        """Choose an observed vertical traversal into new room-local space.

        This is not hidden map knowledge: every candidate comes directly from
        the current collision-derived traversal affordances. Prefer targets whose
        coarse route cell Link has never occupied so ladders/stairs become useful
        exploration frontiers instead of being ignored by the horizontal probes.
        """
        if not game.player or not game.traversal_affordances:
            return None

        player_position = tuple(float(value) for value in game.player.position)
        candidates = []
        for row in game.traversal_affordances:
            target = tuple(float(value) for value in row.target_position)
            approach = tuple(float(value) for value in row.approach_position)
            target_id = _node_id(
                game.scene,
                game.room,
                target,
                mirrored=game.mirrored_world,
                age=game.player.age,
            )
            visited = target_id in self.nodes
            if visited and not include_visited:
                continue

            approach_distance = _distance(player_position, approach)
            kind_penalty = (
                90.0
                if row.kind == "ledge_down"
                else 20.0
                if row.kind in {"stairs_or_slope_up", "stairs_or_slope_down"}
                else 0.0
            )
            visited_penalty = 400.0 if visited else 0.0
            score = approach_distance + kind_penalty + visited_penalty
            candidates.append(
                (
                    score,
                    approach_distance,
                    row,
                    approach,
                    target,
                    target_id,
                    visited,
                )
            )

        if not candidates:
            return None

        (
            _,
            approach_distance,
            row,
            approach,
            target,
            target_id,
            visited,
        ) = min(candidates, key=lambda item: item[0])
        if approach_distance > 55.0:
            phase = "approach"
            waypoint = approach
        else:
            phase = "target"
            waypoint = target

        return {
            "waypoint": waypoint,
            "waypoint_id": (
                f"traversal:{game.scene}:{game.room}:{row.kind}:"
                f"{target_id}:{phase}"
            )[:220],
            "path_nodes": 1,
            "waypoint_index": 0,
            "target_gap": None,
            "confidence": 1.0 if not visited else 0.70,
            "partial": True,
            "traversal": True,
            "traversal_kind": row.kind,
            "traversal_direction": row.direction,
            "traversal_phase": phase,
            "traversal_target_node": target_id,
            "traversal_target_visited": visited,
        }

    def exploration_waypoint(
        self,
        game: GameState,
        *,
        now_s: float | None = None,
    ) -> dict | None:
        """Hold a local observed frontier until success, stall, or expiry.

        Recomputing the best relative probe every motor tick lets the greedy
        selector collapse into a permanent left/right bias. This method instead
        turns one observed open probe into a short-lived world-space objective.
        Rearward probes are considered only when no forward/side option exists.
        """
        if not game.player or not game.navigation_probes:
            self.active_frontier = None
            return None

        now_s = time.monotonic() if now_s is None else float(now_s)
        context = (
            int(game.scene),
            int(game.room),
            bool(game.mirrored_world),
            "adult" if game.player.age == "adult" else "child",
        )
        player_position = tuple(float(v) for v in game.player.position)
        current_node_id = _node_id(
            game.scene,
            game.room,
            player_position,
            mirrored=game.mirrored_world,
            age=game.player.age,
        )

        # Expired failures become eligible again after Link has had time to
        # discover a different approach.
        self.frontier_retry_after = {
            key: until
            for key, until in self.frontier_retry_after.items()
            if until > now_s
        }

        active = self.active_frontier
        if active is not None and active.get("context") == context:
            waypoint = tuple(active["waypoint"])
            distance = _distance(player_position, waypoint)
            reached = (
                distance <= FRONTIER_REACHED_DISTANCE
                or current_node_id == active.get("waypoint_id")
            )
            if reached:
                self.frontier_completed += 1
                waypoint_id = str(active["waypoint_id"])
                previous_failures = self.frontier_failures.get(waypoint_id, 0)
                if previous_failures > 0:
                    if previous_failures == 1:
                        self.frontier_failures.pop(waypoint_id, None)
                    else:
                        self.frontier_failures[waypoint_id] = previous_failures - 1
                    if self.writable:
                        self.persistence_revision += 1
                        self.dirty = True
                self.active_frontier = None
            else:
                best_distance = float(active.get("best_distance", distance))
                if distance <= best_distance - FRONTIER_PROGRESS_DELTA:
                    active["best_distance"] = distance
                    active["last_progress_at"] = now_s
                stalled = (
                    now_s - float(active.get("last_progress_at", now_s))
                    >= FRONTIER_STALL_S
                )
                expired = (
                    now_s - float(active.get("started_at", now_s))
                    >= FRONTIER_MAX_AGE_S
                )
                if not stalled and not expired:
                    return {
                        "waypoint": waypoint,
                        "waypoint_id": active["waypoint_id"],
                        "path_nodes": 1,
                        "waypoint_index": 0,
                        "target_gap": None,
                        "confidence": active["confidence"],
                        "partial": True,
                        "frontier": True,
                        "direction": active["direction"],
                        "stable": True,
                        "age_s": round(
                            now_s - float(active["started_at"]),
                            3,
                        ),
                    }

                self.frontier_abandoned += 1
                failed_id = str(active["waypoint_id"])
                self.frontier_retry_after[failed_id] = (
                    now_s + FRONTIER_RETRY_COOLDOWN_S
                )
                self.frontier_failures[failed_id] = (
                    self.frontier_failures.get(failed_id, 0) + 1
                )
                if self.writable:
                    self.persistence_revision += 1
                    self.dirty = True
                self.active_frontier = None
        elif active is not None:
            self.active_frontier = None

        probe_by_direction = {}
        for probe in game.navigation_probes:
            current = probe_by_direction.get(probe.direction)
            if current is None or probe.distance > current.distance:
                probe_by_direction[probe.direction] = probe

        player = game.player
        player_yaw = player.yaw * math.pi / 32768.0
        candidates = []
        for name, offset in _PROBE_YAW_OFFSETS.items():
            probe = probe_by_direction.get(name)
            if probe is None or not probe.floor_found or probe.wall_hit:
                continue
            delta_y = float(probe.delta_y or 0.0)
            if delta_y > 70.0 or delta_y < -90.0:
                continue

            yaw = player_yaw + offset
            distance = float(probe.distance)
            waypoint = (
                float(player.position[0]) + math.sin(yaw) * distance,
                float(probe.floor_y)
                if probe.floor_y is not None
                else float(player.position[1]),
                float(player.position[2]) + math.cos(yaw) * distance,
            )
            node_id = _node_id(
                game.scene,
                game.room,
                waypoint,
                mirrored=game.mirrored_world,
                age=player.age,
            )
            if self.frontier_retry_after.get(node_id, 0.0) > now_s:
                continue
            node = self.nodes.get(node_id)
            unseen = node is None
            visits = int((node or {}).get("visits") or 0)
            failures = max(0, int(self.frontier_failures.get(node_id, 0)))
            alignment = math.cos(offset)
            candidates.append({
                "name": name,
                "offset": offset,
                "waypoint": waypoint,
                "node_id": node_id,
                "visits": visits,
                "failures": failures,
                "unseen": unseen,
                "alignment": alignment,
                "distance": distance,
            })

        if not candidates:
            return None

        non_rear = [
            row for row in candidates
            if row["name"] not in {"back_left", "back", "back_right"}
        ]
        if non_rear:
            candidates = non_rear

        healthy = [
            row for row in candidates
            if row["failures"] < FRONTIER_AVOID_FAILURES
        ]
        if healthy:
            candidates = healthy

        selected = max(
            candidates,
            key=lambda row: (
                -row["failures"],
                1 if row["unseen"] else 0,
                -row["visits"],
                row["alignment"],
                row["distance"],
            ),
        )
        confidence = 1.0 if selected["visits"] > 0 else 0.5
        self.active_frontier = {
            "context": context,
            "waypoint": tuple(selected["waypoint"]),
            "waypoint_id": selected["node_id"],
            "direction": selected["name"],
            "confidence": confidence,
            "failures": selected["failures"],
            "started_at": now_s,
            "last_progress_at": now_s,
            "best_distance": _distance(
                player_position,
                selected["waypoint"],
            ),
        }
        return {
            "waypoint": tuple(selected["waypoint"]),
            "waypoint_id": selected["node_id"],
            "path_nodes": 1,
            "waypoint_index": 0,
            "target_gap": None,
            "confidence": confidence,
            "partial": True,
            "frontier": True,
            "direction": selected["name"],
            "frontier_failures": selected["failures"],
            "stable": True,
            "age_s": 0.0,
        }

    def room_failure_pressure(self, game: GameState) -> int:
        if game.player is None:
            return 0
        age_key = "adult" if game.player.age == "adult" else "child"
        prefix = (
            f"{int(bool(game.mirrored_world))}:{age_key}:"
            f"{int(game.scene)}:{int(game.room)}:"
        )
        frontier_pressure = sum(
            min(FRONTIER_AVOID_FAILURES, max(0, int(value)))
            for key, value in self.frontier_failures.items()
            if key.startswith(prefix)
        )
        edge_pressure = 0
        for source, destinations in self.edges.items():
            if not source.startswith(prefix):
                continue
            edge_pressure += sum(
                min(3, max(0, int(edge.get("failures") or 0)))
                for edge in destinations.values()
            )
        return frontier_pressure + edge_pressure

    def stats(self) -> dict:
        edge_count = sum(len(rows) for rows in self.edges.values())
        return {
            "nodes": len(self.nodes),
            "edges": edge_count,
            "learned_interactions": len(self.interactions),
            "routes_reused": self.routes_reused,
            "active_frontier": (
                self.active_frontier.get("direction")
                if self.active_frontier
                else None
            ),
            "frontier_completed": self.frontier_completed,
            "frontier_abandoned": self.frontier_abandoned,
            "frontier_failed_cells": len(self.frontier_failures),
            "frontier_failure_total": sum(self.frontier_failures.values()),
            "route_edge_completed": self.route_edge_completed,
            "route_edge_abandoned": self.route_edge_abandoned,
            "route_edges_cooling_down": len(self.route_edge_retry_after),
            "active_route_edge": (
                self.active_route_edge.get("key")
                if self.active_route_edge
                else None
            ),
            "last_path_nodes": self.last_path_nodes,
            "cached_path_nodes": len(self.active_path),
            "exhausted_partial_nodes": len(self.exhausted_partial_nodes),
            "revision": self.revision,
            "persistence_revision": self.persistence_revision,
            "dirty": self.dirty,
            "last_target_gap": (
                round(self.last_target_gap, 1)
                if isinstance(self.last_target_gap, (int, float))
                else None
            ),
            "writable": self.writable,
            "load_error": self.load_error or None,
        }
