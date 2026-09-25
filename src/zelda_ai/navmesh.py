"""Collision-derived local navigation mesh planning.

The native bridge emits a compact walkable grid from SoH collision. This module
keeps route selection deterministic and local: A* chooses a connected path,
while the 16 realtime terrain probes remain the last safety gate before input.
"""
from __future__ import annotations

from dataclasses import dataclass
import heapq
import math

from .models import GameState, NavigationMeshSnapshot


# Bit order must match native/ZeldaAiBridge.cpp.
_DIRECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
_PROBE_NAMES_V2 = (
    "forward", "forward_left", "left", "back_left",
    "back", "back_right", "right", "forward_right",
)
_PROBE_NAMES_LEGACY = (
    "forward", "forward_right", "right", "back_right",
    "back", "back_left", "left", "forward_left",
)


@dataclass(frozen=True)
class NavMeshPlan:
    waypoint: tuple[float, float, float]
    path: tuple[tuple[int, int], ...]
    goal_cell: tuple[int, int]
    target_distance: float
    cost: float
    exact_goal_reachable: bool


def _cell_map(mesh: NavigationMeshSnapshot) -> dict[tuple[int, int], tuple[float, int]]:
    return {(gx, gz): (floor_y, links) for gx, gz, floor_y, links in mesh.cells}


def _cell_world(mesh: NavigationMeshSnapshot, key: tuple[int, int],
                row: tuple[float, int]) -> tuple[float, float, float]:
    return (
        mesh.origin[0] + key[0] * mesh.step,
        row[0],
        mesh.origin[2] + key[1] * mesh.step,
    )


def _target_score(point: tuple[float, float, float], target: tuple[float, float, float]) -> float:
    # XZ dominates route selection; a modest Y penalty avoids selecting a floor
    # directly above/below the target when multiple walkable levels overlap.
    return math.hypot(point[0] - target[0], point[2] - target[2]) + abs(point[1] - target[1]) * 1.5


def _nearest_cell(mesh: NavigationMeshSnapshot,
                  cells: dict[tuple[int, int], tuple[float, int]],
                  point: tuple[float, float, float]) -> tuple[int, int] | None:
    if not cells:
        return None
    return min(cells, key=lambda key: _target_score(_cell_world(mesh, key, cells[key]), point))


def _neighbors(mesh: NavigationMeshSnapshot,
               cells: dict[tuple[int, int], tuple[float, int]],
               key: tuple[int, int]):
    floor_y, links = cells[key]
    for index, (dx, dz) in enumerate(_DIRECTIONS):
        if not (links & (1 << index)):
            continue
        neighbor = (key[0] + dx, key[1] + dz)
        other = cells.get(neighbor)
        if other is None:
            continue
        # Native generation should already emit reciprocal links. Requiring the
        # reverse bit makes a partially-truncated/corrupt snapshot fail closed.
        if not (other[1] & (1 << ((index + 4) % 8))):
            continue
        horizontal = mesh.step * (math.sqrt(2.0) if dx and dz else 1.0)
        vertical = abs(other[0] - floor_y)
        yield neighbor, horizontal + vertical * 0.35


def _heuristic(mesh: NavigationMeshSnapshot, a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot((a[0] - b[0]) * mesh.step, (a[1] - b[1]) * mesh.step)


def _reconstruct(came_from: dict[tuple[int, int], tuple[int, int]],
                 end: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    path = [end]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return tuple(path)


def _astar(mesh: NavigationMeshSnapshot,
           cells: dict[tuple[int, int], tuple[float, int]],
           start: tuple[int, int], goal: tuple[int, int]
           ) -> tuple[tuple[tuple[int, int], ...], float] | None:
    queue: list[tuple[float, float, tuple[int, int]]] = [(0.0, 0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best = {start: 0.0}
    while queue:
        _, cost, current = heapq.heappop(queue)
        if cost > best.get(current, float("inf")) + 1e-6:
            continue
        if current == goal:
            return _reconstruct(came_from, current), cost
        for neighbor, edge_cost in _neighbors(mesh, cells, current):
            new_cost = cost + edge_cost
            if new_cost + 1e-6 >= best.get(neighbor, float("inf")):
                continue
            best[neighbor] = new_cost
            came_from[neighbor] = current
            priority = new_cost + _heuristic(mesh, neighbor, goal)
            heapq.heappush(queue, (priority, new_cost, neighbor))
    return None


def _reachable_frontier(mesh: NavigationMeshSnapshot,
                        cells: dict[tuple[int, int], tuple[float, int]],
                        start: tuple[int, int],
                        target: tuple[float, float, float]
                        ) -> tuple[tuple[tuple[int, int], ...], float] | None:
    """Return the reachable cell that makes the most geometric target progress."""
    queue: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    best_cost = {start: 0.0}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost > best_cost.get(current, float("inf")) + 1e-6:
            continue
        for neighbor, edge_cost in _neighbors(mesh, cells, current):
            new_cost = cost + edge_cost
            if new_cost + 1e-6 >= best_cost.get(neighbor, float("inf")):
                continue
            best_cost[neighbor] = new_cost
            came_from[neighbor] = current
            heapq.heappush(queue, (new_cost, neighbor))

    start_distance = _target_score(_cell_world(mesh, start, cells[start]), target)
    candidates = [key for key in best_cost if key != start]
    if not candidates:
        return None
    best_key = min(
        candidates,
        key=lambda key: (
            _target_score(_cell_world(mesh, key, cells[key]), target),
            best_cost[key],
        ),
    )
    best_distance = _target_score(_cell_world(mesh, best_key, cells[best_key]), target)
    # A wall may force a short lateral move before target distance can improve.
    # Permit a bounded regression so the moving mesh can follow the wall until
    # its endpoint becomes visible, while refusing large detours into unrelated
    # connected space.
    if best_distance > start_distance + max(12.0, mesh.step * 0.35):
        return None
    return _reconstruct(came_from, best_key), best_cost[best_key]


def plan_navmesh(game: GameState, target: tuple[float, float, float] | list[float],
                 *, max_lookahead_edges: int = 2) -> NavMeshPlan | None:
    """Plan one safe local waypoint toward target using the latest full NavMesh."""
    if not game.player or not game.navmesh.available:
        return None
    mesh = game.navmesh
    cells = _cell_map(mesh)
    if not cells:
        return None
    target3 = (float(target[0]), float(target[1]), float(target[2]))
    start = _nearest_cell(mesh, cells, game.player.position)
    goal = _nearest_cell(mesh, cells, target3)
    if start is None or goal is None:
        return None

    exact = _astar(mesh, cells, start, goal)
    exact_goal_reachable = exact is not None
    route = exact or _reachable_frontier(mesh, cells, start, target3)
    if route is None:
        return None
    path, cost = route
    if len(path) < 2:
        # The target can be inside the same 70-unit cell while still outside a
        # small requested stop_distance. Keep that final micro-approach local;
        # waypoint_probe_safe() will still veto walls/cliffs before input.
        player = game.player.position
        horizontal = math.hypot(target3[0] - player[0], target3[2] - player[2])
        if horizontal <= mesh.step * 0.75 and abs(target3[1] - player[1]) <= 24.0:
            return NavMeshPlan(
                waypoint=target3,
                path=path,
                goal_cell=start,
                target_distance=0.0,
                cost=0.0,
                exact_goal_reachable=True,
            )
        return None

    # Look through only collinear, already-connected edges. This smooths the
    # stick command without cutting a corner that the graph intentionally
    # blocked, and limits exposure to the 200 ms full-mesh refresh interval.
    waypoint_index = 1
    if max_lookahead_edges > 1 and len(path) > 2:
        first_step = (path[1][0] - path[0][0], path[1][1] - path[0][1])
        for index in range(2, min(len(path), max_lookahead_edges + 1)):
            step = (path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1])
            if step != first_step:
                break
            waypoint_index = index

    waypoint_key = path[waypoint_index]
    waypoint = _cell_world(mesh, waypoint_key, cells[waypoint_key])
    goal_key = path[-1]
    goal_world = _cell_world(mesh, goal_key, cells[goal_key])
    return NavMeshPlan(
        waypoint=waypoint,
        path=path,
        goal_cell=goal_key,
        target_distance=math.dist(goal_world, target3),
        cost=cost,
        exact_goal_reachable=exact_goal_reachable,
    )


def _probe_direction(game: GameState, waypoint: tuple[float, float, float]) -> str | None:
    if not game.player:
        return None
    dx = waypoint[0] - game.player.position[0]
    dz = waypoint[2] - game.player.position[2]
    if math.hypot(dx, dz) < 1e-4:
        return None
    desired_yaw = math.atan2(dx, dz) * 32768.0 / math.pi
    relative = ((desired_yaw - game.player.yaw + 32768.0) % 65536.0) - 32768.0
    index = int(math.floor((relative + 4096.0) / 8192.0)) % 8
    names = _PROBE_NAMES_V2 if "probe_yaw_v2" in game.capabilities else _PROBE_NAMES_LEGACY
    return names[index]


def waypoint_probe_safe(game: GameState, waypoint: tuple[float, float, float],
                        *, probe_distance: float = 75.0,
                        wall_clearance: float = 42.0) -> bool:
    """Validate the next NavMesh heading against the fast, Link-relative probes."""
    direction = _probe_direction(game, waypoint)
    if direction is None:
        return True
    if not game.navigation_probes:
        return "local_navmesh" not in game.capabilities
    probes = [p for p in game.navigation_probes
              if p.direction == direction and p.distance <= probe_distance]
    if not probes:
        return False
    desired_sample = 140.0 if probe_distance > 100.0 else 70.0
    probe = min(probes, key=lambda row: abs(row.distance - desired_sample))
    if not probe.floor_found or probe.delta_y is None:
        return False
    if abs(probe.delta_y) > 24.0:
        return False
    if probe.wall_hit:
        if probe.wall_distance is None:
            return False
        if game.player:
            waypoint_distance = math.hypot(
                waypoint[0] - game.player.position[0],
                waypoint[2] - game.player.position[2],
            )
        else:
            waypoint_distance = probe.distance
        required_clearance = min(probe.distance, waypoint_distance + 18.0)
        if probe.wall_distance < max(wall_clearance, required_clearance):
            return False
    return True


def primitive_move_safe(game: GameState, direction: str, *,
                        probe_distance: float = 75.0,
                        wall_clearance: float = 42.0) -> bool:
    """Validate the world heading produced by a raw camera-relative move."""
    if not game.player:
        return False
    if not game.navigation_probes:
        return "local_navmesh" not in game.capabilities

    offsets = {
        "forward": 0,
        "back": 0x8000,
        # SoH computes stick angle from Math_Atan2S(relY, -relX):
        # raw negative X is +90 degrees, raw positive X is -90 degrees.
        "left": 0x4000,
        "right": -0x4000,
    }
    offset = offsets.get(direction)
    if offset is None:
        return True
    if game.mirrored_world and direction in {"left", "right"}:
        offset = -offset

    if game.camera_input_yaw is not None:
        world_yaw = game.camera_input_yaw + offset
    elif game.camera_eye is not None and game.camera_at is not None:
        fx = game.camera_at[0] - game.camera_eye[0]
        fz = game.camera_at[2] - game.camera_eye[2]
        if math.hypot(fx, fz) >= 1e-4:
            world_yaw = math.atan2(fx, fz) * 32768.0 / math.pi + offset
        else:
            world_yaw = game.player.yaw + offset
    else:
        world_yaw = game.player.yaw + offset

    angle = world_yaw * math.pi / 32768.0
    distance = 140.0 if probe_distance > 100.0 else 70.0
    x, y, z = game.player.position
    return waypoint_probe_safe(
        game,
        (x + math.sin(angle) * distance, y, z + math.cos(angle) * distance),
        probe_distance=probe_distance,
        wall_clearance=wall_clearance,
    )

