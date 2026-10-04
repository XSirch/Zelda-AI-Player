"""Ephemeral paths over the CURRENT instrumented local collision mesh.

These are proposed paths, never evidence for LearnedRouteGraph. Only actual
movement may persist a directed route. No cells, links, reverse edges or portal
destinations are invented here.
"""
from __future__ import annotations

import heapq
import math

from ..models import GameState

OFFSETS = ((0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1))


def observed_local_path(game: GameState, target, *, minimum_gain=None) -> dict | None:
    mesh = game.navmesh
    if not game.player or not mesh.available or game.dialogue.active or game.pause_menu.active:
        return None
    cells = {(x, z): (y, links) for x, z, y, links in mesh.cells}
    ox, _, oz = mesh.origin
    step = mesh.step

    def point(key):
        return (ox + key[0] * step, cells[key][0], oz + key[1] * step)

    position = game.player.position
    starts = [key for key in cells if abs(cells[key][0] - position[1]) <= 55.0]
    if not starts:
        return None
    start = min(starts, key=lambda key: math.dist(position, point(key)))
    # Never bridge from a distant/lower floor to the nearest convenient cell.
    if math.dist(position, point(start)) > step * 0.8:
        return None
    distance = {start: 0.0}
    previous = {}
    frontier = [(0.0, start)]
    while frontier:
        cost, key = heapq.heappop(frontier)
        if cost != distance[key]:
            continue
        for index, (dx, dz) in enumerate(OFFSETS):
            if not cells[key][1] & (1 << index):
                continue
            neighbor = (key[0] + dx, key[1] + dz)
            if neighbor not in cells or abs(cells[neighbor][0] - cells[key][0]) > 55.0:
                continue
            if dx and dz and ((key[0] + dx, key[1]) not in cells or (key[0], key[1] + dz) not in cells):
                continue
            candidate_cost = cost + math.dist(point(key), point(neighbor))
            if candidate_cost >= distance.get(neighbor, math.inf):
                continue
            distance[neighbor] = candidate_cost
            previous[neighbor] = key
            heapq.heappush(frontier, (candidate_cost, neighbor))
    endpoint = min(distance, key=lambda key: (math.dist(point(key), target), distance[key], key))
    gap = math.dist(point(endpoint), target)
    gain = step * 0.5 if minimum_gain is None else minimum_gain
    if endpoint == start or gap >= math.dist(position, target) - gain:
        return None
    keys = [endpoint]
    while keys[-1] != start:
        keys.append(previous[keys[-1]])
    keys.reverse()
    waypoint = point(keys[1])
    return {
        "waypoint": waypoint,
        "waypoint_id": f"local:{game.scene_epoch}:{tuple(round(v, 1) for v in waypoint)}",
        "path_nodes": len(keys),
        "waypoint_index": 1,
        "target_gap": gap,
        "confidence": 1.0,
        "partial": gap > step * 0.8,
        "local_path": True,
        "path_evidence": "current_collision_links",
        "path_length": distance[endpoint],
    }
