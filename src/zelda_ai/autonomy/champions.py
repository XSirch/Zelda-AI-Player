from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import time
import uuid
from pathlib import Path

_CHAMPION_ID = re.compile(r"^completion-(\d{4,})$")


def _atomic_copy(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        destination.name + ".tmp-" + uuid.uuid4().hex
    )
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _atomic_json(destination: Path, value: dict):
    temporary = destination.with_name(
        destination.name + ".tmp-" + uuid.uuid4().hex
    )
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ChampionStore:
    """Immutable completion snapshots plus small atomic metadata files."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _rows(self) -> list[dict]:
        rows = []
        for path in self.root.glob("completion-*.json"):
            if path.name.endswith(".routes.json"):
                continue
            match = _CHAMPION_ID.match(path.stem)
            if not match:
                continue
            try:
                row = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            checkpoint = self.root / f"{path.stem}.pt"
            if (
                row.get("id") == path.stem
                and checkpoint.is_file()
                and isinstance(row.get("created_at"), (int, float))
            ):
                rows.append(row)
        rows.sort(key=lambda row: (float(row["created_at"]), row["id"]))
        return rows

    def list(self) -> list[dict]:
        return [dict(row) for row in self._rows()]

    def latest(self) -> dict | None:
        rows = self._rows()
        return dict(rows[-1]) if rows else None

    def best_completion(self) -> dict | None:
        rows = self._rows()
        if not rows:
            return None
        # "Best" is narrowly defined as the shortest observed completed run.
        # Latest remains the default evaluation candidate because it generally
        # contains the most recent online training.
        row = min(
            rows,
            key=lambda item: (
                float(item.get("elapsed_s") or float("inf")),
                -float(item["created_at"]),
            ),
        )
        return dict(row)

    def catalog(self) -> dict:
        rows = self._rows()
        latest = dict(rows[-1]) if rows else None
        best = self.best_completion()
        return {
            "count": len(rows),
            "latest": latest,
            "best_completion": best,
        }

    def resolve(self, champion_id: str | None = None) -> tuple[dict, Path]:
        if champion_id in {None, "", "latest"}:
            row = self.latest()
        elif champion_id == "best":
            row = self.best_completion()
        elif _CHAMPION_ID.match(champion_id):
            row = next(
                (item for item in self._rows() if item["id"] == champion_id),
                None,
            )
        else:
            raise ValueError("Invalid champion id")

        if row is None:
            raise ValueError(
                "No completed champion is available. Complete the game in training mode first."
            )
        checkpoint = self.root / f"{row['id']}.pt"
        if not checkpoint.is_file():
            raise ValueError("Champion checkpoint is missing")
        expected_hash = row.get("sha256")
        if not isinstance(expected_hash, str) or _sha256(checkpoint) != expected_hash:
            raise ValueError("Champion checkpoint checksum mismatch")
        if row.get("route_graph_file"):
            self.resolve_route_graph(row)
        if row.get("room_map_file"):
            self.resolve_room_map(row)
        return dict(row), checkpoint

    def resolve_route_graph(self, champion: dict) -> Path | None:
        filename = champion.get("route_graph_file")
        if not filename:
            return None
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Champion route graph filename is invalid")
        route_graph = self.root / filename
        if not route_graph.is_file():
            raise ValueError("Champion route graph is missing")
        expected_hash = champion.get("route_graph_sha256")
        if (
            not isinstance(expected_hash, str)
            or _sha256(route_graph) != expected_hash
        ):
            raise ValueError("Champion route graph checksum mismatch")
        return route_graph

    def resolve_room_map(self, champion: dict) -> Path | None:
        filename = champion.get("room_map_file")
        if not filename:
            return None
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("Champion room map filename is invalid")
        room_map = self.root / filename
        if not room_map.is_file():
            raise ValueError("Champion room map is missing")
        expected_hash = champion.get("room_map_sha256")
        if (
            not isinstance(expected_hash, str)
            or _sha256(room_map) != expected_hash
        ):
            raise ValueError("Champion room map checksum mismatch")
        return room_map

    def capture(
        self,
        source_checkpoint: Path,
        metadata: dict,
        route_graph_source: Path | None = None,
        room_map_source: Path | None = None,
    ) -> dict:
        source_checkpoint = Path(source_checkpoint)
        if not source_checkpoint.is_file():
            raise ValueError("Training checkpoint is missing; champion was not created")

        existing_ids = set()
        for pattern in (
            "completion-*.pt",
            "completion-*.json",
            "completion-*.routes.json",
            "completion-*.room-map.json",
        ):
            for path in self.root.glob(pattern):
                if path.name.endswith(".routes.json"):
                    candidate = path.name[: -len(".routes.json")]
                elif path.name.endswith(".room-map.json"):
                    candidate = path.name[: -len(".room-map.json")]
                else:
                    candidate = path.stem
                if match := _CHAMPION_ID.match(candidate):
                    existing_ids.add(int(match.group(1)))
        sequence = max(existing_ids, default=0) + 1
        champion_id = f"completion-{sequence:04d}"
        checkpoint = self.root / f"{champion_id}.pt"
        metadata_path = self.root / f"{champion_id}.json"
        route_graph_path = self.root / f"{champion_id}.routes.json"
        room_map_path = self.root / f"{champion_id}.room-map.json"

        _atomic_copy(source_checkpoint, checkpoint)
        route_metadata = {}
        if route_graph_source is not None:
            source_routes = Path(route_graph_source)
            if source_routes.is_file():
                _atomic_copy(source_routes, route_graph_path)
                route_metadata = {
                    "route_graph_file": route_graph_path.name,
                    "route_graph_sha256": _sha256(route_graph_path),
                }
        room_map_metadata = {}
        if room_map_source is not None:
            source_room_map = Path(room_map_source)
            if source_room_map.is_file():
                _atomic_copy(source_room_map, room_map_path)
                room_map_metadata = {
                    "room_map_file": room_map_path.name,
                    "room_map_sha256": _sha256(room_map_path),
                }

        row = {
            **metadata,
            "id": champion_id,
            "kind": "game_completed",
            "created_at": time.time(),
            "checkpoint_file": checkpoint.name,
            "sha256": _sha256(checkpoint),
            **route_metadata,
            **room_map_metadata,
        }
        _atomic_json(metadata_path, row)

        best = self.best_completion()
        if best is not None:
            best_source = self.root / f"{best['id']}.pt"
            try:
                if _sha256(best_source) == best.get("sha256"):
                    _atomic_copy(best_source, self.root / "best-completion.pt")
                    best_route_metadata = {}
                    best_route_alias = self.root / "best-completion.routes.json"
                    best_route = self.resolve_route_graph(best)
                    if best_route is not None:
                        _atomic_copy(best_route, best_route_alias)
                        best_route_metadata = {
                            "route_graph_file": best_route_alias.name,
                            "route_graph_sha256": _sha256(best_route_alias),
                        }
                    else:
                        best_route_alias.unlink(missing_ok=True)

                    best_room_map_metadata = {}
                    best_room_map_alias = self.root / "best-completion.room-map.json"
                    best_room_map = self.resolve_room_map(best)
                    if best_room_map is not None:
                        _atomic_copy(best_room_map, best_room_map_alias)
                        best_room_map_metadata = {
                            "room_map_file": best_room_map_alias.name,
                            "room_map_sha256": _sha256(best_room_map_alias),
                        }
                    else:
                        best_room_map_alias.unlink(missing_ok=True)
                    _atomic_json(
                        self.root / "best-completion.json",
                        {
                            "champion_id": best["id"],
                            "elapsed_s": best.get("elapsed_s"),
                            "created_at": best.get("created_at"),
                            "sha256": best.get("sha256"),
                            **best_route_metadata,
                            **best_room_map_metadata,
                        },
                    )
            except (OSError, ValueError):
                # The individual completion is already durable. The best alias
                # is a convenience and must never turn a saved champion into a
                # failed capture.
                pass
        return dict(row)
