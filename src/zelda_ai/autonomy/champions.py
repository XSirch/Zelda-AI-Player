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
        return dict(row), checkpoint

    def capture(self, source_checkpoint: Path, metadata: dict) -> dict:
        source_checkpoint = Path(source_checkpoint)
        if not source_checkpoint.is_file():
            raise ValueError("Training checkpoint is missing; champion was not created")

        rows = self._rows()
        sequence = max(
            (
                int(match.group(1))
                for row in rows
                if (match := _CHAMPION_ID.match(str(row.get("id", ""))))
            ),
            default=0,
        ) + 1
        champion_id = f"completion-{sequence:04d}"
        checkpoint = self.root / f"{champion_id}.pt"
        metadata_path = self.root / f"{champion_id}.json"

        _atomic_copy(source_checkpoint, checkpoint)
        row = {
            **metadata,
            "id": champion_id,
            "kind": "game_completed",
            "created_at": time.time(),
            "checkpoint_file": checkpoint.name,
            "sha256": _sha256(checkpoint),
        }
        _atomic_json(metadata_path, row)

        best = self.best_completion()
        if best is not None:
            best_source = self.root / f"{best['id']}.pt"
            _atomic_copy(best_source, self.root / "best-completion.pt")
            _atomic_json(
                self.root / "best-completion.json",
                {
                    "champion_id": best["id"],
                    "elapsed_s": best.get("elapsed_s"),
                    "created_at": best.get("created_at"),
                    "sha256": best.get("sha256"),
                },
            )
        return dict(row)
