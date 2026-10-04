"""Deduplicate native load events without inferring resets from camera motion."""
from __future__ import annotations


class ObservationLifetime:
    def __init__(self):
        self.instance_id = None
        self.load_id = None

    def observe(self, game) -> bool:
        restarted = self.instance_id is not None and self.instance_id != game.instance_id
        if restarted:
            self.load_id = None
        self.instance_id = game.instance_id
        marker = next((event.id for event in reversed(game.events) if event.kind == "save_loaded"), None)
        loaded = marker is not None and marker != self.load_id
        if loaded:
            self.load_id = marker
        return restarted or loaded
