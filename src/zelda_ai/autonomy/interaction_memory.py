"""Observed physical-button bindings, with read-only evaluation working memory."""
from __future__ import annotations


def context_key(game):
    action = game.context_action
    label = (action.label or "").strip().lower()
    if label == "none" or action.code == 0:
        return None
    actor = game.context_actor
    category = (actor.category_name or "").strip().lower() if actor else "none"
    actor_id = actor.actor_id if actor else -1
    return f"{action.code}:{label}:{category}:{actor_id}"[:200]


class ObservedInteractionMemory:
    """Training persists through the graph; evaluation never changes its bytes."""
    def __init__(self, graph):
        self.graph, self.working = graph, {}

    def interaction_button(self, key):
        if key in self.working:
            return self.working[key]["button"]
        return self.graph.interaction_button(key)

    def _put(self, key, row):
        if key not in self.working and len(self.working) >= 512:
            del self.working[next(iter(self.working))]
        self.working[key] = row

    def record_interaction_success(self, key, button):
        if self.graph.writable:
            self.graph.record_interaction_success(key, button)
            return
        old = self.working.get(key, {})
        count = old.get("successes", 0) if old.get("button") == button else 0
        self._put(key, {"button": button, "successes": count + 1, "failures": 0})

    def record_interaction_failure(self, key, button):
        if self.graph.writable:
            self.graph.record_interaction_failure(key, button)
            return
        if self.interaction_button(key) != button:
            return
        old = self.working.get(key, {})
        failures = old.get("failures", 0) + 1
        self._put(key, {"button": None if failures >= 3 else button,
                        "successes": old.get("successes", 0), "failures": failures})

    def snapshot(self):
        return {"persistence": "training_graph" if self.graph.writable else "ephemeral_observed_control_attempts",
                "working": {key: dict(row) for key, row in self.working.items()}}
