"""Observed attached-mode task; the task verifies physics, not a raw action."""
from __future__ import annotations

import time

from .descent_task import ObservedDescentTask
from .execution import physical_context


class LadderDescentTask(ObservedDescentTask):
    VERSION = "observed-ladder-descent-task-v1"

    @classmethod
    def continue_from(cls, game, parent, *, now=None):
        now = time.monotonic() if now is None else now
        if (not isinstance(parent, ObservedDescentTask) or parent.failure != "unsupported_locomotor_mode"
                or physical_context(game) != parent.context or game.scene_epoch != parent.scene_epoch
                or any(event not in parent.load_events for event in cls._loads(game)) or now >= parent.deadline
                or not game.player or not game.player.climbing_ladder
                or game.player.hanging_ledge or game.player.climbing_ledge
                or game.camera_input_yaw is None):
            raise ValueError("An observed ladder handoff must preserve the live descent context and budget")
        if not any(abs(y - parent.target[1]) <= 4 for _, _, y, _ in game.navmesh.cells):
            raise ValueError("The lower floor must still have current local collision support")
        task = cls._create(game, "ladder_down", parent.target, game.player.position,
                           now=now, budget_s=min(20, parent.deadline - now))
        task.version = cls.VERSION
        return task

    def accepts_attached_mode(self, game):
        return bool(game.player.climbing_ladder and not game.player.hanging_ledge
                    and not game.player.climbing_ledge)

    @staticmethod
    def grounded(game):
        return bool(ObservedDescentTask.grounded(game) and not game.player.climbing_ladder
                    and not game.player.hanging_ledge and not game.player.climbing_ledge)

    def observe(self, game, *, consumed, now=None):
        super().observe(game, consumed=consumed, now=now)
        if (not self.terminal and not game.player.climbing_ladder
                and abs(game.player.position[1] - self.target[1]) > 4):
            self.interrupt("attachment_left_before_landing")

    def reference_stick(self, game):
        # A verifier alone never supplies a walking projection in this mode.
        return (0, 0)
