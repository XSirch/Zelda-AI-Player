"""Bounded causal return from observed first-person control; no button mapping."""
from __future__ import annotations

import time
from dataclasses import dataclass

from .context_tasks import ObservedContextTask
from .execution import physical_context
from .interaction_memory import context_key
from .locomotion import camera_modal_active, grounded


def eligible_camera_return(game):
    return bool(game.source == "soh" and game.in_game and game.player and game.player.health > 0
        and grounded(game.player) and camera_modal_active(game.player) and not game.game_over_state
        and not game.paused and not game.pause_menu.active and not game.dialogue.active
        and not game.cutscene_active and game.context_action.label.lower() == "return"
        and context_key(game) is not None)


@dataclass
class ObservedCameraReturnTask(ObservedContextTask):
    version: str = "observed-camera-return-v1"

    @classmethod
    def create(cls, game, *, now=None, budget_s=20):
        if not eligible_camera_return(game):
            raise ValueError("Camera return needs a current native first-person flag and Return prompt")
        task = cls._create(game, "observed_camera_return", game.player.position,
            game.player.position, now=now, budget_s=budget_s, allow_camera_modal=True)
        task.context_action_key, task.phase = context_key(game), "prepare"
        task.prior_events = tuple(e.id for e in game.events)
        return task

    def accepts_camera_modal(self, game):
        return True

    def allows_context_buttons(self, game):
        return (self.phase == "execute" and eligible_camera_return(game)
                and context_key(game) == self.context_action_key)

    def guidance(self, game):
        result = super().guidance(game)
        result["camera_return_active"] = self.allows_context_buttons(game)
        return result

    def observe(self, game, *, consumed, now=None):
        if self.terminal:
            return
        now = time.monotonic() if now is None else now
        if (physical_context(game) != self.context or game.scene_epoch != self.scene_epoch
                or game.seq < self.origin_seq or any(e not in self.load_events for e in self._loads(game))):
            self.interrupt("camera_return_context_changed")
            return
        if not game.player or not game.in_game or game.player.health <= 0 or game.game_over_state:
            self.interrupt("game_not_playable")
            return
        if game.dialogue.active or game.pause_menu.active or game.paused or game.cutscene_active:
            self.interrupt("modal_owns_control")
            return
        if not grounded(game.player):
            self.interrupt("unsupported_camera_return_mode")
            return
        if game.seq <= self.last_seq or game.seq <= self.origin_seq:
            return
        self.last_seq, self.consumed = game.seq, self.consumed or consumed
        if not camera_modal_active(game.player):
            if not self.consumed:
                self.interrupt("camera_exit_without_button_consumption")
                return
            self.observed_effect, self.effect_seq = "first_person_closed", self.effect_seq or game.seq
            self.phase = "verify"
            self.verification_frames = self.verification_frames + 1 if game.player.speed_xz < .1 else 0
            if self.verification_frames >= 3:
                self.phase = "succeeded"
            elif now >= self.deadline:
                self.phase, self.failure = "failed", "camera_return_not_verified"
            return
        self.verification_frames = 0
        if now >= self.deadline:
            self.phase, self.failure = "failed", "camera_return_no_observed_effect"
        elif context_key(game) != self.context_action_key:
            self.interrupt("camera_return_prompt_lost")
        elif self.phase == "prepare":
            self.stopped_frames = self.stopped_frames + 1 if game.player.speed_xz < .1 else 0
            if self.stopped_frames >= 3:
                self.phase = "execute"
        else:
            self.phase = "execute"
