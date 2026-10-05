"""Stationary contextual probes and linear text, never semantic button macros."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .execution import physical_context
from .interaction_memory import context_key
from .local_tasks import LocalTask
from .locomotion import grounded, water_active


def eligible_context(game):
    return bool(game.source == "soh" and game.in_game and game.player and game.player.health > 0
        and grounded(game.player) and not game.game_over_state and not game.paused
        and not game.dialogue.active and not game.pause_menu.active and not game.cutscene_active
        and game.context_action.label.lower() in {"check", "speak", "open", "enter"}
        and context_key(game) is not None)


@dataclass
class ObservedContextTask(LocalTask):
    context_action_key: str = ""
    button_commands: dict = field(default_factory=dict)
    stopped_frames: int = 0
    observed_effect: str | None = None
    effect_seq: int | None = None
    version: str = "observed-context-control-task-v1"

    @classmethod
    def create(cls, game, *, now=None, budget_s=20):
        if not eligible_context(game):
            raise ValueError("No currently actionable supported ground context")
        task = cls._create(game, "observed_context_interaction", game.player.position,
            game.player.position, now=now, budget_s=budget_s)
        task.context_action_key, task.phase = context_key(game), "prepare"
        return task

    def register_button(self, seq, mask):
        # Physical wiring validation only; it assigns no meaning to a button.
        if seq > 0 and mask and mask & (mask - 1) == 0 and not mask & ~0xE01F:
            if len(self.button_commands) < 256:
                self.button_commands[int(seq)] = int(mask)

    def button_consumed(self, bridge):
        return any(bridge.command_consumed(seq) and seq in bridge.receipts
            and bridge.receipts[seq].pressed == mask for seq, mask in self.button_commands.items())

    def allows_context_buttons(self, game):
        return self.phase == "execute" and context_key(game) == self.context_action_key

    def reference_stick(self, game):
        return (0, 0)

    def guidance(self, game):
        return {"active": not self.terminal, "source": "observed_context_task", "target": self.origin,
            "stick": (0, 0), "strength": 1., "button_quiet": 1., "blocked": False,
            "context_interaction_active": self.allows_context_buttons(game), "local_task": self.identity}

    def observe(self, game, *, consumed, now=None):
        if self.terminal:
            return
        now = time.monotonic() if now is None else now
        if (game.instance_id != self.context[0] or physical_context(game)[3:] != self.context[3:]
                or not game.player or game.player.health <= 0
                or game.game_over_state or any(e not in self.load_events for e in self._loads(game))):
            self.interrupt("context_episode_changed")
            return
        if game.seq <= self.last_seq or game.seq <= self.origin_seq:
            return
        self.last_seq = game.seq
        self.consumed = self.consumed or consumed
        if game.pause_menu.active or game.paused:
            self.interrupt("modal_owns_control")
            return
        effect = ("dialogue_started" if game.dialogue.active else
                  "observed_scene_room_transition" if physical_context(game)[1:3] != self.context[1:3] else None)
        if effect and self.consumed:
            if effect != self.observed_effect:
                self.verification_frames = 0
            self.observed_effect, self.effect_seq = effect, self.effect_seq or game.seq
            self.phase = "verify"
            self.verification_frames += 1
            if self.verification_frames >= 3:
                self.phase = "succeeded"
            return
        if now >= self.deadline:
            self.phase, self.failure = "failed", "context_no_observed_effect"
        elif game.cutscene_active or water_active(game.player):
            self.interrupt("unsupported_context_mode")
        elif context_key(game) != self.context_action_key:
            self.interrupt("context_target_lost")
        elif self.phase == "prepare":
            self.stopped_frames = self.stopped_frames + 1 if grounded(game.player) and game.player.speed_xz < .1 else 0
            if self.stopped_frames >= 3:
                self.phase = "execute"
        elif math.dist(game.player.position, self.origin) > 35:
            self.interrupt("context_anchor_moved")


@dataclass
class LinearDialogueTask(ObservedContextTask):
    observed_text_changes_after_consumption: int = 0
    last_text_signature: tuple | None = None
    version: str = "observed-linear-dialogue-task-v1"

    @classmethod
    def create(cls, game, *, now=None, budget_s=20):
        if (game.source != "soh" or not game.in_game or not game.player or game.player.health <= 0
                or game.game_over_state or not game.dialogue.active or game.pause_menu.active
                or game.paused or not 0 < budget_s <= 30):
            raise ValueError("Linear dialogue needs a playable current text box")
        now = time.monotonic() if now is None else now
        task = cls(f"linear_dialogue:{game.seq}:{game.dialogue.text_id}", "observed_linear_dialogue",
            physical_context(game), game.scene_epoch, game.seq, cls._loads(game), tuple(game.player.position),
            tuple(game.player.position), tuple(game.player.position), now, now+budget_s, now, 0., phase="execute")
        task.last_text_signature = (game.dialogue.text_id, game.dialogue.text, game.dialogue.choice_count)
        return task

    def allows_context_buttons(self, game):
        return False  # Active dialogue has its own higher-priority causal discovery.

    def observe(self, game, *, consumed, now=None):
        if self.terminal:
            return
        now = time.monotonic() if now is None else now
        if (physical_context(game) != self.context or game.scene_epoch != self.scene_epoch
                or not game.player or game.player.health <= 0 or game.game_over_state
                or any(e not in self.load_events for e in self._loads(game))):
            self.interrupt("context_episode_changed")
            return
        if game.seq <= self.last_seq or game.seq <= self.origin_seq:
            return
        self.last_seq = game.seq
        self.consumed = self.consumed or consumed
        if game.dialogue.active and game.dialogue.choice_count:
            self.interrupt("semantic_choice_requires_cognition")
            return
        if game.pause_menu.active or game.paused:
            self.interrupt("modal_owns_control")
            return
        signature = (game.dialogue.text_id, game.dialogue.text, game.dialogue.choice_count)
        if game.dialogue.active and signature != self.last_text_signature and self.consumed:
            self.observed_text_changes_after_consumption += 1
            self.last_text_signature, self.progress_at = signature, now
        if not game.dialogue.active and self.consumed:
            self.observed_effect, self.phase = "dialogue_closed", "verify"
            self.verification_frames += 1
            if self.verification_frames >= 3:
                self.phase = "succeeded"
        elif now >= self.deadline:
            self.phase, self.failure = "failed", "dialogue_no_verified_close"
