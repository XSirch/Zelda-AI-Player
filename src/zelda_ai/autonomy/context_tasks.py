"""Stationary contextual probes and linear text, never semantic button macros."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from .execution import physical_context
from .interaction_memory import context_key
from .local_tasks import LocalTask
from .locomotion import camera_modal_active, grounded, water_active


def eligible_context(game):
    return bool(game.source == "soh" and game.in_game and game.player and game.player.health > 0
        and grounded(game.player) and not camera_modal_active(game.player) and not game.game_over_state and not game.paused
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
    effect_event_id: str | None = None
    prior_events: tuple = ()
    container_context: bool = False
    context_actor_uid: str | None = None
    version: str = "observed-context-control-task-v2"

    @classmethod
    def create(cls, game, *, now=None, budget_s=20):
        if not eligible_context(game):
            raise ValueError("No currently actionable supported ground context")
        task = cls._create(game, "observed_context_interaction", game.player.position,
            game.player.position, now=now, budget_s=budget_s)
        task.context_action_key, task.phase = context_key(game), "prepare"
        task.prior_events = tuple(e.id for e in game.events)
        task.container_context = bool(game.context_actor
            and game.context_actor.category_name.lower() == "chest"
            and game.context_action.label.lower() == "open")
        task.context_actor_uid = game.context_actor.actor_uid if game.context_actor else None
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
        if camera_modal_active(game.player):
            self.interrupt("camera_modal_owns_control")
            return
        chest = next((e for e in game.events if e.kind == "chest_opened"
            and e.id not in self.prior_events and e.detail.startswith(f"{self.context[1]}:")
            and (e.actor_uid is None or e.actor_uid == self.context_actor_uid)
            and physical_context(game)[1:3] == self.context[1:3]), None)
        if not self.consumed:
            self.prior_events = tuple(dict.fromkeys((*self.prior_events, *(e.id for e in game.events))))[-256:]
        if self.container_context and chest and self.consumed:
            self.effect_event_id = chest.id
        effect = ("chest_opened" if self.effect_event_id else "dialogue_started" if game.dialogue.active else
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
        elif game.cutscene_active and self.container_context and self.consumed:
            # Opening animation can precede the actual native treasure flag.
            # Remain neutral and bounded; animation alone earns no success.
            self.phase = "transition"
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
class NativeModalWaitTask(LocalTask):
    observed_effect: str | None = None
    version: str = "observed-native-modal-wait-v1"

    @classmethod
    def create(cls, game, *, now=None, budget_s=20):
        if (game.source != "soh" or not game.in_game or not game.player or game.player.health <= 0
                or game.game_over_state or not game.cutscene_active or game.dialogue.active
                or game.paused or game.pause_menu.active or not 0 < budget_s <= 30):
            raise ValueError("Modal wait requires an actual current native animation")
        now = time.monotonic() if now is None else now
        point = tuple(game.player.position)
        return cls(f"native_modal_wait:{game.seq}", "observed_native_modal_wait",
            physical_context(game), game.scene_epoch, game.seq, cls._loads(game), point, point, point,
            now, now+budget_s, now, 0., phase="execute")

    def observe(self, game, *, consumed=False, now=None):
        if self.terminal:
            return
        now = time.monotonic() if now is None else now
        if (game.instance_id != self.context[0] or physical_context(game)[3:] != self.context[3:]
                or not game.player or game.player.health <= 0 or game.game_over_state
                or not game.in_game or any(e not in self.load_events for e in self._loads(game))):
            self.interrupt("context_episode_changed")
            return
        if game.paused or game.pause_menu.active:
            self.interrupt("modal_owns_control")
            return
        if game.seq <= self.origin_seq or game.seq <= self.last_seq:
            return
        self.last_seq = game.seq
        effect = "dialogue_handoff" if game.dialogue.active else (
            "playable_handoff" if not game.cutscene_active else None)
        self.verification_frames = self.verification_frames + 1 if effect == self.observed_effect and effect else int(bool(effect))
        self.observed_effect = effect
        if self.verification_frames >= 3:
            self.phase = "succeeded"
        elif now >= self.deadline:
            self.phase, self.failure = "failed", "native_modal_wait_timeout"


@dataclass
class LinearDialogueTask(ObservedContextTask):
    observed_text_changes_after_consumption: int = 0
    last_text_signature: tuple | None = None
    read_speaker_contexts: tuple = ()
    version: str = "observed-linear-dialogue-task-v2"

    def note_visible_page(self, game):
        speaker = game.dialogue.speaker
        if (game.dialogue.active and game.dialogue.text_visible is True and game.dialogue.text
                and game.dialogue.can_advance and not game.dialogue.choice_count
                and speaker and speaker.actor_uid):
            key = (*physical_context(game), speaker.actor_uid)
            if key not in self.read_speaker_contexts and len(self.read_speaker_contexts) < 8:
                self.read_speaker_contexts += (key,)

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
        task.context_actor_uid = game.dialogue.speaker.actor_uid if game.dialogue.speaker else None
        task.note_visible_page(game)
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
        self.note_visible_page(game)
        if (game.dialogue.active and game.dialogue.text_visible is not False
                and signature != self.last_text_signature and self.consumed):
            self.observed_text_changes_after_consumption += 1
            self.last_text_signature, self.progress_at = signature, now
        if not game.dialogue.active and self.consumed:
            self.observed_effect, self.phase = "dialogue_closed", "verify"
            self.verification_frames += 1
            if self.verification_frames >= 3:
                self.phase = "succeeded"
        elif now >= self.deadline:
            self.phase, self.failure = "failed", "dialogue_no_verified_close"
