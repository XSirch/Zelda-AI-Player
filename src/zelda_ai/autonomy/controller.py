from __future__ import annotations

import asyncio
import contextlib
import hashlib
import math
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..bridge import Bridge
from .features import (
    BUTTON_NAMES,
    camera_relative_stick,
    camera_world_yaw,
    encode_novelty_state,
    encode_state,
    goal_guidance,
    stack_frames,
    target_point,
)
from .ml_policy import OnlinePPO
from .models import AgentIntent
from .reward import RewardTracker
from .room_map import RoomMapMemory
from .routes import LearnedRouteGraph

# Physical N64 controller wiring only.  These are not semantic skills and the
# policy is never told what any button does.
BUTTON_MASKS = {
    "A": 0x8000,
    "B": 0x4000,
    "Z": 0x2000,
    "START": 0x1000,
    "R": 0x0010,
    "C_UP": 0x0008,
    "C_LEFT": 0x0002,
    "C_DOWN": 0x0004,
    "C_RIGHT": 0x0001,
}

# Contextual interaction discovery probes one physical button at a time.
# START is excluded because opening the pause menu is not a useful contextual
# interaction probe and can take controller authority away from the current task.
INTERACTION_PROBE_BUTTONS = tuple(
    name for name in BUTTON_NAMES if name != "START"
)
EXIT_PRIORITY_DWELL_S = 20.0
ROOM_FAILURE_EXIT_PRESSURE = 6
NO_PROGRESS_TRAINING_CUTOFF_S = 180.0
ESCAPE_PROGRESS_DELTA = 8.0
ESCAPE_STALL_S = 3.5
ESCAPE_MAX_AGE_S = 45.0
ESCAPE_RETRY_COOLDOWN_S = 20.0
INTERACTION_PROBE_COOLDOWN_S = 0.35
INTERACTION_OUTCOME_WINDOW_S = 1.5
DIALOGUE_REENTRY_GUARD_S = 8.0
DIALOGUE_REENTRY_CLEAR_DISTANCE = 120.0

# A camera mode switch can change what the same physical stick vector means
# between two 10 Hz policy decisions.  Detect that at the 20 Hz motor loop,
# briefly neutralize stale movement, then re-project the same world-space
# heading against the new camera without asking the route planner to replan.
CAMERA_TRANSITION_DELTA_RAD = math.radians(12.0)
CAMERA_CUT_DELTA_RAD = math.radians(35.0)
CAMERA_TRANSITION_SETTLE_TICKS = 1
CAMERA_CUT_SETTLE_TICKS = 2


@dataclass(frozen=True)
class Setpoint:
    buttons: int = 0
    stick_x: int = 0
    stick_y: int = 0
    reason: str = "neutral"


class ContinuousController:
    """20 Hz controller with 10 Hz ML action decisions and asynchronous learning."""

    tick_s = 0.05
    action_repeat_ticks = 2
    route_save_interval_s = 20.0

    def __init__(
        self,
        bridge: Bridge,
        checkpoint: Path,
        *,
        rollout_size: int = 256,
        on_achievement: Callable[[dict], None] | None = None,
        training_enabled: bool = True,
        route_graph_path: Path | None = None,
        room_map_path: Path | None = None,
    ):
        self.bridge = bridge
        self.policy = OnlinePPO(
            checkpoint,
            strict_checkpoint=not training_enabled,
        )
        self.starting_updates = self.policy.updates
        self.starting_samples_trained = self.policy.samples_trained
        self.rollout_size = rollout_size
        self.on_achievement = on_achievement
        self.training_enabled = training_enabled
        self.route_graph = LearnedRouteGraph(
            route_graph_path or checkpoint.parent / "route-graph-v1.json",
            writable=training_enabled,
        )
        self.room_map = RoomMapMemory(
            room_map_path or checkpoint.parent / "room-map-v1.json",
            writable=training_enabled,
        )
        self.intent = AgentIntent.bootstrap()
        self.intent_updated_at = time.monotonic()
        self.reward_tracker = RewardTracker()
        self.feature_history = deque(maxlen=4)
        self.novelty_history = deque(maxlen=4)
        self.training_queue: asyncio.Queue = asyncio.Queue(maxsize=2)

        self.last_setpoint = Setpoint()
        self.last_stick = (0.0, 0.0)
        self.last_buttons = tuple(0.0 for _ in BUTTON_NAMES)
        self.pending: dict | None = None
        self.rollout: list[dict] = []
        self.tick = 0
        self.motor_ticks = 0
        self.actions_sampled = 0
        self.total_reward = 0.0
        self.last_reward = 0.0
        self.last_reward_breakdown: dict[str, float] = {}
        self.last_button_probability_mean = 0.0
        self.last_expected_button_count = 0.0
        self.last_guidance_mix = 0.0
        self.reward_window = deque(maxlen=200)
        self.useful_progress_window = deque(maxlen=200)
        self.last_useful_progress_at = time.monotonic()
        self.achievements = deque(maxlen=64)
        self.last_training_stats: dict = {}
        self.route_save_error = ""
        self.room_map_save_error = ""
        self.interaction_probe_index: dict[str, int] = {}
        self.interaction_last_probe_at = 0.0
        self.pending_interaction_probe: dict | None = None
        self.interaction_probe_successes = 0
        self.last_interaction_source = "none"
        self.dialogue_reentry_guard: dict | None = None
        self.dialogue_reentry_suppressed = 0
        self.active_escape_attempt: dict | None = None
        self.escape_retry_after: dict[str, float] = {}
        self.escape_failures = 0
        self.escape_successes = 0
        self.last_camera_yaw: float | None = None
        self.camera_motion_state = "stable"
        self.camera_guard_ticks = 0
        self.camera_cut_count = 0
        self.camera_transition_count = 0
        self.last_guidance = {
            "active": False,
            "stick": (0.0, 0.0),
            "strength": 0.0,
            "button_quiet": 0.0,
            "distance": None,
            "source": "none",
            "target": None,
            "blocked": False,
            "detour": None,
            "direct_probe": None,
            "stuck_scale": 1.0,
            "route_active": False,
            "route_path_nodes": 0,
            "route_confidence": 0.0,
            "route_target_gap": None,
            "route_waypoint": None,
            "route_waypoint_id": None,
            "route_partial": False,
            "route_edge_key": None,
            "route_edge_failures": 0,
            "frontier_active": False,
            "frontier_direction": None,
            "frontier_stable": False,
            "frontier_age_s": None,
            "exit_active": False,
            "exit_index": None,
            "exit_direct_reachable": False,
        }
        self.last_motor_summary = "ML policy is ready to explore raw controller inputs."

    def reset_episode_state(self):
        self.reward_tracker = RewardTracker()
        self.route_graph.reset_trace()
        self.room_map.reset_trace()
        self.feature_history.clear()
        self.novelty_history.clear()
        self.pending = None
        self.rollout = []
        self.tick = 0
        self.last_setpoint = Setpoint()
        self.last_stick = (0.0, 0.0)
        self.last_buttons = tuple(0.0 for _ in BUTTON_NAMES)
        self.interaction_probe_index.clear()
        self.interaction_last_probe_at = 0.0
        self.pending_interaction_probe = None
        self.last_interaction_source = "none"
        self.dialogue_reentry_guard = None
        self.dialogue_reentry_suppressed = 0
        self.active_escape_attempt = None
        self.escape_retry_after.clear()
        self.escape_failures = 0
        self.escape_successes = 0
        self.last_camera_yaw = None
        self.camera_motion_state = "stable"
        self.camera_guard_ticks = 0
        self.camera_cut_count = 0
        self.camera_transition_count = 0
        self.last_guidance = {
            "active": False,
            "stick": (0.0, 0.0),
            "strength": 0.0,
            "button_quiet": 0.0,
            "distance": None,
            "source": "none",
            "target": None,
            "blocked": False,
            "detour": None,
            "direct_probe": None,
            "stuck_scale": 1.0,
            "route_active": False,
            "route_path_nodes": 0,
            "route_confidence": 0.0,
            "route_target_gap": None,
            "route_waypoint": None,
            "route_waypoint_id": None,
            "route_partial": False,
            "route_edge_key": None,
            "route_edge_failures": 0,
            "frontier_active": False,
            "frontier_direction": None,
            "frontier_stable": False,
            "frontier_age_s": None,
            "exit_active": False,
            "exit_index": None,
            "exit_direct_reachable": False,
        }
        self.last_reward = 0.0
        self.last_reward_breakdown = {}
        self.last_button_probability_mean = 0.0
        self.last_expected_button_count = 0.0
        self.last_guidance_mix = 0.0
        self.reward_window.clear()
        self.useful_progress_window.clear()
        self.last_useful_progress_at = time.monotonic()
        self.achievements.clear()

    def set_intent(self, intent: AgentIntent):
        self.intent = intent
        self.intent_updated_at = time.monotonic()

    @staticmethod
    def _angle_delta(current: float, previous: float) -> float:
        return (current - previous + math.pi) % (2.0 * math.pi) - math.pi

    def _reset_camera_guard(self):
        self.last_camera_yaw = None
        self.camera_motion_state = "stable"
        self.camera_guard_ticks = 0

    def _camera_state(self, game) -> str:
        current_yaw = camera_world_yaw(game, game.player)
        if current_yaw is None:
            self._reset_camera_guard()
            return self.camera_motion_state

        previous_yaw = self.last_camera_yaw
        self.last_camera_yaw = current_yaw
        if previous_yaw is None:
            self.camera_motion_state = "stable"
            return self.camera_motion_state

        delta = abs(self._angle_delta(current_yaw, previous_yaw))
        if delta >= CAMERA_CUT_DELTA_RAD:
            self.camera_motion_state = "cut"
            self.camera_guard_ticks = CAMERA_CUT_SETTLE_TICKS
            self.camera_cut_count += 1
            return self.camera_motion_state

        if delta >= CAMERA_TRANSITION_DELTA_RAD:
            self.camera_motion_state = "transition"
            self.camera_guard_ticks = max(
                self.camera_guard_ticks,
                CAMERA_TRANSITION_SETTLE_TICKS,
            )
            self.camera_transition_count += 1
            return self.camera_motion_state

        if self.camera_guard_ticks > 0:
            self.camera_guard_ticks -= 1
            self.camera_motion_state = "transition"
            return self.camera_motion_state

        self.camera_motion_state = "stable"
        return self.camera_motion_state

    @staticmethod
    def _mixed_stick(
        policy_stick,
        guidance_stick,
        guidance_strength: float,
    ) -> tuple[int, int]:
        mix = max(0.0, min(1.0, float(guidance_strength or 0.0)))
        px, py = (float(policy_stick[0]), float(policy_stick[1]))
        gx, gy = (float(guidance_stick[0]), float(guidance_stick[1]))
        executed_x = max(-1.0, min(1.0, px * (1.0 - mix) + gx * mix))
        executed_y = max(-1.0, min(1.0, py * (1.0 - mix) + gy * mix))
        return (
            max(-80, min(80, round(executed_x * 80.0))),
            max(-80, min(80, round(executed_y * 80.0))),
        )

    def _refresh_camera_relative_setpoint(self, game):
        """Refresh only camera projection; never choose/replan a waypoint here."""
        camera_state = self._camera_state(game)

        if camera_state != "stable":
            # Camera cuts only invalidate camera-relative navigation movement.
            # Never steal authority from dialogue/interaction/modal overrides.
            if self.last_setpoint.reason != "ml_policy":
                return
            if self.pending is not None:
                # This action was partially replaced by the camera safety guard;
                # keeping it out of PPO preserves latent-action/log-prob causality.
                self.pending["trainable"] = False
            self.last_setpoint = Setpoint(
                buttons=self.last_setpoint.buttons,
                stick_x=0,
                stick_y=0,
                reason=f"camera_{camera_state}",
            )
            self.last_stick = (0.0, 0.0)
            self.last_motor_summary = (
                f"Camera {camera_state}; stale movement neutralized while "
                "preserving the current world-space waypoint."
            )
            return

        if (
            self.last_setpoint.reason
            not in {"ml_policy", "camera_cut", "camera_transition"}
            or self.pending is None
            or not self.last_guidance.get("active")
        ):
            return

        world_yaw = self.last_guidance.get("world_yaw")
        policy_stick = self.pending.get("policy_stick")
        if not isinstance(world_yaw, (int, float)) or not (
            isinstance(policy_stick, (list, tuple)) and len(policy_stick) == 2
        ):
            return

        guidance_stick = camera_relative_stick(
            game,
            game.player,
            float(world_yaw),
        )
        guidance_strength = float(
            self.pending.get(
                "guidance_strength",
                self.last_guidance.get("strength") or 0.0,
            )
            or 0.0
        )
        stick_x, stick_y = self._mixed_stick(
            policy_stick,
            guidance_stick,
            guidance_strength,
        )
        self.last_setpoint = Setpoint(
            buttons=self.last_setpoint.buttons,
            stick_x=stick_x,
            stick_y=stick_y,
            reason="ml_policy",
        )
        executed = (stick_x / 80.0, stick_y / 80.0)
        self.last_stick = executed
        self.last_guidance["stick"] = guidance_stick
        self.last_guidance["camera_yaw"] = camera_world_yaw(game, game.player)
        self.pending["stick"] = list(executed)
        self.pending["guidance_stick"] = list(guidance_stick)

    def neutralize(self, reason: str = "stopped"):
        self._reset_camera_guard()
        self.last_setpoint = Setpoint(reason=reason)
        self.last_stick = (0.0, 0.0)
        self.last_buttons = tuple(0.0 for _ in BUTTON_NAMES)
        self.last_guidance = {
            "active": False,
            "stick": (0.0, 0.0),
            "strength": 0.0,
            "button_quiet": 0.0,
            "distance": None,
            "source": "none",
            "target": None,
            "blocked": False,
            "detour": None,
            "direct_probe": None,
            "stuck_scale": 1.0,
            "route_active": False,
            "route_path_nodes": 0,
            "route_confidence": 0.0,
            "route_target_gap": None,
            "route_waypoint": None,
            "route_waypoint_id": None,
            "route_partial": False,
            "route_edge_key": None,
            "route_edge_failures": 0,
            "frontier_active": False,
            "frontier_direction": None,
            "frontier_stable": False,
            "frontier_age_s": None,
            "exit_active": False,
            "exit_index": None,
            "exit_direct_reachable": False,
        }
        self.last_motor_summary = "Controller input revoked; no buttons are being held."

    @staticmethod
    def _mask(buttons: list[float]) -> int:
        mask = 0
        for name, active in zip(BUTTON_NAMES, buttons):
            if active > 0.5:
                mask |= BUTTON_MASKS[name]
        return mask

    @staticmethod
    def _interaction_key(game) -> str | None:
        action = game.context_action
        label = (action.label or "").strip()
        if label == "none" or action.code == 0:
            return None
        actor = game.context_actor
        category = (
            (actor.category_name or "").strip().lower()
            if actor is not None
            else "none"
        )
        actor_id = actor.actor_id if actor is not None else -1
        return f"{action.code}:{label.lower()}:{category}:{actor_id}"[:200]

    @staticmethod
    def _interaction_probe_order(key: str) -> list[str]:
        return sorted(
            INTERACTION_PROBE_BUTTONS,
            key=lambda name: hashlib.sha256(
                f"{key}|{name}".encode("utf-8")
            ).digest(),
        )

    def note_dialogue_closed(self, old_game, game):
        """Release the closing button and prevent immediate dialogue re-entry."""
        if not old_game.dialogue.active or game.dialogue.active:
            return
        now = time.monotonic()
        position = (
            tuple(float(v) for v in game.player.position)
            if game.player is not None
            else tuple(float(v) for v in old_game.player.position)
            if old_game.player is not None
            else None
        )
        speaker = old_game.dialogue.speaker
        closing_button = None
        if (
            self.pending_interaction_probe is not None
            and self.pending_interaction_probe.get("kind") == "dialogue"
        ):
            closing_button = self.pending_interaction_probe.get("button")
            if closing_button:
                self.route_graph.record_interaction_success(
                    "dialogue:advance",
                    str(closing_button),
                )
                self.interaction_probe_successes += 1
        if closing_button is None:
            closing_button = self.route_graph.interaction_button(
                "dialogue:advance"
            )

        self.dialogue_reentry_guard = {
            "scene": int(game.scene),
            "room": int(game.room),
            "anchor": position,
            "speaker_uid": speaker.actor_uid if speaker is not None else None,
            "speaker_id": speaker.actor_id if speaker is not None else None,
            "text_id": old_game.dialogue.text_id,
            "closing_button": closing_button,
            "started_at": now,
            "until": now + DIALOGUE_REENTRY_GUARD_S,
        }
        self.pending_interaction_probe = None
        self.interaction_last_probe_at = now

        # Dialogue may close between repeated-action ticks. Explicitly release
        # the closing button now so the old lease cannot reopen the same text
        # before the next ML decision.
        self.last_setpoint = Setpoint(
            buttons=0,
            stick_x=self.last_setpoint.stick_x,
            stick_y=self.last_setpoint.stick_y,
            reason="dialogue_disengage",
        )
        self.last_buttons = tuple(0.0 for _ in BUTTON_NAMES)
        self.last_interaction_source = "dialogue_disengage"

    def _dialogue_reentry_override(
        self,
        game,
        setpoint: Setpoint,
        sample: dict,
    ) -> tuple[Setpoint, dict, bool]:
        guard = self.dialogue_reentry_guard
        if guard is None or game.dialogue.active:
            return setpoint, sample, False

        now = time.monotonic()
        if (
            int(game.scene) != int(guard["scene"])
            or int(game.room) != int(guard["room"])
            or now >= float(guard["until"])
        ):
            self.dialogue_reentry_guard = None
            return setpoint, sample, False

        anchor = guard.get("anchor")
        if anchor is not None and game.player is not None:
            if math.dist(game.player.position, anchor) >= DIALOGUE_REENTRY_CLEAR_DISTANCE:
                self.dialogue_reentry_guard = None
                return setpoint, sample, False

        # Preserve stick authority so route/frontier/exit logic can move Link
        # away, but suppress all buttons while still in the re-entry zone. This
        # avoids assuming which semantic button started the conversation.
        neutral_buttons = [0.0 for _ in BUTTON_NAMES]
        self.dialogue_reentry_suppressed += 1
        self.last_interaction_source = "dialogue_disengage"
        return (
            Setpoint(
                buttons=0,
                stick_x=setpoint.stick_x,
                stick_y=setpoint.stick_y,
                reason="dialogue_disengage",
            ),
            {
                **sample,
                "buttons": neutral_buttons,
            },
            True,
        )

    @staticmethod
    def _dialogue_interaction_key(game) -> str:
        # The mapping is learned empirically; the key names the observed
        # affordance class, not a semantic button assumption.
        return "dialogue:advance"

    def _dialogue_override(
        self,
        game,
        setpoint: Setpoint,
        sample: dict,
    ) -> tuple[Setpoint, dict, bool]:
        if not game.dialogue.active:
            return setpoint, sample, False

        # Active text owns the controller. Exploration/navigation must never
        # continue moving Link underneath a visible message box.
        neutral_buttons = [0.0 for _ in BUTTON_NAMES]
        neutral_sample = {
            **sample,
            "stick": [0.0, 0.0],
            "buttons": neutral_buttons,
        }

        # Choice selection remains cognition-owned. Do not let stochastic PPO
        # accidentally confirm a choice while cognition is deciding.
        if game.dialogue.choice_count > 0:
            desired = (
                self.intent.choice_index
                if self.intent.mode == "dialogue"
                else None
            )
            if desired is None or desired != game.dialogue.choice_index:
                self.last_interaction_source = "dialogue_choice_wait"
                return (
                    Setpoint(reason="dialogue_choice_wait"),
                    neutral_sample,
                    True,
                )

        if not game.dialogue.can_advance:
            self.last_interaction_source = "dialogue_wait"
            return (
                Setpoint(reason="dialogue_wait"),
                neutral_sample,
                True,
            )

        now = time.monotonic()
        if self.pending_interaction_probe is not None:
            return (
                Setpoint(reason="dialogue_probe_wait"),
                neutral_sample,
                True,
            )
        if now - self.interaction_last_probe_at < INTERACTION_PROBE_COOLDOWN_S:
            return (
                Setpoint(reason="dialogue_probe_wait"),
                neutral_sample,
                True,
            )

        key = self._dialogue_interaction_key(game)
        known_button = self.route_graph.interaction_button(key)
        using_known = known_button in BUTTON_MASKS
        if using_known:
            button = known_button
            source = "dialogue_memory"
        else:
            order = self._interaction_probe_order(key)
            index = self.interaction_probe_index.get(key, 0)
            button = order[index % len(order)]
            self.interaction_probe_index[key] = index + 1
            source = "dialogue_probe"

        executed_buttons = [
            1.0 if name == button else 0.0
            for name in BUTTON_NAMES
        ]
        self.interaction_last_probe_at = now
        self.pending_interaction_probe = {
            "kind": "dialogue",
            "key": key,
            "button": button,
            "at": now,
            "scene": game.scene,
            "room": game.room,
            "dialogue_active": True,
            "dialogue_text_id": game.dialogue.text_id,
            "dialogue_text": game.dialogue.text,
            "dialogue_choice_count": game.dialogue.choice_count,
            "known": bool(using_known),
        }
        self.last_interaction_source = f"{source}:{button}"
        return (
            Setpoint(
                buttons=BUTTON_MASKS[button],
                stick_x=0,
                stick_y=0,
                reason=source,
            ),
            {
                **neutral_sample,
                "buttons": executed_buttons,
            },
            True,
        )

    def _observe_interaction_outcome(self, game, reward):
        probe = self.pending_interaction_probe
        if probe is None:
            return

        now = time.monotonic()
        if probe.get("kind") == "dialogue":
            text_changed = bool(
                not game.dialogue.active
                or game.dialogue.text_id != probe.get("dialogue_text_id")
                or game.dialogue.text != probe.get("dialogue_text")
                or game.dialogue.choice_count
                != probe.get("dialogue_choice_count")
            )
            if text_changed:
                self.route_graph.record_interaction_success(
                    probe["key"],
                    probe["button"],
                )
                self.interaction_probe_successes += 1
                self.last_interaction_source = (
                    f"learned:{probe['key']}->{probe['button']}"
                )
                self.pending_interaction_probe = None
                return
            if now - probe["at"] >= INTERACTION_OUTCOME_WINDOW_S:
                if probe.get("known"):
                    self.route_graph.record_interaction_failure(
                        probe["key"],
                        probe["button"],
                    )
                self.pending_interaction_probe = None
            return

        current_key = self._interaction_key(game)
        major_keys = {
            "new_world_transition",
            "new_dialogue",
            "durable_progress",
            "objective_milestone",
            "native_event",
        }
        major_effect = any(
            reward.breakdown.get(key, 0.0) > 0
            for key in major_keys
        )
        scene_changed = (
            (game.scene, game.room)
            != (probe["scene"], probe["room"])
        )
        dialogue_started = (
            game.dialogue.active and not probe["dialogue_active"]
        )
        exit_context_changed = bool(
            (
                probe.get("exit_active")
                or probe.get("actor_is_door")
            )
            and current_key != probe["key"]
        )
        success = (
            scene_changed
            or dialogue_started
            or major_effect
            or exit_context_changed
        )

        if success:
            self.route_graph.record_interaction_success(
                probe["key"],
                probe["button"],
            )
            self.interaction_probe_successes += 1
            self.last_interaction_source = (
                f"learned:{probe['key']}->{probe['button']}"
            )
            self.pending_interaction_probe = None
            return

        if now - probe["at"] >= INTERACTION_OUTCOME_WINDOW_S:
            if probe.get("known"):
                self.route_graph.record_interaction_failure(
                    probe["key"],
                    probe["button"],
                )
            self.pending_interaction_probe = None

    def _interaction_override(
        self,
        game,
        guidance: dict,
        setpoint: Setpoint,
        sample: dict,
    ) -> tuple[Setpoint, dict, bool]:
        key = self._interaction_key(game)
        if (
            key is None
            or game.dialogue.active
            or game.pause_menu.active
            or game.cutscene_active
        ):
            self.last_interaction_source = "none"
            return setpoint, sample, False

        actor = game.context_actor
        actor_is_door = bool(
            actor is not None
            and (actor.category_name or "").strip().lower() == "door"
        )
        should_interact = bool(
            guidance.get("exit_active")
            or actor_is_door
            or self.intent.mode == "interact"
        )
        if not should_interact:
            self.last_interaction_source = "none"
            return setpoint, sample, False

        # While an earlier probe is waiting for an observable result, release
        # all buttons so causality stays one-button-at-a-time.
        now = time.monotonic()
        if self.pending_interaction_probe is not None:
            executed_buttons = [0.0 for _ in BUTTON_NAMES]
            sample = {
                **sample,
                "stick": [0.0, 0.0],
                "buttons": executed_buttons,
            }
            return (
                Setpoint(
                    buttons=0,
                    stick_x=0,
                    stick_y=0,
                    reason="interaction_wait",
                ),
                sample,
                True,
            )

        if now - self.interaction_last_probe_at < INTERACTION_PROBE_COOLDOWN_S:
            executed_buttons = [0.0 for _ in BUTTON_NAMES]
            sample = {
                **sample,
                "stick": [0.0, 0.0],
                "buttons": executed_buttons,
            }
            return (
                Setpoint(
                    buttons=0,
                    stick_x=0,
                    stick_y=0,
                    reason="interaction_wait",
                ),
                sample,
                True,
            )

        known_button = self.route_graph.interaction_button(key)
        using_known = known_button in BUTTON_MASKS
        if using_known:
            button = known_button
            source = "interaction_memory"
        else:
            order = self._interaction_probe_order(key)
            index = self.interaction_probe_index.get(key, 0)
            button = order[index % len(order)]
            self.interaction_probe_index[key] = index + 1
            source = "interaction_probe"

        executed_buttons = [
            1.0 if name == button else 0.0
            for name in BUTTON_NAMES
        ]
        self.interaction_last_probe_at = now
        self.pending_interaction_probe = {
            "kind": "context",
            "key": key,
            "button": button,
            "at": now,
            "scene": game.scene,
            "room": game.room,
            "dialogue_active": bool(game.dialogue.active),
            "exit_active": bool(guidance.get("exit_active")),
            "actor_is_door": bool(actor_is_door),
            "known": bool(using_known),
        }
        self.last_interaction_source = f"{source}:{key}->{button}"
        sample = {
            **sample,
            "stick": [0.0, 0.0],
            "buttons": executed_buttons,
        }
        return (
            Setpoint(
                buttons=BUTTON_MASKS[button],
                stick_x=0,
                stick_y=0,
                reason=source,
            ),
            sample,
            True,
        )

    @staticmethod
    def _room_has_local_actionable_evidence(game) -> bool:
        action = game.context_action
        actor = game.context_actor
        actor_is_door = bool(
            actor is not None
            and (actor.category_name or "").strip().lower() == "door"
        )
        if (
            action.code != 0
            and (action.label or "").strip().lower() not in {"", "none"}
            and not actor_is_door
        ):
            return True
        for candidate in game.room_actors:
            category = (candidate.category_name or "").strip().lower()
            if category in {"chest", "npc", "switch", "enemy", "boss"}:
                return True
        return False

    def _should_prefer_observed_exit(
        self,
        game,
        *,
        actor_is_door: bool,
    ) -> bool:
        if not (
            self.route_graph.has_observed_escape(game)
            or self.room_map.has_known_escape(game)
        ):
            return False
        room_failure_pressure = self.route_graph.room_failure_pressure(game)
        trackable_objective = self.intent.completion.kind != "manual"
        known_successful_exit = self.room_map.has_remembered_transition(game)
        if (
            trackable_objective
            and self.intent.mode == "explore"
            and known_successful_exit
            and not self._room_has_local_actionable_evidence(game)
        ):
            # A previously traversed departure is stronger evidence than another
            # blind frontier cycle. Revisited known rooms may leave immediately
            # when the strategic objective has no local actor/interaction target.
            return True
        return bool(
            self.reward_tracker.local_dwell_seconds >= EXIT_PRIORITY_DWELL_S
            or actor_is_door
            or (
                trackable_objective
                and room_failure_pressure >= ROOM_FAILURE_EXIT_PRESSURE
            )
        )

    @staticmethod
    def _escape_context(game) -> tuple[int, int, bool, str]:
        return (
            int(game.scene),
            int(game.room),
            bool(game.mirrored_world),
            "adult" if game.player and game.player.age == "adult" else "child",
        )

    def _escape_on_cooldown(self, key: str | None, *, now: float) -> bool:
        if not key:
            return False
        until = self.escape_retry_after.get(key)
        if until is None:
            return False
        if until <= now:
            self.escape_retry_after.pop(key, None)
            return False
        return True

    def _update_escape_attempt(self, game):
        now = time.monotonic()
        self.escape_retry_after = {
            key: until
            for key, until in self.escape_retry_after.items()
            if until > now
        }
        attempt = self.active_escape_attempt
        context = self._escape_context(game)
        if attempt is not None and attempt.get("context") != context:
            self.escape_successes += 1
            self.active_escape_attempt = None
            return
        if (
            game.player is None
            or game.dialogue.active
            or game.pause_menu.active
            or game.cutscene_active
        ):
            return

        guidance = self.last_guidance
        if not (
            guidance.get("active")
            and guidance.get("forced_escape")
            and guidance.get("escape_key")
            and guidance.get("target") is not None
        ):
            return

        key = str(guidance["escape_key"])
        target = tuple(float(value) for value in guidance["target"])
        distance = math.dist(game.player.position, target)
        if (
            attempt is None
            or attempt.get("key") != key
            or attempt.get("context") != context
        ):
            self.active_escape_attempt = {
                "key": key,
                "context": context,
                "target": target,
                "started_at": now,
                "last_progress_at": now,
                "best_distance": distance,
            }
            return

        if tuple(attempt.get("target") or ()) != target:
            attempt["target"] = target
            attempt["best_distance"] = distance
            attempt["last_progress_at"] = now
            return

        best_distance = float(attempt.get("best_distance", distance))
        if distance <= best_distance - ESCAPE_PROGRESS_DELTA:
            attempt["best_distance"] = distance
            attempt["last_progress_at"] = now
            return

        stalled = (
            now - float(attempt.get("last_progress_at", now))
            >= ESCAPE_STALL_S
        )
        expired = (
            now - float(attempt.get("started_at", now))
            >= ESCAPE_MAX_AGE_S
        )
        if stalled or expired:
            self.escape_retry_after[key] = now + ESCAPE_RETRY_COOLDOWN_S
            self.escape_failures += 1
            self.active_escape_attempt = None
            self.route_graph.clear_navigation_commitment()

    def _ppo_transition_trainable(
        self,
        *,
        interaction_override: bool,
        guidance: dict,
        sample: dict,
    ) -> bool:
        if interaction_override:
            return False
        if (
            guidance.get("forced_escape")
            and float(sample.get("guidance_strength") or 0.0) >= 0.999
        ):
            # Full-authority escape steering is a structured intervention. The
            # sampled residual did not cause the executed movement.
            return False
        if self.reward_tracker.local_dwell_seconds >= NO_PROGRESS_TRAINING_CUTOFF_S:
            # A few minutes of negative anti-loop experience are enough. Hours
            # of the same failed room should not dominate the PPO checkpoint.
            return False
        return True

    @staticmethod
    def _escape_traversal_hint(game, escape_hint: dict) -> dict:
        """Insert one observed vertical traversal step before an unreachable exit."""
        if (
            game.player is None
            or bool(escape_hint.get("direct_reachable"))
            or int(escape_hint.get("path_nodes") or 0) > 1
            or not game.traversal_affordances
        ):
            return escape_hint

        final_target = escape_hint.get("exit_position") or escape_hint.get("waypoint")
        if not (
            isinstance(final_target, (list, tuple))
            and len(final_target) == 3
        ):
            return escape_hint
        final_target = tuple(float(value) for value in final_target)
        player_position = tuple(float(value) for value in game.player.position)
        current_distance = math.dist(player_position, final_target)
        current_vertical_gap = abs(final_target[1] - player_position[1])
        desired_vertical = final_target[1] - player_position[1]

        candidates = []
        for row in game.traversal_affordances:
            target = tuple(float(value) for value in row.target_position)
            approach = tuple(float(value) for value in row.approach_position)
            target_distance = math.dist(target, final_target)
            vertical_gap = abs(final_target[1] - target[1])
            improves_target = target_distance <= current_distance - 12.0
            improves_vertical = vertical_gap <= current_vertical_gap - 8.0
            if not improves_target and not improves_vertical:
                continue

            direction_penalty = 0.0
            if desired_vertical <= -25.0 and row.direction != "down":
                direction_penalty = 180.0
            elif desired_vertical >= 25.0 and row.direction != "up":
                direction_penalty = 180.0

            kind_penalty = (
                80.0
                if row.kind == "ledge_down"
                else 20.0
                if row.kind in {"stairs_or_slope_up", "stairs_or_slope_down"}
                else 0.0
            )
            approach_distance = math.dist(player_position, approach)
            score = (
                approach_distance
                + target_distance
                + direction_penalty
                + kind_penalty
            )
            candidates.append((score, approach_distance, row, approach, target))

        if not candidates:
            return escape_hint

        _, approach_distance, row, approach, target = min(
            candidates,
            key=lambda item: item[0],
        )
        if approach_distance > 55.0:
            phase = "approach"
            waypoint = approach
        else:
            phase = "target"
            waypoint = target

        escape_key = str(
            escape_hint.get("escape_key")
            or escape_hint.get("waypoint_id")
            or "escape"
        )
        return {
            "waypoint": waypoint,
            "waypoint_id": (
                f"escape-traversal:{row.kind}:{phase}:"
                f"{round(waypoint[0], 1)}:{round(waypoint[1], 1)}:"
                f"{round(waypoint[2], 1)}"
            )[:220],
            "path_nodes": 1,
            "waypoint_index": 0,
            "target_gap": math.dist(waypoint, final_target),
            "confidence": 1.0,
            "partial": True,
            "traversal": True,
            "traversal_kind": row.kind,
            "traversal_direction": row.direction,
            "traversal_phase": phase,
            "forced_escape": True,
            "escape_key": escape_key,
            "escape_final_target": final_target,
        }

    def _escape_waypoint(self, game) -> dict | None:
        """Choose a healthy escape target, preferring proven departures."""
        now = time.monotonic()
        excluded_memory = {
            key.split(":", 2)[2]
            for key, until in self.escape_retry_after.items()
            if until > now and key.startswith("memory:") and key.count(":") >= 2
        }

        # A point where Link actually changed room/scene is stronger evidence
        # than a merely detected exit surface and should win on revisits.
        remembered_transition = self.room_map.remembered_escape_waypoint(
            game,
            self.route_graph,
            kinds={"transition"},
            excluded_keys=excluded_memory,
        )
        if remembered_transition is not None:
            key = str(remembered_transition.get("escape_key") or "")
            if not self._escape_on_cooldown(key, now=now):
                return self._escape_traversal_hint(
                    game,
                    remembered_transition,
                )

        observed_exit = self.route_graph.exit_waypoint(
            game,
            allow_unreachable=True,
        )
        if observed_exit is not None:
            key = f"observed:{observed_exit.get('waypoint_id')}"
            if not self._escape_on_cooldown(key, now=now):
                return self._escape_traversal_hint(
                    game,
                    {
                        **observed_exit,
                        "forced_escape": True,
                        "escape_key": key,
                    },
                )

        observed_door = self.route_graph.door_waypoint(game)
        if observed_door is not None:
            key = f"observed:{observed_door.get('waypoint_id')}"
            if not self._escape_on_cooldown(key, now=now):
                return self._escape_traversal_hint(
                    game,
                    {
                        **observed_door,
                        "forced_escape": True,
                        "escape_key": key,
                    },
                )

        remembered_other = self.room_map.remembered_escape_waypoint(
            game,
            self.route_graph,
            kinds={"exit", "door"},
            excluded_keys=excluded_memory,
        )
        if remembered_other is not None:
            key = str(remembered_other.get("escape_key") or "")
            if not self._escape_on_cooldown(key, now=now):
                return self._escape_traversal_hint(
                    game,
                    remembered_other,
                )
        return None

    def _sample_setpoint(
        self,
        observation: list[float],
        guidance: dict,
    ) -> tuple[Setpoint, dict]:
        sample = self.policy.sample(
            observation,
            deterministic=not self.training_enabled,
            guidance_stick=guidance["stick"] if guidance.get("active") else None,
            guidance_strength=float(guidance.get("strength") or 0.0),
            button_quiet_strength=float(guidance.get("button_quiet") or 0.0),
        )
        stick = sample["stick"]
        buttons = sample["buttons"]
        setpoint = Setpoint(
            buttons=self._mask(buttons),
            stick_x=max(-80, min(80, round(stick[0] * 80))),
            stick_y=max(-80, min(80, round(stick[1] * 80))),
            reason="ml_policy",
        )
        return setpoint, sample

    def _enqueue_rollout(self, observation: list[float], *, done: bool):
        if not self.rollout:
            return
        bootstrap = 0.0 if done else self.policy.actor_value(observation)
        batch = {
            "rollout": self.rollout,
            "bootstrap_value": bootstrap,
            "bootstrap_done": done,
        }
        self.rollout = []
        if self.training_queue.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self.training_queue.get_nowait()
        self.training_queue.put_nowait(batch)

    def _ml_step(self, game) -> Setpoint:
        self.route_graph.observe(game)
        self.room_map.observe(game)
        self._update_escape_attempt(game)
        base_observation = encode_state(
            game,
            self.intent,
            last_stick=self.last_stick,
            last_buttons=self.last_buttons,
        )
        self.feature_history.append(base_observation)
        observation = stack_frames(list(self.feature_history))

        self.novelty_history.append(encode_novelty_state(game))
        novelty_observation = stack_frames(list(self.novelty_history))
        intrinsic = self.policy.intrinsic_reward(novelty_observation)
        reward = self.reward_tracker.step(
            game,
            self.intent,
            intrinsic=intrinsic,
            pressed_buttons=self.last_setpoint.buttons,
            guidance=self.last_guidance,
        )
        self._observe_interaction_outcome(game, reward)
        self.last_reward = reward.reward
        self.total_reward += reward.reward
        self.reward_window.append(reward.reward)
        self.last_reward_breakdown = reward.breakdown
        useful_keys = {
            "new_space",
            "new_macro_region",
            "frontier_progress",
            "route_waypoint",
            "new_actor",
            "new_dialogue",
            "new_context",
            "new_world_transition",
            "resource_rupees",
            "resource_ammo",
            "resource_health",
            "resource_magic",
            "durable_progress",
            "objective_milestone",
            "enemy_damage",
            "native_event",
        }
        useful = any(reward.breakdown.get(key, 0.0) > 0 for key in useful_keys)
        useful = useful or reward.breakdown.get("intent_progress", 0.0) > 0.01
        useful = useful or self.reward_tracker.route_waypoint_advanced
        self.useful_progress_window.append(bool(useful))
        if useful:
            self.last_useful_progress_at = time.monotonic()
        for achievement in reward.achievements:
            row = {
                **achievement,
                "action_index": self.actions_sampled + 1,
            }
            self.achievements.append(row)
            if self.on_achievement is not None:
                self.on_achievement(dict(row))

        if (
            self.training_enabled
            and self.pending is not None
            and self.pending.get("trainable", True)
        ):
            transition = {
                **self.pending,
                "reward": reward.reward,
                "done": reward.done,
            }
            self.rollout.append(transition)

        if self.training_enabled and (
            len(self.rollout) >= self.rollout_size
            or (reward.done and len(self.rollout) >= 16)
        ):
            self._enqueue_rollout(observation, done=reward.done)

        final_target = target_point(game, self.intent)
        if game.dialogue.active or game.pause_menu.active:
            # Modal UI owns physical control. Drop the ephemeral frontier
            # commitment without treating the modal pause as a navigation fail.
            self.route_graph.clear_frontier()
            route_hint = None
        elif (
            final_target is not None
            and self.intent.mode in {"navigate", "explore", "observe"}
        ):
            route_hint = self.route_graph.next_waypoint(
                game,
                final_target,
            )
            if route_hint is not None:
                self.route_graph.clear_frontier()
            elif (
                self.route_graph.route_recovery_needed()
                or self.route_graph.active_frontier is not None
            ):
                # A learned directed edge just failed in real execution. Do not
                # immediately fall back to the same straight-line cognition
                # target; recover from current observed evidence instead.
                route_hint = self._escape_waypoint(game)
                if route_hint is not None:
                    self.route_graph.clear_frontier()
                else:
                    route_hint = self.route_graph.exploration_waypoint(game)
        elif self.intent.mode == "explore":
            actor_is_door = bool(
                game.context_actor is not None
                and (
                    game.context_actor.category_name or ""
                ).strip().lower() == "door"
            )
            prefer_exit = self._should_prefer_observed_exit(
                game,
                actor_is_door=actor_is_door,
            )
            route_hint = (
                self._escape_waypoint(game)
                if prefer_exit
                else None
            )
            if route_hint is not None:
                self.route_graph.clear_frontier()
            else:
                route_hint = self.route_graph.exploration_waypoint(game)
        else:
            route_hint = None
        guidance = goal_guidance(
            game,
            self.intent,
            local_dwell_seconds=self.reward_tracker.local_dwell_seconds,
            route_hint=route_hint,
        )
        setpoint, sample = self._sample_setpoint(observation, guidance)
        setpoint, sample, dialogue_override = self._dialogue_override(
            game,
            setpoint,
            sample,
        )
        if dialogue_override:
            interaction_override = True
        else:
            (
                setpoint,
                sample,
                dialogue_reentry_override,
            ) = self._dialogue_reentry_override(
                game,
                setpoint,
                sample,
            )
            if dialogue_reentry_override:
                interaction_override = True
            else:
                setpoint, sample, interaction_override = self._interaction_override(
                    game,
                    guidance,
                    setpoint,
                    sample,
                )
        transition_trainable = self._ppo_transition_trainable(
            interaction_override=interaction_override,
            guidance=guidance,
            sample=sample,
        )
        self.last_guidance = guidance
        self.last_setpoint = setpoint
        self.last_stick = tuple(sample["stick"])
        self.last_buttons = tuple(sample["buttons"])
        self.last_button_probability_mean = float(
            sample.get("button_probability_mean") or 0.0
        )
        self.last_expected_button_count = float(
            sample.get("expected_button_count") or 0.0
        )
        self.last_guidance_mix = float(
            sample.get("guidance_strength") or 0.0
        )
        self.pending = {
            "observation": observation,
            "novelty_observation": novelty_observation,
            "stick": list(sample["stick"]),
            "policy_stick": list(sample.get("policy_stick", sample["stick"])),
            "buttons": list(sample["buttons"]),
            "log_prob": sample["log_prob"],
            "value": sample["value"],
            "guidance_stick": sample.get("guidance_stick", [0.0, 0.0]),
            "guidance_strength": sample.get("guidance_strength", 0.0),
            "button_quiet_strength": sample.get("button_quiet_strength", 0.0),
            "trainable": transition_trainable,
        }
        self.actions_sampled += 1

        active_names = [
            name for name, value in zip(BUTTON_NAMES, self.last_buttons)
            if value > 0.5
        ]
        buttons_text = "+".join(active_names) if active_names else "none"
        guidance_text = ""
        if guidance.get("active"):
            distance = guidance.get("distance")
            distance_text = (
                f", distance {distance:.0f}u"
                if isinstance(distance, (int, float))
                else ""
            )
            guidance_text = (
                f" Goal guidance {guidance.get('source')}: "
                f"({guidance['stick'][0]:+.2f}, {guidance['stick'][1]:+.2f}) "
                f"mix {sample.get('guidance_strength', 0.0):.2f}{distance_text}."
            )
        if setpoint.reason == "ml_policy":
            self.last_motor_summary = (
                f"ML policy sampled raw controller: stick "
                f"({setpoint.stick_x:+d}, {setpoint.stick_y:+d}), "
                f"buttons {buttons_text}."
                + guidance_text
            )
        else:
            self.last_motor_summary = (
                f"Local controller override {setpoint.reason}: stick "
                f"({setpoint.stick_x:+d}, {setpoint.stick_y:+d}), "
                f"buttons {buttons_text}."
            )
        return setpoint

    async def _learner_loop(self, active: Callable[[], bool], publish: Callable[[], None]):
        while active() or not self.training_queue.empty():
            try:
                batch = await asyncio.wait_for(self.training_queue.get(), timeout=0.2)
            except asyncio.TimeoutError:
                continue
            try:
                stats = await asyncio.to_thread(
                    self.policy.train_rollout,
                    batch["rollout"],
                    bootstrap_value=batch["bootstrap_value"],
                    bootstrap_done=batch["bootstrap_done"],
                )
                self.last_training_stats = stats
                publish()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                # Training failures must not take controller ownership away from
                # the actor loop. Keep playing with the last published weights.
                self.last_training_stats = {
                    "error": f"{type(exc).__name__}: {str(exc)[:180]}"
                }
                publish()

    async def _route_saver_loop(
        self,
        stop_event: asyncio.Event,
        publish: Callable[[], None],
    ):
        """Persist route learning outside the realtime control loop."""
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(
                    stop_event.wait(),
                    timeout=self.route_save_interval_s,
                )
                break
            except asyncio.TimeoutError:
                pass

            route_dirty = self.route_graph.writable and self.route_graph.dirty
            room_map_dirty = self.room_map.writable and self.room_map.dirty
            if not route_dirty and not room_map_dirty:
                continue
            if route_dirty:
                try:
                    await asyncio.to_thread(self.route_graph.save)
                    self.route_save_error = ""
                except Exception as exc:
                    # The graph remains dirty and the next interval retries.
                    self.route_save_error = (
                        f"{type(exc).__name__}: {str(exc)[:180]}"
                    )
            if room_map_dirty:
                try:
                    await asyncio.to_thread(self.room_map.save)
                    self.room_map_save_error = ""
                except Exception as exc:
                    self.room_map_save_error = (
                        f"{type(exc).__name__}: {str(exc)[:180]}"
                    )
            publish()

    async def run(self, active: Callable[[], bool], publish: Callable[[], None]):
        learner = (
            asyncio.create_task(self._learner_loop(active, publish))
            if self.training_enabled
            else None
        )
        route_stop = asyncio.Event()
        route_saver = (
            asyncio.create_task(
                self._route_saver_loop(route_stop, publish)
            )
            if self.training_enabled
            else None
        )
        try:
            while active():
                game = self.bridge.state
                self.motor_ticks += 1
                self.tick += 1

                if not self.bridge.connected or game is None or not game.in_game or game.player is None:
                    self.pending = None
                    self.feature_history.clear()
                    self.novelty_history.clear()
                    self.reward_tracker.break_causal_chain()
                    self.route_graph.reset_trace()
                    self.room_map.reset_trace()
                    self.active_escape_attempt = None
                    self._reset_camera_guard()
                    self.last_setpoint = Setpoint(reason="bridge_wait")
                    self.last_stick = (0.0, 0.0)
                    self.last_buttons = tuple(0.0 for _ in BUTTON_NAMES)
                    self.last_motor_summary = "Waiting for a playable realtime bridge state."
                    self.bridge.release()
                    publish()
                    await asyncio.sleep(0.1)
                    continue

                # During a non-interactive cutscene there is no useful control
                # transition to learn; neutral input avoids polluting the rollout.
                if game.cutscene_active and not game.dialogue.active:
                    self.route_graph.reset_trace()
                    # Keep room-map trace across door/transition cutscenes so the
                    # first playable frame in the next room can persist the real
                    # departure point from the previous room.
                    self.last_setpoint = Setpoint(reason="cutscene")
                    self.last_motor_summary = "Cutscene owns Link; ML actor remains live and resumes immediately."
                elif self.tick % self.action_repeat_ticks == 1 or self.pending is None:
                    self._ml_step(game)

                if not (game.cutscene_active and not game.dialogue.active):
                    # Policy residuals are still sampled at 10 Hz.  Camera-space
                    # projection is refreshed at the 20 Hz motor cadence so a
                    # fixed/indoor camera switch cannot reuse a stale stick.
                    self._refresh_camera_relative_setpoint(game)

                try:
                    self.bridge.send(
                        buttons=self.last_setpoint.buttons,
                        stick_x=self.last_setpoint.stick_x,
                        stick_y=self.last_setpoint.stick_y,
                        lease_ms=150,
                    )
                except RuntimeError:
                    pass

                publish()
                await asyncio.sleep(self.tick_s)
        finally:
            self.bridge.release()
            if route_saver is not None:
                route_stop.set()
                with contextlib.suppress(asyncio.CancelledError):
                    await route_saver
            if self.route_graph.writable:
                try:
                    await asyncio.to_thread(
                        self.route_graph.save,
                        force=True,
                    )
                    self.route_save_error = ""
                except (OSError, ValueError, RuntimeError) as exc:
                    self.route_save_error = (
                        f"{type(exc).__name__}: {str(exc)[:180]}"
                    )
            if self.room_map.writable:
                try:
                    await asyncio.to_thread(
                        self.room_map.save,
                        force=True,
                    )
                    self.room_map_save_error = ""
                except (OSError, ValueError, RuntimeError) as exc:
                    self.room_map_save_error = (
                        f"{type(exc).__name__}: {str(exc)[:180]}"
                    )
            # Let queued/in-flight batches finish after input authority is gone.
            # This cannot move Link because the controller loop has already ended.
            if learner is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(learner)

            # A short training session may stop before rollout_size. Preserve useful
            # experience instead of discarding it; evaluation never updates weights.
            if self.training_enabled and len(self.rollout) >= 16:
                game = self.bridge.state
                if game is not None and self.feature_history:
                    observation = stack_frames(list(self.feature_history))
                    bootstrap = self.policy.actor_value(observation)
                    try:
                        self.last_training_stats = await asyncio.to_thread(
                            self.policy.train_rollout,
                            self.rollout,
                            bootstrap_value=bootstrap,
                            bootstrap_done=False,
                        )
                    except Exception as exc:
                        self.last_training_stats = {
                            "error": f"{type(exc).__name__}: {str(exc)[:180]}"
                        }
                    self.rollout = []
                    publish()

    def telemetry(self) -> dict:
        return {
            "intent": self.intent.model_dump(),
            "intent_age_ms": round((time.monotonic() - self.intent_updated_at) * 1000),
            "motor": self.last_motor_summary,
            "guidance": {
                **self.last_guidance,
                "stick": list(self.last_guidance.get("stick") or (0.0, 0.0)),
                "target": (
                    list(self.last_guidance["target"])
                    if self.last_guidance.get("target") is not None
                    else None
                ),
            },
            "motor_ticks": self.motor_ticks,
            "actions_sampled": self.actions_sampled,
            "camera_control": {
                "state": self.camera_motion_state,
                "yaw": (
                    round(self.last_camera_yaw, 6)
                    if self.last_camera_yaw is not None
                    else None
                ),
                "guard_ticks": self.camera_guard_ticks,
                "cuts": self.camera_cut_count,
                "transitions": self.camera_transition_count,
            },
            "escape_control": {
                "active": bool(self.active_escape_attempt),
                "key": (
                    self.active_escape_attempt.get("key")
                    if self.active_escape_attempt
                    else None
                ),
                "age_s": (
                    round(
                        time.monotonic()
                        - float(self.active_escape_attempt.get("started_at", time.monotonic())),
                        1,
                    )
                    if self.active_escape_attempt
                    else 0.0
                ),
                "failures": self.escape_failures,
                "successes": self.escape_successes,
                "cooling_down": len(self.escape_retry_after),
            },
            "setpoint": {
                "buttons": self.last_setpoint.buttons,
                "button_names": [
                    name for name, mask in BUTTON_MASKS.items()
                    if self.last_setpoint.buttons & mask
                ],
                "stick_x": self.last_setpoint.stick_x,
                "stick_y": self.last_setpoint.stick_y,
                "reason": self.last_setpoint.reason,
            },
            "learning": {
                **self.policy.stats(),
                "training_enabled": self.training_enabled,
                "room_map": {
                    **self.room_map.stats(self.bridge.state),
                    "save_error": self.room_map_save_error or None,
                },
                "run_updates": max(0, self.policy.updates - self.starting_updates),
                "run_samples_trained": max(
                    0, self.policy.samples_trained - self.starting_samples_trained
                ),
                "rollout_steps": len(self.rollout),
                "queued_rollouts": self.training_queue.qsize(),
                "last_reward": round(self.last_reward, 6),
                "total_reward": round(self.total_reward, 4),
                "recent_mean_reward": round(
                    sum(self.reward_window) / len(self.reward_window), 6
                ) if self.reward_window else 0.0,
                "positive_reward_rate": round(
                    sum(1 for value in self.reward_window if value > 0) / len(self.reward_window),
                    4,
                ) if self.reward_window else 0.0,
                "button_probability_mean": round(
                    self.last_button_probability_mean,
                    6,
                ),
                "expected_button_count": round(
                    self.last_expected_button_count,
                    6,
                ),
                "guidance_mix": round(self.last_guidance_mix, 6),
                "interaction_learning": {
                    "learned": len(self.route_graph.interactions),
                    "probe_successes": self.interaction_probe_successes,
                    "pending": bool(self.pending_interaction_probe),
                    "last": self.last_interaction_source,
                    "dialogue_reentry_guard": bool(self.dialogue_reentry_guard),
                    "dialogue_reentry_suppressed": self.dialogue_reentry_suppressed,
                },
                "useful_progress_rate": round(
                    sum(1 for value in self.useful_progress_window if value)
                    / len(self.useful_progress_window),
                    4,
                ) if self.useful_progress_window else 0.0,
                "seconds_since_useful_progress": round(
                    time.monotonic() - self.last_useful_progress_at, 1
                ),
                "objective_score": self.reward_tracker.objective_score,
                "achievements": list(self.achievements),
                "resources": {
                    "chests_opened": self.reward_tracker.chests_opened,
                    "rupees_collected": self.reward_tracker.rupees_collected,
                    "ammo_collected": self.reward_tracker.ammo_collected,
                    "health_recovered": self.reward_tracker.health_recovered,
                    "magic_recovered": self.reward_tracker.magic_recovered,
                },
                "exploration": {
                    "unique_spaces": len(self.reward_tracker.visited_cells),
                    "unique_actors": len(self.reward_tracker.seen_actors),
                    "unique_transitions": len(self.reward_tracker.seen_transitions),
                    "unique_macro_regions": len(
                        self.reward_tracker.seen_macro_regions
                    ),
                    "local_dwell_seconds": round(
                        self.reward_tracker.local_dwell_seconds, 1
                    ),
                    "local_anchor_distance": round(
                        self.reward_tracker.local_anchor_distance, 1
                    ),
                    "local_frontier_radius": round(
                        self.reward_tracker.local_frontier_radius, 1
                    ),
                    "local_dwell_penalty": round(
                        self.reward_tracker.local_dwell_penalty, 6
                    ),
                },
                "reward_breakdown": self.last_reward_breakdown,
                "route_memory": {
                    **self.route_graph.stats(),
                    "waypoints_advanced": self.reward_tracker.route_waypoints_advanced,
                    "room_failure_pressure": (
                        self.route_graph.room_failure_pressure(self.bridge.state)
                        if self.bridge.state is not None
                        else 0
                    ),
                    "save_error": self.route_save_error or None,
                },
                "last_update": self.last_training_stats,
            },
        }
