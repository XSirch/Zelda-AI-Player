from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from ..bridge import Bridge
from .features import (
    BUTTON_NAMES,
    encode_novelty_state,
    encode_state,
    goal_guidance,
    stack_frames,
)
from .ml_policy import OnlinePPO
from .models import AgentIntent
from .reward import RewardTracker

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

    def __init__(
        self,
        bridge: Bridge,
        checkpoint: Path,
        *,
        rollout_size: int = 256,
        on_achievement: Callable[[dict], None] | None = None,
        training_enabled: bool = True,
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
        self.reward_window = deque(maxlen=200)
        self.useful_progress_window = deque(maxlen=200)
        self.last_useful_progress_at = time.monotonic()
        self.achievements = deque(maxlen=64)
        self.last_training_stats: dict = {}
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
        }
        self.last_motor_summary = "ML policy is ready to explore raw controller inputs."

    def reset_episode_state(self):
        self.reward_tracker = RewardTracker()
        self.feature_history.clear()
        self.novelty_history.clear()
        self.pending = None
        self.rollout = []
        self.tick = 0
        self.last_setpoint = Setpoint()
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
        }
        self.last_reward = 0.0
        self.last_reward_breakdown = {}
        self.reward_window.clear()
        self.useful_progress_window.clear()
        self.last_useful_progress_at = time.monotonic()
        self.achievements.clear()

    def set_intent(self, intent: AgentIntent):
        self.intent = intent
        self.intent_updated_at = time.monotonic()

    def neutralize(self, reason: str = "stopped"):
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
        }
        self.last_motor_summary = "Controller input revoked; no buttons are being held."

    @staticmethod
    def _mask(buttons: list[float]) -> int:
        mask = 0
        for name, active in zip(BUTTON_NAMES, buttons):
            if active > 0.5:
                mask |= BUTTON_MASKS[name]
        return mask

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
        )
        self.last_reward = reward.reward
        self.total_reward += reward.reward
        self.reward_window.append(reward.reward)
        self.last_reward_breakdown = reward.breakdown
        useful_keys = {
            "new_space",
            "new_macro_region",
            "frontier_progress",
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

        if self.training_enabled and self.pending is not None:
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

        guidance = goal_guidance(
            game,
            self.intent,
            local_dwell_seconds=self.reward_tracker.local_dwell_seconds,
        )
        setpoint, sample = self._sample_setpoint(observation, guidance)
        self.last_guidance = guidance
        self.last_setpoint = setpoint
        self.last_stick = tuple(sample["stick"])
        self.last_buttons = tuple(sample["buttons"])
        self.pending = {
            "observation": observation,
            "novelty_observation": novelty_observation,
            "stick": list(sample["stick"]),
            "buttons": list(sample["buttons"]),
            "log_prob": sample["log_prob"],
            "value": sample["value"],
            "guidance_stick": sample.get("guidance_stick", [0.0, 0.0]),
            "guidance_strength": sample.get("guidance_strength", 0.0),
            "button_quiet_strength": sample.get("button_quiet_strength", 0.0),
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
                f"strength {guidance['strength']:.2f}{distance_text}."
            )
        self.last_motor_summary = (
            f"ML policy sampled raw controller: stick "
            f"({setpoint.stick_x:+d}, {setpoint.stick_y:+d}), buttons {buttons_text}."
            + guidance_text
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

    async def run(self, active: Callable[[], bool], publish: Callable[[], None]):
        learner = (
            asyncio.create_task(self._learner_loop(active, publish))
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
                    self.last_setpoint = Setpoint(reason="cutscene")
                    self.last_motor_summary = "Cutscene owns Link; ML actor remains live and resumes immediately."
                elif self.tick % self.action_repeat_ticks == 1 or self.pending is None:
                    self._ml_step(game)

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
                "last_update": self.last_training_stats,
            },
        }
