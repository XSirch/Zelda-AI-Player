from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from ..bridge import Bridge
from ..models import ActorObservation, GameState
from ..navmesh import plan_navmesh, waypoint_probe_safe
from ..skills.catalog import BUTTONS
from ..skills.common import _steer_to, _world_yaw_stick
from .models import AgentIntent


BUTTON_ORDER = ("A", "B", "Z", "R", "C_UP", "C_LEFT", "C_DOWN", "C_RIGHT", "START")
ROLE_EFFECTS = {
    "interact": ("dialogue", "world", "context"),
    "attack": ("melee",),
    "target": ("target",),
    "menu": ("menu",),
}


@dataclass(frozen=True)
class Setpoint:
    buttons: int = 0
    stick_x: int = 0
    stick_y: int = 0
    reason: str = "neutral"


class ContinuousController:
    """Realtime motor controller that never waits for model inference.

    It consumes a high-level AgentIntent, reacts to immediate game state, and
    empirically records the effects of physical button probes.  The controller
    emits a new short-lease setpoint at ~20 Hz; if Python stalls, the native
    watchdog releases control.
    """

    tick_s = 0.05

    def __init__(self, bridge: Bridge):
        self.bridge = bridge
        self.intent = AgentIntent.bootstrap()
        self.intent_updated_at = time.monotonic()
        self.last_setpoint = Setpoint()
        self.last_motor_summary = "Bootstrapping continuous control."
        self.motor_ticks = 0
        self.scene_key: tuple[int, int, int] | None = None
        self.scene_entered_at = time.monotonic()
        self.explore_yaw: float | None = None
        self.explore_turns = 0

        self._pulse_name: str | None = None
        self._pulse_mask = 0
        self._pulse_until = 0.0
        self._release_until = 0.0
        self._next_action_at = 0.0
        self._next_generic_probe_at = 0.0
        self._pending_probe: dict | None = None
        self._profile_dirty = False
        self._profile: dict = self._empty_profile()

    @staticmethod
    def _empty_profile() -> dict:
        return {
            "version": 1,
            "buttons": {
                name: {
                    "samples": 0,
                    "effects": {
                        "dialogue": 0,
                        "world": 0,
                        "context": 0,
                        "melee": 0,
                        "target": 0,
                        "menu": 0,
                        "movement": 0,
                        "ocarina": 0,
                    },
                }
                for name in BUTTON_ORDER
            },
            "updated_at": 0.0,
        }

    def reset(self, profile: dict | None = None):
        self.intent = AgentIntent.bootstrap()
        self.intent_updated_at = time.monotonic()
        self.last_setpoint = Setpoint()
        self.last_motor_summary = "Exploring immediately while cognition starts."
        self.motor_ticks = 0
        self.scene_key = None
        self.scene_entered_at = time.monotonic()
        self.explore_yaw = None
        self.explore_turns = 0
        self._pulse_name = None
        self._pulse_mask = 0
        self._pulse_until = 0.0
        self._release_until = 0.0
        self._next_action_at = 0.0
        self._next_generic_probe_at = time.monotonic() + 0.7
        self._pending_probe = None
        self._profile = self._normalize_profile(profile)
        self._profile_dirty = False

    def _normalize_profile(self, profile: dict | None) -> dict:
        base = self._empty_profile()
        if not isinstance(profile, dict):
            return base
        buttons = profile.get("buttons")
        if not isinstance(buttons, dict):
            return base
        for name in BUTTON_ORDER:
            incoming = buttons.get(name)
            if not isinstance(incoming, dict):
                continue
            row = base["buttons"][name]
            row["samples"] = max(0, int(incoming.get("samples") or 0))
            effects = incoming.get("effects")
            if isinstance(effects, dict):
                for effect in row["effects"]:
                    row["effects"][effect] = max(0, int(effects.get(effect) or 0))
        base["updated_at"] = float(profile.get("updated_at") or 0.0)
        return base

    def set_intent(self, intent: AgentIntent):
        self.intent = intent
        self.intent_updated_at = time.monotonic()

    @staticmethod
    def _actor_matches(actor: ActorObservation, intent: AgentIntent) -> bool:
        if intent.target_actor_uid and actor.actor_uid != intent.target_actor_uid:
            return False
        if intent.target_actor_id is not None and actor.actor_id != intent.target_actor_id:
            return False
        if intent.target_actor_params is not None and actor.params != intent.target_actor_params:
            return False
        return intent.target_actor_id is not None or intent.target_actor_uid is not None

    def _target_actor(self, game: GameState) -> ActorObservation | None:
        candidates = list(game.room_actors)
        if game.target_actor:
            candidates.insert(0, game.target_actor)
        matches = [a for a in candidates if self._actor_matches(a, self.intent)]
        return min(matches, key=lambda row: row.distance) if matches else None

    @staticmethod
    def _fingerprint(game: GameState) -> dict:
        player = game.player
        return {
            "scene": game.scene,
            "room": game.room,
            "scene_epoch": game.scene_epoch,
            "position": tuple(player.position) if player else None,
            "dialogue": (game.dialogue.active, game.dialogue.text_id, game.dialogue.state_code),
            "context": (game.context_action.code, game.context_action.label),
            "menu": game.pause_menu.active,
            "target": (
                game.target_actor.actor_uid if game.target_actor else None,
                player.z_target_active_timer if player else 0,
            ),
            "melee": (
                player.melee_weapon_state if player else 0,
                player.melee_weapon_animation if player else 0,
            ),
            "ocarina": (game.ocarina_mode, game.ocarina_action),
        }

    @staticmethod
    def _effects(before: dict, game: GameState) -> set[str]:
        after = ContinuousController._fingerprint(game)
        effects: set[str] = set()
        if before["dialogue"] != after["dialogue"]:
            effects.add("dialogue")
        if (before["scene"], before["room"], before["scene_epoch"]) != (
            after["scene"], after["room"], after["scene_epoch"]
        ):
            effects.add("world")
        if before["context"] != after["context"]:
            effects.add("context")
        if before["menu"] != after["menu"]:
            effects.add("menu")
        if before["target"] != after["target"]:
            effects.add("target")
        if before["melee"] != after["melee"]:
            effects.add("melee")
        if before["ocarina"] != after["ocarina"]:
            effects.add("ocarina")
        if before["position"] and after["position"]:
            if math.dist(before["position"], after["position"]) >= 6.0:
                effects.add("movement")
        return effects

    def _finish_probe_if_ready(self, game: GameState, now: float):
        pending = self._pending_probe
        if not pending or now < pending["evaluate_at"]:
            return
        name = pending["name"]
        row = self._profile["buttons"][name]
        row["samples"] += 1
        for effect in self._effects(pending["before"], game):
            if effect in row["effects"]:
                row["effects"][effect] += 1
        self._profile["updated_at"] = time.time()
        self._profile_dirty = True
        self._pending_probe = None

    def _role_score(self, name: str, role: str) -> int:
        row = self._profile["buttons"][name]
        effects = ROLE_EFFECTS.get(role, ())
        return sum(int(row["effects"].get(effect) or 0) for effect in effects)

    def _learned_button(self, role: str) -> str | None:
        ranked = sorted(
            BUTTON_ORDER,
            key=lambda name: (
                self._role_score(name, role),
                -self._profile["buttons"][name]["samples"],
            ),
            reverse=True,
        )
        if not ranked or self._role_score(ranked[0], role) <= 0:
            return None
        return ranked[0]

    def _probe_candidate(self, candidates: tuple[str, ...]) -> str:
        return min(
            candidates,
            key=lambda name: (
                self._profile["buttons"][name]["samples"],
                -sum(self._profile["buttons"][name]["effects"].values()),
                BUTTON_ORDER.index(name),
            ),
        )

    def _request_button(self, game: GameState, name: str, now: float, *, hold_s: float = 0.09) -> bool:
        if name not in BUTTONS or now < self._next_action_at:
            return False
        self._pulse_name = name
        self._pulse_mask = BUTTONS[name]
        self._pulse_until = now + hold_s
        self._release_until = self._pulse_until + 0.08
        self._next_action_at = self._release_until + 0.08
        self._pending_probe = {
            "name": name,
            "before": self._fingerprint(game),
            "evaluate_at": self._release_until + 0.22,
        }
        return True

    def _role_button_or_probe(
        self,
        game: GameState,
        role: str,
        now: float,
        candidates: tuple[str, ...] = ("A", "B", "Z", "R"),
    ) -> str | None:
        learned = self._learned_button(role)
        if learned:
            return learned
        name = self._probe_candidate(candidates)
        if self._request_button(game, name, now):
            self.last_motor_summary = f"Testing physical button {name} for {role}-like effects."
        return None

    def _pulse_buttons(self, now: float) -> int:
        if now < self._pulse_until:
            return self._pulse_mask
        if now < self._release_until:
            return 0
        self._pulse_name = None
        self._pulse_mask = 0
        return 0

    @staticmethod
    def _nearest_threat(game: GameState) -> ActorObservation | None:
        enemies = [
            actor for actor in game.room_actors
            if actor.category in {5, 9} and actor.distance <= 320.0
        ]
        return min(enemies, key=lambda row: row.distance) if enemies else None

    def _dialogue_setpoint(self, game: GameState, now: float) -> Setpoint:
        dialogue = game.dialogue
        if dialogue.choice_count > 0:
            desired = self.intent.choice_index
            if desired is None:
                desired = dialogue.choice_index
            if desired != dialogue.choice_index and now >= self._next_action_at:
                direction = 60 if desired < dialogue.choice_index else -60
                self._next_action_at = now + 0.18
                self.last_motor_summary = f"Moving dialogue choice toward option {desired + 1}."
                return Setpoint(stick_y=direction, reason="dialogue_choice")
        if dialogue.can_advance and now >= self._next_action_at:
            button = self._learned_button("interact")
            if button:
                self._request_button(game, button, now)
                self.last_motor_summary = "Advancing dialogue with an empirically learned action button."
            else:
                self._role_button_or_probe(game, "interact", now, ("A", "B", "Z", "R"))
        return Setpoint(buttons=self._pulse_buttons(now), reason="dialogue")

    def _gameover_setpoint(self, game: GameState, now: float) -> Setpoint:
        if now >= self._next_action_at:
            button = self._learned_button("interact")
            if button:
                self._request_button(game, button, now)
            else:
                self._role_button_or_probe(game, "interact", now, ("A", "B", "START"))
        self.last_motor_summary = "Recovering from game over without stopping cognition."
        return Setpoint(buttons=self._pulse_buttons(now), reason="gameover")

    def _menu_setpoint(self, game: GameState, now: float) -> Setpoint:
        if self.intent.mode != "menu":
            if now >= self._next_action_at:
                menu_button = self._learned_button("menu")
                if menu_button:
                    self._request_button(game, menu_button, now)
                else:
                    self._request_button(game, "START", now)
            self.last_motor_summary = "Closing an unexpected menu and returning to play."
            return Setpoint(buttons=self._pulse_buttons(now), reason="menu_close")
        self.last_motor_summary = "Menu is open; waiting for a semantic menu objective."
        return Setpoint(buttons=self._pulse_buttons(now), reason="menu")

    def _combat_setpoint(self, game: GameState, enemy: ActorObservation, now: float) -> Setpoint:
        target_button = self._learned_button("target")
        attack_button = self._learned_button("attack")

        # Learn target/attack affordances in parallel with positioning.
        if target_button is None and now >= self._next_action_at:
            self._role_button_or_probe(game, "target", now, ("Z", "A", "B", "R"))
        elif attack_button is None and now >= self._next_action_at:
            self._role_button_or_probe(game, "attack", now, ("B", "A", "Z", "R"))

        target = enemy.focus_position or enemy.position
        stick_x, stick_y, distance = _steer_to(game, target, 0.72)
        buttons = self._pulse_buttons(now)

        if target_button:
            buttons |= BUTTONS[target_button]
        if attack_button and distance <= 135 and now >= self._next_action_at:
            self._request_button(game, attack_button, now, hold_s=0.07)
            buttons = self._pulse_buttons(now) | (BUTTONS[target_button] if target_button else 0)
        elif distance < 80 and target_button:
            # A conservative defensive prior while the controller is still
            # learning which physical actions are useful against this enemy.
            buttons |= BUTTONS["R"]

        self.last_motor_summary = (
            f"Reactive combat against observed actor {enemy.actor_id}; "
            f"distance {distance:.0f}, learning target/attack effects online."
        )
        return Setpoint(buttons=buttons, stick_x=stick_x, stick_y=stick_y, reason="combat")

    def _exploration_target(self, game: GameState, now: float) -> tuple[float, float, float] | None:
        if not game.player:
            return None

        # A model-specified traversal direction is grounded to observed geometry.
        if self.intent.direction in {"up", "down"} and game.traversal_affordances:
            matching = [
                row for row in game.traversal_affordances
                if row.direction == self.intent.direction
            ]
            if matching:
                row = min(matching, key=lambda item: item.distance)
                return row.approach_position

        # Before the first cognition response, walking toward an actually observed
        # reachable exit is more useful than idling in interiors such as Link's house.
        if self.intent.mode == "explore" and now - self.scene_entered_at >= 0.5:
            exits = [row for row in game.scene_exits if row.direct_reachable]
            if exits:
                return min(
                    exits,
                    key=lambda row: math.dist(game.player.position, row.position),
                ).position
        return None

    def _navigation_setpoint(self, game: GameState, now: float) -> Setpoint:
        assert game.player
        actor = self._target_actor(game)
        target = self.intent.target_position
        if actor is not None:
            target = actor.position
        if target is None:
            target = self._exploration_target(game, now)

        # Contextual interactions are opportunistic and do not wait for cognition.
        context_near = game.context_action.label != "none" and (
            game.context_actor is None or game.context_actor.distance <= 95
        )
        if context_near and now >= self._next_action_at:
            button = self._learned_button("interact")
            if button:
                self._request_button(game, button, now)
            else:
                self._role_button_or_probe(game, "interact", now, ("A", "B", "Z", "R"))

        if self.intent.mode == "interact" and actor is not None and actor.distance <= 75 and now >= self._next_action_at:
            button = self._learned_button("interact")
            if button:
                self._request_button(game, button, now)
            else:
                self._role_button_or_probe(game, "interact", now, ("A", "B", "Z", "R"))

        buttons = self._pulse_buttons(now)

        if target is not None:
            target3 = tuple(float(v) for v in target)
            plan = plan_navmesh(game, target3)
            waypoint = plan.waypoint if plan else target3
            safe = waypoint_probe_safe(game, waypoint, known_floor_target=bool(plan))
            if safe:
                stick_x, stick_y, distance = _steer_to(game, waypoint, 0.82)
                self.last_motor_summary = (
                    f"Continuous navigation toward observed target; "
                    f"next waypoint {distance:.0f} units away."
                )
                return Setpoint(buttons=buttons, stick_x=stick_x, stick_y=stick_y, reason="navigate")
            self.explore_yaw = float(game.player.yaw + 0x3000)
            self.explore_turns += 1

        # Free exploration: keep moving, but use realtime probes to avoid repeatedly
        # driving into a wall/cliff. The heading persists across cognition calls.
        if self.explore_yaw is None:
            self.explore_yaw = float(game.player.yaw)
        forward = next(
            (
                row for row in game.navigation_probes
                if row.direction == "forward" and row.distance <= 75
            ),
            None,
        )
        unsafe = bool(
            forward and (
                forward.wall_hit
                or not forward.floor_found
                or forward.delta_y is None
                or abs(forward.delta_y) > 28
            )
        )
        if unsafe:
            self.explore_yaw += 0x2800 if self.explore_turns % 2 == 0 else -0x3000
            self.explore_turns += 1
        stick_x, stick_y = _world_yaw_stick(game, self.explore_yaw, 56)

        if now >= self._next_generic_probe_at and now >= self._next_action_at and not context_near:
            candidate = self._probe_candidate(("A", "B", "Z", "R"))
            self._request_button(game, candidate, now, hold_s=0.07)
            self._next_generic_probe_at = now + 1.4
            buttons = self._pulse_buttons(now)
            self.last_motor_summary = (
                f"Exploring continuously while probing raw button {candidate} "
                "and measuring its effect."
            )
        else:
            self.last_motor_summary = "Exploring reachable space continuously; no inference pause."

        return Setpoint(buttons=buttons, stick_x=stick_x, stick_y=stick_y, reason="explore")

    def next_setpoint(self, game: GameState) -> Setpoint:
        now = time.monotonic()
        self.motor_ticks += 1
        self._finish_probe_if_ready(game, now)

        key = (game.scene, game.room, game.scene_epoch)
        if key != self.scene_key:
            self.scene_key = key
            self.scene_entered_at = now
            self.explore_yaw = float(game.player.yaw) if game.player else None
            self.explore_turns = 0

        if not game.in_game or not game.player:
            self.last_motor_summary = "Waiting for a playable game state."
            return Setpoint(reason="not_in_game")
        if game.game_over_state:
            return self._gameover_setpoint(game, now)
        if game.dialogue.active:
            return self._dialogue_setpoint(game, now)
        if game.pause_menu.active:
            return self._menu_setpoint(game, now)
        if game.cutscene_active or game.paused:
            self.last_motor_summary = "Game currently owns control; maintaining neutral input."
            return Setpoint(reason="game_owned")

        threat = self._nearest_threat(game)
        if threat is not None and (
            self.intent.mode == "combat"
            or threat.distance <= 190
            or game.target_actor is not None
        ):
            return self._combat_setpoint(game, threat, now)

        return self._navigation_setpoint(game, now)

    async def run(
        self,
        active: Callable[[], bool],
        publish: Callable[[], None],
        persist_profile: Callable[[dict], None] | None = None,
    ):
        last_persist = 0.0
        try:
            while active():
                game = self.bridge.state
                if not self.bridge.connected or game is None:
                    self.last_setpoint = Setpoint(reason="bridge_disconnected")
                    self.last_motor_summary = "Waiting for realtime bridge telemetry."
                    publish()
                    await asyncio.sleep(0.1)
                    continue
                setpoint = self.next_setpoint(game)
                self.last_setpoint = setpoint
                try:
                    self.bridge.send(
                        buttons=setpoint.buttons,
                        stick_x=setpoint.stick_x,
                        stick_y=setpoint.stick_y,
                        lease_ms=150,
                    )
                except RuntimeError:
                    # Native lease expiry guarantees neutral input if the bridge
                    # becomes stale between two motor ticks.
                    pass
                publish()
                now = time.monotonic()
                if self._profile_dirty and persist_profile and now - last_persist >= 1.0:
                    persist_profile(self.learning_snapshot())
                    self._profile_dirty = False
                    last_persist = now
                await asyncio.sleep(self.tick_s)
        finally:
            if persist_profile and self._profile_dirty:
                persist_profile(self.learning_snapshot())
                self._profile_dirty = False
            self.bridge.release()

    def learning_snapshot(self) -> dict:
        profile = {
            "version": self._profile["version"],
            "buttons": {},
            "updated_at": self._profile["updated_at"],
        }
        for name in BUTTON_ORDER:
            row = self._profile["buttons"][name]
            profile["buttons"][name] = {
                "samples": row["samples"],
                "effects": dict(row["effects"]),
            }
        profile["roles"] = {
            role: self._learned_button(role)
            for role in ROLE_EFFECTS
        }
        return profile

    def telemetry(self) -> dict:
        return {
            "intent": self.intent.model_dump(),
            "intent_age_ms": round((time.monotonic() - self.intent_updated_at) * 1000),
            "motor": self.last_motor_summary,
            "motor_ticks": self.motor_ticks,
            "setpoint": {
                "buttons": self.last_setpoint.buttons,
                "button_names": [
                    name for name, mask in BUTTONS.items()
                    if self.last_setpoint.buttons & mask
                ],
                "stick_x": self.last_setpoint.stick_x,
                "stick_y": self.last_setpoint.stick_y,
                "reason": self.last_setpoint.reason,
            },
            "control_learning": self.learning_snapshot(),
        }
