import math
from pathlib import Path

import pytest

from zelda_ai.autonomy.features import BASE_FEATURE_DIM, FEATURE_DIM, STACK_FRAMES, encode_state, goal_guidance, stack_frames
from zelda_ai.autonomy.ml_policy import OnlinePPO
from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.autonomy.reward import RewardTracker
from zelda_ai.models import EquipmentObservation, GameEvent, InventoryObservation, NavigationProbe


def test_agent_intent_schema_is_strict_and_not_a_skill_contract():
    schema = AgentIntent.model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == set(schema["properties"])
    assert "skill" not in schema["properties"]
    assert "buttons" not in schema["properties"]
    assert "stick_x" not in schema["properties"]
    target = schema["properties"]["target_position"]["anyOf"][0]
    assert target["type"] == "array"
    assert target["items"] == {"type": "number"}
    assert target["minItems"] == target["maxItems"] == 3
    assert "prefixItems" not in target


def test_uid_only_actor_target_stays_navigate():
    intent = AgentIntent(
        objective="Approach observed actor",
        summary="Approach observed actor",
        mode="navigate",
        target_actor_uid="actor-123",
        horizon_ms=10000,
    )
    assert intent.mode == "navigate"
    assert intent.target_actor_uid == "actor-123"


def test_structured_state_is_stacked_for_temporal_policy(state):
    intent = AgentIntent.bootstrap()
    frame = encode_state(state, intent)
    assert len(frame) == BASE_FEATURE_DIM
    stacked = stack_frames([frame])
    assert len(stacked) == FEATURE_DIM == BASE_FEATURE_DIM * STACK_FRAMES
    assert stacked[-BASE_FEATURE_DIM:] == frame
    assert all(value == 0.0 for value in stacked[:-BASE_FEATURE_DIM])


def test_reward_uses_observed_novelty_without_scripted_route(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    first = tracker.step(state, intent, intrinsic=1.0, pressed_buttons=0)
    assert first.reward > 0
    moved = state.model_copy(deep=True)
    moved.player.position = (140.0, 0.0, 0.0)
    second = tracker.step(moved, intent, intrinsic=0.5, pressed_buttons=0)
    assert second.breakdown.get("new_space", 0) > 0
    assert "checkpoint" not in second.breakdown


def test_raw_controller_source_has_no_legacy_skill_dispatch():
    text = Path("src/zelda_ai/autonomy/controller.py").read_text(encoding="utf-8")
    assert "execute_skill" not in text
    assert "fight_enemy" not in text
    assert "navigate_to" not in text
    assert "explore_area" not in text
    assert "BUTTON_MASKS" in text
    assert "stick_x" in text and "stick_y" in text



def test_goal_guidance_is_camera_relative(state):
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (0.0, 0.0, 300.0),
    })
    state.camera_input_yaw = 0
    guidance = goal_guidance(state, intent)
    assert guidance["active"] is True
    assert abs(guidance["stick"][0]) < 0.05
    assert guidance["stick"][1] > 0.95
    assert guidance["strength"] > 0.8

    east = intent.model_copy(update={"target_position": (300.0, 0.0, 0.0)})
    east_guidance = goal_guidance(state, east)
    assert east_guidance["stick"][0] > 0.95
    assert abs(east_guidance["stick"][1]) < 0.05

    # Rotate the camera 90 degrees: world +X becomes camera-forward.
    state.camera_input_yaw = 0x4000
    rotated = goal_guidance(state, east)
    assert abs(rotated["stick"][0]) < 0.05
    assert rotated["stick"][1] > 0.95


def test_goal_guidance_uses_view_heading_when_input_yaw_is_missing(state):
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (300.0, 0.0, 0.0),
    })
    state.camera_input_yaw = None
    state.camera_eye = (-100.0, 100.0, 0.0)
    state.camera_at = (0.0, 0.0, 0.0)

    guidance = goal_guidance(state, intent)

    # View forward is world +X, so the eastward world target is stick-forward.
    assert guidance["active"] is True
    assert abs(guidance["stick"][0]) < 0.05
    assert guidance["stick"][1] > 0.95
    assert guidance["world_yaw"] == pytest.approx(math.pi / 2.0)


def test_goal_guidance_detours_around_observed_wall(state):
    state.navigation_probes = [
        NavigationProbe(
            direction="forward",
            distance=70.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=True,
            wall_distance=35.0,
        ),
        NavigationProbe(
            direction="forward_right",
            distance=70.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
        NavigationProbe(
            direction="right",
            distance=70.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
    ]
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (0.0, 0.0, 500.0),
    })

    guidance = goal_guidance(state, intent)

    assert guidance["active"] is True
    assert guidance["blocked"] is True
    assert guidance["direct_probe"] == "forward"
    assert guidance["detour"] == "forward_right"
    assert "detour:forward_right" in guidance["source"]
    assert 0.5 < guidance["strength"] < 0.86


def test_goal_guidance_fades_unreachable_target_after_long_dwell(state):
    blocked = []
    for direction in (
        "forward", "forward_right", "right", "back_right",
        "back", "back_left", "left", "forward_left",
    ):
        blocked.append(NavigationProbe(
            direction=direction,
            distance=70.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=True,
            wall_distance=25.0,
        ))
    state.navigation_probes = blocked
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (0.0, 0.0, 500.0),
    })

    guidance = goal_guidance(state, intent, local_dwell_seconds=400.0)

    assert guidance["blocked"] is True
    assert guidance["detour"] is None
    assert guidance["stuck_scale"] == pytest.approx(0.25)
    assert guidance["strength"] < 0.1
    assert guidance["button_quiet"] < 0.05


def test_goal_guidance_keeps_explicit_climbable_wall_target(state):
    from zelda_ai.models import TraversalAffordanceObservation

    state.navigation_probes = [
        NavigationProbe(
            direction="forward",
            distance=70.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=True,
            wall_distance=30.0,
        ),
        NavigationProbe(
            direction="forward_right",
            distance=70.0,
            floor_found=True,
            floor_y=0.0,
            delta_y=0.0,
            wall_hit=False,
        ),
    ]
    state.traversal_affordances = [
        TraversalAffordanceObservation(
            kind="climbable_wall_up",
            direction="up",
            approach_position=(0.0, 0.0, 40.0),
            target_position=(0.0, 100.0, 70.0),
            distance=40.0,
            height_delta=100.0,
            wall_flags=1,
        )
    ]
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "explore",
        "direction": "up",
    })

    guidance = goal_guidance(state, intent)

    assert guidance["source"] == "traversal:climbable_wall_up:target"
    assert guidance["blocked"] is False
    assert guidance["detour"] is None
    assert guidance["stick"][1] > 0.9


def test_goal_guidance_replays_learned_route_waypoint(state):
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (0.0, 0.0, 500.0),
    })
    state.camera_input_yaw = 0

    guidance = goal_guidance(
        state,
        intent,
        route_hint={
            "waypoint": (160.0, 0.0, 0.0),
            "path_nodes": 7,
            "target_gap": 20.0,
            "confidence": 0.75,
        },
    )

    assert guidance["route_active"] is True
    assert guidance["source"] == "learned_route"
    assert guidance["route_path_nodes"] == 7
    assert guidance["route_confidence"] == pytest.approx(0.75)
    assert guidance["route_target_gap"] == pytest.approx(20.0)
    assert guidance["stick"][0] > 0.95
    assert abs(guidance["stick"][1]) < 0.05
    # UI distance remains the final objective distance, not just the next hop.
    assert guidance["distance"] == pytest.approx(500.0)


def test_observed_door_escape_gets_full_steering_authority(state):
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "explore",
        "target_position": None,
        "direction": None,
    })
    state.camera_input_yaw = 0

    guidance = goal_guidance(
        state,
        intent,
        local_dwell_seconds=120.0,
        route_hint={
            "waypoint": (160.0, 0.0, 0.0),
            "waypoint_id": "door:85:0:house-door",
            "path_nodes": 1,
            "target_gap": 0.0,
            "confidence": 0.85,
            "partial": True,
            "exit": True,
            "door": True,
            "forced_escape": True,
            "exit_position": (160.0, 0.0, 0.0),
            "direct_reachable": False,
        },
    )

    assert guidance["active"] is True
    assert guidance["source"] == "observed_door"
    assert guidance["exit_active"] is True
    assert guidance["strength"] == pytest.approx(1.0)
    assert guidance["button_quiet"] == pytest.approx(0.95)
    assert guidance["stick"][0] > 0.95
    assert abs(guidance["stick"][1]) < 0.05


def test_partial_learned_route_does_not_collapse_under_long_dwell(state):
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (0.0, 0.0, 1000.0),
    })
    state.camera_input_yaw = 0

    guidance = goal_guidance(
        state,
        intent,
        local_dwell_seconds=400.0,
        route_hint={
            "waypoint": (80.0, 0.0, 0.0),
            "waypoint_id": "normal:child:85:0:1:0:0",
            "path_nodes": 22,
            "target_gap": 500.0,
            "confidence": 0.25,
            "partial": True,
        },
    )

    # Screenshot regression: the old generic proximity+dwell fade reduced this
    # to roughly 0.05, handing ~95% of stick authority back to PPO residual.
    expected = (0.78 + 0.16 * 0.25) * 0.85
    assert guidance["route_active"] is True
    assert guidance["route_partial"] is True
    assert guidance["stuck_scale"] == pytest.approx(1.0)
    assert guidance["strength"] == pytest.approx(expected)
    assert guidance["strength"] > 0.65


def test_targetless_explore_uses_observed_frontier_guidance(state):
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "explore",
        "target_position": None,
        "direction": None,
    })
    state.camera_input_yaw = 0

    guidance = goal_guidance(
        state,
        intent,
        route_hint={
            "waypoint": (0.0, 0.0, 140.0),
            "waypoint_id": "frontier-node",
            "path_nodes": 1,
            "confidence": 0.5,
            "partial": True,
            "frontier": True,
            "direction": "forward",
        },
    )

    assert guidance["active"] is True
    assert guidance["source"] == "observed_frontier"
    assert guidance["frontier_active"] is True
    assert guidance["frontier_direction"] == "forward"
    assert guidance["route_active"] is False
    assert guidance["stick"][1] > 0.95
    assert guidance["strength"] > 0.5


def test_targetless_explore_prioritizes_observed_scene_exit(state):
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "explore",
        "target_position": None,
        "direction": None,
    })
    state.camera_input_yaw = 0

    guidance = goal_guidance(
        state,
        intent,
        route_hint={
            "waypoint": (0.0, 0.0, 120.0),
            "waypoint_id": "exit:85:0:1:10",
            "path_nodes": 1,
            "confidence": 1.0,
            "partial": False,
            "exit": True,
            "exit_index": 1,
            "entrance_index": 10,
            "exit_position": (0.0, 0.0, 120.0),
            "direct_reachable": True,
        },
    )

    assert guidance["active"] is True
    assert guidance["source"] == "scene_exit"
    assert guidance["exit_active"] is True
    assert guidance["exit_index"] == 1
    assert guidance["exit_direct_reachable"] is True
    assert guidance["stick"][1] > 0.95
    assert guidance["strength"] == pytest.approx(0.92)
    assert guidance["stuck_scale"] == pytest.approx(1.0)
    assert guidance["button_quiet"] < 0.2

    late_guidance = goal_guidance(
        state,
        intent,
        local_dwell_seconds=400.0,
        route_hint={
            "waypoint": (0.0, 0.0, 120.0),
            "waypoint_id": "exit:85:0:1:10",
            "path_nodes": 1,
            "confidence": 1.0,
            "partial": False,
            "exit": True,
            "exit_index": 1,
            "entrance_index": 10,
            "exit_position": (0.0, 0.0, 120.0),
            "direct_reachable": True,
        },
    )
    assert late_guidance["strength"] == pytest.approx(0.92)
    assert late_guidance["stuck_scale"] == pytest.approx(1.0)


def test_active_dialogue_suppresses_navigation_guidance(state):
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "explore",
        "target_position": None,
        "direction": None,
    })
    state.dialogue.active = True
    state.dialogue.can_advance = True

    guidance = goal_guidance(
        state,
        intent,
        route_hint={
            "waypoint": (0.0, 0.0, 140.0),
            "waypoint_id": "frontier-node",
            "path_nodes": 1,
            "confidence": 0.5,
            "partial": True,
            "frontier": True,
            "direction": "forward",
        },
    )

    assert guidance["active"] is False
    assert guidance["source"] == "dialogue_hold"
    assert guidance["stick"] == (0.0, 0.0)
    assert guidance["frontier_active"] is False


def test_goal_guidance_uses_observed_vertical_traversal(state):
    from zelda_ai.models import TraversalAffordanceObservation

    state.traversal_affordances = [
        TraversalAffordanceObservation(
            kind="stairs_or_slope_up",
            direction="up",
            approach_position=(0.0, 0.0, 160.0),
            target_position=(0.0, 70.0, 260.0),
            distance=160.0,
            height_delta=70.0,
            wall_flags=0,
        )
    ]
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "explore",
        "direction": "up",
    })
    guidance = goal_guidance(state, intent)
    assert guidance["active"] is True
    assert guidance["source"].startswith("traversal:stairs_or_slope_up")
    assert guidance["stick"][1] > 0.9


def test_goal_prior_moves_deterministic_distribution_toward_target(tmp_path):
    policy = OnlinePPO(tmp_path / "policy.pt", epochs=1, minibatch_size=8)
    observation = [0.0] * FEATURE_DIM

    forward = policy.sample(
        observation,
        deterministic=True,
        guidance_stick=(0.0, 1.0),
        guidance_strength=0.9,
        button_quiet_strength=0.8,
    )
    backward = policy.sample(
        observation,
        deterministic=True,
        guidance_stick=(0.0, -1.0),
        guidance_strength=0.9,
        button_quiet_strength=0.8,
    )
    assert forward["stick"][1] > backward["stick"][1]
    assert forward["guidance_strength"] == pytest.approx(0.9)
    assert forward["button_quiet_strength"] == pytest.approx(0.8)


def test_guidance_mixes_with_residual_stick_instead_of_being_a_soft_prior(tmp_path):
    policy = OnlinePPO(tmp_path / "policy.pt", epochs=1, minibatch_size=8)
    observation = [0.0] * FEATURE_DIM

    free = policy.sample(observation, deterministic=True)
    guided = policy.sample(
        observation,
        deterministic=True,
        guidance_stick=(0.0, 1.0),
        guidance_strength=0.9,
    )

    assert guided["policy_stick"] == free["policy_stick"]
    expected_y = round(
        (free["policy_stick"][1] * 0.1 + 1.0 * 0.9) * 80.0
    ) / 80.0
    assert guided["stick"][1] == pytest.approx(expected_y)
    assert guided["guidance_strength"] == pytest.approx(0.9)


def test_button_exploration_entropy_decays_to_zero(tmp_path):
    policy = OnlinePPO(
        tmp_path / "policy.pt",
        epochs=1,
        minibatch_size=8,
        exploration_decay_samples=100,
    )
    stick_start, button_start, decay_start = policy._exploration_coefficients()
    assert stick_start == pytest.approx(0.003)
    assert button_start == pytest.approx(0.001)
    assert decay_start == pytest.approx(0.0)

    policy.samples_trained = 100
    stick_end, button_end, decay_end = policy._exploration_coefficients()
    assert stick_end == pytest.approx(0.0003)
    assert button_end == pytest.approx(0.0)
    assert decay_end == pytest.approx(1.0)


def test_deterministic_policy_action_is_repeatable(tmp_path):
    policy = OnlinePPO(tmp_path / "policy.pt", epochs=1, minibatch_size=8)
    observation = [0.0] * FEATURE_DIM
    first = policy.sample(observation, deterministic=True)
    second = policy.sample(observation, deterministic=True)
    assert first["stick"] == second["stick"]
    assert first["buttons"] == second["buttons"]


@pytest.mark.parametrize("button_count", [9])
def test_online_ppo_samples_and_updates_checkpoint(tmp_path, button_count):
    policy = OnlinePPO(
        tmp_path / "policy.pt",
        epochs=1,
        minibatch_size=8,
    )
    observation = [0.0] * FEATURE_DIM
    rollout = []
    for index in range(16):
        sample = policy.sample(
            observation,
            guidance_stick=(0.0, 1.0),
            guidance_strength=0.7,
            button_quiet_strength=0.5,
        )
        assert len(sample["stick"]) == 2
        assert len(sample["buttons"]) == button_count
        rollout.append({
            "observation": observation,
            "stick": sample["stick"],
            "policy_stick": sample["policy_stick"],
            "buttons": sample["buttons"],
            "log_prob": sample["log_prob"],
            "value": sample["value"],
            "guidance_stick": sample["guidance_stick"],
            "guidance_strength": sample["guidance_strength"],
            "button_quiet_strength": sample["button_quiet_strength"],
            "reward": 0.05 + index * 0.001,
            "done": False,
        })
    stats = policy.train_rollout(rollout, bootstrap_value=0.0, bootstrap_done=False)
    assert stats["updates"] == 1
    assert stats["samples_trained"] == 16
    assert "stick_entropy" in stats
    assert "button_entropy" in stats
    assert "expected_button_count" in stats
    assert stats["button_entropy_coef"] <= 0.001
    assert (tmp_path / "policy.pt").is_file()



def test_kokiri_sword_is_objective_score_not_unbounded_ppo_reward(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)

    sword = state.model_copy(deep=True)
    sword.progress.equipment = [
        EquipmentObservation(
            item_id=59,
            name="Kokiri Sword",
            equipment_type="sword",
            value=1,
            equipped=True,
        )
    ]
    sword.progress.owned_equipment = ["Kokiri Sword"]
    result = tracker.step(sword, intent, intrinsic=0.0, pressed_buttons=0)

    achievement = next(row for row in result.achievements if row["title"] == "Kokiri Sword")
    assert achievement["points"] == 100
    assert achievement["training_reward"] == 3.0
    assert tracker.objective_score == 100
    assert result.breakdown["objective_milestone"] == 3.0
    assert result.reward <= 5.0


def test_existing_save_progress_is_baseline_not_free_achievement(state):
    sword = state.model_copy(deep=True)
    sword.progress.equipment = [
        EquipmentObservation(
            item_id=59,
            name="Kokiri Sword",
            equipment_type="sword",
            value=1,
            equipped=True,
        )
    ]
    sword.progress.owned_equipment = ["Kokiri Sword"]
    tracker = RewardTracker()
    first = tracker.step(sword, AgentIntent.bootstrap(), intrinsic=0.0, pressed_buttons=0)
    assert first.achievements == []
    assert tracker.objective_score == 0



def test_consumable_reacquisition_does_not_farm_objective_score(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()

    with_item = state.model_copy(deep=True)
    with_item.inventory_named = [
        InventoryObservation(slot=0, item_id=1, name="Deku Stick", ammo=1)
    ]
    tracker.step(with_item, intent, intrinsic=0.0, pressed_buttons=0)

    depleted = with_item.model_copy(deep=True)
    depleted.inventory_named = []
    tracker.step(depleted, intent, intrinsic=0.0, pressed_buttons=0)

    reacquired = with_item.model_copy(deep=True)
    result = tracker.step(reacquired, intent, intrinsic=0.0, pressed_buttons=0)
    assert not any(row["id"] == "item:1" for row in result.achievements)
    assert tracker.objective_score == 0



def test_rupee_reward_requires_actual_wallet_increase(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    baseline = state.model_copy(deep=True)
    baseline.player.rupees = 20
    tracker.step(baseline, intent, intrinsic=0.0, pressed_buttons=0)

    richer = baseline.model_copy(deep=True)
    richer.player.rupees = 25
    gained = tracker.step(richer, intent, intrinsic=0.0, pressed_buttons=0)
    assert gained.breakdown["resource_rupees"] > 0
    assert tracker.rupees_collected == 5

    full_or_unchanged = richer.model_copy(deep=True)
    unchanged = tracker.step(
        full_or_unchanged,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
    )
    assert "resource_rupees" not in unchanged.breakdown
    assert tracker.rupees_collected == 5


def test_ammo_reward_requires_actual_ammo_increase(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    baseline = state.model_copy(deep=True)
    baseline.inventory_named = [
        InventoryObservation(slot=0, item_id=1, name="Deku Stick", ammo=5)
    ]
    tracker.step(baseline, intent, intrinsic=0.0, pressed_buttons=0)

    refilled = baseline.model_copy(deep=True)
    refilled.inventory_named[0].ammo = 8
    gained = tracker.step(refilled, intent, intrinsic=0.0, pressed_buttons=0)
    assert gained.breakdown["resource_ammo"] > 0
    assert tracker.ammo_collected == 3

    full_pickup = refilled.model_copy(deep=True)
    unchanged = tracker.step(
        full_pickup,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
    )
    assert "resource_ammo" not in unchanged.breakdown
    assert tracker.ammo_collected == 3


def test_new_item_does_not_double_count_initial_ammo(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)

    acquired = state.model_copy(deep=True)
    acquired.inventory_named = [
        InventoryObservation(slot=0, item_id=1, name="Deku Stick", ammo=10)
    ]
    result = tracker.step(acquired, intent, intrinsic=0.0, pressed_buttons=0)
    assert result.breakdown["objective_milestone"] > 0
    assert "resource_ammo" not in result.breakdown
    assert tracker.ammo_collected == 0


def test_health_and_magic_pickups_only_reward_observed_recovery(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    baseline = state.model_copy(deep=True)
    baseline.player.health = 32
    baseline.player.magic = 20
    tracker.step(baseline, intent, intrinsic=0.0, pressed_buttons=0)

    recovered = baseline.model_copy(deep=True)
    recovered.player.health = 48
    recovered.player.magic = 35
    result = tracker.step(recovered, intent, intrinsic=0.0, pressed_buttons=0)
    assert result.breakdown["resource_health"] > 0
    assert result.breakdown["resource_magic"] > 0
    assert tracker.health_recovered == 16
    assert tracker.magic_recovered == 15

    unchanged = tracker.step(
        recovered.model_copy(deep=True),
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
    )
    assert "resource_health" not in unchanged.breakdown
    assert "resource_magic" not in unchanged.breakdown


def test_chest_opened_event_rewards_once(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)

    opened = state.model_copy(deep=True)
    opened.events = [
        GameEvent(id="chest-1", kind="chest_opened", detail="85:3")
    ]
    result = tracker.step(opened, intent, intrinsic=0.0, pressed_buttons=0)
    assert result.breakdown["native_event"] == 0.6
    assert tracker.chests_opened == 1
    assert any(row["title"] == "Baú aberto" for row in result.achievements)

    repeated = tracker.step(
        opened.model_copy(deep=True),
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
    )
    assert repeated.breakdown.get("native_event", 0) == 0
    assert tracker.chests_opened == 1



def test_dungeon_item_and_max_health_gain_are_persistent_rewards(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)

    progressed = state.model_copy(deep=True)
    progressed.progress.dungeon_items = ["Dungeon Map"]
    progressed.player.max_health = state.player.max_health + 16
    progressed.player.health = min(progressed.player.max_health, state.player.health + 16)
    result = tracker.step(
        progressed,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
    )

    assert result.breakdown["durable_progress"] == 2.0
    assert result.breakdown["objective_milestone"] >= 4.0
    titles = {row["title"] for row in result.achievements}
    assert "Dungeon Map" in titles
    assert "Capacidade de vida aumentada" in titles


def test_native_game_completed_event_ends_episode_and_scores_achievement(state):
    tracker = RewardTracker()
    tracker.step(state, AgentIntent.bootstrap(), intrinsic=0.0, pressed_buttons=0)
    completed = state.model_copy(deep=True)
    completed.events = [GameEvent(id="1", kind="game_completed", detail="final_ganon_defeated")]
    result = tracker.step(completed, AgentIntent.bootstrap(), intrinsic=0.0, pressed_buttons=0)
    assert result.done is True
    assert any(row["points"] == 5000 for row in result.achievements)




def test_vertical_movement_counts_as_new_space(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)

    lower_floor = state.model_copy(deep=True)
    lower_floor.player.position = (
        state.player.position[0],
        state.player.position[1] - 90.0,
        state.player.position[2],
    )
    result = tracker.step(lower_floor, intent, intrinsic=0.0, pressed_buttons=0)
    assert result.breakdown.get("new_space", 0) > 0
    assert "stagnation" not in result.breakdown




def test_blocked_guidance_suppresses_straight_line_intent_reward(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (400.0, 0.0, 0.0),
    })
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0, now_s=0.0)

    moved = state.model_copy(deep=True)
    moved.player.position = (20.0, 0.0, 0.0)
    result = tracker.step(
        moved,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        guidance={"blocked": True, "detour": "forward_right"},
        now_s=20.0,
    )

    assert "intent_progress" not in result.breakdown


def test_learned_route_waypoint_rewards_once_without_resetting_coarse_dwell(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (400.0, 0.0, 0.0),
    })
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0, now_s=0.0)

    reached = state.model_copy(deep=True)
    reached.player.position = (80.0, 0.0, 0.0)
    guidance = {
        "route_active": True,
        "blocked": False,
        "route_waypoint_id": "85:0:1:0:0",
        "route_waypoint": [80.0, 0.0, 0.0],
    }
    first = tracker.step(
        reached,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        guidance=guidance,
        now_s=120.0,
    )

    assert first.breakdown["route_waypoint"] == pytest.approx(0.12)
    assert tracker.route_waypoint_advanced is True
    assert tracker.local_dwell_seconds == 120.0
    assert first.breakdown.get("local_dwell", 0.0) < 0

    repeated = reached.model_copy(deep=True)
    repeated.seq += 1
    second = tracker.step(
        repeated,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        guidance=guidance,
        now_s=130.0,
    )
    assert "route_waypoint" not in second.breakdown


def test_replayed_route_waypoint_resets_dwell_without_repaying_reward(state):
    tracker = RewardTracker()
    first_intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (400.0, 0.0, 0.0),
    })
    tracker.step(state, first_intent, intrinsic=0.0, pressed_buttons=0, now_s=0.0)

    waypoint = state.model_copy(deep=True)
    waypoint.player.position = (80.0, 0.0, 0.0)
    guidance = {
        "route_active": True,
        "blocked": False,
        "route_waypoint_id": "normal:child:85:0:1:0:0",
        "route_waypoint": [80.0, 0.0, 0.0],
    }
    first = tracker.step(
        waypoint,
        first_intent,
        intrinsic=0.0,
        pressed_buttons=0,
        guidance=guidance,
        now_s=100.0,
    )
    assert first.breakdown["route_waypoint"] == pytest.approx(0.12)
    assert tracker.route_waypoint_advanced is True

    # Change strategic objective so a later traversal of the same learned node
    # starts a new route episode. It may reset dwell again, but must not pay PPO
    # reward a second time.
    second_intent = first_intent.model_copy(update={
        "target_position": (500.0, 0.0, 0.0),
    })
    reset = waypoint.model_copy(deep=True)
    reset.seq += 1
    tracker.step(
        reset,
        second_intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=220.0,
    )

    replay = reset.model_copy(deep=True)
    replay.seq += 1
    replay_result = tracker.step(
        replay,
        second_intent,
        intrinsic=0.0,
        pressed_buttons=0,
        guidance=guidance,
        now_s=350.0,
    )

    assert "route_waypoint" not in replay_result.breakdown
    assert tracker.route_waypoint_advanced is True
    assert tracker.route_waypoints_advanced == 2
    assert tracker.local_dwell_seconds > 0.0
    assert replay_result.breakdown.get("local_dwell", 0.0) < 0


def test_learned_route_suppresses_straight_line_intent_reward(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (400.0, 0.0, 0.0),
    })
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0, now_s=0.0)

    detouring = state.model_copy(deep=True)
    detouring.player.position = (0.0, 0.0, 40.0)
    result = tracker.step(
        detouring,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        guidance={"route_active": True, "blocked": False},
        now_s=20.0,
    )

    assert "intent_progress" not in result.breakdown


def test_stale_waypoint_stops_rewarding_return_to_same_target(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "navigate",
        "target_position": (400.0, 0.0, 0.0),
    })

    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0, now_s=0.0)

    early = state.model_copy(deep=True)
    early.player.position = (10.0, 0.0, 0.0)
    early_result = tracker.step(
        early, intent, intrinsic=0.0, pressed_buttons=0, now_s=30.0
    )
    assert early_result.breakdown.get("intent_progress", 0.0) > 0

    stale = state.model_copy(deep=True)
    stale.player.position = (20.0, 0.0, 0.0)
    stale_result = tracker.step(
        stale, intent, intrinsic=0.0, pressed_buttons=0, now_s=200.0
    )
    assert "intent_progress" not in stale_result.breakdown
    assert stale_result.breakdown.get("local_dwell", 0.0) < 0


def test_frontier_progress_rewards_only_new_outward_radius(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(
        state,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )

    outward = state.model_copy(deep=True)
    outward.player.position = (40.0, 0.0, 0.0)
    first = tracker.step(
        outward,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=1.0,
    )
    assert first.breakdown.get("frontier_progress", 0) > 0
    best = tracker.local_frontier_radius

    circle = outward.model_copy(deep=True)
    circle.player.position = (0.0, 0.0, 40.0)
    repeated_radius = tracker.step(
        circle,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=2.0,
    )
    assert "frontier_progress" not in repeated_radius.breakdown
    assert tracker.local_frontier_radius == best

    farther = circle.model_copy(deep=True)
    farther.player.position = (90.0, 0.0, 0.0)
    improved = tracker.step(
        farther,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=3.0,
    )
    assert improved.breakdown.get("frontier_progress", 0) > 0
    assert tracker.local_frontier_radius > best


def test_new_macro_region_gets_strong_exploration_reward(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    first = tracker.step(
        state,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )
    assert "new_macro_region" not in first.breakdown

    escaped = state.model_copy(deep=True)
    escaped.player.position = (300.0, 0.0, 0.0)
    result = tracker.step(
        escaped,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=1.0,
    )
    assert result.breakdown["new_macro_region"] == 0.8
    assert len(tracker.seen_macro_regions) == 2


def test_macro_region_grid_does_not_split_tiny_moves_across_zero(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(
        state,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )

    tiny_negative = state.model_copy(deep=True)
    tiny_negative.player.position = (-10.0, 0.0, -10.0)
    result = tracker.step(
        tiny_negative,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=1.0,
    )
    assert "new_macro_region" not in result.breakdown
    assert len(tracker.seen_macro_regions) == 1


def test_intent_target_progress_has_meaningful_dense_weight(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap().model_copy(
        update={
            "mode": "navigate",
            "target_position": (300.0, 0.0, 0.0),
        }
    )
    tracker.step(
        state,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )

    closer = state.model_copy(deep=True)
    closer.player.position = (30.0, 0.0, 0.0)
    result = tracker.step(
        closer,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=1.0,
    )
    assert result.breakdown["intent_progress"] >= 0.19


def test_long_local_dwell_penalizes_circling_even_when_link_keeps_moving(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(
        state,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )

    nearby = state.model_copy(deep=True)
    nearby.player.position = (30.0, 0.0, 30.0)
    early = tracker.step(
        nearby,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=30.0,
    )
    assert "local_dwell" not in early.breakdown

    same_area = nearby.model_copy(deep=True)
    same_area.player.position = (45.0, 0.0, 25.0)
    pressured = tracker.step(
        same_area,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=90.0,
    )
    assert pressured.breakdown["local_dwell"] < 0
    first_penalty = pressured.breakdown["local_dwell"]

    still_circling = same_area.model_copy(deep=True)
    still_circling.player.position = (80.0, 0.0, 10.0)
    late = tracker.step(
        still_circling,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=300.0,
    )
    assert late.breakdown["local_dwell"] < first_penalty
    assert late.breakdown["local_dwell"] >= -0.020001
    assert tracker.local_dwell_seconds == 300.0


def test_local_dwell_resets_after_real_spatial_expansion(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(
        state,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )

    circling = state.model_copy(deep=True)
    circling.player.position = (20.0, 0.0, 20.0)
    pressured = tracker.step(
        circling,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=120.0,
    )
    assert pressured.breakdown["local_dwell"] < 0

    escaped = state.model_copy(deep=True)
    escaped.player.position = (600.0, 0.0, 0.0)
    result = tracker.step(
        escaped,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=130.0,
    )
    assert "local_dwell" not in result.breakdown
    assert tracker.local_dwell_seconds == 0.0
    assert tracker.local_anchor_distance == 0.0



def test_consuming_progress_resource_does_not_clear_local_dwell(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    with_key = state.model_copy(deep=True)
    with_key.progress.small_keys = 1
    tracker.step(
        with_key,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )

    circling = with_key.model_copy(deep=True)
    circling.player.position = (30.0, 0.0, 30.0)
    pressured = tracker.step(
        circling,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=120.0,
    )
    assert pressured.breakdown["local_dwell"] < 0

    spent_key = circling.model_copy(deep=True)
    spent_key.progress.small_keys = 0
    still_pressured = tracker.step(
        spent_key,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=130.0,
    )
    assert "durable_progress" not in still_pressured.breakdown
    assert still_pressured.breakdown["local_dwell"] < 0
    assert tracker.local_dwell_seconds == 130.0


def test_local_dwell_resets_on_meaningful_progress(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(
        state,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )

    circling = state.model_copy(deep=True)
    circling.player.position = (25.0, 0.0, 0.0)
    tracker.step(
        circling,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=120.0,
    )
    assert tracker.local_dwell_seconds == 120.0

    progress = circling.model_copy(deep=True)
    progress.progress.equipment = [
        EquipmentObservation(
            item_id=59,
            name="Kokiri Sword",
            equipment_type="sword",
            value=1,
            equipped=True,
        )
    ]
    progress.progress.owned_equipment = ["Kokiri Sword"]
    result = tracker.step(
        progress,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=130.0,
    )
    assert result.breakdown["objective_milestone"] == 3.0
    assert "local_dwell" not in result.breakdown
    assert tracker.local_dwell_seconds == 0.0


def test_static_state_does_not_keep_earning_reward(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)
    last = None
    for _ in range(20):
        last = tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)
    assert last is not None
    assert last.reward < 0
    assert last.breakdown.get("stagnation", 0) < 0


def test_generic_native_event_does_not_pay_ppo_reward(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)
    changed = state.model_copy(deep=True)
    changed.events = [
        GameEvent(id="99", kind="context_action_changed", detail="none->open")
    ]
    result = tracker.step(changed, intent, intrinsic=0.0, pressed_buttons=0)
    assert result.breakdown.get("native_event", 0) == 0


def test_rnd_relative_novelty_has_zero_familiar_baseline(tmp_path):
    policy = OnlinePPO(tmp_path / "rnd-policy.pt", epochs=1, minibatch_size=8)
    values = [policy._relative_novelty(1.0) for _ in range(40)]
    assert max(values) == 0.0
    assert policy._relative_novelty(2.0) > 0.0


def test_repeated_dialogue_does_not_farm_reward(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    dialogue = state.model_copy(deep=True)
    dialogue.dialogue.active = True
    dialogue.dialogue.text_id = 77
    dialogue.dialogue.text = "Same observed dialogue"

    first = tracker.step(dialogue, intent, intrinsic=0.0, pressed_buttons=0)
    assert first.breakdown.get("new_dialogue", 0) > 0

    closed = dialogue.model_copy(deep=True)
    closed.dialogue.active = False
    tracker.step(closed, intent, intrinsic=0.0, pressed_buttons=0)

    repeated = dialogue.model_copy(deep=True)
    again = tracker.step(repeated, intent, intrinsic=0.0, pressed_buttons=0)
    assert "new_dialogue" not in again.breakdown



def test_scene_epoch_churn_is_not_world_progress_or_dwell_reset(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(
        state,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=0.0,
    )

    circling = state.model_copy(deep=True)
    circling.player.position = (30.0, 0.0, 30.0)
    tracker.step(
        circling,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=120.0,
    )

    epoch_only = circling.model_copy(deep=True)
    epoch_only.scene_epoch += 1
    result = tracker.step(
        epoch_only,
        intent,
        intrinsic=0.0,
        pressed_buttons=0,
        now_s=130.0,
    )
    assert "new_world_transition" not in result.breakdown
    assert result.breakdown["local_dwell"] < 0
    assert tracker.local_dwell_seconds == 130.0


def test_repeated_world_edge_is_not_a_progress_farm(state):
    tracker = RewardTracker()
    intent = AgentIntent.bootstrap()
    tracker.step(state, intent, intrinsic=0.0, pressed_buttons=0)

    next_room = state.model_copy(deep=True)
    next_room.seq += 1
    next_room.room = 1
    next_room.scene_epoch += 1
    first = tracker.step(next_room, intent, intrinsic=0.0, pressed_buttons=0)
    assert first.breakdown.get("new_world_transition", 0) > 0

    back = state.model_copy(deep=True)
    back.seq += 2
    back.scene_epoch += 2
    tracker.step(back, intent, intrinsic=0.0, pressed_buttons=0)

    again = next_room.model_copy(deep=True)
    again.seq += 3
    again.scene_epoch += 3
    repeated = tracker.step(again, intent, intrinsic=0.0, pressed_buttons=0)
    assert repeated.breakdown.get("repeated_transition") == 0.0
    assert "new_world_transition" not in repeated.breakdown


def test_policy_features_include_menu_and_camera_state(state):
    state.camera_input_yaw = 1234
    state.inventory = [1, 2, 0xFF]
    state.equipped = [3, 4, 0xFF, 0xFF]
    state.pause_menu.active = True
    state.pause_menu.page_index = 2
    state.pause_menu.cursor_point = [3, 0, 0, 0, 0]
    state.pause_menu.cursor_item = [2, 0xFF, 0xFF, 0xFF]
    intent = AgentIntent.bootstrap().model_copy(update={
        "mode": "menu",
        "target_item_id": 2,
        "choice_index": 1,
    })
    frame = encode_state(state, intent)
    assert len(frame) == BASE_FEATURE_DIM
    assert any(value != 0.0 for value in frame)


def test_strict_evaluation_checkpoint_is_never_quarantined(tmp_path):
    checkpoint = tmp_path / "champion.pt"
    checkpoint.write_bytes(b"not-a-valid-torch-checkpoint")
    original = checkpoint.read_bytes()
    with pytest.raises(RuntimeError, match="Evaluation checkpoint could not be loaded"):
        OnlinePPO(checkpoint, strict_checkpoint=True)
    assert checkpoint.is_file()
    assert checkpoint.read_bytes() == original
    assert not list(tmp_path.glob("champion.pt.corrupt-*"))
