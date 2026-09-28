from pathlib import Path

import pytest

from zelda_ai.autonomy.features import BASE_FEATURE_DIM, FEATURE_DIM, STACK_FRAMES, encode_state, stack_frames
from zelda_ai.autonomy.ml_policy import OnlinePPO
from zelda_ai.autonomy.models import AgentIntent
from zelda_ai.autonomy.reward import RewardTracker
from zelda_ai.models import EquipmentObservation, GameEvent, InventoryObservation


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
        sample = policy.sample(observation)
        assert len(sample["stick"]) == 2
        assert len(sample["buttons"]) == button_count
        rollout.append({
            "observation": observation,
            "stick": sample["stick"],
            "buttons": sample["buttons"],
            "log_prob": sample["log_prob"],
            "value": sample["value"],
            "reward": 0.05 + index * 0.001,
            "done": False,
        })
    stats = policy.train_rollout(rollout, bootstrap_value=0.0, bootstrap_done=False)
    assert stats["updates"] == 1
    assert stats["samples_trained"] == 16
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
