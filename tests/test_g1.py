import copy

import pytest

from zelda_ai.g1 import (
    isolated_config,
    physical_mappings_disabled,
    portal_crossed,
    summarize,
    variation_target,
)
from zelda_ai.models import GameEvent, NavigationMeshSnapshot, SceneExitObservation


def evidence():
    manifest = {"suite_id": "g1-test", "planned_episodes": 100, "provider_calls": 0,
                "physical_device_mappings_disabled": True}
    episodes = [{"episode_id": f"g1-test:{i}", "success": True, "source": "soh",
                 "settled_crossing": True, "consumed_commands": 20, "total_elapsed_s": 30,
                 "run_updates": 0, "artifacts_unchanged": True, "setup_valid": True,
                 "objective_unchanged": True, "native_build": "rt-input-v3.2", "upstream_revision": "pinned",
                 "initial_context": ["same-process", i % 2, 0, False, "child"],
                 "initial_position": [i % 5 * 25, 0, i % 7 * 25],
                 "initial_camera_yaw": i % 2 * 16000,
                 "setup_heading_bin": i % 4, "setup_heading_bins": [i % 4], "setup_consumed_commands": 12}
                for i in range(100)]
    return manifest, episodes


def test_gate_requires_complete_varied_real_evidence():
    manifest, episodes = evidence()
    assert summarize(manifest, episodes, finished=True)["g1_qualified"]
    assert not summarize(manifest, episodes[:-1], finished=True)["g1_qualified"]
    assert not summarize(manifest, episodes, finished=False)["g1_qualified"]
    episodes[0]["success"] = False
    assert summarize(manifest, episodes, finished=True)["g1_qualified"]
    episodes[1]["success"] = False
    assert not summarize(manifest, episodes, finished=True)["g1_qualified"]


def test_qa_copy_disables_keyboard_and_gamepad_without_changing_game_options():
    original = {"CVars": {"gSettings": {"Controllers": {"Port1": {
        "Buttons": {"32768ButtonMappingIds": "keyboard,gamepad,"},
        "LeftStick": {"UpAxisDirectionMappingIds": "keyboard,gamepad,", "SensitivityPercentage": 100},
        "RightStick": {"UpAxisDirectionMappingIds": "gamepad,"}}}, "Volume": {"Master": 0}}},
        "Window": {"Width": 800}}
    qa = isolated_config(copy.deepcopy(original))
    port = qa["CVars"]["gSettings"]["Controllers"]["Port1"]
    assert port["HasConfig"] == 1
    assert physical_mappings_disabled(qa)
    assert not physical_mappings_disabled(original)
    assert port["Buttons"]["32768ButtonMappingIds"] == ""
    assert port["LeftStick"]["UpAxisDirectionMappingIds"] == ""
    assert port["RightStick"]["UpAxisDirectionMappingIds"] == ""
    assert port["LeftStick"]["SensitivityPercentage"] == 100
    assert qa["Window"] == original["Window"]
    assert original["CVars"]["gSettings"]["Controllers"]["Port1"]["Buttons"]["32768ButtonMappingIds"]


@pytest.mark.parametrize("field,value", [("source", "simulator"), ("consumed_commands", 0),
    ("settled_crossing", False), ("total_elapsed_s", 120.001), ("run_updates", 1),
    ("artifacts_unchanged", False), ("setup_valid", False)])
def test_success_boolean_cannot_replace_physical_proof(field, value):
    manifest, episodes = evidence()
    for e in episodes:
        e[field] = value
    assert not summarize(manifest, episodes, finished=True)["g1_qualified"]


def test_identical_starts_and_cameras_do_not_satisfy_variation():
    manifest, episodes = evidence()
    for e in episodes:
        e["initial_position"] = [0, 0, 0]
        e["initial_camera_yaw"] = 0
        e["setup_heading_bin"] = None
        e["setup_heading_bins"] = []
    report = summarize(manifest, episodes, finished=True)
    assert report["successful_episodes"] == 100
    assert not report["g1_qualified"]


def test_duplicate_evidence_and_mutations_cannot_qualify():
    manifest, episodes = evidence()
    episodes[-1] = copy.deepcopy(episodes[0])
    assert not summarize(manifest, episodes, finished=True)["g1_qualified"]
    manifest, episodes = evidence()
    assert not summarize(manifest, episodes, finished=True, artifacts_unchanged=False)["g1_qualified"]
    manifest["provider_calls"] = 1
    assert not summarize(manifest, episodes, finished=True)["g1_qualified"]


@pytest.mark.parametrize("problem", ["restart", "reload", "dead", "modal", "cutscene", "mirror", "age", "stale"])
def test_reset_or_interrupted_state_is_not_a_portal(state, problem):
    final = state.model_copy(deep=True)
    final.scene += 1
    final.scene_epoch += 1
    final.seq += 1
    assert portal_crossed(state, final)
    if problem == "restart":
        final.instance_id = "other-process"
    elif problem == "reload":
        final.events = [GameEvent(id="11", kind="save_loaded", detail="1")]
    elif problem == "dead":
        final.player.health = 0
    elif problem == "modal":
        final.dialogue.active = True
    elif problem == "cutscene":
        final.cutscene_active = True
    elif problem == "mirror":
        final.mirrored_world = True
    elif problem == "age":
        final.player.age = "adult"
    else:
        final.seq = state.seq
    assert not portal_crossed(state, final)


def test_variation_does_not_invent_reverse_edges_or_choose_portal_cells(state):
    state.navmesh = NavigationMeshSnapshot(step=70, half_extent=2, cells=[
        (0, 0, 0, 1 << 2), (1, 0, 0, 0), (-1, 0, 0, 1 << 2)])
    assert variation_target(state, 4) == (70, 0, 0)  # Only reachable direction, regardless of requested bin.
    state.scene_exits = [SceneExitObservation(exit_index=1, entrance_index=1, position=(70, 0, 0),
                                         samples=1, direct_reachable=True)]
    assert variation_target(state, 4) is None
