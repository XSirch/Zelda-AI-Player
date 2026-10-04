import copy

import pytest

from zelda_ai.autonomy.imitation import FEATURES, VERSION, SurfacePolicy
from zelda_ai.autonomy.ml_policy import torch
from zelda_ai.surface_curriculum import aggregate


def demonstration():
    return {"source": "soh", "controller": "reference", "successful_episode": True,
            "first_tick": 40, "reference_blend": 0, "buttons": 0,
            "features": [0.] * len(FEATURES), "action": [0., .5]}


@pytest.mark.parametrize("change", [
    {"source": "simulator"}, {"first_tick": 0}, {"controller": "candidate"},
    {"successful_episode": False}, {"buttons": 0x8000}, {"reference_blend": .2},
    {"action": [float("nan"), 0]}, {"features": [1.]},
])
def test_imitation_rejects_noncausal_or_wrong_action_domain(change):
    with pytest.raises(ValueError, match="demonstration"):
        SurfacePolicy().fit([{**demonstration(), **change}], epochs=1)


def test_candidate_checkpoint_immutable_and_evaluation_read_only(tmp_path):
    policy = SurfacePolicy(seed=8)
    before = copy.deepcopy(policy.net.state_dict())
    policy.fit([demonstration()], epochs=3)
    assert any(not torch.equal(before[k], value) for k, value in policy.net.state_dict().items())
    path = tmp_path / "candidate.pt"
    policy.save(path, dataset_sha256="a" * 64)
    original = path.read_bytes()
    loaded = SurfacePolicy.load(path)
    assert loaded.training_steps == 3
    assert loaded.net.training is False
    with pytest.raises(FileExistsError):
        loaded.save(path, dataset_sha256="a" * 64)
    assert path.read_bytes() == original
    malformed = torch.load(path, weights_only=True)
    malformed["action_contract"] = "ppo_latent_residual"
    torch.save(malformed, tmp_path / "wrong.pt")
    with pytest.raises(ValueError, match="Incompatible"):
        SurfacePolicy.load(tmp_path / "wrong.pt")


def test_no_family_is_qualified_by_pilot_or_by_other_easy_families():
    good = {"family": "surface_up", "stage": "after", "success": True,
            "consumed_actions": 5, "objective_unchanged": True, "run_updates": 0,
            "reference_blend": 0, "source": "soh", "artifacts_unchanged": True}
    records = [{**good, "index": index} for index in range(99)]
    assert aggregate(records, "surface_up", "after")["g2_qualified"] is False
    records += [{**good, "family": "surface_down", "index": index} for index in range(100)]
    assert aggregate(records, "surface_up", "after")["g2_qualified"] is False
    assert aggregate(records, "surface_down", "after")["g2_qualified"] is True
    records[-1]["critical_control_bug"] = True
    assert aggregate(records, "surface_down", "after")["g2_qualified"] is False
    assert VERSION == "surface-imitation-v1"
