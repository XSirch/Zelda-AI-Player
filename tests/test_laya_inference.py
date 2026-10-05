from types import SimpleNamespace

import pytest

from zelda_ai.laya_data import LADDER_PROFILE, PROFILE, bounded_state
from zelda_ai.laya_inference import NumericInference


def test_graph_boundary_rejects_training_unfrozen_or_wrong_profile():
    import torch

    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.numeric_telemetry = True
    model.observation_profile = PROFILE
    for training, frozen, profile in ((True, True, PROFILE), (False, False, PROFILE), (False, True, LADDER_PROFILE)):
        model.train(training)
        model.encoder.requires_grad_(not frozen)
        with pytest.raises(ValueError, match="frozen matching schema"):
            NumericInference(model, None, profile=profile)
        assert model.training == training
        assert any(p.requires_grad for p in model.encoder.parameters()) is not frozen


def test_cpu_model_is_rejected_instead_of_silent_fallback():
    import torch

    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2).requires_grad_(False)
    model.eval()
    model.numeric_telemetry = True
    model.observation_profile = PROFILE
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        NumericInference(model, None, profile=PROFILE)


def test_changed_or_unbounded_input_rejected_before_any_graph_replay():
    inference = NumericInference.__new__(NumericInference)
    inference.profile = PROFILE
    replayed = []
    inference.graph = SimpleNamespace(replay=lambda: replayed.append(True))
    # No CUDA buffer exists: every invalid state must fail before accessing it.
    with pytest.raises(ValueError, match="profile changed"):
        inference(bounded_state([[0.0] * 9], profile=LADDER_PROFILE))
    invalid = bounded_state([[0.0] * 9])
    invalid["unobserved_route"] = [[100, 0, 100]]
    with pytest.raises(ValueError, match="Unexpected fields"):
        inference(invalid)
    assert replayed == []
