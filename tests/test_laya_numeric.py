import pytest

from zelda_ai.laya_data import bounded_state
from zelda_ai.laya_numeric import ENCODER_SCHEMA, numeric_forward, numeric_values
from zelda_ai.laya_training import decode_stick


def test_numeric_conditioning_uses_current_observation_and_axis_without_labels():
    current = [1.0, 0.0, 0.0, 0.4, 0.14, 0.2, 0.0, 0.0, 1.0]
    state = bounded_state([[0.0] * 9, current])
    expected = [current + [1.0, 0.0], current + [0.0, 1.0]]
    assert numeric_values([{"state": state}]) == expected
    assert numeric_values([{"state": state, "labels": [-80, -80]}]) == expected
    assert numeric_values([{"state": state, "labels": [80, 80]}]) == expected
    other = bounded_state([[-1.0, *current[1:]]])
    assert numeric_values([{"state": other}]) != expected
    assert ENCODER_SCHEMA["values"] == "numeric_conditioning"
    state["hidden_route"] = [1, 2, 3]
    with pytest.raises(ValueError):
        numeric_values([{"state": state}])


def test_cached_schema_keeps_live_values_and_head_gradients_out_of_the_cache():
    from types import SimpleNamespace

    import torch
    from torch import nn

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(16, 8)
            self.calls = 0

        def forward(self, input_ids, attention_mask):
            self.calls += 1
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = nn.Module()
        model.encoder = Encoder().requires_grad_(False)
        model.type_emb = nn.Embedding(3, 8)
        model.telemetry_adapter = nn.Linear(11, 8)
        model.head = None
        model.scorer = nn.Sequential(nn.LayerNorm(8), nn.Linear(8, 1))
        model.numeric_cache = None
        batch = {
            "input_ids": torch.tensor([[0, 1, 2, 3], [0, 4, 5, 3]]),
            "attention_mask": torch.ones((2, 4), dtype=torch.long),
            "qtype": torch.zeros(2, dtype=torch.long),
            "marker_pos": torch.tensor([[1, 2], [1, 2]]),
            "marker_mask": torch.ones((2, 2), dtype=torch.bool),
            "telemetry": torch.zeros((2, 11)),
        }
        with torch.inference_mode():
            first = numeric_forward(model, batch).clone()
        batch["telemetry"][:, 0] = 1
        second = numeric_forward(model, batch)
        assert not torch.allclose(first, second)
        second.sum().backward()
        assert model.telemetry_adapter.weight.grad.abs().sum() > 0
        assert model.encoder.calls == 1
        assert all(p.grad is None for p in model.encoder.parameters())
        batch["input_ids"][0, 0] = 9
        with pytest.raises(ValueError, match="schema"):
            numeric_forward(model, batch)


def test_continuous_decoder_uses_model_probabilities_and_checks_native_bounds():
    import torch

    logits = torch.full((1, 9), -100.0)
    logits[0, 6] = logits[0, 7] = 0  # Equal mass at 40 and 60.
    assert decode_stick(logits, "expectation").tolist() == [50]
    assert decode_stick(logits, "argmax").tolist() == [40]
    assert decode_stick(torch.zeros((2, 9)), "expectation").tolist() == [0, 0]
    with pytest.raises(ValueError):
        decode_stick(logits, "unknown")
    logits[0, 0] = float("nan")
    with pytest.raises(ValueError):
        decode_stick(logits, "expectation")
