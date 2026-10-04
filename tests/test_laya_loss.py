import pytest
import torch

from zelda_ai.laya_training import supervised_stick_loss


def test_physical_loss_distinguishes_near_and_opposite_wrong_stick_values():
    near, opposite = torch.full((2, 9), -3.0), torch.full((2, 9), -3.0)
    near[:, 6], opposite[:, 2] = 3.0, 3.0  # 40 versus -40; real reference is 20.
    labels = torch.tensor([5, 5])
    reference = [[20.0, 20.0]]
    assert torch.allclose(
        supervised_stick_loss(near, labels, reference, mode="choice"),
        supervised_stick_loss(opposite, labels, reference, mode="choice"),
    )
    assert supervised_stick_loss(near, labels, reference, mode="stick_mse") < supervised_stick_loss(
        opposite, labels, reference, mode="stick_mse"
    )


def test_physical_loss_learns_the_unquantized_consumed_action_through_probabilities():
    logits = torch.zeros((2, 9), requires_grad=True)
    labels = torch.tensor([5, 4])
    reference = [[15.0, -7.0]]  # Actual native values, not the rounded choice labels.
    before = supervised_stick_loss(logits, labels, reference, mode="stick_mse")
    before.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
    assert logits.grad[0, 8] < logits.grad[0, 0]
    assert logits.grad[1, 0] < logits.grad[1, 8]
    with torch.no_grad():
        logits -= logits.grad
    assert supervised_stick_loss(logits, labels, reference, mode="stick_mse") < before
    assert supervised_stick_loss(logits, labels, [[20.0, 0.0]], mode="stick_mse") != before


@pytest.mark.parametrize("reference", ([[81, 0]], [[float("nan"), 0]], [[0]], [[0, 0, 0]]))
def test_physical_loss_rejects_invalid_or_misaligned_native_targets(reference):
    with pytest.raises(ValueError):
        supervised_stick_loss(
            torch.zeros((2, 9)), torch.zeros(2, dtype=torch.long), reference, mode="stick_mse"
        )


def test_physical_loss_rejects_unknown_objective_or_nonfinite_logits():
    with pytest.raises(ValueError):
        supervised_stick_loss(torch.zeros((2, 9)), torch.zeros(2, dtype=torch.long), [[0, 0]], mode="unknown")
    logits = torch.zeros((2, 9))
    logits[0, 0] = float("nan")
    with pytest.raises(ValueError):
        supervised_stick_loss(logits, torch.zeros(2, dtype=torch.long), [[0, 0]], mode="stick_mse")
