import pytest
from pydantic import ValidationError

from zelda_ai.budgets import budget_reason, runtime_exhausted
from zelda_ai.models import RunConfig


def config(**changes):
    return RunConfig(provider="openrouter", model="test/model", **changes)


def metrics(**changes):
    return {"calls": 0, "unknown_usage_calls": 0, "total_tokens": 0,
            "unknown_cost_calls": 0, "known_cost_usd": 0.0, **changes}


def test_zero_limits_roundtrip_without_infinity():
    value = config(max_calls=0, max_tokens=0, max_cost_usd=0, max_runtime_s=0)
    assert RunConfig.model_validate_json(value.model_dump_json()) == value
    assert not runtime_exhausted(value, 10**12)
    def no_reservation():
        raise AssertionError("Unlimited USD must not require prices")
    assert budget_reason(value, metrics(calls=10**12, total_tokens=10**12,
        unknown_usage_calls=9, unknown_cost_calls=5, known_cost_usd=10**6), 10**12,
        no_reservation) is None


@pytest.mark.parametrize("field", ["max_calls", "max_tokens", "max_cost_usd", "max_runtime_s"])
def test_negative_limits_rejected(field):
    with pytest.raises(ValidationError):
        config(**{field: -1})


@pytest.mark.parametrize("field", ["max_calls", "max_tokens", "max_cost_usd", "max_runtime_s"])
def test_each_limit_can_be_disabled_independently(field):
    value = config(**{field: 0})
    assert getattr(value, field) == 0
    for other in {"max_calls", "max_tokens", "max_cost_usd", "max_runtime_s"} - {field}:
        assert getattr(value, other) > 0


def test_unlimited_calls_do_not_disable_tokens():
    assert budget_reason(config(max_calls=0, max_tokens=2), metrics(calls=100000,
        total_tokens=2), 0, lambda: 0) == "token_budget_reached"


def test_unlimited_tokens_do_not_disable_calls():
    assert budget_reason(config(max_calls=2, max_tokens=0), metrics(calls=2,
        total_tokens=100000, unknown_usage_calls=1), 0, lambda: 0) == "call_budget_reached"


def test_unknown_usage_only_blocks_finite_token_budget():
    assert budget_reason(config(), metrics(unknown_usage_calls=1), 0, lambda: 0).startswith("unreconciled_token")
    assert budget_reason(config(max_tokens=0), metrics(unknown_usage_calls=1), 0, lambda: 0) is None


def test_unknown_cost_only_blocks_finite_cost_budget():
    assert budget_reason(config(), metrics(unknown_cost_calls=1), 0, lambda: 0).startswith("unreconciled_openrouter")
    assert budget_reason(config(max_cost_usd=0), metrics(unknown_cost_calls=1), 0) is None


def test_reservation_and_runtime_still_apply_with_other_limits_off():
    assert budget_reason(config(max_calls=0, max_tokens=0), metrics(), 0, lambda: 3) == "cost_budget_reservation_exceeded"
    assert budget_reason(config(max_calls=0, max_tokens=0, max_cost_usd=0), metrics(), 43200) == "runtime_budget_reached"


def test_per_response_output_is_not_an_unlimited_run_budget():
    with pytest.raises(ValidationError):
        config(max_output_tokens=0)
