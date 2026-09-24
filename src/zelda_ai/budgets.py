"""Independent run budgets. Zero disables enforcement, not accounting or provider limits."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .models import RunConfig


def runtime_exhausted(config: RunConfig, elapsed_s: float) -> bool:
    return config.max_runtime_s > 0 and elapsed_s >= config.max_runtime_s


def budget_reason(
    config: RunConfig,
    metrics: Mapping[str, Any],
    elapsed_s: float,
    reserve: Callable[[], float] | None = None,
) -> str | None:
    """Return the first applicable limit; never treat missing usage as a measured zero.

    Unknown usage prevents enforcing a *finite* token/cost budget. When that budget
    is disabled it stays visible in metrics, but is not a reason to block the run.
    Provider failures are handled separately; this function does not retry calls.
    """
    if runtime_exhausted(config, elapsed_s):
        return "runtime_budget_reached"
    if config.max_calls > 0 and metrics["calls"] >= config.max_calls:
        return "call_budget_reached"
    if config.max_tokens > 0:
        if metrics["unknown_usage_calls"]:
            return "unreconciled_token_usage; audit this run before continuing"
        if metrics["total_tokens"] >= config.max_tokens:
            return "token_budget_reached"
    if config.provider == "openrouter" and config.max_cost_usd > 0:
        if metrics["unknown_cost_calls"]:
            return "unreconciled_openrouter_cost; stop this run and audit provider billing"
        if reserve is None:
            return "cost_reservation_unavailable"
        if metrics["known_cost_usd"] + reserve() > config.max_cost_usd:
            return "cost_budget_reservation_exceeded"
    return None
