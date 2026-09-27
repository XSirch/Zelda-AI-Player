"""Continuous autonomous player primitives.

The language model publishes high-level intent.  A local motor loop keeps
controlling Link at realtime bridge cadence while cognition runs concurrently.
"""
from .controller import ContinuousController
from .models import AgentIntent, IntentMode
from .prompt import AUTONOMY_SYSTEM_PROMPT, build_cognition_observation

__all__ = [
    "AgentIntent",
    "IntentMode",
    "ContinuousController",
    "AUTONOMY_SYSTEM_PROMPT",
    "build_cognition_observation",
]
