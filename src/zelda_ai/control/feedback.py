"""Feedback helpers shared by skills; no model, UI, SQL or open-loop clock in RT mode."""
from __future__ import annotations

import asyncio


def is_realtime(bridge) -> bool:
    return bool(getattr(bridge, "realtime", False))


def consumed(bridge, command_id) -> bool:
    check = getattr(bridge, "command_consumed", None)
    if check is not None:
        return check(command_id)
    return bool(command_id is not None and bridge.state and bridge.state.last_command_seq >= command_id)


async def feedback(bridge, previous, legacy_delay: float = .1):
    if is_realtime(bridge):
        return await bridge.next_state(previous.seq, timeout=.35)
    await asyncio.sleep(legacy_delay)
    return bridge.state
