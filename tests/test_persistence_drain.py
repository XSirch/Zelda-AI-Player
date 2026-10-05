"""A subprocess bounds an event-loop spin that wait_for cannot interrupt."""
import asyncio
import subprocess
import sys

import pytest


def test_completed_persistence_jobs_cannot_starve_their_cleanup_callbacks():
    code = """
import asyncio
from zelda_ai.autonomy.runtime import AutonomyRuntime

async def scenario():
    runtime = object.__new__(AutonomyRuntime)
    completed = asyncio.get_running_loop().create_future()
    completed.set_result(None)
    runtime.persist_tasks = {completed}
    completed.add_done_callback(runtime.persist_tasks.discard)
    await runtime._drain_persistence()
    assert not runtime.persist_tasks

asyncio.run(scenario())
"""
    try:
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    except subprocess.TimeoutExpired:
        pytest.fail("Completed persistence jobs spun without yielding to cleanup")
    assert result.returncode == 0, result.stderr


@pytest.mark.asyncio
async def test_drain_waits_for_jobs_added_while_the_first_snapshot_settles():
    from zelda_ai.autonomy.runtime import AutonomyRuntime

    runtime = object.__new__(AutonomyRuntime)
    release_first, release_second, second_started = (asyncio.Event() for _ in range(3))
    written = []

    async def second():
        second_started.set()
        await release_second.wait()
        written.append("second")

    async def first():
        await release_first.wait()
        job = asyncio.create_task(second())
        runtime.persist_tasks.add(job)
        job.add_done_callback(runtime.persist_tasks.discard)
        written.append("first")

    job = asyncio.create_task(first())
    runtime.persist_tasks = {job}
    job.add_done_callback(runtime.persist_tasks.discard)
    drain = asyncio.create_task(runtime._drain_persistence())
    try:
        await asyncio.sleep(0)
        release_first.set()
        await asyncio.wait_for(second_started.wait(), timeout=1)
        assert not drain.done()
        release_second.set()
        await asyncio.wait_for(drain, timeout=1)
        assert written == ["first", "second"] and not runtime.persist_tasks
    finally:
        release_first.set()
        release_second.set()
        await asyncio.gather(drain, return_exceptions=True)
