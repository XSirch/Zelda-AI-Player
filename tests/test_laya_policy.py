import asyncio
import json
import time

from zelda_ai.autonomy.laya_policy import DecisionLease, LayaWalkingPolicy
from zelda_ai.autonomy.local_tasks import LocalTask
from zelda_ai.models import TraversalAffordanceObservation


def test_decision_lease_expires_and_rejects_old_context_or_direction():
    features = (0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0)
    lease = DecisionLease(("instance", 2, "task"), 10.0, features, (0, 60))
    assert lease.resolve(lease.owner, features, now=10.08, budget_s=0.1) == (0, 60)
    assert lease.resolve(lease.owner, features, now=10.101, budget_s=0.1) == (0, 0)
    assert lease.resolve(("instance", 3, "task"), features, now=10.08, budget_s=0.1) == (0, 0)
    assert lease.resolve(lease.owner, (1.0, 0.0, *features[2:]), now=10.08, budget_s=0.1) == (0, 0)
    assert lease.resolve(lease.owner, features, now=9.99, budget_s=0.1) == (0, 0)


class FakeWorker:
    def __init__(self):
        self.stdin = self
        self.stdout = self
        self.sent = []
        self.replies = asyncio.Queue()

    def write(self, line):
        self.sent.append(json.loads(line))

    async def drain(self):
        pass

    async def readline(self):
        return (json.dumps(await self.replies.get()) + "\n").encode()

    def close(self):
        pass

    async def wait(self):
        return 0


def task_for(state):
    affordance = TraversalAffordanceObservation(
        kind="stairs_or_slope_up",
        direction="up",
        approach_position=state.player.position,
        target_position=(0, 14, 70),
        distance=0,
        height_delta=14,
    )
    state.traversal_affordances = [affordance]
    return LocalTask.traversal(state, affordance)


def test_inflight_response_cannot_reenter_a_changed_context_or_stopped_policy(state):
    async def scenario():
        process = FakeWorker()
        policy = LayaWalkingPolicy(process)
        try:
            assert policy(state, task_for(state)) == (0, 0)
            await asyncio.sleep(0)
            old_id = process.sent[0]["id"]
            state.scene_epoch += 1
            state.seq += 1
            assert policy(state, task_for(state)) == (0, 0)
            await process.replies.put({"id": old_id, "stick": [20, -40], "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy.metrics["context_rejections"] == 1
            assert policy.lease.owner is None
        finally:
            await policy.close()
        assert policy(state, task_for(state)) == (0, 0)

    asyncio.run(scenario())


def test_expired_or_mismatched_worker_reply_never_becomes_controller_input(state):
    async def scenario():
        process = FakeWorker()
        policy = LayaWalkingPolicy(process)
        try:
            task = task_for(state)
            policy(state, task)
            identity, owner, _, features, telemetry = policy.pending
            policy.pending = (identity, owner, time.monotonic() - 1, features, telemetry)
            await process.replies.put({"id": identity, "stick": [20, -40], "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy.metrics["expired_responses"] == 1
            assert policy(state, task) == (0, 0)
            state.seq += 1
            policy(state, task)
            await process.replies.put({"id": -1, "stick": [20, -40], "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy.failure is not None
            assert policy(state, task) == (0, 0)
        finally:
            await policy.close()

    asyncio.run(scenario())
