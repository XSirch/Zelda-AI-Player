import asyncio
import json
import math
import time
from dataclasses import replace

import pytest

from zelda_ai.autonomy.collectible_task import ObservedCollectibleApproachTask
from zelda_ai.autonomy.container_task import ObservedContainerApproachTask
from zelda_ai.autonomy.controller import ContinuousController
from zelda_ai.autonomy.ground_descent_task import GroundDescentApproachTask
from zelda_ai.autonomy.imitation import encode_surface
from zelda_ai.autonomy.laya_policy import DecisionLease, LayaWalkingPolicy
from zelda_ai.autonomy.local_tasks import LocalTask
from zelda_ai.models import ActorObservation, NavigationMeshSnapshot, TraversalAffordanceObservation


def test_container_approach_uses_dry_walking_contract_and_lost_actor_revokes_reply(state):
    async def scenario():
        state.player.bg_check_flags = 1
        state.camera_input_yaw = 0
        state.navmesh = NavigationMeshSnapshot(step=70, half_extent=4,
            cells=[(0, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 64)])
        actor = ActorObservation(actor_uid="container", actor_id=999, category=10, category_name="chest",
            params=0, position=(180, 0, 0), distance=180, drawn=True)
        state.nearby_actors = [actor]
        task = ObservedContainerApproachTask.create(state, actor)
        process = FakeWorker()
        policy = LayaWalkingPolicy(process)
        try:
            policy(state, task)
            await asyncio.sleep(0)
            request = process.sent[0]
            assert encode_surface(state, task)[-1] == 1
            state.nearby_actors = []
            assert policy(state, task) == (0, 0)
            await process.replies.put({"id": request["id"], "stick": [20, 60], "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy.metrics["context_rejections"] == 1 and policy.lease.owner is None
        finally:
            await policy.close()
    asyncio.run(scenario())


def test_collectible_approach_uses_frozen_walking_and_neutralizes_when_target_is_lost(state):
    async def scenario():
        state.player.bg_check_flags, state.camera_input_yaw = 1, 0
        state.navmesh = NavigationMeshSnapshot(step=70, half_extent=4,
            cells=[(0, 0, 0., 4), (1, 0, 0., 68), (2, 0, 0., 64)])
        actor = ActorObservation(actor_uid="current-collectible", actor_id=21, category=6,
            category_name="misc", name="En_Item00", params=-32768,
            position=(180, 1, 0), distance=180, drawn=True)
        state.nearby_actors = [actor]
        task = ObservedCollectibleApproachTask.create(state, actor)
        process, policy = FakeWorker(), None
        policy = LayaWalkingPolicy(process)
        try:
            policy(state, task)
            await asyncio.sleep(0)
            request = process.sent[0]
            assert request["state"]["history_oldest_first"][-1][-1] == 1
            state.nearby_actors = []
            assert policy(state, task) == (0, 0)
            await process.replies.put({"id": request["id"], "stick": [20, 60], "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy.metrics["context_rejections"] == 1 and policy.lease.owner is None
        finally:
            await policy.close()
    asyncio.run(scenario())


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
    state.camera_input_yaw = 0
    return LocalTask.traversal(state, affordance)


@pytest.mark.parametrize("yaw", (1024, 16384, -32768))
@pytest.mark.parametrize("mirrored", (False, True))
def test_laya_inflight_heading_survives_indoor_camera_orbit_and_top_down(
    state, tmp_path, yaw, mirrored,
):
    async def scenario():
        class Bridge:
            def command_consumed(self, seq):
                return True

        process = FakeWorker()
        policy = LayaWalkingPolicy(process)
        state.mirrored_world = mirrored
        bridge = Bridge()
        bridge.state = state
        controller = ContinuousController(bridge, tmp_path / "policy.pt")
        # Model/optimizer cold initialization is outside the physical task's
        # bounded attempt. Camera handling is what this test exercises.
        task = task_for(state)
        controller.start_local_task(task, stick_policy=policy)
        try:
            controller._ml_step(state)
            await asyncio.sleep(0)
            request_id = process.sent[0]["id"]
            # Camera changes while the model is still computing. The vertical
            # view supplies no horizontal heading; native input yaw does.
            state.camera_input_yaw = yaw
            state.camera_eye, state.camera_at = (0, 200, 0), (0, 0, 0)
            state.seq += 1
            await process.replies.put({"id": request_id, "stick": [20, 60], "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            controller._refresh_camera_relative_setpoint(state)
            actual = (controller.last_setpoint.stick_x, controller.last_setpoint.stick_y)
            sign = 1 if mirrored else -1
            world_heading = math.atan2(sign * 20, 60)
            relative = world_heading - yaw * math.pi / 32768
            magnitude = math.hypot(20, 60)
            expected = (round(sign * math.sin(relative) * magnitude),
                        round(math.cos(relative) * magnitude))
            assert actual == expected
            assert controller.last_setpoint.buttons == 0
            assert controller.pending["trainable"] is False
            assert controller.pending["stick"] == [v / 80 for v in expected]
            assert controller.rollout == []
            # A second orbit after completion must also preserve the chosen
            # world movement, even when the observation sequence is unchanged.
            state.camera_input_yaw = 0
            controller._refresh_camera_relative_setpoint(state)
            assert (controller.last_setpoint.stick_x, controller.last_setpoint.stick_y) == (20, 60)
            task.target = (70, 14, 0)
            controller._refresh_camera_relative_setpoint(state)
            assert (controller.last_setpoint.stick_x, controller.last_setpoint.stick_y) == (0, 0)
        finally:
            await policy.close()

    asyncio.run(scenario())


def test_rotated_diagonal_saturation_preserves_direction():
    features = (0., 1., 0., .5, 0., 0., 0., 0., 1.)
    lease = DecisionLease(("instance", 2, "task"), 10., features, (80, 80))
    angle = math.pi / 4
    current = (-math.sin(angle), math.cos(angle), *features[2:])
    assert lease.resolve(lease.owner, current, now=10.05, budget_s=.1,
                         camera_yaw=angle) == (80, 0)


@pytest.mark.parametrize("state_1,state_2", ((1 << 27, 0), (1 << 27, 1 << 10),
                                             (1 << 27, 1 << 11), (1 << 18, 0), (1 << 19, 0)))
def test_ground_reply_cannot_control_water_or_airborne_mode(state, state_1, state_2):
    async def scenario():
        process = FakeWorker()
        policy = LayaWalkingPolicy(process)
        try:
            task = task_for(state)
            policy(state, task)
            await asyncio.sleep(0)
            identity = process.sent[0]["id"]
            state.player.state_flags_1 = state_1
            state.player.state_flags_2 = state_2
            assert policy(state, task) == (0, 0)
            assert policy.pending is None
            await process.replies.put({"id": identity, "stick": [20, 60], "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy.lease.owner is None
            assert policy.metrics["context_rejections"] == 1
            assert policy(state, task) == (0, 0)
            assert len(process.sent) == 1
        finally:
            await policy.close()

    asyncio.run(scenario())


def test_top_down_without_native_input_yaw_cannot_reuse_a_lease(state):
    async def scenario():
        process = FakeWorker()
        policy = LayaWalkingPolicy(process)
        try:
            task = task_for(state)
            policy(state, task)
            await asyncio.sleep(0)
            await process.replies.put({"id": process.sent[0]["id"], "stick": [20, 60],
                                       "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy(state, task) == (20, 60)
            state.camera_input_yaw = None
            state.camera_eye, state.camera_at = (0, 200, 0), (0, 0, 0)
            state.player.yaw = 16384
            assert policy(state, task) == (0, 0)
            assert policy.lease.owner is None
            assert policy.pending is None
        finally:
            await policy.close()

    asyncio.run(scenario())


def test_walking_candidate_cannot_reuse_a_reply_for_an_untrained_descent_mode(state):
    async def scenario():
        process = FakeWorker()
        policy = LayaWalkingPolicy(process)
        try:
            task = task_for(state)
            policy(state, task)
            await asyncio.sleep(0)
            await process.replies.put({"id": process.sent[0]["id"], "stick": [20, 60],
                                       "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy(state, task) == (20, 60)
            task.kind = "ledge_down"
            assert policy(state, task) == (0, 0)
            assert policy.pending is None
            assert policy.lease.owner is None
            assert len(process.sent) == 1
        finally:
            await policy.close()

    asyncio.run(scenario())


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
            identity = policy.pending.identity
            policy.pending = replace(policy.pending, submitted_at=time.monotonic() - 1)
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


def test_walking_candidate_owns_only_dry_phase_of_observed_descent(state):
    async def scenario():
        state.player.position, state.player.floor_height, state.player.bg_check_flags = (0,100,0),100,1
        state.camera_input_yaw = 0
        state.navmesh = NavigationMeshSnapshot(origin=(0,100,0),step=70,half_extent=3,
            cells=[(0,0,100.,0),(1,0,0.,0)])
        row = TraversalAffordanceObservation(kind="ledge_down",direction="down",
            approach_position=state.player.position,target_position=(70,0,0),distance=0,height_delta=-100)
        state.traversal_affordances = [row]
        task = GroundDescentApproachTask.create(state,row)
        policy = LayaWalkingPolicy(FakeWorker())
        owner = policy.key(state,task)
        policy.owner = owner
        policy.lease = DecisionLease(owner,time.monotonic(),tuple(encode_surface(state,task)),(0,60))
        try:
            assert policy(state,task) != (0,0)
            state.player.climbing_ladder = True
            state.seq += 1
            assert policy(state,task) == (0,0) and policy.owner is None
        finally:
            await policy.close()
    asyncio.run(scenario())
