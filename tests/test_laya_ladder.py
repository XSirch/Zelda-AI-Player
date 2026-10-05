import asyncio
import json
import time

import pytest

from zelda_ai.autonomy.attached_descent import AttachedDescentTask
from zelda_ai.autonomy.ladder_task import LadderDescentTask
from zelda_ai.autonomy.laya_ladder_policy import AttachedDecisionLease, LayaLadderPolicy, encode_ladder
from zelda_ai.laya_data import LADDER_PROFILE, PROFILE, bounded_state, profile_features, validate_record
from zelda_ai.laya_numeric import encoder_schema, numeric_values


def attached_task(state, task_type=LadderDescentTask):
    state.player.climbing_ladder = True
    state.player.position = (0, 100, 0)
    state.player.bg_check_flags = 1
    state.camera_input_yaw = 0
    task = task_type._create(state, "ladder_down", (0, -80, 0), state.player.position,
                             now=time.monotonic(), budget_s=20)
    task.version = task_type.VERSION
    return task


def test_attached_schema_is_separate_bounded_and_does_not_reinterpret_walking(state):
    task = attached_task(state)
    row = {"state": bounded_state([encode_ladder(state, task)], profile=LADDER_PROFILE)}
    assert row["state"]["features"] == list(profile_features(LADDER_PROFILE))
    assert row["state"] != bounded_state(row["state"]["history_oldest_first"])
    assert encoder_schema(LADDER_PROFILE) != encoder_schema(PROFILE)
    assert numeric_values([row])[0][-2:] == [1, 0]
    row["state"]["hidden_route"] = "unobserved"
    with pytest.raises(ValueError):
        numeric_values([row])
    with pytest.raises(ValueError):
        profile_features("invented")


def test_ladder_lease_returns_raw_vector_and_rejects_frame_drift(state):
    task = attached_task(state)
    features = tuple(encode_ladder(state, task))
    lease = AttachedDecisionLease(("context",), 10., features, (20, -60))
    assert lease.resolve(lease.owner, features, now=10.05, budget_s=.1) == (20, -60)
    for attribute in ("camera_input_yaw", "player"):
        if attribute == "player":
            state.player.yaw = 4096
        else:
            state.camera_input_yaw = 4096
        assert lease.resolve(lease.owner, encode_ladder(state, task), now=10.05, budget_s=.1) == (0, 0)
        state.player.yaw = state.camera_input_yaw = 0
    assert lease.resolve(lease.owner, features, now=10.2, budget_s=.1) == (0, 0)
    assert lease.resolve(("new-context",), features, now=10.05, budget_s=.1) == (0, 0)


def test_candidate_task_has_no_calibration_or_reference_analog(state):
    task = attached_task(state)
    state.seq += 1
    task.observe(state, consumed=True, now=task.started + .5)
    assert task.reference_stick(state) == (0, 0)
    assert "calibration" not in task.snapshot()
    state.player.climbing_ladder = False
    state.seq += 1
    task.observe(state, consumed=True, now=task.started + .6)
    assert task.failure == "attachment_left_before_landing"


@pytest.mark.parametrize("bad", ("uncalibrated", "unattached", "walking_schema", "candidate"))
def test_ladder_labels_cannot_credit_calibration_probes_or_candidate_actions(state, bad):
    task = attached_task(state)
    row = {"profile": LADDER_PROFILE, "source": "soh", "controller": "reference", "first_tick": 3,
        "buttons": 0, "successful_episode": True, "reference_blend": 0, "episode_id": "one",
        "observation_seq": 1, "command_seq": 2, "family": "ladder_down",
        "reference_calibrated": True, "ladder_attached": True,
        "state": bounded_state([encode_ladder(state, task)], profile=LADDER_PROFILE),
        "executed_stick": [0, -60], "labels": [0, -60]}
    validate_record(row, profile=LADDER_PROFILE)
    if bad == "uncalibrated":
        row["reference_calibrated"] = False
    elif bad == "unattached":
        row["ladder_attached"] = False
    elif bad == "walking_schema":
        row["state"] = bounded_state(row["state"]["history_oldest_first"])
    else:
        row["controller"] = "candidate"
    with pytest.raises(ValueError):
        validate_record(row, profile=LADDER_PROFILE)


class Worker:
    def __init__(self):
        self.stdin = self.stdout = self
        self.sent, self.replies = [], asyncio.Queue()

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


def test_attached_ipc_never_rotates_a_raw_action_into_a_walking_vector(state):
    async def scenario():
        worker, task = Worker(), attached_task(state)
        policy = LayaLadderPolicy(worker)
        try:
            assert policy(state, task) == (0, 0)
            await asyncio.sleep(0)
            assert worker.sent[0]["state"]["profile"] == LADDER_PROFILE
            await worker.replies.put({"id": worker.sent[0]["id"], "stick": [20, -60], "inference_ms": 5})
            for _ in range(5):
                await asyncio.sleep(0)
            assert policy(state, task) == (20, -60)
            state.camera_input_yaw = 16384
            state.seq += 1
            assert policy(state, task) == (0, 0)
            assert policy.metrics["camera_reprojected_outputs"] == 0
            state.player.climbing_ladder = False
            assert policy(state, task) == (0, 0)
            assert policy.lease.owner is None and policy.pending is None
        finally:
            await policy.close()
    asyncio.run(scenario())


def test_candidate_cannot_run_under_a_calibrating_reference_task(state):
    async def scenario():
        worker, task = Worker(), attached_task(state, AttachedDescentTask)
        policy = LayaLadderPolicy(worker)
        try:
            assert policy(state, task) == (0, 0)
            await asyncio.sleep(0)
            assert not worker.sent
        finally:
            await policy.close()
    asyncio.run(scenario())
