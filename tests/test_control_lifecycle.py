import asyncio
import json
import time

import pytest

from zelda_ai.bridge import Bridge
from zelda_ai.control.authority import ControlRevoked
from zelda_ai.models import ModelInfo, RunConfig, SwitchConfig, Usage
from zelda_ai.providers.base import InferenceResult
from zelda_ai.runtime import Runtime, controller_input, BUTTONS


class Transport:
    def __init__(self):
        self.sent = []

    def sendto(self, data, addr):
        self.sent.append(json.loads(data))


def connected(state):
    bridge = Bridge('x' * 32)
    bridge.state, bridge.peer, bridge.last_seen = state, ('127.0.0.1', 9999), time.monotonic()
    bridge.transport = Transport()
    return bridge


class Provider:
    def __init__(self, decision):
        self.decision = decision
        self.catalog_gate = asyncio.Event()
        self.catalog_gate.set()
        self.catalog_entered = asyncio.Event()
        self.inference_gate = asyncio.Event()
        self.inference_entered = asyncio.Event()
        self.ignore_cancel = False

    async def status(self):
        self.catalog_entered.set()
        await self.catalog_gate.wait()
        return {'connected': True}

    async def models(self):
        return [ModelInfo(id='test', name='test'), ModelInfo(id='other', name='other')]

    async def decide(self, config, prompt):
        self.inference_entered.set()
        try:
            await self.inference_gate.wait()
        except asyncio.CancelledError:
            if not self.ignore_cancel:
                raise
            await self.inference_gate.wait()
        return InferenceResult(self.decision.model_dump_json(), Usage(input_tokens=3, output_tokens=4))


def unlimited(**changes):
    return RunConfig(provider='codex', model='test', max_calls=0, max_tokens=0,
                     max_runtime_s=0, max_cost_usd=0, **changes)


async def settle(runtime):
    tasks = list(runtime.retired_tasks)
    if tasks:
        await asyncio.wait(tasks, timeout=0.5)


@pytest.mark.asyncio
async def test_stop_does_not_wait_for_slow_switch(state, store, decision):
    bridge = connected(state)
    provider = Provider(decision)
    rt = Runtime(bridge, store, {'codex': provider})
    await rt.start(unlimited())
    await provider.inference_entered.wait()
    provider.catalog_gate.clear()
    provider.catalog_entered.clear()
    change = asyncio.create_task(rt.switch(SwitchConfig(provider='codex', model='other')))
    await provider.catalog_entered.wait()
    await asyncio.wait_for(rt.control('stop'), timeout=0.2)
    assert rt.state == 'stopped'
    assert bridge.transport.sent[-1]['active'] is False
    provider.catalog_gate.set()
    with pytest.raises(ValueError, match='discarded'):
        await change
    assert rt.pending_switch is None
    await settle(rt)


@pytest.mark.asyncio
async def test_stop_during_start_never_creates_run(state, store, decision):
    provider = Provider(decision)
    provider.catalog_gate.clear()
    rt = Runtime(connected(state), store, {'codex': provider})
    start = asyncio.create_task(rt.start(unlimited()))
    await provider.catalog_entered.wait()
    await asyncio.wait_for(rt.control('stop'), timeout=0.2)
    provider.catalog_gate.set()
    with pytest.raises(ValueError, match='cancelled'):
        await start
    assert rt.run_id is None
    assert store.list_runs() == []


@pytest.mark.asyncio
async def test_provider_ignoring_cancel_cannot_move_link(state, store, decision):
    bridge = connected(state)
    provider = Provider(decision)
    provider.ignore_cancel = True
    rt = Runtime(bridge, store, {'codex': provider})
    await rt.start(unlimited())
    await provider.inference_entered.wait()
    run_id = rt.run_id
    await asyncio.wait_for(rt.control('stop'), timeout=0.2)
    await asyncio.sleep(0.01)
    count = len(bridge.transport.sent)
    provider.inference_gate.set()
    await asyncio.sleep(0.03)
    await settle(rt)
    assert len(bridge.transport.sent) == count
    assert rt.state == 'stopped'
    metrics = store.metrics(run_id)
    assert metrics['calls'] == 1
    assert metrics['total_tokens'] == 7
    assert metrics['unknown_usage_calls'] == 0
    assert store.detail(run_id)['calls'][0]['status'] == 'cancelled'


@pytest.mark.asyncio
async def test_old_owner_cleanup_cannot_release_successor(state):
    bridge = connected(state)
    ready, finish = asyncio.Event(), asyncio.Event()
    async def old():
        with bridge.input_scope('old'):
            bridge.send(stick_y=40)
            ready.set()
            await finish.wait()
            with pytest.raises(ControlRevoked):
                bridge.send(stick_y=80)
    task = asyncio.create_task(old())
    await ready.wait()
    with bridge.input_scope('new'):
        bridge.send(buttons=BUTTONS['R'])
        count = len(bridge.transport.sent)
        finish.set()
        await task
        assert len(bridge.transport.sent) == count
        assert bridge.transport.sent[-1]['buttons'] == BUTTONS['R']


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['cutscene', 'dialogue', 'inference'])
async def test_deadline_covers_non_planning_paths(state, store, decision, mode):
    if mode == 'cutscene':
        state.cutscene_active = True
    if mode == 'dialogue':
        state.dialogue.active = True
    provider = Provider(decision)
    rt = Runtime(connected(state), store, {'codex': provider})
    config = unlimited().model_copy(update={'max_runtime_s': 1})
    await rt.start(config)
    # The task has not begun yet: make its remaining deadline small without a one-second test.
    rt.started = time.monotonic() - 0.98
    await asyncio.sleep(0.08)
    assert rt.state == 'paused'
    assert rt.reason == 'runtime_budget_reached'
    assert not rt.bridge.authority.enabled
    await settle(rt)


@pytest.mark.asyncio
async def test_zero_duration_stays_unlimited_in_cutscene(state, store, decision):
    state.cutscene_active = True
    rt = Runtime(connected(state), store, {'codex': Provider(decision)})
    await rt.start(unlimited())
    rt.started = time.monotonic() - 9999999
    await asyncio.sleep(0.02)
    assert rt.state == 'running'
    await rt.control('stop')
    await settle(rt)


def test_runtime_budget_uses_zero_semantics(state, store, decision):
    rt = Runtime(connected(state), store, {'codex': Provider(decision)})
    rt.config = unlimited()
    rt.selected_model = ModelInfo(id='test', name='test')
    rt.run_id = store.new_run(rt.config.model_dump(), 'soh', 'test')
    rt.segment_id = store.segment(rt.run_id, rt.config.model_dump(), 'test')
    store.begin_call(rt.run_id, rt.segment_id, {})  # Unknown usage is retained, not fabricated zero.
    rt.budget_check('{}')
    assert store.metrics(rt.run_id)['unknown_usage_calls'] == 1


def test_backflip_and_sidestep_have_action_edge(decision):
    backflip = decision.model_copy(update={'skill': 'backflip'})
    assert controller_input(backflip)[0] == BUTTONS['Z'] | BUTTONS['A']
    side = decision.model_copy(update={'skill': 'sidestep', 'args': decision.args.model_copy(update={'direction': 'left'})})
    buttons, x, y = controller_input(side)
    assert buttons == BUTTONS['Z'] | BUTTONS['A']
    assert x <= -50 and y == 0


@pytest.mark.asyncio
async def test_run_start_is_blocked_while_diagnostic_active(state, store, decision):
    bridge = connected(state)
    provider = Provider(decision)
    rt = Runtime(bridge, store, {'codex': provider})
    rt.diagnostic_active = True
    with pytest.raises(ValueError, match='diagnostic'):
        await rt.start(unlimited())


def test_diagnostic_release_revokes_without_runtime_lock(state, store, decision):
    bridge = connected(state)
    rt = Runtime(bridge, store, {'codex': Provider(decision)})
    rt.diagnostic_active = True
    bridge.enable_control()
    before = bridge.authority.generation
    result = rt.release_diagnostic_control()
    assert result['released'] is True
    assert bridge.authority.generation > before
    assert not bridge.authority.enabled
