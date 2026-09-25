import asyncio
import json
import time

import pytest

from zelda_ai.bridge import Bridge
from zelda_ai.models import Decision, GameState, InputReceipt, RealtimeState
from zelda_ai.runtime import _matching_actor, execute_skill

TOKEN = 'x' * 32
PEER = ('127.0.0.1', 9999)


class Transport:
    def __init__(self): self.sent = []
    def sendto(self, raw, addr): self.sent.append(json.loads(raw))


def full(state, **changes):
    values = state.model_dump()
    values.update(protocol=2, kind='full', seq=10, full_seq=10, context_epoch=1,
                  bridge_build='rt-input-v2.4', capabilities=['fast_state', 'input_sequence', 'consumed_receipts',
                  'player_relative_dodge_state', 'control_stick_direction'],
                  event_floor=1, event_seq=0)
    values.update(changes)
    return values


def fast(state, **changes):
    values = full(state)
    values.update(seq=11, kind='fast', camera_input_yaw=0, mirrored_world=False,
                  player={**values['player'], 'control_stick_direction': values['player'].get('control_stick_direction', -1)})
    values.update(changes)
    return {name: values[name] for name in RealtimeState.model_fields}


def feed(bridge, value, token=TOKEN, peer=PEER):
    bridge.datagram_received(json.dumps({**value, 'token': token}).encode(), peer)


def ready(state):
    bridge = Bridge(TOKEN)
    bridge.transport = Transport()
    feed(bridge, full(state))
    return bridge


def receipt(seq, **changes):
    value = dict(seq=seq, owner_epoch=1, status='accepted', first_tick=0, last_tick=0,
                 pressed=0, released=0, apply_latency_ms=None, client_to_consume_ms=None, reason='')
    value.update(changes)
    return value


def actor(uid, distance):
    return dict(actor_uid=uid, actor_id=9, category=5, params=0, distance=distance,
                position=[0, 0, distance], targeted=False)


def test_fast_updates_pose_without_slow_callback(state):
    bridge = ready(state)
    callbacks = []
    bridge.on_state = lambda new, old: callbacks.append((new, old))
    feed(bridge, fast(state, player={**state.player.model_dump(), 'position': [4, 0, 0]}))
    assert bridge.state.player.position == (4, 0, 0)
    assert bridge.state.full_seq == 10
    assert not callbacks
    feed(bridge, full(state, seq=12, full_seq=12))
    assert len(callbacks) == 1


def test_fast_requires_exact_base_and_context(state):
    bridge = ready(state)
    old_seen = bridge.last_seen
    for changes in [{'full_seq': 9}, {'context_epoch': 2}, {'room': 4}, {'scene_epoch': 2}]:
        feed(bridge, fast(state, **changes))
    assert bridge.last_seen == old_seen and bridge.state.seq == 10
    assert bridge.fast_base_misses == 4
    assert any(p.get('request_full') for p in bridge.transport.sent)


def test_fast_before_full_requests_resync_without_ownership(state):
    bridge = Bridge(TOKEN); bridge.transport = Transport()
    feed(bridge, fast(state))
    assert bridge.state is None and not bridge.connected
    assert bridge.transport.sent[-1]['kind'] == 'observe_ack'


def test_metadata_only_follows_same_lifetime(state):
    bridge = Bridge(TOKEN); bridge.transport = Transport()
    feed(bridge, full(state, room_actors=[{**actor('a', 40), 'name': 'original'}]))
    feed(bridge, fast(state, room_actors=[actor('a', 50), actor('b', 10)]))
    assert bridge.state.room_actors[0].name == 'original'
    assert bridge.state.room_actors[1].name == ''
    assert _matching_actor(bridge.state, 9, 0, 'a').distance == 50
    assert _matching_actor(bridge.state, 9, 0, 'missing') is None


def test_event_bootstrap_does_not_replay_old_victories(state):
    bridge = Bridge(TOKEN); bridge.transport = Transport()
    feed(bridge, full(state, event_seq=3, events=[{'id': '3', 'kind': 'game_completed'}]))
    assert bridge.state.events == []
    assert bridge.transport.sent[-1]['event_cursor'] == 3


def test_events_deliver_once_and_ack_contiguous_prefix(state):
    bridge = ready(state); seen = []
    bridge.on_state = lambda new, old: seen.append([e.id for e in new.events])
    feed(bridge, fast(state, event_seq=2, events=[{'id': '2', 'kind': 'enemy_defeated'}]))
    assert bridge._event_cursor == 0 and seen == []
    feed(bridge, fast(state, seq=12, event_seq=2, events=[{'id': '1', 'kind': 'health_changed'},
         {'id': '2', 'kind': 'enemy_defeated', 'actor_uid': 'a'}]))
    assert bridge._event_cursor == 2 and len(seen) == 1
    feed(bridge, fast(state, seq=13, event_seq=2, events=[{'id': '2', 'kind': 'enemy_defeated'}]))
    assert len(seen) == 1


def test_event_gap_is_reported_not_fabricated_as_success(state):
    bridge = ready(state)
    feed(bridge, fast(state, event_floor=5, event_seq=5, events=[{'id': '5', 'kind': 'health_changed'}]))
    assert bridge.event_gaps == 1
    assert [e.kind for e in bridge.state.events] == ['bridge_event_gap', 'health_changed']
    assert bridge._event_cursor == 5


def test_acceptance_is_not_consumption_and_seq_comparison_is_not_an_ack(state):
    bridge = ready(state)
    feed(bridge, fast(state, last_command_seq=100, input_receipts=[receipt(90),
        receipt(91, status='superseded'), receipt(92, status='completed', first_tick=1, last_tick=1,
                                             apply_latency_ms=2)]))
    assert not bridge.command_consumed(90)
    assert not bridge.command_consumed(91)
    assert bridge.command_consumed(92)
    assert not bridge.command_consumed(89)
    assert bridge.status()['realtime']['native_apply_p95_ms'] == 2


@pytest.mark.parametrize('mutation', ['token', 'peer', 'unknown_field', 'nan', 'reordered'])
def test_invalid_packets_do_not_keep_control_alive(state, mutation):
    bridge = ready(state); before = bridge.last_seen
    value = fast(state); token = TOKEN; peer = PEER
    if mutation == 'token': token = 'wrong'
    elif mutation == 'peer': peer = ('127.0.0.1', 9998)
    elif mutation == 'unknown_field': value['unexpected'] = 10
    elif mutation == 'nan': value['player']['position'] = [float('nan'), 0, 0]
    else: value['seq'] = 9
    feed(bridge, value, token=token, peer=peer)
    assert bridge.last_seen == before
    assert bridge.rejected_packets == 1


def test_native_owner_is_not_reused_after_backend_restart(state):
    bridge = Bridge(TOKEN); bridge.transport = Transport()
    feed(bridge, full(state, owner_epoch=100, last_received_seq=200))
    bridge.send(stick_y=50)
    command = bridge.transport.sent[-1]
    assert command['owner_epoch'] > 100 and command['seq'] > 200
    bridge.revoke()
    assert bridge.transport.sent[-1]['kind'] == 'release'
    assert bridge.transport.sent[-1]['owner_epoch'] > command['owner_epoch']


def test_release_neutral_is_distinct_from_handoff(state):
    bridge = ready(state)
    bridge.release()
    assert bridge.transport.sent[-1]['kind'] == 'cancel'
    bridge.revoke()
    assert bridge.transport.sent[-1]['kind'] == 'release'


@pytest.mark.asyncio
async def test_next_state_does_not_return_same_snapshot(state):
    bridge = ready(state)
    with pytest.raises(RuntimeError, match='feedback_timeout'):
        await bridge.next_state(10, timeout=.01)


@pytest.mark.asyncio
async def test_pulse_waits_for_exact_consumption(state):
    bridge = ready(state)
    task = asyncio.create_task(bridge.pulse(buttons=0x8000, timeout=.3))
    await asyncio.sleep(0)
    sequence = next(p for p in bridge.transport.sent if p.get('kind') == 'sequence')
    seq = sequence['seq']
    feed(bridge, fast(state, input_receipts=[receipt(seq)]))
    await asyncio.sleep(0)
    assert not task.done()
    feed(bridge, fast(state, seq=12, input_receipts=[receipt(seq, status='completed',
        first_tick=2, last_tick=3, pressed=0x8000, released=0x8000, apply_latency_ms=3)]))
    assert await task
    assert bridge.transport.sent[-1]['kind'] == 'cancel'


@pytest.mark.asyncio
async def test_pulse_stops_renewing_when_feedback_is_lost(state):
    bridge = ready(state)
    with pytest.raises(RuntimeError, match='feedback_timeout'):
        await bridge.pulse(buttons=0x8000, timeout=.015)
    assert sum(p.get('kind') == 'sequence' for p in bridge.transport.sent) == 1
    assert not any(p.get('kind') == 'renew' for p in bridge.transport.sent)


@pytest.mark.asyncio
async def test_xz_overlap_on_different_floor_is_not_arrival(state, decision):
    bridge = ready(state)
    d = decision.model_copy(update={'skill': 'navigate_to', 'args': decision.args.model_copy(
        update={'target_position': (0, 150, 0), 'duration_ms': 1000})})
    result = await execute_skill(bridge, d, bridge.state)
    assert result['reason'] == 'target_on_different_floor' and result['status'] == 'failed'
    assert not any(p.get('stick_y') for p in bridge.transport.sent)


def test_provider_schema_requires_nullable_uid_but_legacy_decisions_load(decision):
    args = Decision.model_json_schema()['$defs']['SkillArgs']
    assert 'target_actor_uid' in args['required']
    assert 'default' not in args['properties']['target_actor_uid']
    legacy = decision.model_dump(); legacy['args'].pop('target_actor_uid', None)
    assert Decision.model_validate(legacy).args.target_actor_uid is None


@pytest.mark.asyncio
async def test_pulse_receipt_exposes_exact_native_edges(state):
    bridge = ready(state)
    task = asyncio.create_task(bridge.pulse_receipt(buttons=0x8000, timeout=.3))
    await asyncio.sleep(0)
    sequence = next(p for p in bridge.transport.sent if p.get('kind') == 'sequence')
    seq = sequence['seq']
    feed(bridge, fast(state, input_receipts=[receipt(seq, status='completed', first_tick=4, last_tick=5,
        pressed=0x8000, released=0x8000, apply_latency_ms=7)]))
    row = await task
    assert row and row.seq == seq and row.apply_latency_ms == 7
    assert row.pressed == row.released == 0x8000
    status = bridge.status()['realtime']
    assert status['native_apply_p99_ms'] == 7
    assert status['last_receipt']['seq'] == seq


def test_status_separates_client_latency_from_native_queue(state):
    bridge = ready(state)
    feed(bridge, fast(state, input_receipts=[receipt(150, status='completed', first_tick=2, last_tick=3,
        pressed=0x8000, released=0x8000, apply_latency_ms=.18, client_to_consume_ms=17.25)]))
    rt = bridge.status()['realtime']
    assert rt['native_apply_p95_ms'] == .18
    assert rt['client_to_consume_p95_ms'] == 17.25


@pytest.mark.asyncio
async def test_sequence_receipt_preserves_priming_step(state):
    bridge = ready(state)
    task = asyncio.create_task(bridge.sequence_receipt([
        {'buttons': 0x2000, 'stick_x': 0, 'stick_y': -60, 'ticks': 1},
        {'buttons': 0xA000, 'stick_x': 0, 'stick_y': -60, 'ticks': 1},
        {'buttons': 0x2000, 'stick_x': 0, 'stick_y': -60, 'ticks': 2},
    ], baseline_buttons=0x2000, edge_buttons=0x8000, timeout=.3))
    await asyncio.sleep(0)
    packet = next(p for p in bridge.transport.sent if p.get('kind') == 'sequence')
    assert packet['steps'][0]['buttons'] == 0x2000
    assert packet['steps'][1]['buttons'] == 0xA000
    seq = packet['seq']
    feed(bridge, fast(state, input_receipts=[receipt(seq, status='completed', first_tick=2, last_tick=5,
        pressed=0xA000, released=0x8000, apply_latency_ms=.11, client_to_consume_ms=14.7)]))
    row = await task
    assert row and row.client_to_consume_ms == 14.7
