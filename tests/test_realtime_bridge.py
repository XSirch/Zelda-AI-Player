import asyncio
import json
import time

import pytest

from zelda_ai.bridge import Bridge
from zelda_ai.models import GameState, InputReceipt, RealtimeState

TOKEN = 'x' * 32
PEER = ('127.0.0.1', 9999)


class Transport:
    def __init__(self): self.sent = []
    def sendto(self, raw, addr): self.sent.append(json.loads(raw))


def full(state, **changes):
    values = state.model_dump()
    values.update(protocol=3, kind='full', seq=10, full_seq=10, context_epoch=1,
                  bridge_build='rt-input-v3.0', capabilities=['fast_state', 'input_sequence', 'consumed_receipts',
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


def test_startup_sequence_waits_within_requested_budget_without_replaying_consumed_input(state):
    bridge = ready(state)
    waits = []

    async def feedback(after_seq, *, timeout):
        waits.append(timeout)
        # Replay the native title cadence (388 ms), without sleeping or
        # pretending this software receipt is physical gameplay evidence.
        if timeout < .388:
            raise RuntimeError("state_feedback_timeout")
        bridge.receipts[1] = InputReceipt(seq=1, owner_epoch=1, status="completed",
            first_tick=251, last_tick=252, pressed=0x4000, released=0x4000)
        bridge.state.seq += 1
        return bridge.state

    bridge.next_state = feedback
    row = asyncio.run(bridge.sequence_receipt([
        {"buttons": 0x4000, "stick_x": 0, "stick_y": 0, "ticks": 1},
        {"buttons": 0, "stick_x": 0, "stick_y": 0, "ticks": 1}],
        edge_buttons=0x4000, startup_slot=1, timeout=.8))
    assert row.first_tick == 251 and .388 <= waits[0] <= .8
    assert sum(p["kind"] == "sequence" for p in bridge.transport.sent) == 1
    assert not any(p["kind"] == "renew" for p in bridge.transport.sent)


def receipt(seq, **changes):
    value = dict(seq=seq, owner_epoch=1, status='accepted', first_tick=0, last_tick=0,
                 pressed=0, released=0, apply_latency_ms=None, client_to_consume_ms=None, reason='')
    value.update(changes)
    return value


def actor(uid, distance):
    return dict(actor_uid=uid, actor_id=9, category=5, params=0, distance=distance,
                position=[0, 0, distance], targeted=False)




def test_fast_keeps_navmesh_from_matching_full_snapshot(state):
    bridge = Bridge(TOKEN)
    bridge.transport = Transport()
    mesh = {"origin": [0, 0, 0], "step": 70, "half_extent": 4,
            "cells": [[0, 0, 0, 1], [0, 1, 0, 16]]}
    exits = [{"exit_index": 1, "entrance_index": 0x211,
              "position": [0, 0, 70], "samples": 4}]
    affordances = [{
        "kind": "stairs_or_slope_down", "direction": "down",
        "approach_position": [0, 0, 70], "target_position": [0, -40, 140],
        "distance": 70, "height_delta": -40, "wall_flags": 0,
    }]
    progress = {
        **state.progress.model_dump(),
        "story_flags": {"greeted_by_saria": True},
    }
    autosave = {
        "pending": False, "target_scene": -1, "last_scene": 85,
        "count": 2, "last_saved_at_ms": 1234,
    }
    feed(bridge, full(state, navmesh=mesh, scene_exits=exits,
        traversal_affordances=affordances, progress=progress, autosave=autosave,
        capabilities=["fast_state", "input_sequence", "consumed_receipts",
                      "local_navmesh", "scene_exit_surfaces", "traversal_affordances_v1",
                      "story_progress_v1", "scene_autosave_v1"]))
    assert bridge.state.navmesh.available
    assert bridge.state.scene_exits[0].entrance_index == 0x211
    assert bridge.state.traversal_affordances[0].kind == "stairs_or_slope_down"
    assert bridge.state.progress.story_flags["greeted_by_saria"] is True
    assert bridge.state.autosave.count == 2
    feed(bridge, fast(state, player={**state.player.model_dump(), "position": [0, 0, 8]}))
    assert bridge.state.player.position == (0, 0, 8)
    assert bridge.state.navmesh.cells == [(0, 0, 0.0, 1), (0, 1, 0.0, 16)]
    assert bridge.state.scene_exits[0].position == (0.0, 0.0, 70.0)
    assert bridge.state.traversal_affordances[0].approach_position == (0.0, 0.0, 70.0)
    assert bridge.state.progress.story_flags["greeted_by_saria"] is True
    assert bridge.state.autosave.last_scene == 85


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
    actors = {row.actor_uid: row for row in bridge.state.room_actors}
    assert actors["a"].distance == 50
    assert "missing" not in actors


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


