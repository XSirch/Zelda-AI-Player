import json
import time

import pytest

from zelda_ai.bridge import Bridge


def connected(state):
    state.protocol = 3
    state.capabilities = ['input_sequence','navmesh_refinement']
    bridge = Bridge('x'*32)
    bridge.state,bridge.last_seen = state,time.monotonic()
    bridge.peer = ('127.0.0.1',8766)
    packets=[]
    class Transport:
        def sendto(self,raw,peer):
            packets.append(json.loads(raw))
    bridge.transport = Transport()
    return bridge,packets


def test_finer_collision_request_is_read_only_current_context_and_rate_limited(state):
    bridge,packets=connected(state)
    assert bridge.request_navigation_refinement(request_id=state.seq)
    assert not bridge.request_navigation_refinement(request_id=state.seq)
    assert len(packets)==1 and bridge.command_seq==0
    packet=packets[0]
    assert packet['kind']=='observe_ack' and packet['refinement_request_id']==state.seq
    assert packet['base_seq']==state.seq and packet['scene_epoch']==state.scene_epoch
    assert packet['context_epoch']==state.context_epoch
    assert not {'buttons','stick_x','stick_y','lease_ms'} & packet.keys()


@pytest.mark.parametrize('change',['old_adapter','stale','disconnect','simulator'])
def test_unsupported_or_stale_peer_gets_no_finer_query(state,change):
    bridge,packets=connected(state)
    if change=='old_adapter':
        state.capabilities.remove('navmesh_refinement')
    elif change=='stale':
        bridge.last_seen -= .4
    elif change=='disconnect':
        bridge.peer = None
    else:
        state.source = 'simulator'
    assert not bridge.request_navigation_refinement(request_id=state.seq)
    assert not packets and bridge.command_seq==0


@pytest.mark.parametrize('request_id',[0,-1,True,1.5,11])
def test_navigation_identity_cannot_be_invalid_or_future(state,request_id):
    bridge,packets=connected(state)
    with pytest.raises(ValueError):
        bridge.request_navigation_refinement(request_id=request_id)
    assert not packets
