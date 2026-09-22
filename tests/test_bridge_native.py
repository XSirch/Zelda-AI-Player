import shutil
import subprocess
import time
from pathlib import Path

import pytest

from conftest import packet
from zelda_ai.bridge import Bridge


class Transport:
    def __init__(self): self.sent = []
    def sendto(self, data, addr): self.sent.append((data, addr))
    def close(self): pass


def test_bridge_auth_order_and_source(state):
    bridge = Bridge("x" * 32)
    bridge.connection_made(Transport())
    bridge.datagram_received(packet(state, token="bad"), ("127.0.0.1", 5000))
    assert bridge.state is None
    bridge.datagram_received(packet(state), ("127.0.0.1", 5000))
    assert bridge.connected
    bridge.datagram_received(packet(state, seq=9), ("127.0.0.1", 5000))
    bridge.datagram_received(packet(state, seq=11), ("127.0.0.1", 5001))
    assert bridge.state.seq == 10
    bridge.last_seen = time.monotonic() - 3
    bridge.datagram_received(packet(state, seq=9), ("127.0.0.1", 5000))
    assert bridge.state.seq == 10
    bridge.datagram_received(packet(state, seq=12, source="simulator"), ("127.0.0.1", 5000))
    assert bridge.state.seq == 10
    bridge.datagram_received(packet(state, seq=12), ("127.0.0.1", 5000))
    bridge.send(stick_y=40)
    bridge.release()
    assert b'"active":false' in bridge.transport.sent[-1][0]
    with pytest.raises(ValueError): bridge.send(lease_ms=501)
    assert bridge.rejected_packets == 5


def test_rejects_nonlocal_peer(state):
    bridge = Bridge("x" * 32)
    bridge.datagram_received(packet(state), ("192.168.1.2", 5000))
    assert not bridge.connected


def test_cpp_input_lease(tmp_path):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    root = Path(__file__).resolve().parents[1]
    source = tmp_path / "lease.cpp"
    source.write_text('''#include "InputLease.hpp"
#include <cassert>
int main() {
    zelda_ai::InputLease lease;
    zelda_ai::InputCommand cmd;
    cmd.seq=1; cmd.sceneEpoch=2; cmd.baseSeq=10; cmd.leaseMs=300;
    cmd.active=true; cmd.stickY=40; cmd.buttons=0x8000;
    uint16_t b=0; int8_t x=0,y=0;
    assert(lease.Apply(cmd,2,10,1000));
    assert(lease.Read(1299,b,x,y) && b==0x8000 && y==40);
    assert(!lease.Read(1300,b,x,y));
    assert(!lease.Apply(cmd,2,10,1400));
    cmd.seq=2; assert(!lease.Apply(cmd,3,10,1400));
    cmd.leaseMs=501; assert(!lease.Apply(cmd,2,10,1400));
    cmd.leaseMs=300; cmd.baseSeq=11; assert(!lease.Apply(cmd,2,10,1400));
    cmd.baseSeq=10; assert(lease.Apply(cmd,2,10,1400));
    lease.Release(); assert(!lease.Read(1401,b,x,y));
}''')
    binary = tmp_path / "lease-test"
    subprocess.run([compiler, "-std=c++20", "-Wall", "-Wextra", "-Werror", "-I", str(root / "native"),
        str(source), "-o", str(binary)], check=True, capture_output=True)
    subprocess.run([str(binary)], check=True)


def test_native_room_actor_scan_is_not_camera_gated():
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "ZeldaAiBridge.cpp").read_text()
    start = source.index("json RoomActors(Player* player)")
    end = source.index("std::vector<std::string> DecodeChoices", start)
    room_scan = source[start:end]
    assert "!actor->isDrawn" not in room_scan
    assert "MAX_ROOM_ACTORS" in room_scan
    assert '{"category_name", ActorCategoryName(actor->category)}' in source
    assert '{"room", actor->room}' in source


def test_native_traversal_state_and_terrain_probes_are_exposed():
    root = Path(__file__).resolve().parents[1]
    source = (root / "native" / "ZeldaAiBridge.cpp").read_text()
    assert "SurfaceType_GetWallFlags" in source
    assert '{"climbing_ladder", (player->stateFlags1 & PLAYER_STATE1_CLIMBING_LADDER) != 0}' in source
    assert '{"hanging_ledge", (player->stateFlags1 & PLAYER_STATE1_HANGING_OFF_LEDGE) != 0}' in source
    assert "json NavigationProbes(Player* player)" in source
    assert 'state["navigation_probes"] = NavigationProbes(player);' in source
    assert "70.0f, 140.0f" in source
