"""Regressions for real SoH's eight-slot equipment payload (including D-pad)."""
from __future__ import annotations

import asyncio
import json
import socket

import pytest
from pydantic import ValidationError

from zelda_ai.bridge import Bridge, bind_bridge
from zelda_ai.models import GameState

TOKEN = "test-bridge-credential-not-a-real-secret"
PEER = ("127.0.0.1", 45678)


def menu_packet() -> dict:
    # Same fields as native/ZeldaAiBridge.cpp before a save is loaded.
    return {
        "protocol": 1, "source": "soh", "instance_id": "equipment-regression",
        "seq": 653, "scene_epoch": 0, "scene": -1, "room": -1,
        "in_game": False, "player": None, "paused": False, "events": [],
        "last_command_seq": 0,
        "upstream_revision": "d30fc192f2eb01ceea45bd1e12de61636cafbf86",
    }


def gameplay_packet() -> dict:
    # ItemEquips.buttonItems[8]: B, three C buttons, four D-pad entries.
    return {
        **menu_packet(), "seq": 654, "scene_epoch": 1, "scene": 85, "room": 0,
        "in_game": True,
        "player": {"position": [12.5, 0.0, -8.25], "yaw": -100,
                   "health": 48, "max_health": 48, "rupees": 0,
                   "magic": 0, "age": "child"},
        "camera_eye": [10.0, 40.0, 100.0], "camera_at": [12.5, 30.0, -8.25],
        "inventory": [255] * 24, "equipped": [255] * 8, "message_id": 0,
    }


def encode(packet: dict, token: str = TOKEN) -> bytes:
    return json.dumps({**packet, "token": token}).encode()


@pytest.mark.parametrize("slots", [0, 4, 8])
def test_equipped_accepts_empty_legacy_and_native_payloads(slots):
    packet = gameplay_packet()
    packet["equipped"] = [255] * slots
    state = GameState.model_validate(packet)
    assert state.equipped == [255] * slots


def test_equipped_keeps_a_bounded_contract():
    packet = gameplay_packet()
    packet["equipped"] = [255] * 9
    with pytest.raises(ValidationError) as exc:
        GameState.model_validate(packet)
    assert any(e["loc"] == ("equipped",) and e["type"] == "too_long"
               for e in exc.value.errors())


def test_title_to_gameplay_keeps_telemetry_alive(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("zelda_ai.bridge.time.monotonic", lambda: now[0])
    bridge = Bridge(TOKEN)
    bridge.datagram_received(encode(menu_packet()), PEER)
    assert bridge.connected
    assert bridge.state.scene == -1
    for offset in range(20):
        now[0] += 0.2
        packet = gameplay_packet()
        packet["seq"] += offset
        bridge.datagram_received(encode(packet), PEER)
    assert bridge.connected
    assert bridge.rejected_packets == 0
    assert bridge.state.seq == 673
    assert bridge.state.in_game
    assert bridge.state.scene == 85
    assert bridge.state.player.health == 48
    assert len(bridge.state.equipped) == 8


def test_schema_rejection_is_diagnosable_without_refreshing_valid_state(monkeypatch, caplog):
    now = [100.0]
    monkeypatch.setattr("zelda_ai.bridge.time.monotonic", lambda: now[0])
    bridge = Bridge(TOKEN)
    bridge.datagram_received(encode(menu_packet()), PEER)
    bad = gameplay_packet()
    bad["equipped"] = [255] * 9
    now[0] += 3
    for _ in range(5):
        bridge.datagram_received(encode(bad), PEER)
    assert not bridge.connected
    assert bridge.state.seq == 653
    assert bridge.rejected_packets == 5
    assert bridge.status()["last_validation_error"] == {
        "fields": ["equipped"], "types": ["too_long"],
    }
    warnings = [r for r in caplog.records if "SoH state rejected" in r.message]
    assert len(warnings) == 1  # Repeated invalid frames do not flood the terminal.
    assert TOKEN not in caplog.text
    bridge.datagram_received(encode(gameplay_packet()), PEER)
    assert bridge.connected
    assert bridge.status()["last_validation_error"] is None
    assert bridge.rejected_packets == 5


def test_diagnostics_do_not_echo_packet_values_or_unknown_keys(caplog):
    bridge = Bridge(TOKEN)
    bad = gameplay_packet()
    bad["player"]["health"] = "sensitive-payload-value"
    bad["secret-in-unknown-field-name"] = "do-not-log"
    bridge.datagram_received(encode(bad), PEER)
    exposed = json.dumps(bridge.status()) + caplog.text
    assert bridge.status()["last_validation_error"]["fields"] == ["<unknown>", "player"]
    for forbidden in (TOKEN, "sensitive-payload-value", "secret-in-unknown-field-name", "do-not-log"):
        assert forbidden not in exposed


def test_authentication_replay_and_owner_checks_remain_enforced():
    bridge = Bridge(TOKEN)
    packet = gameplay_packet()
    bridge.datagram_received(encode(packet, "wrong-credential"), PEER)
    assert bridge.state is None
    bridge.datagram_received(encode(packet), PEER)
    assert bridge.connected
    bridge.datagram_received(encode(packet), PEER)  # Replay.
    bridge.datagram_received(encode({**packet, "seq": 655, "instance_id": "other"}), PEER)
    bridge.datagram_received(encode({**packet, "seq": 655}), ("192.0.2.1", 45678))
    assert bridge.rejected_packets == 4
    assert bridge.state.seq == 654
    assert bridge.state.instance_id == "equipment-regression"


@pytest.mark.parametrize("raw", [b"[]", b"null", b"{invalid", b'"string"'])
def test_malformed_packets_are_still_rejected(raw):
    bridge = Bridge(TOKEN)
    bridge.datagram_received(raw, PEER)
    assert bridge.rejected_packets == 1
    assert not bridge.connected


def test_eight_slot_payload_and_command_roundtrip_over_loopback_udp():
    async def exercise():
        bridge = Bridge(TOKEN)
        await bind_bridge(bridge, 0)
        address = bridge.transport.get_extra_info("sockname")
        loop = asyncio.get_running_loop()
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sender:
            sender.bind(("127.0.0.1", 0))
            sender.setblocking(False)
            try:
                await loop.sock_sendto(sender, encode(menu_packet()), address)
                await asyncio.wait_for(bridge.changed.wait(), 2)
                bridge.changed.clear()
                await loop.sock_sendto(sender, encode(gameplay_packet()), address)
                await asyncio.wait_for(bridge.changed.wait(), 2)
                assert bridge.connected
                assert len(bridge.state.equipped) == 8
                command_seq = bridge.send(buttons=0x8000, lease_ms=100)
                raw = await asyncio.wait_for(loop.sock_recv(sender, 4096), 2)
                command = json.loads(raw)
                assert command["seq"] == command_seq
                assert command["base_seq"] == 654
                assert command["scene_epoch"] == 1
                assert command["buttons"] == 0x8000
            finally:
                bridge.close()
    asyncio.run(exercise())
