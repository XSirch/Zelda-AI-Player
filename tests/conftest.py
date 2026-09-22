import json
import socket

import pytest

from zelda_ai.models import Decision, GameState
from zelda_ai.store import Store


@pytest.fixture
def state():
    return GameState(source="soh", instance_id="test-game", seq=10, scene_epoch=1,
        scene=85, room=0, in_game=True, player={"position": [0, 0, 0], "yaw": 0,
        "health": 48, "max_health": 48, "rupees": 0})


@pytest.fixture
def decision():
    return Decision(goal="Explore", summary="Move toward an untested direction.", skill="move",
        args={"direction": "forward", "duration_ms": 100, "strength": 0.5, "slot": None, "choice_index": None, "song": None, "target_actor_id": None,
            "target_actor_params": None, "target_position": None, "stop_distance": None, "item_id": None}, memory_note=None)


@pytest.fixture
def store(tmp_path):
    database = Store(f"sqlite:///{tmp_path}/test.sqlite3")
    yield database
    database.close()


def free_udp_port():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def packet(state, token="x" * 32, **changes):
    return json.dumps({**state.model_dump(), **changes, "token": token}).encode()
