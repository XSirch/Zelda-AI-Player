import pytest

from zelda_ai.models import GameState
from zelda_ai.runtime import _safe_escape_target


def with_probe(state, **changes):
    probe = dict(direction='back', distance=70, floor_found=True, floor_y=0, delta_y=0,
                 wall_hit=False, wall_distance=None, wall_flags=0)
    probe.update(changes)
    return GameState.model_validate({**state.model_dump(), 'navigation_probes': [probe]})


def test_escape_requires_observed_floor(state):
    assert _safe_escape_target(state, 0) is None
    assert _safe_escape_target(with_probe(state, floor_found=False), 0) is None


@pytest.mark.parametrize('changes', [{'delta_y': -100}, {'delta_y': 40}, {'wall_hit': True}])
def test_escape_rejects_drop_step_and_wall(state, changes):
    assert _safe_escape_target(with_probe(state, **changes), 0) is None


def test_escape_uses_player_world_yaw_not_camera(state):
    game = with_probe(state)
    game.player.yaw = 0
    target = _safe_escape_target(game, 0)
    assert target == pytest.approx((0, 0, -25))
    game.player.yaw = 16384
    assert _safe_escape_target(game, 0) == pytest.approx((-25, 0, 0))


def test_consumer_hook_ignores_non_consuming_reads():
    from pathlib import Path
    native = (Path(__file__).parents[1]/'native/ZeldaAiBridge.cpp').read_text(encoding='utf-8')
    assert 'if (controller != 0 || !rawInput || mode == 0) return;' in native
