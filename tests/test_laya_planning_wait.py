import asyncio
import time
from types import SimpleNamespace

import pytest

from zelda_ai.autonomy.execution import ExecutionSupervisor
from zelda_ai.laya_planning_wait import wait_for_replan


def setup(state):
    state.protocol = 3
    state.player.bg_check_flags, state.player.floor_height = 1, 0
    sent, released = [], []
    bridge = SimpleNamespace(state=state,connected=True,receipts={},
        release=lambda:released.append(True),send=lambda **p:sent.append(p) or len(sent))
    controller = SimpleNamespace(training_enabled=False,local_task=None,executor=ExecutionSupervisor())
    return controller,bridge,sent,released


def run(controller,bridge,directory):
    now = time.monotonic()
    return asyncio.run(wait_for_replan(controller,bridge,directory,retry_at=now+.1,deadline=now+2))


@pytest.mark.parametrize("modal", ["dialogue","pause","cutscene","first_person","disconnected"])
def test_wait_never_sends_input_under_another_control_owner(state,tmp_path,modal):
    controller,bridge,sent,released = setup(state)
    if modal == "dialogue":
        state.dialogue.active = True
    elif modal == "pause":
        state.pause_menu.active = True
    elif modal == "cutscene":
        state.cutscene_active = True
    elif modal == "first_person":
        state.player.state_flags_1 = 1<<20
    else:
        bridge.connected = False
    result = run(controller,bridge,tmp_path)
    assert not sent and len(released)==2 and not result['game_progress']
    assert result['reason'] == ('bridge_unavailable' if modal=='disconnected' else 'control_mode_changed')


def test_wait_preserves_macro_dwell_supervision_instead_of_pausing_progress_clock(state,tmp_path):
    controller,bridge,sent,released = setup(state)
    supervisor = controller.executor
    supervisor.observe(state,{})
    supervisor.room_budget_s = .05
    supervisor.progress_at = time.monotonic()-.1
    prior = supervisor.progress_at
    result = run(controller,bridge,tmp_path)
    assert result['reason']=='supervisor:no_durable_or_macro_progress'
    assert supervisor.progress_at == prior and supervisor.state=='blocked'
    assert sent == [{'buttons':0,'stick_x':0,'stick_y':0,'lease_ms':150}]
    assert released and result['game_progress'] is False and result['consumed_actions']==0


@pytest.mark.parametrize('invalid',['training','task','simulator','protocol'])
def test_wait_rejects_incompatible_execution_before_discarding_control(state,tmp_path,invalid):
    controller,bridge,sent,released = setup(state)
    if invalid=='training':
        controller.training_enabled = True
    elif invalid=='task':
        controller.local_task = object()
    elif invalid=='simulator':
        state.source = 'simulator'
    else:
        state.protocol = 1
    with pytest.raises(ValueError):
        run(controller,bridge,tmp_path)
    assert not sent and not released
