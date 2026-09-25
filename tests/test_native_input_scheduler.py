"""Compile the actual native controller, then exercise deterministic clocks/consumers."""
from pathlib import Path
import shutil
import subprocess

import pytest

CASES = ['button_width', 'end_to_end_latency', 'one_tap', 'repeated_press', 'release_before_consume', 'latest_setpoint',
         'duplicate_once', 'duplicate_does_not_renew', 'stale_owner', 'scene_change',
         'context_change', 'watchdog_not_frames', 'old_sample', 'emergency_stale_state',
         'bad_values', 'busy_does_not_drop_action', 'bounded_receipts', 'renewed_sequence_deadline']


@pytest.fixture(scope='module')
def scheduler(tmp_path_factory):
    compiler = shutil.which('g++') or shutil.which('clang++')
    if compiler is None:
        pytest.skip('C++20 compiler required; native behavior was not tested')
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path_factory.mktemp('native') / 'scheduler'
    subprocess.run([compiler, '-std=c++20', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-I', str(root / 'native'), str(root / 'tests/native_input_scheduler.cpp'),
                    '-o', str(binary)], check=True, capture_output=True, timeout=30)
    return binary


@pytest.mark.parametrize('case', CASES)
def test_native_scheduler(scheduler, case):
    result = subprocess.run([str(scheduler), case], capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stdout + result.stderr
