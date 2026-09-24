import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('integrate_soh', Path(__file__).parents[1] / 'scripts/integrate_soh.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def original(monkeypatch):
    data = b'void PadMgr_RequestPadData(PadMgr* p, Input* inputs, s32 mode) {\n        ogInput++;\n}\n'
    monkeypatch.setattr(module, 'PADMGR_BLOB', module.git_blob(data))
    return data


def test_fresh_and_repeated_patch(original):
    patched = module.patched_padmgr(original)
    assert patched.count(b'ZeldaAiBridge_ConsumeInput(') == 1
    assert module.patched_padmgr(patched) == patched
    assert module.patched_padmgr(patched.replace(b'\n', b'\r\n')) == patched


def test_legacy_migration_preserves_single_hook(original):
    old = module.INCLUDE.encode() + original.replace(module.ANCHOR.encode(), module.LEGACY_CALL.encode() + module.ANCHOR.encode())
    new = module.patched_padmgr(old)
    assert b'OverrideInput' not in new
    assert new.count(b'ConsumeInput') == 1


@pytest.mark.parametrize('extra', [b'// local edit\n', module.INCLUDE.encode(), module.CALL.encode()])
def test_unknown_or_duplicate_changes_are_not_overwritten(original, extra):
    with pytest.raises(ValueError):
        module.patched_padmgr(module.patched_padmgr(original) + extra)


def test_manifest_covers_included_project_headers():
    native = Path(__file__).parents[1] / 'native'
    for line in (native / 'ZeldaAiBridge.cpp').read_text().splitlines():
        if line.startswith('#include "'):
            name = line.split('"')[1]
            if '/' not in name and (native / name).exists():
                assert name in module.NATIVE_FILES


def test_input_is_hooked_at_consumer_not_raw_poll():
    assert 'newInput' in module.CALL
    assert 'ogInput++' in module.ANCHOR
    assert 'input->cur' not in module.CALL
