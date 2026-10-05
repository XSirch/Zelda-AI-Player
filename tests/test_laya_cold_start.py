import asyncio
from types import SimpleNamespace

import pytest

from zelda_ai import laya_curriculum


def test_walking_reference_is_warm_before_any_native_launch(tmp_path, monkeypatch):
    async def scenario():
        (tmp_path / "seed-home").mkdir()
        warmed, launches = [], []

        async def warm(frozen):
            warmed.append(frozen)
            return {"elapsed_ms": 10, "native_commands": 0, "run_updates": 0}

        async def bind(bridge, port):
            bridge.transport = SimpleNamespace(get_extra_info=lambda _: ("127.0.0.1", 9999))

        def launch(*args):
            launches.append(bool(warmed))
            raise RuntimeError("test_stop_before_native_launch")

        monkeypatch.setattr(laya_curriculum, "warm_reference", warm, raising=False)
        monkeypatch.setattr(laya_curriculum, "prepare_suite", lambda *args, **kwargs:
                            (tmp_path, {"executable": "unit-test-no-executable"}))
        monkeypatch.setattr(laya_curriculum, "bind_bridge", bind)
        monkeypatch.setattr(laya_curriculum, "bridge_secret", lambda _: "unit-test-only")
        monkeypatch.setattr(laya_curriculum, "OwnedProcess", launch)
        with pytest.raises(RuntimeError, match="test_stop_before_native_launch"):
            await laya_curriculum.collect(SimpleNamespace(), tmp_path, tmp_path, sessions=3, tasks=3)
        assert launches == [True]
        assert warmed == [tmp_path / "frozen"]

    asyncio.run(scenario())
