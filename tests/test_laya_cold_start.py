import asyncio
import json
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


@pytest.mark.parametrize("report_write_fails", (False, True))
def test_pre_input_bridge_failure_preserves_unknowns_and_always_closes(tmp_path, monkeypatch, report_write_fails):
    async def scenario():
        (tmp_path / "seed-home").mkdir()
        closed, transports = [], []

        class Transport:
            def get_extra_info(self, name):
                return ("127.0.0.1", 9999)

            def close(self):
                transports.append("closed")

        class Process:
            child = SimpleNamespace(poll=lambda: None)

            def __init__(self, *args):
                pass

            async def close(self):
                closed.append(True)

        async def warm(*args):
            return {"elapsed_ms": 0, "native_commands": 0, "run_updates": 0}

        async def bind(bridge, port):
            bridge.connection_made(Transport())

        times = iter((0., 16.))  # Elapsed setup only; asyncio keeps its real clock.
        monkeypatch.setattr(laya_curriculum, "time", SimpleNamespace(monotonic=lambda: next(times, 16.)))
        monkeypatch.setattr(laya_curriculum, "warm_reference", warm)
        monkeypatch.setattr(laya_curriculum, "prepare_suite", lambda *args, **kwargs:
                            (tmp_path, {"executable": "unit-test-no-executable"}))
        monkeypatch.setattr(laya_curriculum, "bind_bridge", bind)
        monkeypatch.setattr(laya_curriculum, "bridge_secret", lambda _: "unit-test-only")
        monkeypatch.setattr(laya_curriculum, "OwnedProcess", Process)
        if report_write_fails:
            original_write = laya_curriculum.write_json

            def write(path, data):
                if path.name == "session-failure.json":
                    raise OSError("unit_report_write_failed")
                original_write(path, data)

            monkeypatch.setattr(laya_curriculum, "write_json", write)
        expected = OSError if report_write_fails else RuntimeError
        reason = "unit_report_write_failed" if report_write_fails else "Real native input-receipt bridge unavailable"
        with pytest.raises(expected, match=reason):
            await laya_curriculum.collect(SimpleNamespace(), tmp_path, tmp_path, sessions=3, tasks=3)
        assert closed == [True] and transports == ["closed"]
        if not report_write_fails:
            path = tmp_path / "native-session-1/session-failure.json"
            raw = path.read_text(encoding="utf-8")
            result = json.loads(raw)
            assert result["phase"] == "await_native_bridge"
            assert result["last_observation"] is None
            assert result["native_process_exit_code"] is None
            assert result["command_seq_at_failure"] == 0
            assert result["bridge"]["rejected_packets"] == 0
            assert result["bridge"]["realtime"]["last_receipt"] is None
            assert "unit-test-only" not in raw and '"token"' not in raw

    asyncio.run(scenario())
