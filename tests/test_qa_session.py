import json
from types import SimpleNamespace

import pytest

from zelda_ai import qa_session
from zelda_ai.qa_reset import QA_RESET_BUTTONS


class FakeBridge:
    def __init__(self, token, **kwargs):
        assert kwargs == {"allow_simulator": False}
        self.transport = SimpleNamespace(get_extra_info=lambda _: ("127.0.0.1", 1234))
        self.releases = self.closed = 0

    def release(self):
        self.releases += 1

    def close(self):
        self.closed += 1


class FakeProcess:
    def __init__(self, executable, home, environment):
        assert environment["SHIP_HOME"] == str(home)
        self.child = SimpleNamespace(poll=lambda: None)
        self.closed = 0

    async def close(self):
        self.closed += 1


async def bind(*_):
    pass


def pool(tmp_path, *, reuse):
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "shipofharkinian.json").write_text('{"CVars": {"gSettings": {"ResetBtn": 128}}}', encoding="utf-8")
    (seed / "Save").mkdir()
    (seed / "Save" / "file2.sav").write_bytes(b"fixture-not-a-real-save")
    return qa_session.QaProcessPool(tmp_path / "soh.exe", seed, "test-token", reuse=reuse,
        bridge_factory=FakeBridge, process_factory=FakeProcess, bind=bind)


@pytest.mark.asyncio
async def test_evaluation_keeps_one_owned_process_and_preserves_seed(tmp_path, monkeypatch):
    harness = pool(tmp_path, reuse=True)
    resets = []

    async def reset(bridge, *, configured_buttons):
        assert bridge is harness.bridge and configured_buttons == QA_RESET_BUTTONS
        resets.append(bridge)
        return {"success": True, "reason": "observed_normal_title_reset"}

    monkeypatch.setattr(qa_session, "reset_to_title", reset)
    async with harness:
        bridge, process, evidence = await harness.acquire(tmp_path / "episode1")
        assert evidence is None
        await harness.finish_episode()
        assert bridge.closed == process.closed == 0
        next_bridge, next_process, evidence = await harness.acquire(tmp_path / "episode2")
        assert next_bridge is bridge and next_process is process
        assert evidence["success"] and harness.launches == 1
        # A reset is NOT silently represented as restoration of the save fixture.
        assert not (tmp_path / "episode2" / "Save").exists()
        assert json.loads((tmp_path / "episode2" / "reset.json").read_text())["success"]
    assert len(resets) == 1 and process.closed == bridge.closed == 1
    seed_config = json.loads((harness.seed_home / "shipofharkinian.json").read_text())
    assert seed_config["CVars"]["gSettings"]["ResetBtn"] == 128
    assert (harness.seed_home / "Save" / "file2.sav").read_bytes() == b"fixture-not-a-real-save"


@pytest.mark.asyncio
async def test_failed_reset_is_retained_and_never_falls_back_to_relaunch(tmp_path, monkeypatch):
    harness = pool(tmp_path, reuse=True)

    async def failed(*args, **kwargs):
        return {"success": False, "reason": "reset_effect_timeout"}

    monkeypatch.setattr(qa_session, "reset_to_title", failed)
    with pytest.raises(RuntimeError, match="reset_effect_timeout"):
        async with harness:
            bridge, process, _ = await harness.acquire(tmp_path / "episode1")
            await harness.finish_episode()
            await harness.acquire(tmp_path / "episode2")
    assert harness.launches == 1 and process.closed == bridge.closed == 1
    assert not json.loads((tmp_path / "episode2" / "reset.json").read_text())["success"]


@pytest.mark.asyncio
async def test_demonstration_groups_get_separate_processes_and_copies(tmp_path):
    harness = pool(tmp_path, reuse=False)
    async with harness:
        first_bridge, first, _ = await harness.acquire(tmp_path / "group1")
        await harness.finish_episode()
        second_bridge, second, _ = await harness.acquire(tmp_path / "group2")
        assert first is not second and first_bridge is not second_bridge
        assert first.closed == first_bridge.closed == 1
        assert (tmp_path / "group2" / "Save" / "file2.sav").exists()
    assert harness.launches == 2 and second.closed == second_bridge.closed == 1


def test_reset_episodes_cannot_be_used_as_independent_training_split_groups():
    assert qa_session.evaluation_reuses_process(evaluating=True, requested=None)
    assert not qa_session.evaluation_reuses_process(evaluating=False, requested=None)
    assert not qa_session.evaluation_reuses_process(evaluating=True, requested=False)
    with pytest.raises(ValueError, match="independent native-instance"):
        qa_session.evaluation_reuses_process(evaluating=False, requested=True)
