"""Owned QA process lifecycle; ordinary reset stays outside the game policy."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from .g1 import write_json
from .qa_reset import reset_to_title, soft_reset_config


class QaProcessPool:
    """Reuse one owned game for evaluation, or isolate demonstration groups."""

    def __init__(self, executable, seed_home, token, *, reuse, bridge_factory, process_factory, bind):
        self.executable, self.seed_home, self.token = executable, seed_home, token
        self.reuse = reuse
        self.bridge_factory, self.process_factory, self.bind = bridge_factory, process_factory, bind
        self.bridge = self.process = self.native_home = None
        self.launches = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        await self.close()

    async def acquire(self, episode_home):
        if self.process is not None:
            if not self.reuse or self.process.child.poll() is not None:
                raise RuntimeError("Owned QA process cannot be reused")
            episode_home.mkdir()
            # Check the actual QA configuration. Never install a reset binding
            # into the operator's original home or silently relaunch on failure.
            config = json.loads((self.native_home / "shipofharkinian.json").read_text(encoding="utf-8"))
            buttons = config.get("CVars", {}).get("gSettings", {}).get("ResetBtn")
            reset = await reset_to_title(self.bridge, configured_buttons=buttons)
            write_json(episode_home / "reset.json", reset)
            if not reset["success"]:
                raise RuntimeError(f"Normal QA reset failed: {reset['reason']}")
            return self.bridge, self.process, reset
        shutil.copytree(self.seed_home, episode_home)
        if self.reuse:
            config_path = episode_home / "shipofharkinian.json"
            write_json(config_path, soft_reset_config(json.loads(config_path.read_text(encoding="utf-8"))))
        self.native_home = episode_home
        self.bridge = self.bridge_factory(self.token, allow_simulator=False)
        await self.bind(self.bridge, 0)
        environment = dict(os.environ)
        environment.update(ZELDA_BRIDGE_TOKEN=self.token,
                           ZELDA_BRIDGE_PORT=str(self.bridge.transport.get_extra_info("sockname")[1]),
                           SHIP_HOME=str(episode_home))
        self.process = self.process_factory(Path(self.executable), episode_home, environment)
        self.launches += 1
        return self.bridge, self.process, None

    async def finish_episode(self):
        if self.bridge:
            self.bridge.release()
        if not self.reuse:
            await self.close()

    async def close(self):
        try:
            if self.bridge:
                self.bridge.close()
        finally:
            self.bridge = None
            if self.process:
                try:
                    await self.process.close()
                finally:
                    self.process = None


def evaluation_reuses_process(*, evaluating, requested):
    reuse = evaluating if requested is None else requested
    if reuse and not evaluating:
        raise ValueError("Demonstrations require independent native-instance groups for dataset splits")
    return reuse
