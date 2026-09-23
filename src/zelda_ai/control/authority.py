"""Task-local ownership fences. Old cleanup cannot release a successor's input."""
from __future__ import annotations

import asyncio
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from collections.abc import Iterator


class ControlRevoked(asyncio.CancelledError):
    """The originating run/skill no longer owns the controller."""


class InputAuthority:
    def __init__(self) -> None:
        self.generation = 0
        self.epoch = 0
        self.enabled = True  # Direct/manual skill tests are supported before a run.
        self.label = "idle"
        self._run: ContextVar[int | None] = ContextVar("input_run", default=None)
        self._owner: ContextVar[tuple[int, int] | None] = ContextVar("input_owner", default=None)

    def enable(self) -> int:
        self.generation += 1
        self.epoch += 1
        self.enabled = True
        self.label = "idle"
        return self.generation

    def revoke(self) -> int:
        self.generation += 1
        self.epoch += 1
        self.enabled = False
        self.label = "human"
        return self.generation

    def valid(self, *, owner: bool = True) -> bool:
        run = self._run.get()
        token = self._owner.get()
        return (self.enabled and (run is None or run == self.generation)
                and (not owner or token is None or token == (self.generation, self.epoch)))

    def check(self, *, owner: bool = True) -> None:
        if not self.valid(owner=owner):
            raise ControlRevoked("input_ownership_revoked")

    @contextmanager
    def run_scope(self) -> Iterator[None]:
        self.check(owner=False)
        token = self._run.set(self.generation)
        try:
            yield
        finally:
            self._run.reset(token)

    @contextmanager
    def scope(self, label: str) -> Iterator[None]:
        self.check(owner=False)
        self.epoch += 1
        epoch = self.epoch
        self.label = label
        token = self._owner.set((self.generation, epoch))
        try:
            yield
        finally:
            if self.valid() and self.epoch == epoch:
                self.label = "idle"
            self._owner.reset(token)


def owned(bridge, label: str):
    """Allow minimal test/simulator bridges while fencing the real UDP bridge."""
    scope = getattr(bridge, "input_scope", None)
    return scope(label) if scope is not None else nullcontext()
