from dataclasses import dataclass

from ..models import Usage


@dataclass
class InferenceResult:
    text: str
    usage: Usage


class ProviderFailure(RuntimeError):
    def __init__(self, message: str, usage: Usage | None = None):
        super().__init__(message)
        self.usage = usage or Usage()
