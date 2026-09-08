"""Deterministic structured model provider for unit tests and offline fixtures."""

from __future__ import annotations

from college_builder.providers.base import ModelClassificationRequest, ModelDecision


class FakeStructuredModelProvider:
    """Return scripted decisions in order and record every domain request."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        decisions: tuple[ModelDecision, ...],
    ) -> None:
        self.provider = _nonempty_identity(provider, "provider")
        self.model = _nonempty_identity(model, "model")
        self._decisions = tuple(decisions)
        self._requests: list[ModelClassificationRequest] = []
        self._next_decision = 0

    @property
    def requests(self) -> tuple[ModelClassificationRequest, ...]:
        return tuple(self._requests)

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        self._requests.append(request)
        if self._next_decision >= len(self._decisions):
            raise RuntimeError("no scripted decision remains for fake provider")
        decision = self._decisions[self._next_decision]
        self._next_decision += 1
        return decision


def _nonempty_identity(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value
