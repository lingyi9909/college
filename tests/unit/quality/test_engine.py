from __future__ import annotations

from datetime import UTC, datetime

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.quality.engine import (
    GateContext,
    GateEngine,
    GateResultEvidence,
)


class PassingGate:
    name = "gate_test"
    provider = "deterministic"
    model = "test_rule_v1"
    prompt_version = "not_applicable"

    def evaluate(self, candidate: object, context: GateContext) -> GateResultEvidence:
        return GateResultEvidence(
            gate_name=self.name,
            verdict=GateVerdict.PASS,
            score=1.0,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code="TEST_PASS",
            evidence_payload={"candidate": str(candidate)},
            timestamp=datetime.now(UTC),
        )


class RejectingGate(PassingGate):
    name = "gate_reject"

    def evaluate(self, candidate: object, context: GateContext) -> GateResultEvidence:
        return GateResultEvidence(
            gate_name=self.name,
            verdict=GateVerdict.REJECT,
            score=0.0,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code="TEST_REJECT",
            evidence_payload={},
            timestamp=datetime.now(UTC),
        )


class ExplodingGate(PassingGate):
    name = "gate_explodes"

    def evaluate(self, candidate: object, context: GateContext) -> GateResultEvidence:
        raise RuntimeError("boom")


class MissingScoreGate(PassingGate):
    name = "gate_missing_score"

    def evaluate(self, candidate: object, context: GateContext) -> object:
        return {
            "gate_name": self.name,
            "verdict": "PASS",
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "config_version": context.config_version,
            "reason_code": "INVALID",
            "evidence_payload": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }


class UnknownVerdictGate(PassingGate):
    name = "gate_unknown_verdict"

    def evaluate(self, candidate: object, context: GateContext) -> object:
        return {
            "gate_name": self.name,
            "verdict": "MAYBE",
            "score": 1.0,
            "provider": self.provider,
            "model": self.model,
            "prompt_version": self.prompt_version,
            "config_version": context.config_version,
            "reason_code": "INVALID",
            "evidence_payload": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }


class WrongNameGate(PassingGate):
    name = "declared_name"

    def evaluate(self, candidate: object, context: GateContext) -> GateResultEvidence:
        result = super().evaluate(candidate, context)
        return result.model_copy(update={"gate_name": "different_name"})


class CountingGate(PassingGate):
    name = "gate_counting"

    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, candidate: object, context: GateContext) -> GateResultEvidence:
        self.calls += 1
        return super().evaluate(candidate, context)


class MissingIdentityGate:
    name = "gate_identity"
    provider = "deterministic"
    model = "identity_rule_v1"
    prompt_version = "not_applicable"

    def __init__(self, missing: tuple[str, ...]) -> None:
        self._missing = missing
        for attribute in missing:
            delattr(self, attribute)

    def __getattribute__(self, name: str) -> object:
        if name in {"name", "provider", "model", "prompt_version"}:
            missing = object.__getattribute__(self, "_missing")
            if name in missing:
                raise AttributeError(name)
        return object.__getattribute__(self, name)

    def evaluate(self, candidate: object, context: GateContext) -> object:
        fallback = {
            "name": "unknown_gate",
            "provider": "unknown_provider",
            "model": "unknown_model",
            "prompt_version": "unknown_prompt",
        }
        identity = {
            field: fallback[field] if field in self._missing else getattr(self, field)
            for field in fallback
        }
        return {
            "gate_name": identity["name"],
            "verdict": "PASS",
            "score": 1.0,
            "provider": identity["provider"],
            "model": identity["model"],
            "prompt_version": identity["prompt_version"],
            "config_version": context.config_version,
            "reason_code": "SHOULD_NOT_PASS",
            "evidence_payload": {},
            "timestamp": datetime.now(UTC).isoformat(),
        }


class ConstructedMissingScoreGate(PassingGate):
    name = "gate_construct_missing_score"

    def evaluate(self, candidate: object, context: GateContext) -> GateResultEvidence:
        return GateResultEvidence.model_construct(
            gate_name=self.name,
            verdict=GateVerdict.PASS,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code="INVALID_CONSTRUCTED",
            evidence_payload={},
            timestamp=datetime.now(UTC),
        )


class ConstructedUnknownVerdictGate(PassingGate):
    name = "gate_construct_unknown_verdict"

    def evaluate(self, candidate: object, context: GateContext) -> GateResultEvidence:
        return GateResultEvidence.model_construct(
            gate_name=self.name,
            verdict="MAYBE",
            score=1.0,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code="INVALID_CONSTRUCTED",
            evidence_payload={},
            timestamp=datetime.now(UTC),
        )


def _context() -> GateContext:
    return GateContext(config_version="pilot-v1")


def test_engine_preserves_complete_auditable_gate_evidence() -> None:
    result = GateEngine(gates=(PassingGate(),), context=_context()).run("candidate-1")

    assert result.verdict is GateVerdict.PASS
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.gate_name == "gate_test"
    assert evidence.score == 1.0
    assert evidence.reason_code == "TEST_PASS"
    assert evidence.evidence_payload == {"candidate": "candidate-1"}
    assert evidence.provider == "deterministic"
    assert evidence.model == "test_rule_v1"
    assert evidence.prompt_version == "not_applicable"
    assert evidence.config_version == "pilot-v1"
    assert evidence.timestamp.tzinfo is not None


def test_gate_exception_fails_closed_to_reject() -> None:
    result = GateEngine(gates=(ExplodingGate(),), context=_context()).run("candidate")

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].verdict is GateVerdict.REJECT
    assert result.evidence[0].score == 0.0
    assert result.evidence[0].reason_code == "GATE_EXCEPTION"
    assert result.evidence[0].evidence_payload["exception_type"] == "RuntimeError"


def test_missing_score_fails_closed_to_reject() -> None:
    result = GateEngine(gates=(MissingScoreGate(),), context=_context()).run("candidate")

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "MALFORMED_GATE_OUTPUT"


def test_unknown_verdict_fails_closed_to_reject() -> None:
    result = GateEngine(gates=(UnknownVerdictGate(),), context=_context()).run("candidate")

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "MALFORMED_GATE_OUTPUT"


def test_gate_name_mismatch_fails_closed_to_reject() -> None:
    result = GateEngine(gates=(WrongNameGate(),), context=_context()).run("candidate")

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "GATE_NAME_MISMATCH"


def test_reject_short_circuits_later_gates() -> None:
    later = CountingGate()
    result = GateEngine(
        gates=(RejectingGate(), later),
        context=_context(),
    ).run("candidate")

    assert result.verdict is GateVerdict.REJECT
    assert later.calls == 0
    assert len(result.evidence) == 1


@pytest.mark.parametrize(
    "missing",
    [
        ("name",),
        ("provider",),
        ("model",),
        ("prompt_version",),
        ("name", "provider", "model", "prompt_version"),
    ],
)
def test_missing_gate_execution_identity_fails_closed(missing: tuple[str, ...]) -> None:
    result = GateEngine(gates=(MissingIdentityGate(missing),), context=_context()).run("candidate")

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "GATE_IDENTITY_MISSING"


def test_model_construct_missing_score_fails_closed_to_reject() -> None:
    result = GateEngine(gates=(ConstructedMissingScoreGate(),), context=_context()).run("candidate")

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "MALFORMED_GATE_OUTPUT"


def test_model_construct_unknown_verdict_fails_closed_to_reject() -> None:
    result = GateEngine(gates=(ConstructedUnknownVerdictGate(),), context=_context()).run("candidate")

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "MALFORMED_GATE_OUTPUT"
