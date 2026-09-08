"""Fail-closed quality gate orchestration and auditable gate evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Annotated, Protocol, TypeVar, runtime_checkable

from pydantic import ConfigDict, Field, ValidationError, field_validator

from college_builder.domain.evidence import GateEvidence, GateVerdict
from college_builder.domain.source import JsonValue, RawSourceRecord

NonEmptyStr = Annotated[str, Field(min_length=1)]


class GateResultEvidence(GateEvidence):
    """Complete audit record required for every GateEngine-consumed result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason_code: NonEmptyStr
    evidence_payload: dict[str, JsonValue] = Field(default_factory=dict)
    timestamp: datetime

    @field_validator("timestamp")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("gate evidence timestamp must be timezone-aware")
        return value


@dataclass(frozen=True, slots=True)
class GateContext:
    """Immutable execution context shared by deterministic and model-backed gates."""

    config_version: str
    raw_records: Mapping[str, RawSourceRecord] = field(default_factory=dict)
    missing_image_sources: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.config_version.strip():
            raise ValueError("config_version must be non-empty")
        object.__setattr__(self, "raw_records", MappingProxyType(dict(self.raw_records)))
        object.__setattr__(self, "missing_image_sources", frozenset(self.missing_image_sources))


CandidateT_contra = TypeVar("CandidateT_contra", contravariant=True)


@runtime_checkable
class QualityGate(Protocol[CandidateT_contra]):
    """Provider-neutral gate contract consumed by GateEngine."""

    name: str
    provider: str
    model: str
    prompt_version: str

    def evaluate(
        self,
        candidate: CandidateT_contra,
        context: GateContext,
    ) -> GateEvidence: ...


@dataclass(frozen=True, slots=True)
class GateRunResult:
    """Aggregate verdict plus ordered audit evidence from a gate run."""

    verdict: GateVerdict
    evidence: tuple[GateResultEvidence, ...]


class GateEngine[CandidateT]:
    """Run gates in order and convert every unsafe path into an explicit REJECT."""

    def __init__(
        self,
        *,
        gates: Sequence[QualityGate[CandidateT]],
        context: GateContext,
    ) -> None:
        self._gates = tuple(gates)
        self._context = context

    def run(self, candidate: CandidateT) -> GateRunResult:
        if not self._gates:
            evidence = self._engine_reject("NO_GATES_CONFIGURED")
            return GateRunResult(verdict=GateVerdict.REJECT, evidence=(evidence,))

        evidence_items: list[GateResultEvidence] = []
        aggregate = GateVerdict.PASS
        for gate in self._gates:
            evidence = self._safe_evaluate(gate, candidate)
            evidence_items.append(evidence)
            if evidence.verdict is GateVerdict.REJECT:
                return GateRunResult(
                    verdict=GateVerdict.REJECT,
                    evidence=tuple(evidence_items),
                )
            if evidence.verdict is GateVerdict.VERIFY:
                aggregate = GateVerdict.VERIFY

        return GateRunResult(verdict=aggregate, evidence=tuple(evidence_items))

    def _safe_evaluate(
        self,
        gate: QualityGate[CandidateT],
        candidate: CandidateT,
    ) -> GateResultEvidence:
        identity = _gate_identity(gate)
        if identity is None:
            return self._engine_reject("GATE_IDENTITY_MISSING")

        try:
            output = gate.evaluate(candidate, self._context)
        except Exception as exc:  # noqa: BLE001 - gate boundary must fail closed.
            return self._reject_for_gate(
                identity,
                reason_code="GATE_EXCEPTION",
                evidence_payload={"exception_type": type(exc).__name__},
            )

        try:
            validation_input: object = output
            if isinstance(output, GateEvidence):
                validation_input = output.model_dump(mode="python", warnings=False)
            evidence = GateResultEvidence.model_validate(validation_input)
        except (ValidationError, TypeError, ValueError):
            return self._reject_for_gate(
                identity,
                reason_code="MALFORMED_GATE_OUTPUT",
                evidence_payload={},
            )

        if evidence.gate_name != identity.name:
            return self._reject_for_gate(
                identity,
                reason_code="GATE_NAME_MISMATCH",
                evidence_payload={"reported_gate_name": evidence.gate_name},
            )
        if evidence.config_version != self._context.config_version:
            return self._reject_for_gate(
                identity,
                reason_code="GATE_CONFIG_VERSION_MISMATCH",
                evidence_payload={"reported_config_version": evidence.config_version},
            )
        if (
            evidence.provider != identity.provider
            or evidence.model != identity.model
            or evidence.prompt_version != identity.prompt_version
        ):
            return self._reject_for_gate(
                identity,
                reason_code="GATE_IDENTITY_MISMATCH",
                evidence_payload={},
            )
        return evidence

    def _engine_reject(self, reason_code: str) -> GateResultEvidence:
        return GateResultEvidence(
            gate_name="gate_engine",
            verdict=GateVerdict.REJECT,
            score=0.0,
            provider="deterministic",
            model="gate_engine_v1",
            prompt_version="not_applicable",
            config_version=self._context.config_version,
            reason_code=reason_code,
            evidence_payload={},
            timestamp=datetime.now(UTC),
        )

    def _reject_for_gate(
        self,
        identity: _GateIdentity,
        *,
        reason_code: str,
        evidence_payload: dict[str, JsonValue],
    ) -> GateResultEvidence:
        return GateResultEvidence(
            gate_name=identity.name,
            verdict=GateVerdict.REJECT,
            score=0.0,
            provider=identity.provider,
            model=identity.model,
            prompt_version=identity.prompt_version,
            config_version=self._context.config_version,
            reason_code=reason_code,
            evidence_payload=evidence_payload,
            timestamp=datetime.now(UTC),
        )


@dataclass(frozen=True, slots=True)
class _GateIdentity:
    name: str
    provider: str
    model: str
    prompt_version: str


def _gate_identity(gate: object) -> _GateIdentity | None:
    name = _identity_value(gate, "name")
    provider = _identity_value(gate, "provider")
    model = _identity_value(gate, "model")
    prompt_version = _identity_value(gate, "prompt_version")
    if name is None or provider is None or model is None or prompt_version is None:
        return None
    return _GateIdentity(
        name=name,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
    )


def _identity_value(gate: object, attribute: str) -> str | None:
    value = getattr(gate, attribute, None)
    if isinstance(value, str) and value.strip():
        return value
    return None
