"""Evidence-based University/STEM and problem classification quality gates."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

from pydantic import ValidationError

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import JsonValue, NormalizedQA
from college_builder.providers.base import (
    ModelClassificationRequest,
    ModelDecision,
    StructuredModelProvider,
)
from college_builder.quality.engine import GateContext, GateResultEvidence

UNIVERSITY_LABELS = (
    "UNIVERSITY_STEM",
    "NON_UNIVERSITY_STEM",
    "NON_STEM",
    "K12",
    "UNCERTAIN",
)
PROBLEM_POSITIVE_LABELS = frozenset(
    {
        "CALCULATION",
        "PROOF",
        "DERIVATION",
        "CONCEPTUAL",
        "ALGORITHM",
        "PROGRAMMING",
        "ENGINEERING",
    }
)
PROBLEM_LABELS = tuple(
    sorted(
        PROBLEM_POSITIVE_LABELS
        | {
            "SOFTWARE_USE",
            "DEBUG_HELP",
            "OPINION",
            "RESOURCE_REQUEST",
            "META",
            "NON_PROBLEM",
            "UNCERTAIN",
        }
    )
)


class _DualProviderClassificationGate:
    name: ClassVar[str]
    task: ClassVar[str]
    allowed_labels: ClassVar[tuple[str, ...]]
    positive_labels: ClassVar[frozenset[str]]
    negative_reason: ClassVar[str]
    low_confidence_reason: ClassVar[str]
    pass_reason: ClassVar[str]
    verify_reason: ClassVar[str]

    def __init__(
        self,
        *,
        primary: StructuredModelProvider,
        verifier: StructuredModelProvider,
        prompt: str,
        prompt_version: str,
        pass_threshold: float = 0.98,
        verify_threshold: float = 0.90,
    ) -> None:
        self.primary = primary
        self.verifier = verifier
        self.provider, self.model = _provider_identity(primary, "primary")
        _provider_identity(verifier, "verifier")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("classification prompt must be non-empty")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("prompt_version must be non-empty")
        if not 0.0 <= verify_threshold < pass_threshold <= 1.0:
            raise ValueError("classification thresholds must satisfy 0 <= verify < pass <= 1")
        self.prompt = prompt
        self.prompt_version = prompt_version
        self.pass_threshold = pass_threshold
        self.verify_threshold = verify_threshold

    def evaluate(self, candidate: NormalizedQA, context: GateContext) -> GateResultEvidence:
        primary_identity = _provider_identity(self.primary, "primary")
        verifier_identity = _provider_identity(self.verifier, "verifier")
        if primary_identity == verifier_identity:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="VERIFIER_NOT_INDEPENDENT",
                evidence_payload={
                    "primary_identity": list(primary_identity),
                    "verifier_identity": list(verifier_identity),
                },
            )

        request = ModelClassificationRequest(
            task=self.task,
            prompt=self.prompt,
            inputs=_classification_inputs(candidate),
            allowed_labels=self.allowed_labels,
        )
        primary = self._call_provider(self.primary, request, context, "primary")
        if isinstance(primary, GateResultEvidence):
            return primary
        verifier = self._call_provider(self.verifier, request, context, "verifier")
        if isinstance(verifier, GateResultEvidence):
            return verifier

        evidence_payload = {
            "primary": _decision_evidence(self.primary, primary),
            "verifier": _decision_evidence(self.verifier, verifier),
        }
        if primary.label not in self.allowed_labels or verifier.label not in self.allowed_labels:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="MALFORMED_MODEL_DECISION",
                evidence_payload=evidence_payload,
            )
        score = min(primary.score, verifier.score)
        if primary.label != verifier.label:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code="MODEL_DECISION_CONFLICT",
                evidence_payload=evidence_payload,
            )
        if primary.label not in self.positive_labels:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code=self.negative_reason,
                evidence_payload=evidence_payload,
            )
        if score >= self.pass_threshold:
            verdict = GateVerdict.PASS
            reason_code = self.pass_reason
        elif score >= self.verify_threshold:
            verdict = GateVerdict.VERIFY
            reason_code = self.verify_reason
        else:
            verdict = GateVerdict.REJECT
            reason_code = self.low_confidence_reason
        return self._result(
            context,
            verdict=verdict,
            score=score,
            reason_code=reason_code,
            evidence_payload=evidence_payload,
        )

    def _call_provider(
        self,
        provider: StructuredModelProvider,
        request: ModelClassificationRequest,
        context: GateContext,
        role: str,
    ) -> ModelDecision | GateResultEvidence:
        try:
            output = provider.classify(request)
        except Exception as exc:  # noqa: BLE001 - model boundary is fail-closed.
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="MODEL_PROVIDER_ERROR",
                evidence_payload={
                    "provider_role": role,
                    "exception_type": type(exc).__name__,
                },
            )
        try:
            validation_input: object = output
            if isinstance(output, ModelDecision):
                validation_input = output.model_dump(mode="python", warnings=False)
            return ModelDecision.model_validate(validation_input)
        except (ValidationError, TypeError, ValueError):
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="MALFORMED_MODEL_DECISION",
                evidence_payload={"provider_role": role},
            )

    def _result(
        self,
        context: GateContext,
        *,
        verdict: GateVerdict,
        score: float,
        reason_code: str,
        evidence_payload: dict[str, JsonValue],
    ) -> GateResultEvidence:
        return GateResultEvidence(
            gate_name=self.name,
            verdict=verdict,
            score=score,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code=reason_code,
            evidence_payload=evidence_payload,
            timestamp=datetime.now(UTC),
        )


class UniversityStemGate(_DualProviderClassificationGate):
    """Gate 1: require independently supported university-level STEM evidence."""

    name = "gate_1_university_stem"
    task = "gate_1_university_stem"
    allowed_labels = UNIVERSITY_LABELS
    positive_labels = frozenset({"UNIVERSITY_STEM"})
    negative_reason = "NOT_UNIVERSITY_STEM"
    low_confidence_reason = "UNIVERSITY_STEM_LOW_CONFIDENCE"
    pass_reason = "UNIVERSITY_STEM_CONFIRMED"
    verify_reason = "UNIVERSITY_STEM_REVIEW_REQUIRED"


class ProblemGate(_DualProviderClassificationGate):
    """Gate 2: require a concrete STEM problem/exercise rather than general discussion."""

    name = "gate_2_problem"
    task = "gate_2_problem"
    allowed_labels = PROBLEM_LABELS
    positive_labels = PROBLEM_POSITIVE_LABELS
    negative_reason = "NOT_A_PROBLEM"
    low_confidence_reason = "PROBLEM_LOW_CONFIDENCE"
    pass_reason = "PROBLEM_CONFIRMED"
    verify_reason = "PROBLEM_REVIEW_REQUIRED"


def _classification_inputs(candidate: NormalizedQA) -> dict[str, JsonValue]:
    dump = candidate.model_dump(mode="json")
    metadata = dump["metadata"]
    if not isinstance(metadata, dict):
        metadata = {}
    return {
        "question": candidate.question,
        "source_hints": {
            "subject_candidates": list(candidate.subject_candidates),
            "metadata": metadata,
        },
    }


def _provider_identity(provider: StructuredModelProvider, role: str) -> tuple[str, str]:
    provider_name = getattr(provider, "provider", None)
    model = getattr(provider, "model", None)
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError(f"{role} provider identity must be non-empty")
    if not isinstance(model, str) or not model.strip():
        raise ValueError(f"{role} model identity must be non-empty")
    return provider_name, model


def _decision_evidence(
    provider: StructuredModelProvider,
    decision: ModelDecision,
) -> dict[str, JsonValue]:
    return {
        "provider": provider.provider,
        "model": provider.model,
        "label": decision.label,
        "score": decision.score,
        "reason_code": decision.reason_code,
        "evidence_references": list(decision.evidence_references),
    }
