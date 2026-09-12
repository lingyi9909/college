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
        "MULTIPLE_CHOICE",
        "PROGRAMMING",
        "ALGORITHM",
        "ENGINEERING",
    }
)
PROBLEM_NEGATIVE_LABELS = frozenset(
    {
        "SOFTWARE_USAGE",
        "DEBUG_HELP",
        "CAREER_ADVICE",
        "OPINION",
        "RESOURCE_REQUEST",
        "DISCUSSION",
        "NEWS",
        "META",
    }
)
PROBLEM_LABELS = tuple(
    sorted(PROBLEM_POSITIVE_LABELS | PROBLEM_NEGATIVE_LABELS | {"UNCERTAIN"})
)
_CONTENT_EVIDENCE_PREFIXES = ("question:", "content:")


class _DualProviderClassificationGate:
    name: ClassVar[str]
    task: ClassVar[str]
    allowed_labels: ClassVar[tuple[str, ...]]
    positive_labels: ClassVar[frozenset[str]]
    reject_reasons: ClassVar[dict[str, str]]
    uncertain_reason: ClassVar[str]
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
        request = ModelClassificationRequest(
            task=self.task,
            prompt=self.prompt,
            inputs=_classification_inputs(candidate),
            allowed_labels=self.allowed_labels,
        )
        primary = self._call_provider(self.primary, request, context, "primary")
        if isinstance(primary, GateResultEvidence):
            return primary

        primary_evidence: dict[str, JsonValue] = {
            "primary": _decision_evidence(self.primary, primary),
        }
        if primary.label not in self.allowed_labels:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="MALFORMED_MODEL_DECISION",
                evidence_payload=primary_evidence,
            )

        if primary.score >= self.pass_threshold:
            return self._finalize_high_band(primary, candidate, context, primary_evidence)

        if primary.score < self.verify_threshold:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=primary.score,
                reason_code=self.uncertain_reason,
                evidence_payload=primary_evidence,
            )

        primary_identity = _provider_identity(self.primary, "primary")
        verifier_identity = _provider_identity(self.verifier, "verifier")
        if primary_identity == verifier_identity:
            identity_evidence = dict(primary_evidence)
            identity_evidence.update(
                {
                    "primary_identity": list(primary_identity),
                    "verifier_identity": list(verifier_identity),
                }
            )
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=primary.score,
                reason_code="VERIFIER_NOT_INDEPENDENT",
                evidence_payload=identity_evidence,
            )

        verifier = self._call_provider(self.verifier, request, context, "verifier")
        if isinstance(verifier, GateResultEvidence):
            return verifier

        evidence_payload: dict[str, JsonValue] = {
            "primary": _decision_evidence(self.primary, primary),
            "verifier": _decision_evidence(self.verifier, verifier),
        }
        if verifier.label not in self.allowed_labels:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="MALFORMED_MODEL_DECISION",
                evidence_payload=evidence_payload,
            )

        score = min(primary.score, verifier.score)
        labels_conflict = primary.label != verifier.label and not (
            primary.label in self.positive_labels and verifier.label in self.positive_labels
        )
        if labels_conflict:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code="MODEL_DECISION_CONFLICT",
                evidence_payload=evidence_payload,
            )

        if verifier.score < self.verify_threshold:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code=self.uncertain_reason,
                evidence_payload=evidence_payload,
            )

        if primary.label not in self.positive_labels:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code=self._semantic_reject_reason(primary.label),
                evidence_payload=evidence_payload,
            )

        if not self._has_positive_evidence(primary, candidate) or not self._has_positive_evidence(
            verifier,
            candidate,
        ):
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code=self.uncertain_reason,
                evidence_payload=evidence_payload,
            )

        return self._result(
            context,
            verdict=GateVerdict.VERIFY,
            score=score,
            reason_code=self.verify_reason,
            evidence_payload=evidence_payload,
        )

    def _finalize_high_band(
        self,
        primary: ModelDecision,
        candidate: NormalizedQA,
        context: GateContext,
        evidence_payload: dict[str, JsonValue],
    ) -> GateResultEvidence:
        if primary.label not in self.positive_labels:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=primary.score,
                reason_code=self._semantic_reject_reason(primary.label),
                evidence_payload=evidence_payload,
            )
        if not self._has_positive_evidence(primary, candidate):
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=primary.score,
                reason_code=self.uncertain_reason,
                evidence_payload=evidence_payload,
            )
        return self._result(
            context,
            verdict=GateVerdict.PASS,
            score=primary.score,
            reason_code=self.pass_reason,
            evidence_payload=evidence_payload,
        )

    def _has_positive_evidence(
        self,
        decision: ModelDecision,
        candidate: NormalizedQA,
    ) -> bool:
        del candidate
        return _has_content_evidence(decision)

    def _semantic_reject_reason(self, label: str) -> str:
        return self.reject_reasons.get(label, self.uncertain_reason)

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
    reject_reasons = {
        "NON_STEM": "NON_STEM",
        "NON_UNIVERSITY_STEM": "NOT_UNIVERSITY_LEVEL",
        "K12": "NOT_UNIVERSITY_LEVEL",
        "UNCERTAIN": "UNIVERSITY_LEVEL_UNCERTAIN",
    }
    uncertain_reason = "UNIVERSITY_LEVEL_UNCERTAIN"
    pass_reason = "UNIVERSITY_STEM_CONFIRMED"
    verify_reason = "UNIVERSITY_STEM_REVIEW_REQUIRED"


class ProblemGate(_DualProviderClassificationGate):
    """Gate 2: require a concrete STEM problem/exercise rather than general discussion."""

    name = "gate_2_problem"
    task = "gate_2_problem"
    allowed_labels = PROBLEM_LABELS
    positive_labels = PROBLEM_POSITIVE_LABELS
    reject_reasons = {
        **{label: "NOT_PROBLEM" for label in PROBLEM_NEGATIVE_LABELS},
        "UNCERTAIN": "PROBLEM_TYPE_UNCERTAIN",
    }
    uncertain_reason = "PROBLEM_TYPE_UNCERTAIN"
    pass_reason = "PROBLEM_CONFIRMED"
    verify_reason = "PROBLEM_REVIEW_REQUIRED"

    def _has_positive_evidence(
        self,
        decision: ModelDecision,
        candidate: NormalizedQA,
    ) -> bool:
        return _has_grounded_problem_evidence(decision, candidate.question)


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


def _has_content_evidence(decision: ModelDecision) -> bool:
    return any(
        reference.strip().lower().startswith(_CONTENT_EVIDENCE_PREFIXES)
        for reference in decision.evidence_references
    )


def _has_grounded_problem_evidence(decision: ModelDecision, question: str) -> bool:
    return any(
        _is_grounded_problem_reference(reference, question)
        for reference in decision.evidence_references
    )


def _is_grounded_problem_reference(reference: str, question: str) -> bool:
    stripped = reference.strip()
    lowered = stripped.lower()
    payload: str | None = None
    for prefix in _CONTENT_EVIDENCE_PREFIXES:
        if lowered.startswith(prefix):
            payload = stripped[len(prefix) :].strip()
            break
    if not payload:
        return False

    if len(payload) >= 2 and payload[0] == payload[-1] and payload[0] in {'"', "'"}:
        payload = payload[1:-1].strip()
        if not payload:
            return False

    span_parts = payload.split("-")
    if len(span_parts) == 2 and all(part.isdigit() for part in span_parts):
        start, end = (int(part) for part in span_parts)
        return 0 <= start < end <= len(question) and bool(question[start:end].strip())

    if "..." not in payload:
        return payload in question

    segments = [segment.strip() for segment in payload.split("...") if segment.strip()]
    if not segments:
        return False
    cursor = 0
    for segment in segments:
        position = question.find(segment, cursor)
        if position < 0:
            return False
        cursor = position + len(segment)
    return True
