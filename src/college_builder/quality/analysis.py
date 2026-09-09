"""Gate 4 source-analysis completeness classification with independent verification."""

from __future__ import annotations

import re
from datetime import UTC, datetime

from pydantic import ValidationError

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.question import AnalysisType
from college_builder.domain.source import JsonValue, NormalizedQA
from college_builder.providers.base import (
    ModelClassificationRequest,
    ModelDecision,
    StructuredModelProvider,
)
from college_builder.quality.engine import GateContext, GateResultEvidence

_VALID_ANALYSIS_LABELS = tuple(item.value for item in AnalysisType)
_NEGATIVE_LABELS = ("TOO_SHALLOW", "UNRELATED", "UNCERTAIN")
ANALYSIS_LABELS = _VALID_ANALYSIS_LABELS + _NEGATIVE_LABELS
_ANALYSIS_SPAN_RE = re.compile(r"(?i)^analysis:(?P<start>\d+)-(?P<end>\d+)$")
_URL_ONLY_RE = re.compile(r"(?is)^\s*(?:https?://\S+|\[[^\]]*\]\(https?://[^)]+\))\s*$")
_PAGE_ONLY_RE = re.compile(
    r"(?is)^\s*(?:see\s+)?(?:page\s*|p\.\s*)?\d+(?:\s*[-–]\s*\d+)?\.?\s*$"
)
_CN_PAGE_ONLY_RE = re.compile(r"^\s*第?\s*\d+\s*页\s*[。.]?\s*$")
_ANSWER_ONLY_RE = re.compile(
    r"(?is)^\s*(?:answer\s*[:：]|final\s+answer\s*[:：]|therefore|thus|hence|"
    r"因此|所以|故)\s*(?P<answer>.+?)\s*$"
)
_SAME_AS_ABOVE = frozenset({"同上", "same as above", "as above"})


class AnalysisGate:
    """Gate 4: require material source reasoning without rewriting source analysis."""

    name = "gate_4_original_analysis"

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
            raise ValueError("analysis prompt must be non-empty")
        if not isinstance(prompt_version, str) or not prompt_version.strip():
            raise ValueError("prompt_version must be non-empty")
        if not 0.0 <= verify_threshold < pass_threshold <= 1.0:
            raise ValueError("analysis thresholds must satisfy 0 <= verify < pass <= 1")
        self.prompt = prompt
        self.prompt_version = prompt_version
        self.pass_threshold = pass_threshold
        self.verify_threshold = verify_threshold

    def evaluate(self, candidate: NormalizedQA, context: GateContext) -> GateResultEvidence:
        precheck_reason = _deterministic_reject_reason(candidate)
        if precheck_reason is not None:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code=precheck_reason,
                evidence_payload={"analysis": candidate.analysis},
            )

        request = ModelClassificationRequest(
            task=self.name,
            prompt=self.prompt,
            inputs={
                "question": candidate.question,
                "answer": candidate.answer,
                "analysis": candidate.analysis,
            },
            allowed_labels=ANALYSIS_LABELS,
        )
        primary = self._call_provider(self.primary, request, context, "primary")
        if isinstance(primary, GateResultEvidence):
            return primary

        primary_evidence: dict[str, JsonValue] = {
            "primary": _decision_evidence(self.primary, primary),
        }
        if primary.label not in ANALYSIS_LABELS:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="MALFORMED_MODEL_DECISION",
                evidence_payload=primary_evidence,
            )

        if primary.score >= self.pass_threshold:
            return self._finalize_high_band(
                primary,
                candidate.analysis,
                context,
                primary_evidence,
            )

        if primary.score < self.verify_threshold:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=primary.score,
                reason_code="ANALYSIS_UNCERTAIN",
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
        if verifier.label not in ANALYSIS_LABELS:
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
        if verifier.score < self.verify_threshold:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code="ANALYSIS_UNCERTAIN",
                evidence_payload=evidence_payload,
            )
        if primary.label in _NEGATIVE_LABELS:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code=_semantic_reject_reason(primary.label),
                evidence_payload=evidence_payload,
            )
        if not _has_analysis_evidence(
            primary,
            candidate.analysis,
        ) or not _has_analysis_evidence(verifier, candidate.analysis):
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=score,
                reason_code="ANALYSIS_UNCERTAIN",
                evidence_payload=evidence_payload,
            )

        evidence_payload["analysis_type"] = primary.label
        return self._result(
            context,
            verdict=GateVerdict.VERIFY,
            score=score,
            reason_code="ANALYSIS_REVIEW_REQUIRED",
            evidence_payload=evidence_payload,
        )

    def _finalize_high_band(
        self,
        primary: ModelDecision,
        analysis: str,
        context: GateContext,
        evidence_payload: dict[str, JsonValue],
    ) -> GateResultEvidence:
        if primary.label in _NEGATIVE_LABELS:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=primary.score,
                reason_code=_semantic_reject_reason(primary.label),
                evidence_payload=evidence_payload,
            )
        if not _has_analysis_evidence(primary, analysis):
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=primary.score,
                reason_code="ANALYSIS_UNCERTAIN",
                evidence_payload=evidence_payload,
            )
        evidence_payload["analysis_type"] = primary.label
        return self._result(
            context,
            verdict=GateVerdict.PASS,
            score=primary.score,
            reason_code="ANALYSIS_CONFIRMED",
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
        except Exception as exc:  # noqa: BLE001 - model boundary must fail closed.
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


def _deterministic_reject_reason(candidate: NormalizedQA) -> str | None:
    stripped = candidate.analysis.strip()
    if not stripped or stripped == "略":
        return "ANALYSIS_MISSING"

    source_answer = candidate.answer.strip()
    if source_answer and _canonical_answer(stripped) == _canonical_answer(source_answer):
        return "ANALYSIS_TOO_SHALLOW"
    wrapped_answer = _answer_only_payload(stripped)
    if wrapped_answer is not None:
        if not source_answer:
            return "ANALYSIS_TOO_SHALLOW"
        if _canonical_answer(wrapped_answer) == _canonical_answer(source_answer):
            return "ANALYSIS_TOO_SHALLOW"

    lowered = stripped.lower()
    if lowered in _SAME_AS_ABOVE:
        return "ANALYSIS_TOO_SHALLOW"
    if _URL_ONLY_RE.fullmatch(stripped):
        return "ANALYSIS_TOO_SHALLOW"
    if _PAGE_ONLY_RE.fullmatch(stripped) or _CN_PAGE_ONLY_RE.fullmatch(stripped):
        return "ANALYSIS_TOO_SHALLOW"
    return None


def _answer_only_payload(text: str) -> str | None:
    match = _ANSWER_ONLY_RE.fullmatch(text)
    if match is None:
        return None
    payload = match.group("answer").strip()
    return payload or None


def _canonical_answer(text: str) -> str:
    normalized = text.strip().rstrip(".。").strip().casefold()
    return re.sub(r"\s+", "", normalized)


def _semantic_reject_reason(label: str) -> str:
    return {
        "TOO_SHALLOW": "ANALYSIS_TOO_SHALLOW",
        "UNRELATED": "ANALYSIS_UNRELATED",
        "UNCERTAIN": "ANALYSIS_UNCERTAIN",
    }.get(label, "ANALYSIS_UNCERTAIN")


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


def _has_analysis_evidence(decision: ModelDecision, analysis: str) -> bool:
    for reference in decision.evidence_references:
        match = _ANALYSIS_SPAN_RE.fullmatch(reference.strip())
        if match is None:
            continue
        start = int(match.group("start"))
        end = int(match.group("end"))
        if 0 <= start < end <= len(analysis) and analysis[start:end].strip():
            return True
    return False
