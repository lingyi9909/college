"""Gate 3 source-grounded answer extraction without model generation."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import JsonValue, NormalizedQA
from college_builder.quality.engine import GateContext, GateResultEvidence

_CONCLUSION_RE = re.compile(
    r"(?is)(?:therefore|thus|hence|final\s+answer\s*[:：]|answer\s*[:：]|因此|所以|故)\s*"
    r"(?P<answer>[^\n]+?)\s*$"
)
_REFERENTIAL_ANSWER_RE = re.compile(
    r"(?is)(?:\b(?:shown|given|stated)\s+above\b|\b(?:see|refer\s+to)\b|"
    r"(?:如上|上述|上面|见上))"
)
_URL_ONLY_RE = re.compile(r"(?is)^\s*(?:https?://\S+|\[[^\]]*\]\(https?://[^)]+\))\s*$")
_PAGE_ONLY_RE = re.compile(
    r"(?is)^\s*(?:see\s+)?(?:page\s*|p\.\s*)\d+(?:\s*[-–]\s*\d+)?\.?\s*$"
)
_CN_PAGE_ONLY_RE = re.compile(r"^\s*(?:见\s*)?第?\s*\d+\s*页\s*[。.]?\s*$")
_REFERENTIAL_ONLY_RE = re.compile(
    r"(?is)^\s*(?:"
    r"(?:see|refer\s+to)\s+(?:the\s+)?(?:(?:answer|result|solution)\s+)?(?:above|below)|"
    r"(?:the\s+)?(?:answer|result|solution)\s+(?:shown|given|stated)\s+(?:above|below)|"
    r"(?:答案|结果|解答|解)?\s*(?:见|如)\s*(?:上|上文|上面|上述|下文|下方|如下)|"
    r"(?:上述|上面|以上|下述|下面)\s*(?:答案|结果|解答|解)?"
    r")\s*[。.]?\s*$"
)
_PLACEHOLDERS = frozenset({"略"})


class AnswerExtraction(BaseModel):
    """Auditable deterministic extraction result tied to an exact source span."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    final_answer: str | None
    source_field: Literal["answer", "analysis"] | None
    start_offset: int | None = Field(default=None, ge=0)
    end_offset: int | None = Field(default=None, ge=0)
    source_span: str | None
    answer_source_exists: bool
    answer_extract_score: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def require_complete_span_for_extracted_answer(self) -> AnswerExtraction:
        has_answer = self.final_answer is not None
        span_fields = (
            self.source_field,
            self.start_offset,
            self.end_offset,
            self.source_span,
        )
        if has_answer and any(value is None for value in span_fields):
            raise ValueError("extracted answer requires complete source-span evidence")
        if not has_answer and any(value is not None for value in span_fields):
            raise ValueError("missing answer cannot carry partial source-span evidence")
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.end_offset <= self.start_offset
        ):
            raise ValueError("answer source span must be non-empty")
        return self


def extract_source_answer(candidate: NormalizedQA) -> AnswerExtraction:
    """Extract only literal source content; never solve, infer, or call a model."""

    answer_span = _trimmed_span(candidate.answer)
    if answer_span is not None:
        start, end, text = answer_span
        if text not in _PLACEHOLDERS and not _is_unextractable_direct_answer(text):
            return _extraction(
                final_answer=text,
                source_field="answer",
                start=start,
                end=end,
                score=1.0,
            )

    analysis_text = candidate.analysis
    analysis_span = _conclusion_span(analysis_text)
    if analysis_span is not None:
        start, end, text = analysis_span
        return _extraction(
            final_answer=text,
            source_field="analysis",
            start=start,
            end=end,
            score=0.995,
        )

    source_exists = _has_meaningful_source(candidate.answer) or _has_meaningful_source(
        candidate.analysis
    )
    return AnswerExtraction(
        final_answer=None,
        source_field=None,
        start_offset=None,
        end_offset=None,
        source_span=None,
        answer_source_exists=source_exists,
        answer_extract_score=0.0,
    )


class AnswerGate:
    """Gate 3: require a traceable source answer at Pilot precision threshold."""

    name = "gate_3_original_answer"
    provider = "deterministic"
    model = "source_answer_v1"
    prompt_version = "not_applicable"

    def __init__(self, *, extract_threshold: float = 0.995) -> None:
        if not 0.0 <= extract_threshold <= 1.0:
            raise ValueError("answer extraction threshold must be within [0, 1]")
        self.extract_threshold = extract_threshold

    def evaluate(self, candidate: NormalizedQA, context: GateContext) -> GateResultEvidence:
        extraction = extract_source_answer(candidate)
        evidence_payload: dict[str, JsonValue] = {
            "answer_source_exists": extraction.answer_source_exists,
            "answer_extract_score": extraction.answer_extract_score,
            "final_answer": extraction.final_answer,
            "source_field": extraction.source_field,
            "start_offset": extraction.start_offset,
            "end_offset": extraction.end_offset,
            "source_span": extraction.source_span,
        }

        if not extraction.answer_source_exists:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="ANSWER_MISSING",
                evidence_payload=evidence_payload,
            )
        if extraction.final_answer is None:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="ANSWER_NOT_EXTRACTABLE",
                evidence_payload=evidence_payload,
            )
        if extraction.source_span is None:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=extraction.answer_extract_score,
                reason_code="ANSWER_SOURCE_UNTRACEABLE",
                evidence_payload=evidence_payload,
            )
        if extraction.answer_extract_score < self.extract_threshold:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=extraction.answer_extract_score,
                reason_code="ANSWER_NOT_EXTRACTABLE",
                evidence_payload=evidence_payload,
            )
        return self._result(
            context,
            verdict=GateVerdict.PASS,
            score=extraction.answer_extract_score,
            reason_code="ANSWER_SOURCE_CONFIRMED",
            evidence_payload=evidence_payload,
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


def _trimmed_span(text: str) -> tuple[int, int, str] | None:
    if not text.strip():
        return None
    start = len(text) - len(text.lstrip())
    end = len(text.rstrip())
    return start, end, text[start:end]


def _conclusion_span(text: str) -> tuple[int, int, str] | None:
    if not text.strip():
        return None
    match = _CONCLUSION_RE.search(text)
    if match is None:
        return None
    start, end = match.span("answer")
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start >= end:
        return None
    answer = text[start:end]
    if answer in _PLACEHOLDERS or _REFERENTIAL_ANSWER_RE.search(answer):
        return None
    return start, end, answer


def _is_unextractable_direct_answer(text: str) -> bool:
    return bool(
        _URL_ONLY_RE.fullmatch(text)
        or _PAGE_ONLY_RE.fullmatch(text)
        or _CN_PAGE_ONLY_RE.fullmatch(text)
        or _REFERENTIAL_ONLY_RE.fullmatch(text)
    )


def _has_meaningful_source(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped and stripped not in _PLACEHOLDERS)


def _extraction(
    *,
    final_answer: str,
    source_field: Literal["answer", "analysis"],
    start: int,
    end: int,
    score: float,
) -> AnswerExtraction:
    return AnswerExtraction(
        final_answer=final_answer,
        source_field=source_field,
        start_offset=start,
        end_offset=end,
        source_span=f"{source_field}:{start}:{end}",
        answer_source_exists=True,
        answer_extract_score=score,
    )
