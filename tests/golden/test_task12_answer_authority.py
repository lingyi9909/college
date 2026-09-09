from __future__ import annotations

from typing import Any

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelClassificationRequest, ModelDecision
from college_builder.quality.answer import extract_source_answer
from college_builder.quality.engine import GateContext, GateEngine, GateResultEvidence
from college_builder.quality.verify import (
    AlignmentGate,
    CorrectnessVerifier,
    DeterministicMathVerifier,
    DeterministicVerificationStatus,
)


class _Provider:
    provider = "authority-test"
    model = "authority-v1"

    def __init__(self, refs: tuple[str, ...]) -> None:
        self.refs = refs
        self.requests: list[ModelClassificationRequest] = []

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        self.requests.append(request)
        return ModelDecision(
            label="PASS",
            score=0.999,
            evidence_references=self.refs,
            reason_code="SOURCE_GROUNDED",
        )


class _FailIfCalledProvider:
    provider = "authority-test"
    model = "must-not-run"

    def __init__(self) -> None:
        self.requests: list[ModelClassificationRequest] = []

    def classify(self, request: ModelClassificationRequest) -> Any:
        self.requests.append(request)
        raise AssertionError("provider must not be called")


def _candidate(
    *,
    answer: str,
    analysis: str = "Subtract 3 from both sides and divide by 2. Therefore x = 4",
    question: str = "Solve 2*x + 3 = 11.",
) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-authority-1",
        source_record_id="raw-authority-1",
        question=question,
        answer=answer,
        analysis=analysis,
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _run(gate: object, candidate: NormalizedQA) -> GateResultEvidence:
    result = GateEngine(
        gates=(gate,),  # type: ignore[arg-type]
        context=GateContext(config_version="pilot-v1"),
    ).run(candidate)
    assert len(result.evidence) == 1
    return result.evidence[0]


def _analysis_authority_refs(candidate: NormalizedQA) -> tuple[str, ...]:
    return (
        f"question:0-{len(candidate.question)}",
        f"analysis:0-{len(candidate.analysis)}",
    )


def _direct_answer_refs(candidate: NormalizedQA) -> tuple[str, ...]:
    return (
        f"question:0-{len(candidate.question)}",
        f"answer:0-{len(candidate.answer)}",
        f"analysis:0-{len(candidate.analysis)}",
    )


@pytest.mark.parametrize("raw_answer", ["", "See above"])
def test_analysis_conclusion_is_formal_answer_authority_for_alignment_and_correctness(
    raw_answer: str,
) -> None:
    candidate = _candidate(answer=raw_answer)
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer == "x = 4"
    assert extraction.source_field == "analysis"
    assert extraction.answer_extract_score >= 0.995

    alignment_provider = _Provider(_analysis_authority_refs(candidate))
    alignment = _run(
        AlignmentGate(
            provider=alignment_provider,
            prompt="alignment",
            prompt_version="v1",
        ),
        candidate,
    )
    assert alignment.verdict is GateVerdict.PASS
    assert alignment_provider.requests[0].inputs["answer"] == "x = 4"
    assert alignment.evidence_payload["answer_authority"] == extraction.model_dump(mode="json")

    correctness_provider = _Provider(_analysis_authority_refs(candidate))
    correctness = _run(
        CorrectnessVerifier(
            provider=correctness_provider,
            prompt="correctness",
            prompt_version="v1",
            calibration_id="identity-v1",
        ),
        candidate,
    )
    assert correctness.verdict is GateVerdict.PASS
    assert correctness_provider.requests[0].inputs["answer"] == "x = 4"
    assert correctness.evidence_payload["answer_authority"] == extraction.model_dump(mode="json")


def test_direct_answer_authority_behavior_is_unchanged() -> None:
    candidate = _candidate(answer="x = 4")
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer == "x = 4"
    assert extraction.source_field == "answer"

    provider = _Provider(_direct_answer_refs(candidate))
    evidence = _run(
        AlignmentGate(provider=provider, prompt="alignment", prompt_version="v1"),
        candidate,
    )

    assert evidence.verdict is GateVerdict.PASS
    assert provider.requests[0].inputs["answer"] == "x = 4"
    assert evidence.evidence_payload["answer_authority"] == extraction.model_dump(mode="json")


def test_missing_extractable_final_answer_fails_closed_before_model() -> None:
    candidate = _candidate(
        answer="See above",
        analysis="Subtract 3 from both sides, but the source gives no final conclusion.",
    )
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer is None

    alignment_provider = _FailIfCalledProvider()
    alignment = _run(
        AlignmentGate(
            provider=alignment_provider,
            prompt="alignment",
            prompt_version="v1",
        ),
        candidate,
    )
    assert alignment.verdict is GateVerdict.REJECT
    assert alignment.reason_code == "ANSWER_NOT_EXTRACTABLE"
    assert alignment_provider.requests == []

    correctness_provider = _FailIfCalledProvider()
    correctness = _run(
        CorrectnessVerifier(
            provider=correctness_provider,
            prompt="correctness",
            prompt_version="v1",
            calibration_id="identity-v1",
        ),
        candidate,
    )
    assert correctness.verdict is GateVerdict.REJECT
    assert correctness.reason_code == "ANSWER_NOT_EXTRACTABLE"
    assert correctness_provider.requests == []


@pytest.mark.parametrize(
    "analysis_ref",
    [
        "analysis:999-1000",
        "analysis:0-10",
        "analysis:5-5",
    ],
)
def test_analysis_answer_authority_requires_evidence_covering_real_origin_span(
    analysis_ref: str,
) -> None:
    candidate = _candidate(answer="")
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer == "x = 4"
    assert extraction.source_field == "analysis"
    refs = (f"question:0-{len(candidate.question)}", analysis_ref)

    alignment = _run(
        AlignmentGate(
            provider=_Provider(refs),
            prompt="alignment",
            prompt_version="v1",
        ),
        candidate,
    )
    assert alignment.verdict is GateVerdict.REJECT
    assert alignment.reason_code == "QA_ALIGNMENT_EVIDENCE_INVALID"

    correctness = _run(
        CorrectnessVerifier(
            provider=_Provider(refs),
            prompt="correctness",
            prompt_version="v1",
            calibration_id="identity-v1",
        ),
        candidate,
    )
    assert correctness.verdict is GateVerdict.REJECT
    assert correctness.reason_code == "CORRECTNESS_EVIDENCE_INVALID"


def test_deterministic_verification_uses_extracted_final_answer() -> None:
    candidate = _candidate(
        answer="See above",
        analysis="Subtract 3 from both sides and divide by 2. Therefore x = 5",
    )
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer == "x = 5"
    assert extraction.source_field == "analysis"

    deterministic = DeterministicMathVerifier().verify(candidate)
    assert deterministic.status is DeterministicVerificationStatus.CONTRADICTED

    provider = _FailIfCalledProvider()
    evidence = _run(
        CorrectnessVerifier(
            provider=provider,
            prompt="correctness",
            prompt_version="v1",
            calibration_id="identity-v1",
        ),
        candidate,
    )
    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "DETERMINISTIC_CORRECTNESS_MISMATCH"
    assert provider.requests == []
    assert evidence.evidence_payload["answer_authority"] == extraction.model_dump(mode="json")
