from __future__ import annotations

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.answer import extract_source_answer
from college_builder.quality.engine import GateContext, GateEngine, GateResultEvidence
from college_builder.quality.verify import AlignmentGate, CorrectnessVerifier

_TASK15_DOWNSTREAM_RUN_ID = 34590949726
_TASK15_REAL100_RUN_ID = 34558426996
_TASK15_REAL100_ARTIFACT_ID = 10183643305
_TASK15_RECORD_ID = (
    "raw_stackmathqa_acad4b62acbb4a6e4284ac9777d02fe98b6180808a3e923e2ef17a9709c0f5e6"
)
_TASK15_SOURCE_ID = "math:868000:1"
_TASK15_RAW_SHA256 = "4deb3f4f77b724cc0da2e6722eba372272ba86e2ff6be089c89bf450375e9406"
_ANSWER = "logically equivalent"
_QUESTION = (
    "Alternate translation for: “Every real number except zero has a multiplicative inverse.” "
    "A given text states, “Every real number except zero has a multiplicative inverse\" "
    "(where mul-\ntiplicative inverse of a real number x is a real number y such that xy = 1).\n"
    "It offers the following translation:\n"
    "$$\\forall x((x\\neq 0) \\rightarrow \\exists y(xy = 1)).$$\n"
    "I personally translated the statement as:\n"
    "$$\\forall x \\exists y((x\\neq 0)\\rightarrow (xy = 1)).$$\n"
    "Are these two statements logically equivalent?\n"
    "My reasoning being, for every real number x, there exists a real number y, such that if x "
    "does not equal zero, then the product of x and y equals 1.\n"
)
_ANALYSIS = (
    "Yes : \n\n"
    "$∀x((x \\ne 0 ) → ∃y(xy = 1 ))$\n\n"
    "and \n\n"
    "$∀x∃y((x \\ne 0) → (xy = 1))$\n\n"
    "are logically equivalent, because :\n\n\n"
    "$\\vdash \\exists y (\\alpha \\rightarrow \\beta) \\leftrightarrow "
    "(\\alpha \\rightarrow \\exists y \\beta) \\quad $ if $y$ is not free in $\\alpha$.\n\n\n"
    "In your case, $\\alpha$ is $(x \\ne 0 )$ and $y$ is not free in it.\n"
)


def _candidate() -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16a4-task15-downstream",
        source_record_id=_TASK15_RECORD_ID,
        question=_QUESTION,
        answer=_ANSWER,
        analysis=_ANALYSIS,
        subject_candidates=("mathematics",),
        images=(),
        metadata={
            "task15_source_id": _TASK15_SOURCE_ID,
            "task15_raw_sha256": _TASK15_RAW_SHA256,
            "task15_real100_run_id": _TASK15_REAL100_RUN_ID,
            "task15_real100_artifact_id": _TASK15_REAL100_ARTIFACT_ID,
            "task15_downstream_run_id": _TASK15_DOWNSTREAM_RUN_ID,
        },
        normalization_evidence={},
    )


def _analysis_extracted_candidate() -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16a4-analysis-answer",
        source_record_id="raw-task16a4-analysis-answer",
        question="Compute two plus two.",
        answer="",
        analysis="The source computes the sum directly; therefore 4",
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _valid_refs(candidate: NormalizedQA) -> tuple[str, ...]:
    return (
        f"question:0-{len(candidate.question)}",
        f"answer:0-{len(candidate.answer)}",
        f"analysis:0-{len(candidate.analysis)}",
    )


def _decision(refs: tuple[str, ...]) -> ModelDecision:
    return ModelDecision(
        label="PASS",
        score=1.0,
        evidence_references=refs,
        reason_code="TASK15_SEMANTIC_PASS",
    )


def _run(gate: object, candidate: NormalizedQA) -> GateResultEvidence:
    result = GateEngine(
        gates=(gate,),  # type: ignore[arg-type]
        context=GateContext(config_version="task16-recertification-v2"),
    ).run(candidate)
    assert len(result.evidence) == 1
    return result.evidence[0]


def _alignment(refs: tuple[str, ...]) -> GateResultEvidence:
    candidate = _candidate()
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="deepseek-v4-pro",
        decisions=(_decision(refs),),
    )
    return _run(
        AlignmentGate(
            provider=provider,
            prompt="qa alignment v2 protocol",
            prompt_version="v2",
            pass_threshold=0.995,
        ),
        candidate,
    )


def _correctness(refs: tuple[str, ...]) -> GateResultEvidence:
    candidate = _candidate()
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="deepseek-v4-pro",
        decisions=(_decision(refs),),
    )
    return _run(
        CorrectnessVerifier(
            provider=provider,
            prompt="correctness v2 protocol",
            prompt_version="v2",
        ),
        candidate,
    )


def _analysis_alignment(candidate: NormalizedQA, refs: tuple[str, ...]) -> GateResultEvidence:
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="deepseek-v4-pro",
        decisions=(_decision(refs),),
    )
    return _run(
        AlignmentGate(
            provider=provider,
            prompt="qa alignment v2 protocol",
            prompt_version="v2",
            pass_threshold=0.995,
        ),
        candidate,
    )


def _analysis_correctness(candidate: NormalizedQA, refs: tuple[str, ...]) -> GateResultEvidence:
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="deepseek-v4-pro",
        decisions=(_decision(refs),),
    )
    return _run(
        CorrectnessVerifier(
            provider=provider,
            prompt="correctness v2 protocol",
            prompt_version="v2",
        ),
        candidate,
    )


@pytest.mark.parametrize(
    ("runner", "reason"),
    [
        (_alignment, "QA_ALIGNMENT_CONFIRMED"),
        (_correctness, "CORRECTNESS_CONFIRMED"),
    ],
)
def test_task15_downstream_record_valid_half_open_spans_pass(
    runner: object,
    reason: str,
) -> None:
    candidate = _candidate()

    evidence = runner(_valid_refs(candidate))  # type: ignore[operator]

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.score == 1.0
    assert evidence.reason_code == reason


@pytest.mark.parametrize(
    ("runner", "reason"),
    [
        (_alignment, "QA_ALIGNMENT_EVIDENCE_INVALID"),
        (_correctness, "CORRECTNESS_EVIDENCE_INVALID"),
    ],
)
def test_task15_downstream_semantic_pass_does_not_reinterpret_inclusive_answer_end(
    runner: object,
    reason: str,
) -> None:
    candidate = _candidate()
    refs = (
        f"question:0-{len(candidate.question)}",
        f"answer:0-{len(candidate.answer) - 1}",
        f"analysis:0-{len(candidate.analysis)}",
    )

    evidence = runner(refs)  # type: ignore[operator]

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.score == 1.0
    assert evidence.reason_code == reason
    payload_key = "alignment" if runner is _alignment else "verifier"
    assert evidence.evidence_payload[payload_key]["evidence_references"] == list(refs)


@pytest.mark.parametrize(
    ("bad_ref", "runner", "reason"),
    [
        ("answer:<span>logically equivalent</span>", _alignment, "QA_ALIGNMENT_EVIDENCE_INVALID"),
        ("answer:<span>logically equivalent</span>", _correctness, "CORRECTNESS_EVIDENCE_INVALID"),
        (f"analysis:0-{len(_ANALYSIS) + 1}", _alignment, "QA_ALIGNMENT_EVIDENCE_INVALID"),
        (f"analysis:0-{len(_ANALYSIS) + 1}", _correctness, "CORRECTNESS_EVIDENCE_INVALID"),
    ],
)
def test_task15_downstream_malformed_or_out_of_range_spans_remain_fail_closed(
    bad_ref: str,
    runner: object,
    reason: str,
) -> None:
    candidate = _candidate()
    refs = list(_valid_refs(candidate))
    if bad_ref.startswith("answer:"):
        refs[1] = bad_ref
    else:
        refs[2] = bad_ref

    evidence = runner(tuple(refs))  # type: ignore[operator]

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.score == 1.0
    assert evidence.reason_code == reason
    payload_key = "alignment" if runner is _alignment else "verifier"
    assert evidence.evidence_payload[payload_key]["invalid_evidence_reference_count"] == 1


@pytest.mark.parametrize(
    ("runner", "reason"),
    [
        (_analysis_alignment, "QA_ALIGNMENT_EVIDENCE_INVALID"),
        (_analysis_correctness, "CORRECTNESS_EVIDENCE_INVALID"),
    ],
)
def test_analysis_extracted_answer_missing_verification_answer_span_rejects(
    runner: object,
    reason: str,
) -> None:
    candidate = _analysis_extracted_candidate()
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer == "4"
    assert extraction.source_field == "analysis"
    assert extraction.start_offset is not None
    assert extraction.end_offset is not None
    refs = (
        f"question:0-{len(candidate.question)}",
        f"analysis:{extraction.start_offset}-{extraction.end_offset}",
    )

    evidence = runner(candidate, refs)  # type: ignore[operator]

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.score == 1.0
    assert evidence.reason_code == reason


@pytest.mark.parametrize(
    ("runner", "reason"),
    [
        (_analysis_alignment, "QA_ALIGNMENT_CONFIRMED"),
        (_analysis_correctness, "CORRECTNESS_CONFIRMED"),
    ],
)
def test_analysis_extracted_answer_pass_requires_all_fields_and_authority_coverage(
    runner: object,
    reason: str,
) -> None:
    candidate = _analysis_extracted_candidate()
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer == "4"
    assert extraction.source_field == "analysis"
    assert extraction.start_offset is not None
    assert extraction.end_offset is not None
    refs = (
        f"question:0-{len(candidate.question)}",
        f"answer:0-{len(extraction.final_answer)}",
        f"analysis:{extraction.start_offset}-{extraction.end_offset}",
    )

    evidence = runner(candidate, refs)  # type: ignore[operator]

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.score == 1.0
    assert evidence.reason_code == reason


@pytest.mark.parametrize(
    ("refs_factory", "runner", "reason"),
    [
        (
            lambda candidate, answer, start, end: (
                f"answer:0-{len(answer)}",
                f"analysis:{start}-{end}",
            ),
            _analysis_alignment,
            "QA_ALIGNMENT_EVIDENCE_INVALID",
        ),
        (
            lambda candidate, answer, start, end: (
                f"answer:0-{len(answer)}",
                f"analysis:{start}-{end}",
            ),
            _analysis_correctness,
            "CORRECTNESS_EVIDENCE_INVALID",
        ),
        (
            lambda candidate, answer, start, end: (
                f"question:0-{len(candidate.question)}",
                f"answer:0-{len(answer)}",
            ),
            _analysis_alignment,
            "QA_ALIGNMENT_EVIDENCE_INVALID",
        ),
        (
            lambda candidate, answer, start, end: (
                f"question:0-{len(candidate.question)}",
                f"answer:0-{len(answer)}",
            ),
            _analysis_correctness,
            "CORRECTNESS_EVIDENCE_INVALID",
        ),
        (
            lambda candidate, answer, start, end: (
                f"question:0-{len(candidate.question)}",
                f"answer:0-{len(answer)}",
                "analysis:0-1",
            ),
            _analysis_alignment,
            "QA_ALIGNMENT_EVIDENCE_INVALID",
        ),
        (
            lambda candidate, answer, start, end: (
                f"question:0-{len(candidate.question)}",
                f"answer:0-{len(answer)}",
                "analysis:0-1",
            ),
            _analysis_correctness,
            "CORRECTNESS_EVIDENCE_INVALID",
        ),
    ],
)
def test_analysis_extracted_answer_requires_question_analysis_and_authority_coverage(
    refs_factory: object,
    runner: object,
    reason: str,
) -> None:
    candidate = _analysis_extracted_candidate()
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer == "4"
    assert extraction.start_offset is not None
    assert extraction.end_offset is not None
    refs = refs_factory(  # type: ignore[operator]
        candidate,
        extraction.final_answer,
        extraction.start_offset,
        extraction.end_offset,
    )

    evidence = runner(candidate, refs)  # type: ignore[operator]

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.score == 1.0
    assert evidence.reason_code == reason
