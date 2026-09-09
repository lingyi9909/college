from __future__ import annotations

import json
from pathlib import Path

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.quality.answer import AnswerGate, extract_source_answer
from college_builder.quality.engine import GateContext, GateEngine, GateResultEvidence


def _candidate(*, answer: str, analysis: str) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-1",
        source_record_id="raw-1",
        question="Solve 2x + 3 = 11.",
        answer=answer,
        analysis=analysis,
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _run(candidate: NormalizedQA) -> GateResultEvidence:
    result = GateEngine(
        gates=(AnswerGate(),),
        context=GateContext(config_version="pilot-v1"),
    ).run(candidate)
    assert len(result.evidence) == 1
    return result.evidence[0]


def test_literal_source_answer_is_extracted_with_exact_span_evidence() -> None:
    candidate = _candidate(answer="  x = 4  ", analysis="Subtract 3, then divide by 2.")

    extraction = extract_source_answer(candidate)

    assert extraction.answer_source_exists is True
    assert extraction.final_answer == "x = 4"
    assert extraction.source_field == "answer"
    assert extraction.start_offset == 2
    assert extraction.end_offset == 7
    source_answer = candidate.answer[extraction.start_offset : extraction.end_offset]
    assert source_answer == extraction.final_answer
    assert extraction.source_span == "answer:2:7"
    assert extraction.answer_extract_score == 1.0


def test_deterministic_analysis_conclusion_may_supply_literal_final_answer() -> None:
    analysis = "Subtract 3 from both sides and divide by 2; therefore x = 4"
    candidate = _candidate(answer="", analysis=analysis)

    extraction = extract_source_answer(candidate)

    assert extraction.answer_source_exists is True
    assert extraction.final_answer == "x = 4"
    assert extraction.source_field == "analysis"
    assert extraction.source_span == f"analysis:{analysis.index('x = 4')}:{len(analysis)}"
    assert analysis[extraction.start_offset : extraction.end_offset] == "x = 4"
    assert extraction.answer_extract_score >= 0.995


def test_answer_gate_never_infers_missing_result_from_analysis() -> None:
    candidate = _candidate(
        answer="",
        analysis="Solving gives the result shown above",
    )

    extraction = extract_source_answer(candidate)
    evidence = _run(candidate)

    assert extraction.final_answer is None
    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "ANSWER_NOT_EXTRACTABLE"
    assert evidence.evidence_payload["final_answer"] is None


def test_referential_conclusion_is_not_treated_as_explicit_answer() -> None:
    candidate = _candidate(
        answer="",
        analysis="After simplification, therefore the result shown above",
    )

    extraction = extract_source_answer(candidate)
    evidence = _run(candidate)

    assert extraction.final_answer is None
    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "ANSWER_NOT_EXTRACTABLE"


@pytest.mark.parametrize(
    "answer",
    [
        "See above",
        "See page 42.",
        "https://example.com/answer",
        "[answer](https://example.com/answer)",
        "The result shown above",
        "见上",
        "见第42页",
        "答案见上文",
        "结果如上",
    ],
)
def test_direct_answer_reference_or_placeholder_is_not_extractable(answer: str) -> None:
    candidate = _candidate(
        answer=answer,
        analysis="Subtract 3 from both sides and divide by 2.",
    )

    extraction = extract_source_answer(candidate)
    evidence = _run(candidate)

    assert extraction.answer_source_exists is True
    assert extraction.final_answer is None
    assert extraction.source_span is None
    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "ANSWER_NOT_EXTRACTABLE"


def test_unextractable_direct_answer_does_not_break_explicit_analysis_conclusion() -> None:
    analysis = "Subtract 3 from both sides and divide by 2; therefore x = 4"
    candidate = _candidate(answer="See above", analysis=analysis)

    extraction = extract_source_answer(candidate)
    evidence = _run(candidate)

    assert extraction.final_answer == "x = 4"
    assert extraction.source_field == "analysis"
    assert extraction.source_span == f"analysis:{analysis.index('x = 4')}:{len(analysis)}"
    assert evidence.verdict is GateVerdict.PASS


def test_answer_gate_rejects_when_source_answer_and_solution_are_missing() -> None:
    evidence = _run(_candidate(answer="", analysis=""))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "ANSWER_MISSING"


def test_answer_gate_requires_source_span_and_threshold_for_pass() -> None:
    evidence = _run(_candidate(answer="x = 4", analysis="Subtract 3, then divide by 2."))

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.score == 1.0
    assert evidence.reason_code == "ANSWER_SOURCE_CONFIRMED"
    assert evidence.provider == "deterministic"
    assert evidence.model == "source_answer_v1"
    assert evidence.prompt_version == "not_applicable"
    assert evidence.evidence_payload["answer_source_exists"] is True
    assert evidence.evidence_payload["answer_extract_score"] == 1.0
    assert evidence.evidence_payload["final_answer"] == "x = 4"
    assert evidence.evidence_payload["source_span"] == "answer:0:5"


def test_answer_analysis_golden_reject_cases_remain_fail_closed() -> None:
    cases_path = Path(__file__).parents[2] / "golden" / "answer_analysis_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    answer_cases = [case for case in cases if case["gate"] == "answer"]
    assert answer_cases
    for case in answer_cases:
        evidence = _run(_candidate(answer=case["answer"], analysis=case["analysis"]))
        assert evidence.reason_code == case["expected_reason"]
        assert evidence.verdict is GateVerdict.REJECT
