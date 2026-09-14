from __future__ import annotations

from pathlib import Path

import pytest

from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.answer import extract_source_answer
from college_builder.quality.engine import GateContext, GateEngine
from college_builder.quality.verify import AlignmentGate, CorrectnessVerifier


def _analysis_extracted_candidate() -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16c-authority-hint",
        source_record_id="raw-task16c-authority-hint",
        question="Why does the stated result follow?",
        answer="",
        analysis="The source premise is sufficient; therefore the stated result follows.",
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _source_answer_candidate() -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16c-source-answer-hint",
        source_record_id="raw-task16c-source-answer-hint",
        question="Compute two plus two.",
        answer="4",
        analysis="The source adds two and two directly.",
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _pass_decision(candidate: NormalizedQA) -> ModelDecision:
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer is not None
    assert extraction.source_field is not None
    assert extraction.start_offset is not None
    assert extraction.end_offset is not None
    refs = (
        f"question:0-{len(candidate.question)}",
        f"answer:0-{len(extraction.final_answer)}",
        f"analysis:0-{len(candidate.analysis)}"
        if extraction.source_field == "answer"
        else (
            f"analysis:{extraction.start_offset}-{extraction.end_offset}"
        ),
    )
    return ModelDecision(
        label="PASS",
        score=1.0,
        evidence_references=refs,
        reason_code="grounded_pass",
    )


def _run_alignment(candidate: NormalizedQA) -> FakeStructuredModelProvider:
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="deepseek-v4-pro",
        decisions=(_pass_decision(candidate),),
    )
    gate = AlignmentGate(
        provider=provider,
        prompt="alignment prompt",
        prompt_version="v2",
        pass_threshold=0.995,
    )
    GateEngine(
        gates=(gate,),
        context=GateContext(config_version="task16-recertification-v2"),
    ).run(candidate)
    return provider


def _run_correctness(candidate: NormalizedQA) -> FakeStructuredModelProvider:
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="deepseek-v4-pro",
        decisions=(_pass_decision(candidate),),
    )
    gate = CorrectnessVerifier(
        provider=provider,
        prompt="correctness prompt",
        prompt_version="v2",
    )
    GateEngine(
        gates=(gate,),
        context=GateContext(config_version="task16-recertification-v2"),
    ).run(candidate)
    return provider


@pytest.mark.parametrize("runner", [_run_alignment, _run_correctness])
def test_analysis_extracted_answer_request_exposes_exact_authority_reference(
    runner: object,
) -> None:
    candidate = _analysis_extracted_candidate()
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer is not None
    assert extraction.source_field == "analysis"
    assert extraction.start_offset is not None
    assert extraction.end_offset is not None

    provider = runner(candidate)  # type: ignore[operator]

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.inputs["answer"] == extraction.final_answer
    assert request.inputs["answer_source_reference"] == (
        f"analysis:{extraction.start_offset}-{extraction.end_offset}"
    )


@pytest.mark.parametrize("runner", [_run_alignment, _run_correctness])
def test_source_answer_request_exposes_exact_answer_authority_reference(
    runner: object,
) -> None:
    candidate = _source_answer_candidate()
    extraction = extract_source_answer(candidate)
    assert extraction.final_answer == "4"
    assert extraction.source_field == "answer"

    provider = runner(candidate)  # type: ignore[operator]

    assert len(provider.requests) == 1
    request = provider.requests[0]
    assert request.inputs["answer_source_reference"] == "answer:0-1"


@pytest.mark.parametrize(
    "path",
    [
        Path("prompts/qa_alignment/v2.txt"),
        Path("prompts/correctness_verify/v2.txt"),
    ],
)
def test_gate5_prompts_require_exact_authority_reference_for_pass(path: Path) -> None:
    prompt = path.read_text(encoding="utf-8")

    assert "answer_source_reference" in prompt
    assert "copy it exactly" in prompt.lower()
    assert "PASS" in prompt
