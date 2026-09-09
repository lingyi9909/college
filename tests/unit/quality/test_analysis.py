from __future__ import annotations

import json
from pathlib import Path

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.question import AnalysisType
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.analysis import AnalysisGate
from college_builder.quality.engine import GateContext, GateEngine, GateResultEvidence


def _candidate(*, analysis: str, answer: str = "x = 4") -> NormalizedQA:
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


def _decision(
    label: str,
    score: float,
    *,
    evidence_references: tuple[str, ...] = ("analysis:0-40",),
) -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        evidence_references=evidence_references,
        reason_code="TEST",
    )


def _gate(
    primary_decision: ModelDecision,
    verifier_decision: ModelDecision | None = None,
    *,
    primary_identity: tuple[str, str] = ("fake", "analysis-primary-v1"),
    verifier_identity: tuple[str, str] = ("fake", "analysis-verifier-v2"),
) -> tuple[AnalysisGate, FakeStructuredModelProvider, FakeStructuredModelProvider]:
    primary = FakeStructuredModelProvider(
        provider=primary_identity[0],
        model=primary_identity[1],
        decisions=(primary_decision,),
    )
    verifier = FakeStructuredModelProvider(
        provider=verifier_identity[0],
        model=verifier_identity[1],
        decisions=() if verifier_decision is None else (verifier_decision,),
    )
    return (
        AnalysisGate(
            primary=primary,
            verifier=verifier,
            prompt="analysis prompt",
            prompt_version="v1",
        ),
        primary,
        verifier,
    )


def _run(gate: AnalysisGate, candidate: NormalizedQA) -> GateResultEvidence:
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="pilot-v1"),
    ).run(candidate)
    assert len(result.evidence) == 1
    return result.evidence[0]


@pytest.mark.parametrize(
    ("analysis", "reason"),
    [
        ("", "ANALYSIS_MISSING"),
        ("略", "ANALYSIS_MISSING"),
        ("x = 4", "ANALYSIS_TOO_SHALLOW"),
        ("Answer: x = 4", "ANALYSIS_TOO_SHALLOW"),
        ("Therefore x = 4", "ANALYSIS_TOO_SHALLOW"),
        ("https://example.com/solution", "ANALYSIS_TOO_SHALLOW"),
        ("See page 42.", "ANALYSIS_TOO_SHALLOW"),
        ("同上", "ANALYSIS_TOO_SHALLOW"),
    ],
)
def test_obviously_incomplete_analysis_rejects_before_model_call(
    analysis: str,
    reason: str,
) -> None:
    gate, primary, verifier = _gate(_decision("DERIVATION", 0.99))

    evidence = _run(gate, _candidate(analysis=analysis))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == reason
    assert len(primary.requests) == 0
    assert len(verifier.requests) == 0


@pytest.mark.parametrize("analysis_type", list(AnalysisType))
def test_high_band_valid_analysis_type_passes_without_verifier(
    analysis_type: AnalysisType,
) -> None:
    gate, primary, verifier = _gate(_decision(analysis_type.value, 0.99))

    evidence = _run(
        gate,
        _candidate(analysis="Subtract 3 from both sides, then divide by 2 to obtain x = 4."),
    )

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.reason_code == "ANALYSIS_CONFIRMED"
    assert evidence.evidence_payload["analysis_type"] == analysis_type.value
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 0


def test_low_band_rejects_without_verifier() -> None:
    gate, primary, verifier = _gate(_decision("DERIVATION", 0.89))

    evidence = _run(
        gate,
        _candidate(analysis="Subtract 3 from both sides, then divide by 2 to obtain x = 4."),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "ANALYSIS_UNCERTAIN"
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 0


def test_middle_band_calls_independent_verifier_and_routes_to_verify() -> None:
    gate, primary, verifier = _gate(
        _decision("STEP_BY_STEP", 0.95),
        _decision("STEP_BY_STEP", 0.99),
    )

    evidence = _run(
        gate,
        _candidate(analysis="First subtract 3. Next divide by 2. This gives x = 4."),
    )

    assert evidence.verdict is GateVerdict.VERIFY
    assert evidence.score == 0.95
    assert evidence.reason_code == "ANALYSIS_REVIEW_REQUIRED"
    assert evidence.evidence_payload["analysis_type"] == "STEP_BY_STEP"
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 1


def test_middle_band_conflict_rejects() -> None:
    gate, _, verifier = _gate(
        _decision("DERIVATION", 0.95),
        _decision("PROOF", 0.99),
    )

    evidence = _run(
        gate,
        _candidate(analysis="Subtract 3 and divide by 2, therefore x = 4."),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "MODEL_DECISION_CONFLICT"
    assert len(verifier.requests) == 1


def test_middle_band_verifier_must_be_independent() -> None:
    gate, primary, verifier = _gate(
        _decision("DERIVATION", 0.95),
        _decision("DERIVATION", 0.99),
        primary_identity=("fake", "same-model"),
        verifier_identity=("fake", "same-model"),
    )

    evidence = _run(
        gate,
        _candidate(analysis="Subtract 3 and divide by 2, therefore x = 4."),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "VERIFIER_NOT_INDEPENDENT"
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 0


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        ("TOO_SHALLOW", "ANALYSIS_TOO_SHALLOW"),
        ("UNRELATED", "ANALYSIS_UNRELATED"),
        ("UNCERTAIN", "ANALYSIS_UNCERTAIN"),
    ],
)
def test_high_band_negative_analysis_taxonomy_rejects(label: str, reason: str) -> None:
    gate, _, verifier = _gate(_decision(label, 0.99))

    evidence = _run(
        gate,
        _candidate(analysis="This paragraph contains source commentary that needs classification."),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == reason
    assert len(verifier.requests) == 0


def test_analysis_positive_decision_requires_analysis_content_evidence() -> None:
    gate, _, verifier = _gate(
        _decision(
            "DERIVATION",
            0.99,
            evidence_references=("metadata.site",),
        )
    )

    evidence = _run(\ate,
        _candidate(analysis="Subtract 3 and divide by 2, therefore x = 4."),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "ANALYSIS_UNCERTAIN"
    assert len(verifier.requests) == 0


def test_analysis_request_contains_source_content_only() -> None:
    gate, primary, _ = _gate(_decision("DERIVATION", 0.99))
    candidate = _candidate(analysis="Subtract 3 and divide by 2, therefore x = 4.")

    _run(gate, candidate)

    request = primary.requests[0]
    assert request.task == "gate_4_original_analysis"
    assert request.inputs == {
        "question": candidate.question,
        "answer": candidate.answer,
        "analysis": candidate.analysis,
    }
    assert "final_answer" not in request.inputs


def test_answer_analysis_golden_analysis_reject_cases_remain_fail_closed() -> None:
    cases_path = Path(__file__).parents[2] / "golden" / "answer_analysis_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))

    analysis_cases = [case for case in cases if case["gate"] == "analysis"]
    assert analysis_cases
    for case in analysis_cases:
        gate, primary, verifier = _gate(_decision("DERIVATION", 0.99))
        evidence = _run(
            gate,
            _candidate(answer=case["answer"], analysis=case["analysis"]),
        )
        assert evidence.reason_code == case["expected_reason"]
        assert evidence.verdict is GateVerdict.REJECT
        assert len(primary.requests) == 0
        assert len(verifier.requests) == 0
