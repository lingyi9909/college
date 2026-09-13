from __future__ import annotations

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.classify import ProblemGate
from college_builder.quality.engine import GateContext, GateEngine


def _candidate(question: str) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16c3",
        source_record_id="raw-task16c3",
        question=question,
        answer="source answer",
        analysis="source analysis",
        subject_candidates=("university_stem",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _decision(
    label: str,
    score: float,
    question: str,
    *,
    grounded: bool = True,
) -> ModelDecision:
    references = (
        (f"question:0-{min(len(question), 40)}",)
        if grounded
        else ("source_hints.metadata.tags[0]",)
    )
    return ModelDecision(
        label=label,
        score=score,
        reason_code="TASK16C3_REGRESSION",
        evidence_references=references,
    )


def _run(
    question: str,
    primary_decision: ModelDecision,
    verifier_decision: ModelDecision | None,
):
    primary = FakeStructuredModelProvider(
        provider="fake-primary",
        model="problem-primary-v1",
        decisions=(primary_decision,),
    )
    verifier = FakeStructuredModelProvider(
        provider="fake-verifier",
        model="problem-verifier-v1",
        decisions=() if verifier_decision is None else (verifier_decision,),
    )
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="task16-recertification-v2"),
    ).run(_candidate(question))
    return result, primary, verifier


@pytest.mark.parametrize(
    ("question", "label"),
    [
        (
            "Explain why det(A) != 0 is equivalent to Ax = b having a unique solution.",
            "CONCEPTUAL",
        ),
        ("Prove that the stated complex function is holomorphic.", "PROOF"),
        ("Evaluate the definite integral and justify each step.", "CALCULATION"),
        ("Derive the plasma permeability relation from Maxwell's equations.", "DERIVATION"),
    ],
)
def test_gate2_grounded_positive_low_middle_band_calls_verifier_and_confirms(
    question: str,
    label: str,
) -> None:
    result, primary, verifier = _run(
        question,
        _decision(label, 0.85, question),
        _decision(label, 0.99, question),
    )

    assert result.verdict is GateVerdict.PASS
    assert result.evidence[0].reason_code == "PROBLEM_CONFIRMED"
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 1


def test_gate2_low_positive_missing_grounded_evidence_still_fails_closed() -> None:
    question = "Discuss whether this mathematical claim is true."
    result, _, verifier = _run(
        question,
        _decision("CONCEPTUAL", 0.85, question, grounded=False),
        _decision("CONCEPTUAL", 0.99, question),
    )

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "PROBLEM_TYPE_UNCERTAIN"
    assert len(verifier.requests) == 0


def test_gate2_low_positive_verifier_conflict_still_fails_closed() -> None:
    question = "Explain the determinant criterion for invertibility."
    result, _, verifier = _run(
        question,
        _decision("CONCEPTUAL", 0.85, question),
        _decision("DISCUSSION", 0.99, question),
    )

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "MODEL_DECISION_CONFLICT"
    assert len(verifier.requests) == 1


def test_gate2_sub_floor_positive_still_rejects_without_verifier() -> None:
    question = "Explain a simple algebraic identity."
    result, _, verifier = _run(
        question,
        _decision("CONCEPTUAL", 0.69, question),
        _decision("CONCEPTUAL", 0.99, question),
    )

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "PROBLEM_TYPE_UNCERTAIN"
    assert len(verifier.requests) == 0


def test_gate2_historical_config_preserves_existing_threshold_behavior() -> None:
    question = "Explain the determinant criterion for invertibility."
    primary = FakeStructuredModelProvider(
        provider="fake-primary",
        model="problem-primary-v1",
        decisions=(_decision("CONCEPTUAL", 0.85, question),),
    )
    verifier = FakeStructuredModelProvider(
        provider="fake-verifier",
        model="problem-verifier-v1",
        decisions=(_decision("CONCEPTUAL", 0.99, question),),
    )
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="pilot-v1"),
    ).run(_candidate(question))

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "PROBLEM_TYPE_UNCERTAIN"
    assert len(verifier.requests) == 0
