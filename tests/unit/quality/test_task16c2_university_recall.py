from __future__ import annotations

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.classify import UniversityStemGate
from college_builder.quality.engine import GateContext, GateEngine


def _candidate(question: str) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16c2",
        source_record_id="raw-task16c2",
        question=question,
        answer="source answer",
        analysis="source analysis",
        subject_candidates=("university_stem",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _decision(label: str, score: float, question: str) -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        reason_code="TASK16C2_REGRESSION",
        evidence_references=(f"question:0-{min(len(question), 40)}",),
    )


def _run(
    question: str,
    primary_decision: ModelDecision,
    verifier_decision: ModelDecision,
):
    primary = FakeStructuredModelProvider(
        provider="fake-primary",
        model="university-primary-v1",
        decisions=(primary_decision,),
    )
    verifier = FakeStructuredModelProvider(
        provider="fake-verifier",
        model="university-verifier-v1",
        decisions=(verifier_decision,),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="task16-recertification-v2"),
    ).run(_candidate(question))
    return result, primary, verifier


@pytest.mark.parametrize(
    "question",
    [
        "Explain how microscopic particle motion determines the observed gas pressure.",
        (
            "A hydroelectric plant converts gravitational potential energy into "
            "electrical energy; calculate the output power."
        ),
        "Explain the recoil of a body using Newton's third law and conservation of momentum.",
        "Why can ordinary random cross-validation be biased for an autocorrelated time series?",
        "Interpret the threshold parameters in an ordinal regression model.",
    ],
)
def test_gate1_dual_independent_positive_middle_band_confirms_clear_university_stem(
    question: str,
) -> None:
    result, primary, verifier = _run(
        question,
        _decision("UNIVERSITY_STEM", 0.95, question),
        _decision("UNIVERSITY_STEM", 0.95, question),
    )

    assert result.verdict is GateVerdict.PASS
    assert result.evidence[0].verdict is GateVerdict.PASS
    assert result.evidence[0].reason_code == "UNIVERSITY_STEM_CONFIRMED"
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 1


def test_gate1_dual_positive_missing_content_evidence_still_fails_closed() -> None:
    question = "Derive the relation between force and momentum."
    primary = ModelDecision(
        label="UNIVERSITY_STEM",
        score=0.95,
        reason_code="HINT_ONLY",
        evidence_references=("source_hints.metadata.book",),
    )
    verifier = ModelDecision(
        label="UNIVERSITY_STEM",
        score=0.95,
        reason_code="HINT_ONLY",
        evidence_references=("source_hints.metadata.book",),
    )

    result, _, _ = _run(question, primary, verifier)

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "UNIVERSITY_LEVEL_UNCERTAIN"


def test_gate1_conflicting_middle_band_decisions_still_fail_closed() -> None:
    question = "Explain the recoil of a body using conservation of momentum."
    result, _, _ = _run(
        question,
        _decision("UNIVERSITY_STEM", 0.95, question),
        _decision("K12", 0.99, question),
    )

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "MODEL_DECISION_CONFLICT"


def test_gate1_sub_verify_primary_still_rejects_without_verifier() -> None:
    question = "Interpret a regression coefficient."
    primary = FakeStructuredModelProvider(
        provider="fake-primary",
        model="university-primary-v1",
        decisions=(_decision("UNIVERSITY_STEM", 0.89, question),),
    )
    verifier = FakeStructuredModelProvider(
        provider="fake-verifier",
        model="university-verifier-v1",
        decisions=(_decision("UNIVERSITY_STEM", 0.99, question),),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="task16-recertification-v2"),
    ).run(_candidate(question))

    assert result.verdict is GateVerdict.REJECT
    assert result.evidence[0].reason_code == "UNIVERSITY_LEVEL_UNCERTAIN"
    assert len(verifier.requests) == 0
