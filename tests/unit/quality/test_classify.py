from __future__ import annotations

from typing import cast

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision, StructuredModelProvider
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.classify import ProblemGate, UniversityStemGate
from college_builder.quality.engine import GateContext, GateEngine, GateResultEvidence


def _candidate(
    question: str,
    *,
    subject_candidates: tuple[str, ...] = (),
    metadata: dict[str, object] | None = None,
) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-1",
        source_record_id="raw-1",
        question=question,
        answer="source answer",
        analysis="source analysis",
        subject_candidates=subject_candidates,
        images=(),
        metadata=metadata or {},
        normalization_evidence={},
    )


def _decision(
    label: str,
    score: float,
    reason: str = "TEST",
    *,
    evidence_references: tuple[str, ...] = ("question:0-20",),
) -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        evidence_references=evidence_references,
        reason_code=reason,
    )


def _providers(
    primary: ModelDecision,
    verifier: ModelDecision | None,
    *,
    primary_identity: tuple[str, str] = ("fake", "primary-v1"),
    verifier_identity: tuple[str, str] = ("fake", "verifier-v2"),
) -> tuple[FakeStructuredModelProvider, FakeStructuredModelProvider]:
    return (
        FakeStructuredModelProvider(
            provider=primary_identity[0],
            model=primary_identity[1],
            decisions=(primary,),
        ),
        FakeStructuredModelProvider(
            provider=verifier_identity[0],
            model=verifier_identity[1],
            decisions=() if verifier is None else (verifier,),
        ),
    )


def _run(
    gate: UniversityStemGate | ProblemGate,
    candidate: NormalizedQA,
) -> GateResultEvidence:
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="pilot-v1"),
    ).run(candidate)
    assert len(result.evidence) == 1
    return result.evidence[0]


@pytest.mark.parametrize("score", [0.98, 0.99, 1.0])
def test_university_high_band_passes_without_verifier(score: float) -> None:
    primary, verifier = _providers(_decision("UNIVERSITY_STEM", score), None)
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Compute the eigenvalues of this matrix."))

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.score == score
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 0
    assert "verifier" not in evidence.evidence_payload


def test_university_low_band_rejects_without_verifier() -> None:
    primary, verifier = _providers(_decision("UNIVERSITY_STEM", 0.89), None)
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Prove the asymptotic complexity bound."))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "UNIVERSITY_LEVEL_UNCERTAIN"
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 0


@pytest.mark.parametrize("score", [0.90, 0.95, 0.979999])
def test_university_middle_band_calls_verifier_and_routes_to_verify(score: float) -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", score),
        _decision("UNIVERSITY_STEM", 0.99),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(
        gate,
        _candidate("Derive the wave equation from Maxwell's equations."),
    )

    assert evidence.verdict is GateVerdict.VERIFY
    assert evidence.score == score
    assert len(verifier.requests) == 1


def test_primary_verifier_label_conflict_fails_closed_in_middle_band() -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", 0.95),
        _decision("K12", 0.99),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Evaluate the determinant."))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "MODEL_DECISION_CONFLICT"
    assert len(verifier.requests) == 1


def test_middle_band_verifier_must_have_independent_provider_model_identity() -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", 0.95),
        _decision("UNIVERSITY_STEM", 0.99),
        primary_identity=("fake", "same-model"),
        verifier_identity=("fake", "same-model"),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Find an eigenbasis."))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "VERIFIER_NOT_INDEPENDENT"
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 0


def test_gate_evidence_records_both_middle_band_execution_decisions() -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", 0.95, "PRIMARY_REASON"),
        _decision("UNIVERSITY_STEM", 0.985, "VERIFIER_REASON"),
        primary_identity=("provider-a", "model-a"),
        verifier_identity=("provider-b", "model-b"),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Diagonalize the matrix."))

    assert evidence.verdict is GateVerdict.VERIFY
    assert evidence.provider == "provider-a"
    assert evidence.model == "model-a"
    assert evidence.prompt_version == "v1"
    assert evidence.evidence_payload["primary"] == {
        "provider": "provider-a",
        "model": "model-a",
        "label": "UNIVERSITY_STEM",
        "score": 0.95,
        "reason_code": "PRIMARY_REASON",
        "evidence_references": ["question:0-20"],
    }
    assert evidence.evidence_payload["verifier"] == {
        "provider": "provider-b",
        "model": "model-b",
        "label": "UNIVERSITY_STEM",
        "score": 0.985,
        "reason_code": "VERIFIER_REASON",
        "evidence_references": ["question:0-20"],
    }


@pytest.mark.parametrize(
    "question",
    [
        "Compute the eigenvalues and eigenvectors of A.",
        "Derive the electromagnetic wave equation from Maxwell's equations.",
        "Prove that this divide-and-conquer algorithm runs in O(n log n).",
    ],
)
def test_university_stem_positive_taxonomy_can_pass_with_content_evidence(
    question: str,
) -> None:
    primary, verifier = _providers(_decision("UNIVERSITY_STEM", 0.99), None)
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate(question))

    assert evidence.verdict is GateVerdict.PASS
    assert len(verifier.requests) == 0


@pytest.mark.parametrize(
    ("question", "label", "reason_code"),
    [
        (
            "Which IDE do you recommend for Python?",
            "NON_UNIVERSITY_STEM",
            "NOT_UNIVERSITY_LEVEL",
        ),
        (
            "How do I install this software package?",
            "NON_UNIVERSITY_STEM",
            "NOT_UNIVERSITY_LEVEL",
        ),
        (
            "Should I choose a software engineering career?",
            "NON_UNIVERSITY_STEM",
            "NOT_UNIVERSITY_LEVEL",
        ),
        ("What is 7 + 5?", "K12", "NOT_UNIVERSITY_LEVEL"),
        ("Discuss Renaissance painting.", "NON_STEM", "NON_STEM"),
        (
            "This item cannot be classified confidently.",
            "UNCERTAIN",
            "UNIVERSITY_LEVEL_UNCERTAIN",
        ),
    ],
)
def test_university_stem_semantic_reject_reason_taxonomy(
    question: str,
    label: str,
    reason_code: str,
) -> None:
    primary, verifier = _providers(_decision(label, 0.99), None)
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate(question))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == reason_code
    assert len(verifier.requests) == 0


def test_university_site_and_tag_only_evidence_cannot_support_positive_pass() -> None:
    hint_only = (
        "source_hints.metadata.site",
        "source_hints.metadata.tags[0]",
    )
    primary, verifier = _providers(
        _decision(
            "UNIVERSITY_STEM",
            0.99,
            evidence_references=hint_only,
        ),
        _decision(
            "UNIVERSITY_STEM",
            0.99,
            evidence_references=hint_only,
        ),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )
    candidate = _candidate(
        "What is 7 + 5?",
        subject_candidates=("mathematics",),
        metadata={"site": "math.stackexchange.com", "tags": ["arithmetic"]},
    )

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "UNIVERSITY_LEVEL_UNCERTAIN"
    assert len(verifier.requests) == 0
    assert primary.requests[0].inputs["source_hints"] == {
        "subject_candidates": ["mathematics"],
        "metadata": {"site": "math.stackexchange.com", "tags": ["arithmetic"]},
    }


@pytest.mark.parametrize(
    "label",
    [
        "CALCULATION",
        "PROOF",
        "DERIVATION",
        "CONCEPTUAL",
        "MULTIPLE_CHOICE",
        "PROGRAMMING",
        "ALGORITHM",
        "ENGINEERING",
    ],
)
def test_problem_positive_taxonomy_can_pass(label: str) -> None:
    primary, verifier = _providers(_decision(label, 0.99), None)
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Solve the stated STEM problem."))

    assert evidence.verdict is GateVerdict.PASS
    assert len(verifier.requests) == 0


@pytest.mark.parametrize(
    "label",
    [
        "SOFTWARE_USAGE",
        "DEBUG_HELP",
        "CAREER_ADVICE",
        "OPINION",
        "RESOURCE_REQUEST",
        "DISCUSSION",
        "NEWS",
        "META",
    ],
)
def test_problem_formal_negative_taxonomy_rejects(label: str) -> None:
    primary, verifier = _providers(_decision(label, 0.99), None)
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Please recommend a debugging resource."))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "NOT_PROBLEM"
    assert len(verifier.requests) == 0


def test_problem_uncertain_label_uses_formal_uncertain_reason() -> None:
    primary, verifier = _providers(_decision("UNCERTAIN", 0.99), None)
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Ambiguous source content."))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "PROBLEM_TYPE_UNCERTAIN"
    assert len(verifier.requests) == 0


def test_problem_gate_uses_middle_band_verifier_path() -> None:
    primary, verifier = _providers(
        _decision("CALCULATION", 0.95),
        _decision("CALCULATION", 0.99),
    )
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Calculate the electric field."))

    assert evidence.verdict is GateVerdict.VERIFY
    assert evidence.score == 0.95
    assert len(verifier.requests) == 1


def test_problem_site_and_tag_only_evidence_cannot_support_positive_pass() -> None:
    hint_only = (
        "source_hints.metadata.site",
        "source_hints.metadata.tags[0]",
    )
    primary, verifier = _providers(
        _decision("CALCULATION", 0.99, evidence_references=hint_only),
        _decision("CALCULATION", 0.99, evidence_references=hint_only),
    )
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )
    candidate = _candidate(
        "What is 7 + 5?",
        metadata={"site": "math.stackexchange.com", "tags": ["arithmetic"]},
    )

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "PROBLEM_TYPE_UNCERTAIN"
    assert len(verifier.requests) == 0


class ConstructedMalformedProvider:
    provider = "fake"
    model = "malformed-v1"

    def classify(self, request: object) -> ModelDecision:
        return ModelDecision.model_construct(
            label="UNIVERSITY_STEM",
            score=1.5,
            evidence_references=("question:0-20",),
            reason_code="MALFORMED",
        )


def test_gate_revalidates_typed_model_decision_at_provider_boundary() -> None:
    malformed = cast(StructuredModelProvider, ConstructedMalformedProvider())
    verifier = FakeStructuredModelProvider(
        provider="fake",
        model="verifier-v2",
        decisions=(_decision("UNIVERSITY_STEM", 0.99),),
    )
    gate = UniversityStemGate(
        primary=malformed,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Compute the eigenvalues."))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "MALFORMED_MODEL_DECISION"
    assert len(verifier.requests) == 0
