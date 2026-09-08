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


def _decision(label: str, score: float, reason: str = "TEST") -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        evidence_references=("question:0-20",),
        reason_code=reason,
    )


def _providers(
    primary: ModelDecision,
    verifier: ModelDecision,
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
            decisions=(verifier,),
        ),
    )


def _run(gate: UniversityStemGate | ProblemGate, candidate: NormalizedQA) -> GateResultEvidence:
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="pilot-v1"),
    ).run(candidate)
    assert len(result.evidence) == 1
    return result.evidence[0]


@pytest.mark.parametrize("score", [0.98, 0.99, 1.0])
def test_university_gate_passes_only_at_or_above_pilot_threshold(score: float) -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", score),
        _decision("UNIVERSITY_STEM", score),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Compute the eigenvalues of this matrix."))

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.score == score


@pytest.mark.parametrize("score", [0.90, 0.95, 0.979999])
def test_university_gate_routes_mid_confidence_to_verify(score: float) -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", score),
        _decision("UNIVERSITY_STEM", score),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Derive the wave equation from Maxwell's equations."))

    assert evidence.verdict is GateVerdict.VERIFY
    assert evidence.score == score


def test_university_gate_rejects_below_verify_threshold() -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", 0.899999),
        _decision("UNIVERSITY_STEM", 0.899999),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Prove the asymptotic complexity bound."))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "UNIVERSITY_STEM_LOW_CONFIDENCE"


def test_primary_verifier_label_conflict_fails_closed() -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", 0.99),
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


def test_primary_and_verifier_must_have_independent_provider_model_identity() -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", 0.99),
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


def test_gate_evidence_records_both_execution_identities_and_decisions() -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", 0.99, "PRIMARY_REASON"),
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

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.provider == "provider-a"
    assert evidence.model == "model-a"
    assert evidence.prompt_version == "v1"
    assert evidence.evidence_payload["primary"] == {
        "provider": "provider-a",
        "model": "model-a",
        "label": "UNIVERSITY_STEM",
        "score": 0.99,
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
def test_university_stem_positive_taxonomy_can_pass_with_independent_evidence(
    question: str,
) -> None:
    primary, verifier = _providers(
        _decision("UNIVERSITY_STEM", 0.99),
        _decision("UNIVERSITY_STEM", 0.99),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate(question))

    assert evidence.verdict is GateVerdict.PASS


@pytest.mark.parametrize(
    ("question", "label"),
    [
        ("Which IDE do you recommend for Python?", "NON_UNIVERSITY_STEM"),
        ("How do I install this software package?", "NON_UNIVERSITY_STEM"),
        ("Should I choose a software engineering career?", "NON_UNIVERSITY_STEM"),
        ("What is 7 + 5?", "K12"),
    ],
)
def test_university_stem_negative_taxonomy_rejects(question: str, label: str) -> None:
    primary, verifier = _providers(
        _decision(label, 0.99),
        _decision(label, 0.99),
    )
    gate = UniversityStemGate(
        primary=primary,
        verifier=verifier,
        prompt="university prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate(question))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "NOT_UNIVERSITY_STEM"


def test_math_stackexchange_site_hint_never_forces_university_pass() -> None:
    primary, verifier = _providers(
        _decision("K12", 0.99),
        _decision("K12", 0.99),
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
    assert evidence.reason_code == "NOT_UNIVERSITY_STEM"
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
        "ALGORITHM",
        "PROGRAMMING",
        "ENGINEERING",
    ],
)
def test_problem_positive_taxonomy_can_pass(label: str) -> None:
    primary, verifier = _providers(_decision(label, 0.99), _decision(label, 0.99))
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Solve the stated STEM problem."))

    assert evidence.verdict is GateVerdict.PASS


@pytest.mark.parametrize(
    "label",
    ["SOFTWARE_USE", "DEBUG_HELP", "OPINION", "RESOURCE_REQUEST", "META"],
)
def test_problem_negative_taxonomy_rejects(label: str) -> None:
    primary, verifier = _providers(_decision(label, 0.99), _decision(label, 0.99))
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )

    evidence = _run(gate, _candidate("Please recommend a debugging resource."))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "NOT_A_PROBLEM"


def test_problem_gate_applies_same_threshold_bands() -> None:
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
