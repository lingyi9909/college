from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelClassificationRequest, ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.engine import GateContext, GateEngine, GateResultEvidence
from college_builder.quality.verify import (
    AlignmentGate,
    CorrectnessVerifier,
    DeterministicMathVerifier,
    DeterministicVerificationStatus,
)


def _candidate(
    *,
    question: str = "Solve 2*x + 3 = 11.",
    answer: str = "x = 4",
    analysis: str = "Subtract 3 from both sides and divide by 2, so x = 4.",
) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-1",
        source_record_id="raw-1",
        question=question,
        answer=answer,
        analysis=analysis,
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _source_refs(candidate: NormalizedQA) -> tuple[str, ...]:
    return (
        f"question:0-{len(candidate.question)}",
        f"answer:0-{len(candidate.answer)}",
        f"analysis:0-{len(candidate.analysis)}",
    )


def _decision(
    label: str,
    score: float,
    candidate: NormalizedQA,
    *,
    evidence_references: tuple[str, ...] | None = None,
    reason_code: str = "TEST",
) -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        evidence_references=evidence_references or _source_refs(candidate),
        reason_code=reason_code,
    )


def _run(gate: object, candidate: NormalizedQA) -> GateResultEvidence:
    result = GateEngine(
        gates=(gate,),  # type: ignore[arg-type]
        context=GateContext(config_version="pilot-v1"),
    ).run(candidate)
    assert len(result.evidence) == 1
    return result.evidence[0]


class _ArbitraryProvider:
    def __init__(self, output: object) -> None:
        self.provider = "fake"
        self.model = "arbitrary-v1"
        self.output = output
        self.requests: list[ModelClassificationRequest] = []

    def classify(self, request: ModelClassificationRequest) -> Any:
        self.requests.append(request)
        return self.output


class _RaisingProvider:
    provider = "fake"
    model = "raising-v1"

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        raise RuntimeError("provider unavailable")


def _alignment_gate(output: ModelDecision) -> tuple[AlignmentGate, FakeStructuredModelProvider]:
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="alignment-v1",
        decisions=(output,),
    )
    return (
        AlignmentGate(
            provider=provider,
            prompt="alignment prompt",
            prompt_version="v1",
        ),
        provider,
    )


def _correctness_gate(
    output: ModelDecision,
    *,
    calibrator: object | None = None,
) -> tuple[CorrectnessVerifier, FakeStructuredModelProvider]:
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="correctness-v2",
        decisions=(output,),
    )
    kwargs: dict[str, object] = {}
    if calibrator is not None:
        kwargs["calibrator"] = calibrator
    return (
        CorrectnessVerifier(
            provider=provider,
            prompt="correctness prompt",
            prompt_version="v1",
            calibration_id="gold-cal-v1",
            **kwargs,  # type: ignore[arg-type]
        ),
        provider,
    )


def test_alignment_pass_requires_same_question_answer_and_analysis_support() -> None:
    candidate = _candidate()
    gate, provider = _alignment_gate(_decision("PASS", 0.998, candidate))

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.reason_code == "QA_ALIGNMENT_CONFIRMED"
    assert evidence.score == 0.998
    assert evidence.evidence_payload["alignment"] == {
        "provider": "fake",
        "model": "alignment-v1",
        "verdict": "PASS",
        "alignment_score": 0.998,
        "reason_code": "TEST",
        "evidence_references": list(_source_refs(candidate)),
    }
    request = provider.requests[0]
    assert request.task == "gate_5_qa_alignment"
    assert request.allowed_labels == ("PASS", "FAIL", "UNCERTAIN")
    assert request.inputs == {
        "question": candidate.question,
        "answer": candidate.answer,
        "analysis": candidate.analysis,
    }


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        ("FAIL", "QA_ALIGNMENT_MISMATCH"),
        ("UNCERTAIN", "QA_ALIGNMENT_UNCERTAIN"),
    ],
)
def test_alignment_fail_or_uncertain_rejects(label: str, reason: str) -> None:
    candidate = _candidate()
    gate, _ = _alignment_gate(_decision(label, 0.999, candidate))

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == reason


def test_alignment_below_0995_rejects_even_when_label_is_pass() -> None:
    candidate = _candidate()
    gate, _ = _alignment_gate(_decision("PASS", 0.9949, candidate))

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "QA_ALIGNMENT_UNCERTAIN"


@pytest.mark.parametrize(
    "reference",
    [
        "question:999-1000",
        "answer:999-1000",
        "analysis:999-1000",
        "metadata.site",
    ],
)
def test_alignment_pass_requires_real_spans_for_all_source_fields(reference: str) -> None:
    candidate = _candidate()
    refs = list(_source_refs(candidate))
    if reference.startswith("question:"):
        refs[0] = reference
    elif reference.startswith("answer:"):
        refs[1] = reference
    elif reference.startswith("analysis:"):
        refs[2] = reference
    else:
        refs = [reference]
    gate, _ = _alignment_gate(
        _decision("PASS", 0.999, candidate, evidence_references=tuple(refs))
    )

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "QA_ALIGNMENT_EVIDENCE_INVALID"


@pytest.mark.parametrize(
    "output",
    [
        ModelDecision.model_construct(
            label="PASS",
            score=2.0,
            evidence_references=("question:0-1",),
            reason_code="BYPASS",
        ),
        ModelDecision.model_construct(
            label="PASS",
            score=0.999,
            evidence_references=(),
            reason_code="BYPASS",
        ),
        {"label": "PASS", "score": 0.999},
        None,
        object(),
    ],
)
def test_alignment_revalidates_provider_contract_and_never_crashes(output: object) -> None:
    candidate = _candidate()
    provider = _ArbitraryProvider(output)
    gate = AlignmentGate(provider=provider, prompt="alignment", prompt_version="v1")

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "MALFORMED_MODEL_DECISION"


def test_alignment_provider_exception_fails_closed() -> None:
    evidence = _run(
        AlignmentGate(provider=_RaisingProvider(), prompt="alignment", prompt_version="v1"),
        _candidate(),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "MODEL_PROVIDER_ERROR"


def test_correctness_pass_is_required_and_calibrated_score_is_stored() -> None:
    candidate = _candidate()
    gate, provider = _correctness_gate(
        _decision("PASS", 0.98, candidate),
        calibrator=lambda score: min(1.0, score + 0.01),
    )

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.reason_code == "CORRECTNESS_CONFIRMED"
    assert evidence.score == 0.99
    verifier = evidence.evidence_payload["verifier"]
    assert verifier == {
        "provider": "fake",
        "model": "correctness-v2",
        "verdict": "PASS",
        "raw_score": 0.98,
        "calibrated_score": 0.99,
        "calibration_id": "gold-cal-v1",
        "reason_code": "TEST",
        "evidence_references": list(_source_refs(candidate)),
    }
    assert len(provider.requests) == 1


@pytest.mark.parametrize(
    ("label", "reason"),
    [
        ("FAIL", "CORRECTNESS_MISMATCH"),
        ("UNCERTAIN", "CORRECTNESS_UNCERTAIN"),
    ],
)
def test_correctness_fail_or_uncertain_rejects(label: str, reason: str) -> None:
    candidate = _candidate(question="Explain why the sky is blue.", answer="Rayleigh scattering")
    gate, _ = _correctness_gate(_decision(label, 0.99, candidate))

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == reason


def test_correctness_pass_requires_source_evidence_and_never_accepts_generated_solution() -> None:
    candidate = _candidate(question="Explain conservation of energy.", answer="Energy is conserved")
    decision = _decision(
        "PASS",
        0.99,
        candidate,
        evidence_references=("model_solution:0-20",),
    )
    gate, _ = _correctness_gate(decision)

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "CORRECTNESS_EVIDENCE_INVALID"
    assert "solution" not in json.dumps(evidence.evidence_payload).lower()


def test_correctness_request_contains_only_source_content() -> None:
    candidate = _candidate(question="Explain conservation of energy.", answer="Energy is conserved")
    gate, provider = _correctness_gate(_decision("PASS", 0.99, candidate))

    _run(gate, candidate)

    request = provider.requests[0]
    assert request.task == "independent_correctness_verification"
    assert request.inputs == {
        "question": candidate.question,
        "answer": candidate.answer,
        "analysis": candidate.analysis,
    }
    assert "generated_solution" not in request.inputs


@pytest.mark.parametrize(
    "output",
    [
        ModelDecision.model_construct(
            label="PASS",
            score=-0.1,
            evidence_references=("question:0-1",),
            reason_code="BYPASS",
        ),
        ModelDecision.model_construct(
            label="PASS",
            score=0.99,
            evidence_references=(),
            reason_code="BYPASS",
        ),
        {"label": "PASS", "score": 0.99},
        None,
        object(),
    ],
)
def test_correctness_revalidates_provider_contract_and_never_crashes(output: object) -> None:
    candidate = _candidate(question="Explain conservation of energy.", answer="Energy is conserved")
    provider = _ArbitraryProvider(output)
    gate = CorrectnessVerifier(
        provider=provider,
        prompt="correctness",
        prompt_version="v1",
        calibration_id="gold-cal-v1",
    )

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "MALFORMED_MODEL_DECISION"


def test_invalid_calibrated_score_fails_closed() -> None:
    candidate = _candidate(question="Explain conservation of energy.", answer="Energy is conserved")
    gate, _ = _correctness_gate(
        _decision("PASS", 0.99, candidate),
        calibrator=lambda _: 2.0,
    )

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "CALIBRATION_ERROR"


def test_deterministic_arithmetic_confirms_without_modifying_source_answer() -> None:
    candidate = _candidate(
        question="Compute 2 + 3.",
        answer="5",
        analysis="Adding two and three gives five.",
    )

    result = DeterministicMathVerifier().verify(candidate)

    assert result.status is DeterministicVerificationStatus.CONFIRMED
    assert candidate.answer == "5"


def test_deterministic_arithmetic_contradiction_rejects_before_model() -> None:
    candidate = _candidate(
        question="Compute 2 + 3.",
        answer="6",
        analysis="The source claims the result is six.",
    )
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="correctness-v2",
        decisions=(_decision("PASS", 0.99, candidate),),
    )
    gate = CorrectnessVerifier(
        provider=provider,
        prompt="correctness",
        prompt_version="v1",
        calibration_id="identity-v1",
    )

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "DETERMINISTIC_CORRECTNESS_MISMATCH"
    assert evidence.evidence_payload["deterministic"]["status"] == "CONTRADICTED"
    assert provider.requests == ()


def test_deterministic_simple_equation_confirms_source_answer() -> None:
    candidate = _candidate(
        question="Solve 2*x + 3 = 11.",
        answer="x = 4",
        analysis="Subtract 3 and divide by 2.",
    )

    result = DeterministicMathVerifier().verify(candidate)

    assert result.status is DeterministicVerificationStatus.CONFIRMED


def test_unparseable_math_is_not_deterministically_verified_and_still_uses_model() -> None:
    candidate = _candidate(
        question="Prove that every finite subgroup of a field's multiplicative group is cyclic.",
        answer="The subgroup is cyclic.",
        analysis="Use the polynomial root bound and the structure theorem.",
    )
    gate, provider = _correctness_gate(_decision("PASS", 0.99, candidate))

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.PASS
    assert evidence.evidence_payload["deterministic"]["status"] == "NOT_VERIFIED"
    assert len(provider.requests) == 1


def test_deterministic_confirmation_does_not_bypass_independent_model_verification() -> None:
    candidate = _candidate(
        question="Compute 2 + 3.",
        answer="5",
        analysis="Adding two and three gives five.",
    )
    gate, provider = _correctness_gate(_decision("UNCERTAIN", 0.99, candidate))

    evidence = _run(gate, candidate)

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "CORRECTNESS_UNCERTAIN"
    assert evidence.evidence_payload["deterministic"]["status"] == "CONFIRMED"
    assert len(provider.requests) == 1


def test_verification_golden_cases() -> None:
    cases_path = Path(__file__).parents[2] / "golden" / "verification_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert cases

    for case in cases:
        candidate = _candidate(
            question=case["question"],
            answer=case["answer"],
            analysis=case["analysis"],
        )
        if case["gate"] == "alignment":
            gate, _ = _alignment_gate(
                _decision(case["label"], case["score"], candidate)
            )
        else:
            gate, _ = _correctness_gate(
                _decision(case["label"], case["score"], candidate)
            )
        evidence = _run(gate, candidate)
        assert evidence.verdict.value == case["expected_verdict"]
        assert evidence.reason_code == case["expected_reason"]
