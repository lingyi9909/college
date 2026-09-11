from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.classify import ProblemGate
from college_builder.quality.engine import GateContext, GateEngine


def _decision(label: str, score: float = 0.95) -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        evidence_references=("question:0-20",),
        reason_code="TASK16_REGRESSION",
    )


def _candidate() -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16-gate2",
        source_record_id="raw-task16-gate2",
        question="Prove the stated identity and derive the requested expression.",
        answer="source answer",
        analysis="source analysis",
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _run(primary_label: str, verifier_label: str):
    primary = FakeStructuredModelProvider(
        provider="fake",
        model="primary-v1",
        decisions=(_decision(primary_label),),
    )
    verifier = FakeStructuredModelProvider(
        provider="fake",
        model="verifier-v2",
        decisions=(_decision(verifier_label, 0.99),),
    )
    gate = ProblemGate(
        primary=primary,
        verifier=verifier,
        prompt="problem prompt",
        prompt_version="v1",
    )
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="task16-red"),
    ).run(_candidate())
    return result.evidence[0]


def test_gate2_compatible_positive_labels_are_not_model_conflict() -> None:
    evidence = _run("PROOF", "DERIVATION")

    assert evidence.verdict is GateVerdict.VERIFY
    assert evidence.reason_code == "PROBLEM_REVIEW_REQUIRED"


def test_gate2_positive_negative_disagreement_still_fails_closed() -> None:
    evidence = _run("PROOF", "DISCUSSION")

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "MODEL_DECISION_CONFLICT"
