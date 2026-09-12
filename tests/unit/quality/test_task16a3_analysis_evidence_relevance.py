from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.analysis import AnalysisGate
from college_builder.quality.engine import GateContext, GateEngine


def test_gate4_rejects_in_range_alphanumeric_but_irrelevant_evidence_span() -> None:
    analysis = (
        "Background 2024. Since 2x + 3 = 11, subtract 3 and divide by 2 "
        "to obtain x = 4."
    )
    irrelevant = "Background 2024"
    start = analysis.index(irrelevant)
    end = start + len(irrelevant)

    primary = FakeStructuredModelProvider(
        provider="fake",
        model="analysis-primary-v1",
        decisions=(
            ModelDecision(
                label="DERIVATION",
                score=0.99,
                evidence_references=(f"analysis:{start}-{end}",),
                reason_code="TEST",
            ),
        ),
    )
    verifier = FakeStructuredModelProvider(
        provider="fake",
        model="analysis-verifier-v2",
        decisions=(),
    )
    candidate = NormalizedQA(
        record_id="norm-task16a3-relevance",
        source_record_id="raw-task16a3-relevance",
        question="Solve 2x + 3 = 11.",
        answer="x = 4",
        analysis=analysis,
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )
    gate = AnalysisGate(
        primary=primary,
        verifier=verifier,
        prompt="analysis prompt",
        prompt_version="v3",
    )

    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="task16-recertification-v1"),
    ).run(candidate)
    evidence = result.evidence[0]

    assert analysis[start:end] == irrelevant
    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "ANALYSIS_UNCERTAIN"
    assert len(primary.requests) == 1
    assert len(verifier.requests) == 0
