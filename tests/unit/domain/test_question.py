import pytest
from pydantic import ValidationError

from college_builder.domain.evidence import GateEvidence, GateVerdict
from college_builder.domain.question import (
    AnalysisContent,
    AnalysisType,
    AnswerContent,
    Classification,
    DedupState,
    Discipline,
    ProblemType,
    QualityState,
    QuestionContent,
    QuestionMetadata,
    QuestionProvenance,
    UniversityLevel,
    UniversityQuestionIR,
)


def _gate_evidence() -> GateEvidence:
    return GateEvidence(
        gate_name="university",
        verdict=GateVerdict.PASS,
        score=0.99,
        provider="fake",
        model="fake-v1",
        prompt_version="university_v1",
        config_version="pilot_v1",
    )


def test_university_question_ir_keeps_raw_and_normalized_question_separately() -> None:
    ir = UniversityQuestionIR(
        candidate_id="uq_42",
        source_record_id="raw_stackmathqa_42",
        question=QuestionContent(raw="<p>Q</p>", normalized="Q"),
        answer=AnswerContent(raw="A", final_answer="A"),
        analysis=AnalysisContent(raw="Because A", type=AnalysisType.DERIVATION),
        classification=Classification(
            discipline=Discipline.MATHEMATICS,
            course="Linear Algebra",
            level=UniversityLevel.UNDERGRADUATE,
            problem_type=ProblemType.CALCULATION,
        ),
        metadata=QuestionMetadata(language="en"),
        quality=QualityState(gates=(_gate_evidence(),)),
        provenance=QuestionProvenance(
            source_dataset="stackmathqa",
            source_id="42",
            source_url="https://math.stackexchange.com/questions/42",
            raw_sha256="a" * 64,
        ),
        dedup=DedupState(),
    )

    assert ir.question.raw == "<p>Q</p>"
    assert ir.question.normalized == "Q"
    assert ir.question.raw != ir.question.normalized


def test_core_ir_sections_are_typed_objects_not_dict_contracts() -> None:
    with pytest.raises(ValidationError):
        UniversityQuestionIR(
            candidate_id="uq_42",
            source_record_id="raw_stackmathqa_42",
            question={"raw": "Q", "normalized": "Q"},
            answer={"raw": "A", "final_answer": "A"},
            analysis={"raw": "Because A", "type": "DERIVATION"},
            classification={
                "discipline": "MATHEMATICS",
                "course": "Linear Algebra",
                "level": "UNDERGRADUATE",
                "problem_type": "CALCULATION",
            },
            metadata={"language": "en"},
            quality={"gates": []},
            provenance={
                "source_dataset": "stackmathqa",
                "source_id": "42",
                "raw_sha256": "a" * 64,
            },
            dedup={},
        )


def test_gate_evidence_requires_execution_identity_and_versions() -> None:
    required = {
        "gate_name",
        "verdict",
        "score",
        "provider",
        "model",
        "prompt_version",
        "config_version",
    }

    assert required <= set(GateEvidence.model_fields)

    with pytest.raises(ValidationError):
        GateEvidence(  # type: ignore[call-arg]
            gate_name="university",
            verdict=GateVerdict.PASS,
            score=0.99,
        )


def test_gate_evidence_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        GateEvidence(
            gate_name="university",
            verdict=GateVerdict.PASS,
            score=1.01,
            provider="fake",
            model="fake-v1",
            prompt_version="university_v1",
            config_version="pilot_v1",
        )
