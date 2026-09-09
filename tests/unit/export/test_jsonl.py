from __future__ import annotations

import hashlib
import json

import pytest

from college_builder.domain.evidence import GateEvidence, GateVerdict
from college_builder.domain.final_record import FinalQuestionRecord, slim_question_md5_v1
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
from college_builder.export.profile import to_final_record


def _gate() -> GateEvidence:
    return GateEvidence(
        gate_name="gate_5_qa_alignment",
        verdict=GateVerdict.PASS,
        score=0.999,
        provider="internal-provider-secret",
        model="internal-model-secret",
        prompt_version="v1",
        config_version="pilot-v1",
    )


def _ir(
    *,
    discipline: Discipline = Discipline.MATHEMATICS,
    level: UniversityLevel = UniversityLevel.UNDERGRADUATE,
    problem_type: ProblemType = ProblemType.CALCULATION,
    course: str | None = "Linear Algebra",
    question: str = "Solve 2*x + 3 = 11.",
    assets: tuple[str, ...] = (),
    raw_answer: str = "x = 4",
    final_answer: str | None = "x = 4",
    source_span: str | None = "answer:0:5",
) -> UniversityQuestionIR:
    analysis = "Subtract 3 from both sides and divide by 2. Therefore x = 4"
    return UniversityQuestionIR(
        candidate_id="candidate-1",
        source_record_id="raw-1",
        question=QuestionContent(raw=question, normalized=question, assets=assets),
        answer=AnswerContent(
            raw=raw_answer,
            final_answer=final_answer,
            source_span=source_span,
        ),
        analysis=AnalysisContent(raw=analysis, type=AnalysisType.DERIVATION),
        classification=Classification(
            discipline=discipline,
            course=course,
            level=level,
            problem_type=problem_type,
        ),
        metadata=QuestionMetadata(
            knowledge_points=("linear equations", "algebra"),
            exam_points=("equation solving",),
            language="en",
        ),
        quality=QualityState(gates=(_gate(),)),
        provenance=QuestionProvenance(
            source_dataset="stackmathqa",
            source_id="42",
            source_url="https://math.stackexchange.com/questions/42",
            raw_sha256="a" * 64,
        ),
        dedup=DedupState(exact_hash=slim_question_md5_v1(question)),
    )


def test_to_final_record_maps_metadata_without_fabricating_unknown_fields() -> None:
    record = to_final_record(_ir())

    assert isinstance(record, FinalQuestionRecord)
    assert record.text_question == "Solve 2*x + 3 = 11."
    assert record.text_answer == "x = 4"
    assert record.answer_analysis.endswith("Therefore x = 4")
    assert record.text_course == "数学"
    assert record.text_grade_level == "本科"
    assert record.text_grade == "大学"
    assert record.knowledge_points == "linear equations; algebra"
    assert record.exam_points == "equation solving"
    assert record.publisher == ""
    assert record.textbook_version == ""
    assert record.text_year == ""
    assert record.text_city == ""
    assert record.competition_event == ""
    assert record.entrance_exam_type == "未知"
    assert record.text_paper == "Linear Algebra"
    assert record.question_type == "问答题"


def test_static_info_contains_required_source_traceability_and_no_provider_secrets() -> None:
    ir = _ir()
    record = to_final_record(ir)
    static_info = json.loads(record.static_info)

    required = {
        "slim_question_md5",
        "copyright",
        "contract_version",
        "profile",
        "source_dataset",
        "source_id",
        "source_url",
        "source_license",
        "source_question_hash",
        "source_answer_hash",
        "pipeline_version",
        "quality_tier",
    }
    assert required <= set(static_info)
    assert static_info["slim_question_md5"] == slim_question_md5_v1(ir.question.normalized)
    assert static_info["contract_version"] == "question_record_v1"
    assert static_info["profile"] == "university_stem_v1"
    assert static_info["source_dataset"] == "stackmathqa"
    assert static_info["source_id"] == "42"
    assert static_info["source_url"] == "https://math.stackexchange.com/questions/42"
    assert static_info["source_license"] == "UNKNOWN"
    assert static_info["source_question_hash"] == hashlib.sha256(
        ir.question.raw.encode("utf-8")
    ).hexdigest()
    assert static_info["source_answer_hash"] == hashlib.sha256(
        ir.answer.raw.encode("utf-8")
    ).hexdigest()
    serialized = json.dumps(static_info)
    assert "internal-provider-secret" not in serialized
    assert "internal-model-secret" not in serialized


def test_direct_answer_valid_canonical_span_passes() -> None:
    record = to_final_record(
        _ir(raw_answer="x = 4", final_answer="x = 4", source_span="answer:0:5")
    )

    assert record.text_answer == "x = 4"


def test_analysis_origin_valid_canonical_span_passes_and_hashes_authority_source() -> None:
    ir = _ir(raw_answer="See above", source_span="analysis:54:59")
    record = to_final_record(ir)
    static_info = json.loads(record.static_info)

    assert record.text_answer == "x = 4"
    assert static_info["source_answer_hash"] == hashlib.sha256(
        ir.analysis.raw.encode("utf-8")
    ).hexdigest()


def test_final_answer_must_equal_claimed_source_span() -> None:
    with pytest.raises(ValueError, match="source span"):
        to_final_record(
            _ir(raw_answer="x = 4", final_answer="x = 5", source_span="answer:0:5")
        )


@pytest.mark.parametrize(
    "source_span",
    [
        "answer:999:1000",
        "answer:0:0",
        "answer:5:0",
    ],
)
def test_out_of_range_zero_length_or_reversed_source_span_rejects(
    source_span: str,
) -> None:
    with pytest.raises(ValueError, match="source span"):
        to_final_record(_ir(source_span=source_span))


@pytest.mark.parametrize(
    "source_span",
    [
        "answer:0-5",
        "answer:zero:5",
        "answer:0",
        "answer:0:5:extra",
        ":0:5",
    ],
)
def test_malformed_source_span_rejects(source_span: str) -> None:
    with pytest.raises(ValueError, match="source span"):
        to_final_record(_ir(source_span=source_span))


def test_unsupported_source_span_field_rejects() -> None:
    with pytest.raises(ValueError, match="source span"):
        to_final_record(_ir(source_span="question:0:5"))


def test_analysis_span_must_cover_the_authorized_final_answer() -> None:
    with pytest.raises(ValueError, match="source span"):
        to_final_record(_ir(raw_answer="See above", source_span="analysis:0:3"))


@pytest.mark.parametrize(
    ("discipline", "expected"),
    [
        (Discipline.MATHEMATICS, "数学"),
        (Discipline.STATISTICS, "统计学"),
        (Discipline.PHYSICS, "物理"),
        (Discipline.CHEMISTRY, "化学"),
        (Discipline.COMPUTER_SCIENCE, "计算机"),
        (Discipline.ELECTRONICS, "电子信息"),
        (Discipline.ELECTRICAL_ENGINEERING, "电气工程"),
        (Discipline.AUTOMATION, "自动化"),
        (Discipline.MECHANICAL_ENGINEERING, "机械工程"),
        (Discipline.MATERIALS, "材料科学"),
        (Discipline.CIVIL_ENGINEERING, "土木工程"),
        (Discipline.BIOLOGICAL_SCIENCE, "生物科学"),
        (Discipline.OTHER_STEM, "其他理工"),
    ],
)
def test_first_level_discipline_mapping_is_mechanical(
    discipline: Discipline,
    expected: str,
) -> None:
    assert to_final_record(_ir(discipline=discipline)).text_course == expected


@pytest.mark.parametrize(
    ("level", "expected"),
    [
        (UniversityLevel.UNDERGRADUATE, "本科"),
        (UniversityLevel.MASTERS, "硕士"),
        (UniversityLevel.DOCTORAL, "博士"),
        (UniversityLevel.GRADUATE, "研究生"),
        (UniversityLevel.UNIVERSITY_UNKNOWN, "大学未知"),
    ],
)
def test_university_level_mapping_is_mechanical(
    level: UniversityLevel,
    expected: str,
) -> None:
    assert to_final_record(_ir(level=level)).text_grade_level == expected


def test_multiple_choice_maps_to_final_choice_question_type() -> None:
    record = to_final_record(_ir(problem_type=ProblemType.MULTIPLE_CHOICE))
    assert record.question_type == "选择题"


@pytest.mark.parametrize(
    ("discipline", "level", "problem_type"),
    [
        (Discipline.NON_STEM, UniversityLevel.UNDERGRADUATE, ProblemType.CALCULATION),
        (Discipline.MATHEMATICS, UniversityLevel.NON_UNIVERSITY, ProblemType.CALCULATION),
        (Discipline.MATHEMATICS, UniversityLevel.UNDERGRADUATE, ProblemType.DEBUG_HELP),
    ],
)
def test_invalid_formal_classification_fails_closed(
    discipline: Discipline,
    level: UniversityLevel,
    problem_type: ProblemType,
) -> None:
    with pytest.raises(ValueError):
        to_final_record(
            _ir(discipline=discipline, level=level, problem_type=problem_type)
        )


def test_final_answer_is_required_and_never_generated_from_analysis() -> None:
    with pytest.raises(ValueError, match="final_answer"):
        to_final_record(_ir(raw_answer="See above", final_answer=None, source_span=None))


def test_picture_flag_is_computed_from_question_image_reference_or_assets() -> None:
    with_ref = _ir(
        question='Use the diagram. <img src="image/diagram.png">',
        assets=("image/diagram.png",),
    )
    assert to_final_record(with_ref).is_pic_included == 1
    assert to_final_record(_ir()).is_pic_included == 0
