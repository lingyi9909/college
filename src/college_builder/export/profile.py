"""University STEM validation profile and exact 19-field mapping."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Literal

from pydantic import field_validator

from college_builder.domain.final_record import FinalQuestionRecord, slim_question_md5_v1
from college_builder.domain.question import (
    Discipline,
    ProblemType,
    UniversityLevel,
    UniversityQuestionIR,
)

UNIVERSITY_COURSES: frozenset[str] = frozenset(
    {
        "数学",
        "统计学",
        "物理",
        "化学",
        "计算机",
        "电子信息",
        "电气工程",
        "自动化",
        "机械工程",
        "材料科学",
        "土木工程",
        "生物科学",
        "其他理工",
        "未知",
    }
)
UNIVERSITY_GRADE_LEVELS: frozenset[str] = frozenset(
    {
        "本科一年级",
        "本科二年级",
        "本科三年级",
        "本科四年级",
        "本科",
        "硕士",
        "博士",
        "研究生",
        "大学未知",
    }
)

UniversityCourse = Literal[
    "数学",
    "统计学",
    "物理",
    "化学",
    "计算机",
    "电子信息",
    "电气工程",
    "自动化",
    "机械工程",
    "材料科学",
    "土木工程",
    "生物科学",
    "其他理工",
    "未知",
]
UniversityGradeLevel = Literal[
    "本科一年级",
    "本科二年级",
    "本科三年级",
    "本科四年级",
    "本科",
    "硕士",
    "博士",
    "研究生",
    "大学未知",
]

PIPELINE_VERSION = "university_dataset_builder_0.1.0"
STATIC_INFO_REQUIRED_KEYS: frozenset[str] = frozenset(
    {
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
)
_IMAGE_SRC_RE = re.compile(
    r"<img\b[^>]*\bsrc\s*=\s*['\"](?P<src>[^'\"]+)['\"][^>]*>",
    re.IGNORECASE,
)
_SOURCE_SPAN_RE = re.compile(
    r"^(?P<field>answer|analysis):(?P<start>\d+):(?P<end>\d+)$"
)
_DISCIPLINE_TO_COURSE: dict[Discipline, UniversityCourse] = {
    Discipline.MATHEMATICS: "数学",
    Discipline.STATISTICS: "统计学",
    Discipline.PHYSICS: "物理",
    Discipline.CHEMISTRY: "化学",
    Discipline.COMPUTER_SCIENCE: "计算机",
    Discipline.ELECTRONICS: "电子信息",
    Discipline.ELECTRICAL_ENGINEERING: "电气工程",
    Discipline.AUTOMATION: "自动化",
    Discipline.MECHANICAL_ENGINEERING: "机械工程",
    Discipline.MATERIALS: "材料科学",
    Discipline.CIVIL_ENGINEERING: "土木工程",
    Discipline.BIOLOGICAL_SCIENCE: "生物科学",
    Discipline.OTHER_STEM: "其他理工",
}
_LEVEL_TO_GRADE: dict[UniversityLevel, UniversityGradeLevel] = {
    UniversityLevel.UNDERGRADUATE: "本科",
    UniversityLevel.MASTERS: "硕士",
    UniversityLevel.DOCTORAL: "博士",
    UniversityLevel.GRADUATE: "研究生",
    UniversityLevel.UNIVERSITY_UNKNOWN: "大学未知",
}
_POSITIVE_PROBLEM_TYPES: frozenset[ProblemType] = frozenset(
    {
        ProblemType.CALCULATION,
        ProblemType.PROOF,
        ProblemType.DERIVATION,
        ProblemType.CONCEPTUAL,
        ProblemType.MULTIPLE_CHOICE,
        ProblemType.PROGRAMMING,
        ProblemType.ALGORITHM,
        ProblemType.ENGINEERING,
    }
)


class UniversitySTEMProfile(FinalQuestionRecord):
    """Strict university-only validator without widening the K12 profile."""

    text_course: UniversityCourse
    text_grade_level: UniversityGradeLevel
    text_grade: Literal["大学"]
    entrance_exam_type: Literal["未知"]

    @field_validator("answer_analysis")
    @classmethod
    def require_source_analysis(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped or stripped == "略":
            raise ValueError("university answer_analysis must contain real source analysis")
        return value

    @field_validator("static_info")
    @classmethod
    def require_profile_identity(cls, value: str) -> str:
        parsed = json.loads(value)
        if parsed.get("contract_version") != "question_record_v1":
            raise ValueError("static_info contract_version must be question_record_v1")
        if parsed.get("profile") != "university_stem_v1":
            raise ValueError("static_info profile must be university_stem_v1")
        return value


def image_references(text: str) -> tuple[str, ...]:
    """Return formal image/... references in final-question order."""
    return tuple(
        source
        for source in _all_image_sources(text)
        if source.startswith("image/")
    )


def to_final_record(ir: UniversityQuestionIR) -> FinalQuestionRecord:
    """Mechanically map an already-qualified university IR into QuestionRecordV1."""
    text_course = _map_discipline(ir.classification.discipline)
    text_grade_level = _map_level(ir.classification.level)
    question_type = _map_problem_type(ir.classification.problem_type)

    final_answer = ir.answer.final_answer
    if final_answer is None or not final_answer.strip():
        raise ValueError("formal export requires source-grounded final_answer")
    if ir.answer.source_span is None or not ir.answer.source_span.strip():
        raise ValueError("formal export requires final_answer source span")

    source_answer_text = _validated_answer_authority_source_text(ir, final_answer)
    question_text = ir.question.normalized
    references = _validate_image_inventory(question_text, ir.question.assets)
    static_info = {
        "slim_question_md5": slim_question_md5_v1(question_text),
        "copyright": "0",
        "contract_version": "question_record_v1",
        "profile": "university_stem_v1",
        "source_dataset": ir.provenance.source_dataset,
        "source_id": ir.provenance.source_id,
        "source_url": ir.provenance.source_url or "",
        "source_license": "UNKNOWN",
        "source_question_hash": _sha256(ir.question.raw),
        "source_answer_hash": _sha256(source_answer_text),
        "pipeline_version": PIPELINE_VERSION,
        "quality_tier": "HIGH_CONFIDENCE",
        "raw_sha256": ir.provenance.raw_sha256.lower(),
        "course_name": ir.classification.course or "",
        "answer_source_span": ir.answer.source_span,
    }
    if STATIC_INFO_REQUIRED_KEYS.difference(static_info):
        raise ValueError("Task 13 static_info provenance is incomplete")
    text_paper = _source_title(ir)

    return UniversitySTEMProfile(
        text_question=question_text,
        is_pic_included=1 if references else 0,
        text_answer=final_answer,
        answer_analysis=ir.analysis.raw,
        text_course=text_course,
        text_grade_level=text_grade_level,
        text_grade="大学",
        knowledge_points="; ".join(ir.metadata.knowledge_points),
        exam_points="; ".join(ir.metadata.exam_points),
        publisher="",
        text_paper=text_paper,
        textbook_version="",
        static_info=json.dumps(static_info, ensure_ascii=False, separators=(",", ":")),
        language=ir.metadata.language,
        text_year="",
        entrance_exam_type="未知",
        text_city="",
        question_type=question_type,
        competition_event="",
    )


def _map_discipline(discipline: Discipline) -> UniversityCourse:
    try:
        return _DISCIPLINE_TO_COURSE[discipline]
    except KeyError as exc:
        raise ValueError(f"discipline {discipline.value} is not exportable") from exc


def _map_level(level: UniversityLevel) -> UniversityGradeLevel:
    try:
        return _LEVEL_TO_GRADE[level]
    except KeyError as exc:
        raise ValueError(f"university level {level.value} is not exportable") from exc


def _map_problem_type(problem_type: ProblemType) -> Literal["选择题", "问答题"]:
    if problem_type not in _POSITIVE_PROBLEM_TYPES:
        raise ValueError(f"problem type {problem_type.value} is not exportable")
    if problem_type is ProblemType.MULTIPLE_CHOICE:
        return "选择题"
    return "问答题"


def _source_title(ir: UniversityQuestionIR) -> str:
    course = ir.classification.course
    if course is not None and course.strip():
        return course
    return ir.provenance.source_dataset


def _validated_answer_authority_source_text(
    ir: UniversityQuestionIR,
    final_answer: str,
) -> str:
    span = ir.answer.source_span
    assert span is not None
    match = _SOURCE_SPAN_RE.fullmatch(span)
    if match is None:
        raise ValueError("final_answer source span must use answer:<start>:<end> or analysis:<start>:<end>")

    source_field = match.group("field")
    source_text = ir.answer.raw if source_field == "answer" else ir.analysis.raw
    start = int(match.group("start"))
    end = int(match.group("end"))
    if not 0 <= start < end <= len(source_text):
        raise ValueError("final_answer source span is empty, reversed, or out of range")

    source_slice = source_text[start:end]
    if not source_slice:
        raise ValueError("final_answer source span must resolve to non-empty source text")
    if source_slice != final_answer:
        raise ValueError("final_answer does not match its authorized source span")
    return source_text


def _all_image_sources(text: str) -> tuple[str, ...]:
    return tuple(match.group("src") for match in _IMAGE_SRC_RE.finditer(text))


def _validate_image_inventory(
    question_text: str,
    declared_assets: tuple[str, ...],
) -> tuple[str, ...]:
    placed_sources = _all_image_sources(question_text)
    for path in (*declared_assets, *placed_sources):
        _validate_formal_image_path(path)

    if set(declared_assets) != set(placed_sources):
        raise ValueError(
            "declared image assets must exactly match final question image references"
        )
    return placed_sources


def _validate_formal_image_path(path: str) -> None:
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or len(pure.parts) < 2
        or pure.parts[0] != "image"
        or ".." in pure.parts
        or not path.startswith("image/")
    ):
        raise ValueError(f"invalid formal image path: {path}")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
