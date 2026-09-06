"""University STEM validation profile for the exact 19-field contract."""

from __future__ import annotations

import json
from typing import Literal

from pydantic import field_validator

from college_builder.domain.final_record import FinalQuestionRecord

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
