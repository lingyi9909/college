from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from college_builder.export.profile import (
    UNIVERSITY_COURSES,
    UNIVERSITY_GRADE_LEVELS,
    UniversitySTEMProfile,
)


def _payload() -> dict[str, object]:
    return {
        "text_question": (
            "Prove that the eigenvalues of a triangular matrix are its diagonal entries."
        ),
        "is_pic_included": 0,
        "text_answer": "The eigenvalues are exactly the diagonal entries.",
        "answer_analysis": (
            "The characteristic polynomial is the product of the diagonal terms of A-λI."
        ),
        "text_course": "数学",
        "text_grade_level": "本科",
        "text_grade": "大学",
        "knowledge_points": "特征值; 三角矩阵",
        "exam_points": "证明",
        "publisher": "",
        "text_paper": "Linear Algebra",
        "textbook_version": "",
        "static_info": json.dumps(
            {
                "contract_version": "question_record_v1",
                "profile": "university_stem_v1",
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        "language": "en",
        "text_year": "",
        "entrance_exam_type": "未知",
        "text_city": "",
        "question_type": "问答题",
        "competition_event": "",
    }


def test_university_profile_accepts_formal_university_record() -> None:
    record = UniversitySTEMProfile.model_validate(_payload())

    assert record.text_grade == "大学"
    assert record.text_grade_level == "本科"
    assert record.text_course == "数学"
    assert len(UniversitySTEMProfile.model_fields) == 19


@pytest.mark.parametrize("grade_level", sorted(UNIVERSITY_GRADE_LEVELS))
def test_university_profile_accepts_approved_grade_levels(grade_level: str) -> None:
    payload = _payload()
    payload["text_grade_level"] = grade_level

    assert UniversitySTEMProfile.model_validate(payload).text_grade_level == grade_level


@pytest.mark.parametrize("course", sorted(UNIVERSITY_COURSES))
def test_university_profile_accepts_approved_first_level_disciplines(course: str) -> None:
    payload = _payload()
    payload["text_course"] = course

    assert UniversitySTEMProfile.model_validate(payload).text_course == course


@pytest.mark.parametrize("invalid", ["小学", "初中", "高中", "未知"])
def test_university_profile_does_not_accept_k12_or_unknown_formal_grade(invalid: str) -> None:
    payload = _payload()
    payload["text_grade"] = invalid

    with pytest.raises(ValidationError):
        UniversitySTEMProfile.model_validate(payload)


@pytest.mark.parametrize("invalid", ["小学一年级", "初中三年级", "高中三年级", "未知"])
def test_university_profile_does_not_accept_k12_grade_levels(invalid: str) -> None:
    payload = _payload()
    payload["text_grade_level"] = invalid

    with pytest.raises(ValidationError):
        UniversitySTEMProfile.model_validate(payload)


@pytest.mark.parametrize("invalid", ["语文", "英语", "历史", "地理", "政治"])
def test_university_profile_does_not_silently_widen_k12_courses(invalid: str) -> None:
    payload = _payload()
    payload["text_course"] = invalid

    with pytest.raises(ValidationError):
        UniversitySTEMProfile.model_validate(payload)


@pytest.mark.parametrize("invalid", ["", " ", "略", "  略  "])
def test_university_profile_requires_real_analysis(invalid: str) -> None:
    payload = _payload()
    payload["answer_analysis"] = invalid

    with pytest.raises(ValidationError):
        UniversitySTEMProfile.model_validate(payload)


@pytest.mark.parametrize("invalid", ["高考", "中考", "小升初", "GRE"])
def test_university_course_profile_requires_unknown_entrance_exam_type(invalid: str) -> None:
    payload = _payload()
    payload["entrance_exam_type"] = invalid

    with pytest.raises(ValidationError):
        UniversitySTEMProfile.model_validate(payload)


@pytest.mark.parametrize(
    "static_info",
    [
        {},
        {"contract_version": "question_record_v1"},
        {"profile": "university_stem_v1"},
        {"contract_version": "question_record_v2", "profile": "university_stem_v1"},
        {"contract_version": "question_record_v1", "profile": "k12_v1"},
    ],
)
def test_university_profile_requires_static_profile_identity(static_info: dict[str, str]) -> None:
    payload = _payload()
    payload["static_info"] = json.dumps(static_info)

    with pytest.raises(ValidationError):
        UniversitySTEMProfile.model_validate(payload)


def test_profile_yaml_matches_mechanical_university_contract() -> None:
    profile_path = Path("config/profiles/university_stem_v1.yaml")
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))

    assert profile["profile"] == "university_stem_v1"
    assert profile["contract_version"] == "question_record_v1"
    assert set(profile["text_course"]["allowed"]) == UNIVERSITY_COURSES
    assert set(profile["text_grade_level"]["allowed"]) == UNIVERSITY_GRADE_LEVELS
    assert profile["text_grade"]["allowed"] == ["大学"]
    assert profile["entrance_exam_type"]["allowed"] == ["未知"]
    assert profile["answer_analysis"]["required"] is True
    assert "略" in profile["answer_analysis"]["reject_values"]
