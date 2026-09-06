from __future__ import annotations

import json
import re

import pytest
from pydantic import ValidationError

from college_builder.domain.final_record import (
    FinalQuestionRecord,
    slim_question_md5_v1,
)

EXPECTED_FIELDS = {
    "text_question",
    "is_pic_included",
    "text_answer",
    "answer_analysis",
    "text_course",
    "text_grade_level",
    "text_grade",
    "knowledge_points",
    "exam_points",
    "publisher",
    "text_paper",
    "textbook_version",
    "static_info",
    "language",
    "text_year",
    "entrance_exam_type",
    "text_city",
    "question_type",
    "competition_event",
}


def _record_payload() -> dict[str, object]:
    return {
        "text_question": "Compute $1+1$.",
        "is_pic_included": 0,
        "text_answer": "2",
        "answer_analysis": "Adding one and one gives two.",
        "text_course": "数学",
        "text_grade_level": "本科",
        "text_grade": "大学",
        "knowledge_points": "基础运算",
        "exam_points": "计算",
        "publisher": "",
        "text_paper": "Calculus I",
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
        "text_year": "2026",
        "entrance_exam_type": "未知",
        "text_city": "",
        "question_type": "问答题",
        "competition_event": "",
    }


def test_final_record_has_exact_19_fields() -> None:
    assert set(FinalQuestionRecord.model_fields) == EXPECTED_FIELDS
    assert len(FinalQuestionRecord.model_fields) == 19


def test_final_record_rejects_extra_fields() -> None:
    payload = _record_payload()
    payload["unexpected"] = "must fail closed"

    with pytest.raises(ValidationError):
        FinalQuestionRecord.model_validate(payload)


@pytest.mark.parametrize("invalid", [True, False, -1, 2, "0", "1", 0.0, 1.0])
def test_picture_flag_accepts_only_integer_zero_or_one(invalid: object) -> None:
    payload = _record_payload()
    payload["is_pic_included"] = invalid

    with pytest.raises(ValidationError):
        FinalQuestionRecord.model_validate(payload)


@pytest.mark.parametrize("valid", [0, 1])
def test_picture_flag_accepts_integer_zero_or_one(valid: int) -> None:
    payload = _record_payload()
    payload["is_pic_included"] = valid

    assert FinalQuestionRecord.model_validate(payload).is_pic_included == valid


@pytest.mark.parametrize(
    "invalid",
    ["not-json", "[]", "null", "42", 42, ["not", "a", "string"]],
)
def test_static_info_must_be_a_serialized_json_object(invalid: object) -> None:
    payload = _record_payload()
    payload["static_info"] = invalid

    with pytest.raises(ValidationError):
        FinalQuestionRecord.model_validate(payload)


@pytest.mark.parametrize("valid", ["en", "zh", "de", "fr"])
def test_language_accepts_lowercase_iso_639_1_codes(valid: str) -> None:
    payload = _record_payload()
    payload["language"] = valid

    assert FinalQuestionRecord.model_validate(payload).language == valid


@pytest.mark.parametrize("invalid", ["EN", "eng", "e", "12", "xx"])
def test_language_rejects_non_iso_639_1_codes(invalid: str) -> None:
    payload = _record_payload()
    payload["language"] = invalid

    with pytest.raises(ValidationError):
        FinalQuestionRecord.model_validate(payload)


def test_slim_question_md5_v1_uses_nfc_line_endings_trim_and_blank_line_collapse() -> None:
    decomposed = "  Cafe\u0301\r\n\r\n\r\n$x^2$\r<img src=\"image/a.png\">  "
    composed = "Café\n\n$x^2$\n<img src=\"image/a.png\">"

    digest = slim_question_md5_v1(decomposed)

    assert digest == slim_question_md5_v1(composed)
    assert re.fullmatch(r"[0-9a-f]{32}", digest)


def test_slim_question_md5_v1_preserves_semantic_content() -> None:
    baseline = 'Problem 2: $x^2$?\n\n<img src="image/a.png">'

    variants = (
        'Problem 3: $x^2$?\n\n<img src="image/a.png">',
        'Problem 2: $x^3$?\n\n<img src="image/a.png">',
        'Problem 2: $x^2$!\n\n<img src="image/a.png">',
        'Problem 2: $x^2$?\n\n<img src="image/b.png">',
    )

    baseline_digest = slim_question_md5_v1(baseline)
    assert all(slim_question_md5_v1(variant) != baseline_digest for variant in variants)


def test_slim_question_md5_v1_rejects_empty_question() -> None:
    with pytest.raises(ValueError):
        slim_question_md5_v1(" \r\n\t ")
