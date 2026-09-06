"""Exact 19-field final question contract shared by export profiles."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

QuestionType = Literal["判断题", "选择题", "填空题", "问答题", "其他题型", "未知"]

_ISO_639_1_CODES: frozenset[str] = frozenset(
    """
    aa ab ae af ak am an ar as av ay az ba be bg bi
    bm bn bo br bs ca ce ch co cr cs cu cv cy da de
    dv dz ee el en eo es et eu fa ff fi fj fo fr fy
    ga gd gl gn gu gv ha he hi ho hr ht hu hy hz ia
    id ie ig ii ik io is it iu ja jv ka kg ki kj kk
    kl km kn ko kr ks ku kv kw ky la lb lg li ln lo
    lt lu lv mg mh mi mk ml mn mr ms mt my na nb nd
    ne ng nl nn no nr nv ny oc oj om or os pa pi pl
    ps pt qu rm rn ro ru rw sa sc sd se sg sh si sk
    sl sm sn so sq sr ss st su sv sw ta te tg th ti
    tk tl tn to tr ts tt tw ty ug uk ur uz ve vi vo
    wa wo xh yi yo za zh zu
    """.split()
)


def _canonicalize_question(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("text_question must be non-empty")
    normalized = unicodedata.normalize("NFC", text)
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.strip()
    return re.sub(r"\n(?:[ \t]*\n)+", "\n\n", normalized)


def slim_question_md5_v1(text: str) -> str:
    """Hash the approved loss-minimizing canonical representation of a question."""
    canonical = _canonicalize_question(text)
    return hashlib.md5(canonical.encode("utf-8"), usedforsecurity=False).hexdigest()


class FinalQuestionRecord(BaseModel):
    """Profile-neutral exact 19-field QuestionRecordV1 shape."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    text_question: str = Field(min_length=1)
    is_pic_included: Literal[0, 1]
    text_answer: str = Field(min_length=1)
    answer_analysis: str = Field(min_length=1)
    text_course: str
    text_grade_level: str
    text_grade: str
    knowledge_points: str = ""
    exam_points: str = ""
    publisher: str = ""
    text_paper: str = Field(min_length=1)
    textbook_version: str = ""
    static_info: str
    language: str
    text_year: str = ""
    entrance_exam_type: str
    text_city: str = ""
    question_type: QuestionType
    competition_event: str = ""

    @field_validator("is_pic_included", mode="before")
    @classmethod
    def validate_picture_flag(cls, value: Any) -> Any:
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise ValueError("is_pic_included must be integer 0 or 1")
        return value

    @field_validator("static_info", mode="before")
    @classmethod
    def validate_static_info(cls, value: Any) -> str:
        if not isinstance(value, str):
            raise ValueError("static_info must be a JSON object serialized as a string")
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("static_info must contain valid JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("static_info JSON must be an object")
        return value

    @field_validator("language")
    @classmethod
    def validate_language(cls, value: str) -> str:
        if value not in _ISO_639_1_CODES:
            raise ValueError("language must be a lowercase ISO 639-1 code")
        return value

    @field_validator("text_year")
    @classmethod
    def validate_text_year(cls, value: str) -> str:
        if value and re.fullmatch(r"[0-9]{4}", value) is None:
            raise ValueError("text_year must be empty or a four-digit year string")
        return value
