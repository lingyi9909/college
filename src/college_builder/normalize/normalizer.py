"""RawSourceRecord to NormalizedQA deterministic normalization entrypoint."""

from __future__ import annotations

import hashlib
import json

from college_builder.domain.source import JsonLike, NormalizedQA, RawSourceRecord
from college_builder.normalize.html import HtmlNormalizationResult, normalize_html

NORMALIZATION_VERSION = "normalization_v1"


def normalize(record: RawSourceRecord) -> NormalizedQA:
    """Normalize formatting only; never infer, rewrite, classify, or score content."""
    question = normalize_html(record.raw_question)
    answer = normalize_html(record.raw_answer)
    analysis = normalize_html(record.raw_analysis)

    input_hashes = _content_hashes(
        record.raw_question,
        record.raw_answer,
        record.raw_analysis,
    )
    output_hashes = _content_hashes(question.text, answer.text, analysis.text)
    unresolved = _unresolved_evidence(question, answer, analysis)
    transformations: dict[str, JsonLike] = {
        "question": list(question.transformations),
        "answer": list(answer.transformations),
        "analysis": list(analysis.transformations),
    }
    evidence: dict[str, JsonLike] = {
        "version": NORMALIZATION_VERSION,
        "source_record_id": record.record_id,
        "source_raw_sha256": record.raw_sha256,
        "input_hashes": input_hashes,
        "output_hashes": output_hashes,
        "transformations": transformations,
        "unresolved": unresolved,
    }

    return NormalizedQA.model_validate(
        {
            "record_id": _normalized_record_id(record.record_id, output_hashes["content"]),
            "source_record_id": record.record_id,
            "question": question.text,
            "answer": answer.text,
            "analysis": analysis.text,
            "subject_candidates": (),
            "images": _merge_images(question, answer, analysis),
            "metadata": record.metadata,
            "normalization_evidence": evidence,
        }
    )


def _content_hashes(question: str, answer: str, analysis: str) -> dict[str, JsonLike]:
    return {
        "question": _sha256(question),
        "answer": _sha256(answer),
        "analysis": _sha256(analysis),
        "content": _sha256("\0".join((question, answer, analysis))),
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _normalized_record_id(source_record_id: str, content_hash: JsonLike) -> str:
    identity = json.dumps(
        {
            "normalization_version": NORMALIZATION_VERSION,
            "source_record_id": source_record_id,
            "content_hash": content_hash,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"normalized_{hashlib.sha256(identity.encode()).hexdigest()}"


def _merge_images(*results: HtmlNormalizationResult) -> tuple[str, ...]:
    seen: set[str] = set()
    images: list[str] = []
    for result in results:
        for image in result.images:
            if image in seen:
                continue
            seen.add(image)
            images.append(image)
    return tuple(images)


def _unresolved_evidence(
    question: HtmlNormalizationResult,
    answer: HtmlNormalizationResult,
    analysis: HtmlNormalizationResult,
) -> list[JsonLike]:
    unresolved: list[JsonLike] = []
    for field_name, result in (
        ("question", question),
        ("answer", answer),
        ("analysis", analysis),
    ):
        for math_item in result.unresolved_math:
            unresolved.append(
                {
                    "field": field_name,
                    "code": math_item["code"],
                    "reason": math_item["reason"],
                    "source_mathml": math_item["source_mathml"],
                }
            )
        for structure_item in result.unresolved_structure:
            unresolved.append(
                {
                    "field": field_name,
                    "code": structure_item["code"],
                    "reason": structure_item["reason"],
                    "source_markup": structure_item["source_markup"],
                }
            )
    return unresolved
