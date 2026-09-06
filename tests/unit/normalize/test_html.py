from __future__ import annotations

import hashlib
import json
from pathlib import Path

from college_builder.domain.source import RawSourceRecord
from college_builder.normalize.html import normalize_html
from college_builder.normalize.normalizer import normalize

GOLDEN_CASES = Path("tests/golden/normalization_cases.json")


def _raw_record(*, question: str, answer: str, analysis: str) -> RawSourceRecord:
    return RawSourceRecord.model_validate(
        {
            "record_id": "raw_gold_task7_fixture",
            "source_type": "dataset",
            "source_dataset": "stemq",
            "source_id": "fixture-7",
            "source_url": "https://example.test/stemq/fixture-7",
            "raw_question": question,
            "raw_answer": answer,
            "raw_analysis": analysis,
            "raw_payload": {
                "question": question,
                "answer": answer,
                "analysis": analysis,
            },
            "metadata": {
                "gold_role": "CALIBRATION_GOLD",
                "calibration_holdout": True,
                "quality_tier_candidate": "GOLD",
                "institution": "Example University",
            },
            "license_metadata": {"declared": "CC-BY-4.0"},
            "raw_sha256": "a" * 64,
        }
    )


def test_html_normalization_matches_golden_cases() -> None:
    cases = json.loads(GOLDEN_CASES.read_text(encoding="utf-8"))

    for case in cases:
        result = normalize_html(case["input"])
        assert result.text == case["expected"], case["name"]
        assert list(result.images) == case["images"], case["name"]


def test_normalize_preserves_raw_duality_gold_role_and_hash_evidence() -> None:
    raw = _raw_record(
        question="<p>Solve 2x+3=7.</p>",
        answer="B",
        analysis="<p>Keep $x^2$ and the source expression 2x+3=7 unchanged.</p>",
    )
    before = raw.model_dump(mode="json")

    normalized = normalize(raw)
    dumped = normalized.model_dump(mode="json")

    assert normalized.source_record_id == raw.record_id
    assert normalized.question == "Solve 2x+3=7."
    assert normalized.answer == "B"
    assert normalized.analysis == "Keep $x^2$ and the source expression 2x+3=7 unchanged."
    assert "x=2" not in normalized.question
    assert "x=2" not in normalized.analysis
    assert "$x^2$" in normalized.analysis
    assert normalized.metadata["gold_role"] == "CALIBRATION_GOLD"
    assert normalized.metadata["calibration_holdout"] is True
    assert normalized.metadata["quality_tier_candidate"] == "GOLD"
    assert raw.model_dump(mode="json") == before

    evidence = dumped["normalization_evidence"]
    assert evidence["version"] == "normalization_v1"
    assert evidence["source_record_id"] == raw.record_id
    assert evidence["input_hashes"]["question"] == hashlib.sha256(
        raw.raw_question.encode()
    ).hexdigest()
    assert evidence["output_hashes"]["question"] == hashlib.sha256(
        normalized.question.encode()
    ).hexdigest()
    assert "content" in evidence["input_hashes"]
    assert "content" in evidence["output_hashes"]
    assert evidence["unresolved"] == []


def test_answer_option_is_not_expanded_or_inferred() -> None:
    raw = _raw_record(
        question="<p>Choose the correct option.</p>",
        answer="B",
        analysis="<p>The source solution says option B.</p>",
    )

    normalized = normalize(raw)

    assert normalized.answer == "B"
    assert normalized.answer not in {"Option B", "B. inferred answer"}
    assert normalized.analysis == "The source solution says option B."


def test_images_are_structurally_preserved_without_classification() -> None:
    raw = _raw_record(
        question=(
            '<p>Inspect <img src="https://example.test/q.png" alt="source diagram"></p>'
        ),
        answer="B",
        analysis=(
            '<p>See <img src="https://example.test/q.png" alt="same diagram"> and '
            '<img src="https://example.test/a.png" alt="analysis diagram"></p>'
        ),
    )

    normalized = normalize(raw)

    assert normalized.images == (
        "https://example.test/q.png",
        "https://example.test/a.png",
    )
    assert normalized.subject_candidates == ()
