from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from college_builder.domain.source import RawSourceRecord
from college_builder.source.base import SourceAdapter
from college_builder.source.stackmathqa import StackMathQAAdapter, stratified_sample

GOLDEN = Path("tests/golden/stackmathqa_mapping.json")


def _config() -> dict[str, object]:
    return {
        "data_file": str(GOLDEN),
        "revision": "stackmathqa-fixture-v1",
        "dataset_url": "https://huggingface.co/datasets/math-ai/StackMathQA",
        "license_metadata": {
            "declared": "CC-BY-SA",
            "status": "UNREVIEWED",
        },
    }


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sample_record(site: str, index: int) -> RawSourceRecord:
    payload = {
        "site": site,
        "question_id": index,
        "answer_id": index + 100_000,
        "question_body": f"question-{site}-{index}",
        "answer_body": f"analysis-{site}-{index}",
    }
    return RawSourceRecord.model_validate(
        {
            "record_id": f"raw_stackmathqa_{site}_{index}",
            "source_type": "dataset",
            "source_dataset": "stackmathqa",
            "source_id": f"{site}:{index}:{index + 100_000}",
            "source_url": f"https://example.test/{site}/{index}",
            "raw_question": payload["question_body"],
            "raw_answer": "",
            "raw_analysis": payload["answer_body"],
            "raw_payload": payload,
            "metadata": {"source_site": site},
            "license_metadata": {"declared": "UNREVIEWED"},
            "raw_sha256": _canonical_sha256(payload),
        }
    )


def test_stackmathqa_adapter_maps_four_approved_source_sites_losslessly() -> None:
    adapter = StackMathQAAdapter()
    assert isinstance(adapter, SourceAdapter)

    descriptors = tuple(adapter.discover(_config()))
    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor.source_dataset == "stackmathqa"
    assert descriptor.source_revision == "stackmathqa-fixture-v1"

    records = list(adapter.acquire(descriptor))
    assert [record.metadata["source_site"] for record in records] == [
        "math",
        "mathoverflow",
        "statistics",
        "physics",
    ]

    golden_rows = json.loads(GOLDEN.read_text(encoding="utf-8"))
    first = records[0]
    dumped = first.model_dump(mode="json")
    assert first.source_id == "math:1001:2001"
    assert first.source_url == "https://math.stackexchange.com/questions/1001"
    assert first.raw_question == "  Let $A$ be a 2x2 matrix. Find its determinant.  "
    assert first.raw_answer == ""
    assert first.raw_analysis == golden_rows[0]["answer_body"]
    assert dumped["raw_payload"] == golden_rows[0]
    assert dumped["metadata"]["tags"] == ["linear-algebra", "matrices"]
    assert dumped["metadata"]["question_score"] == 18
    assert dumped["metadata"]["answer_score"] == 27
    assert dumped["metadata"]["accepted_answer"] is True
    assert dumped["metadata"]["question_created_at"] == "2020-01-02T03:04:05Z"
    assert dumped["metadata"]["answer_created_at"] == "2020-01-02T04:05:06Z"
    assert dumped["license_metadata"] == {
        "declared": "CC-BY-SA",
        "status": "UNREVIEWED",
    }
    assert first.raw_sha256 == _canonical_sha256(golden_rows[0])
    assert "university_level" not in dumped["metadata"]
    assert "course" not in dumped["metadata"]
    assert "quality" not in dumped["metadata"]


def test_stackmathqa_long_answer_remains_source_analysis_for_task10() -> None:
    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config()))[0]

    records = list(adapter.acquire(descriptor))

    for record in records:
        source_answer_body = record.model_dump(mode="json")["raw_payload"]["answer_body"]
        assert record.raw_answer == ""
        assert record.raw_analysis == source_answer_body
        assert record.raw_analysis


def test_stackmathqa_mapping_and_identity_are_stable_across_runs() -> None:
    first_adapter = StackMathQAAdapter()
    second_adapter = StackMathQAAdapter()
    first_descriptor = tuple(first_adapter.discover(_config()))[0]
    second_descriptor = tuple(second_adapter.discover(_config()))[0]

    first = list(first_adapter.acquire(first_descriptor))
    second = list(second_adapter.acquire(second_descriptor))

    assert [record.model_dump(mode="json") for record in first] == [
        record.model_dump(mode="json") for record in second
    ]
    assert first_adapter.checkpoint().next_row_offset == 4
    assert first_adapter.checkpoint().source_revision == "stackmathqa-fixture-v1"


def test_stratified_sample_returns_exact_source_site_quotas_deterministically() -> None:
    records = [
        *[_sample_record("math", index) for index in range(40)],
        *[_sample_record("mathoverflow", index) for index in range(20)],
        *[_sample_record("statistics", index) for index in range(25)],
        *[_sample_record("physics", index) for index in range(25)],
    ]
    quotas = {"math": 20, "mathoverflow": 5, "statistics": 10, "physics": 10}

    first = stratified_sample(records, quotas, seed=20260906)
    second = stratified_sample(list(reversed(records)), quotas, seed=20260906)

    assert [record.record_id for record in first] == [record.record_id for record in second]
    counts = Counter(str(record.metadata["source_site"]) for record in first)
    assert counts == quotas
    assert len(first) == 45


def test_stratified_sample_fails_closed_when_requested_stratum_is_short() -> None:
    records = [
        *[_sample_record("math", index) for index in range(19)],
        *[_sample_record("mathoverflow", index) for index in range(5)],
        *[_sample_record("statistics", index) for index in range(10)],
        *[_sample_record("physics", index) for index in range(10)],
    ]
    quotas = {"math": 20, "mathoverflow": 5, "statistics": 10, "physics": 10}

    with pytest.raises(ValueError, match=r"math.*requested 20.*available 19"):
        stratified_sample(records, quotas, seed=7)


def test_stratified_sample_uses_source_site_not_tags_or_inferred_course() -> None:
    physics = _sample_record("physics", 1)
    misleading = physics.model_copy(
        update={
            "metadata": {
                "source_site": "physics",
                "tags": ["mathematics", "algebra"],
                "course": "MATHEMATICS",
            }
        }
    )

    sample = stratified_sample([misleading], {"physics": 1}, seed=1)

    assert sample == [misleading]
