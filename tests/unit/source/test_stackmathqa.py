from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from college_builder.domain.source import RawSourceRecord
from college_builder.source.base import SourceAdapter
from college_builder.source.stackmathqa import StackMathQAAdapter, stratified_sample

GOLDEN_DIR = Path("tests/golden/stackmathqa")
SITE_CASES = (
    ("math.stackexchange.com.jsonl", "math", "1001", 2001),
    ("mathoverflow.net.jsonl", "mathoverflow", "1002", 3001),
    ("stats.stackexchange.com.jsonl", "statistics", "1003", 4001),
    ("physics.stackexchange.com.jsonl", "physics", "1004", 5001),
)


def _config(data_file: Path) -> dict[str, object]:
    return {
        "data_file": str(data_file),
        "revision": "stackmathqafull-1q1a-fixture-v1",
        "dataset_url": "https://huggingface.co/datasets/math-ai/StackMathQA",
        "license_metadata": {
            "declared": "CC-BY-4.0",
            "status": "UNREVIEWED",
        },
    }


def _jsonl_rows(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


@pytest.mark.parametrize(("filename", "site", "question_id", "answer_id"), SITE_CASES)
def test_official_stackmathqa_source_files_map_q_a_meta_losslessly(
    filename: str,
    site: str,
    question_id: str,
    answer_id: int,
) -> None:
    path = GOLDEN_DIR / filename
    adapter = StackMathQAAdapter()
    assert isinstance(adapter, SourceAdapter)

    descriptor = tuple(adapter.discover(_config(path)))[0]
    record = list(adapter.acquire(descriptor))[0]
    official_row = _jsonl_rows(path)[0]
    meta = official_row["meta"]
    assert isinstance(meta, dict)

    dumped = record.model_dump(mode="json")
    assert record.source_id == f"{site}:{question_id}:{answer_id}"
    assert record.source_url == meta["url"]
    assert record.raw_question == official_row["Q"]
    assert record.raw_answer == ""
    assert record.raw_analysis == official_row["A"]
    assert dumped["raw_payload"] == official_row
    assert dumped["metadata"]["source_site"] == site
    for key, value in meta.items():
        assert dumped["metadata"][key] == value
    assert dumped["license_metadata"] == {
        "declared": "CC-BY-4.0",
        "status": "UNREVIEWED",
    }
    assert record.raw_sha256 == _canonical_sha256(official_row)
    assert "university_level" not in dumped["metadata"]
    assert "course" not in dumped["metadata"]
    assert "quality" not in dumped["metadata"]


def test_official_jsonl_loader_reads_multiple_lines() -> None:
    path = GOLDEN_DIR / "math.stackexchange.com.jsonl"
    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    records = list(adapter.acquire(descriptor))

    assert [record.source_id for record in records] == [
        "math:1001:2001",
        "math:1001:2002",
    ]
    assert adapter.checkpoint().next_row_offset == 2
    assert adapter.checkpoint().source_revision == "stackmathqafull-1q1a-fixture-v1"


def test_official_q_a_meta_remain_raw_provenance_and_long_a_is_analysis() -> None:
    path = GOLDEN_DIR / "physics.stackexchange.com.jsonl"
    official_row = _jsonl_rows(path)[0]
    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    record = list(adapter.acquire(descriptor))[0]
    dumped = record.model_dump(mode="json")

    assert dumped["raw_payload"] == official_row
    assert record.raw_question == official_row["Q"]
    assert record.raw_analysis == official_row["A"]
    assert record.raw_answer == ""


def test_source_site_is_derived_from_official_filename_not_meta_labels(tmp_path: Path) -> None:
    path = tmp_path / "physics.stackexchange.com.jsonl"
    row = _jsonl_rows(GOLDEN_DIR / "physics.stackexchange.com.jsonl")[0]
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta["tags"] = ["mathematics", "algebra"]
    meta["course"] = "MATHEMATICS"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]
    record = list(adapter.acquire(descriptor))[0]

    assert record.metadata["source_site"] == "physics"


def test_source_site_meta_url_mismatch_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "physics.stackexchange.com.jsonl"
    row = _jsonl_rows(GOLDEN_DIR / "physics.stackexchange.com.jsonl")[0]
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta["url"] = "https://math.stackexchange.com/questions/1004/wrong-site"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    with pytest.raises(ValueError, match=r"source site.*meta\.url"):
        list(adapter.acquire(descriptor))


def test_missing_official_answer_identity_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "math.stackexchange.com.jsonl"
    row = _jsonl_rows(GOLDEN_DIR / "math.stackexchange.com.jsonl")[0]
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta.pop("answer_id")
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    with pytest.raises(ValueError, match=r"answer_id"):
        list(adapter.acquire(descriptor))


def test_answer_id_zero_generates_stable_source_and_record_identity() -> None:
    path = GOLDEN_DIR / "mathoverflow.net.jsonl"
    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    zero_record = list(adapter.acquire(descriptor))[1]
    expected_source_id = "mathoverflow:1002:0"
    expected_digest = hashlib.sha256(
        f"stackmathqa\0{expected_source_id}".encode()
    ).hexdigest()

    assert zero_record.source_id == expected_source_id
    assert zero_record.record_id == f"raw_stackmathqa_{expected_digest}"
    assert zero_record.metadata["answer_id"] == 0


def test_negative_answer_id_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "mathoverflow.net.jsonl"
    row = _jsonl_rows(GOLDEN_DIR / "mathoverflow.net.jsonl")[1]
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta["answer_id"] = -1
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    with pytest.raises(ValueError, match=r"answer_id"):
        list(adapter.acquire(descriptor))


@pytest.mark.parametrize("invalid_answer_id", [True, "0", 0.0])
def test_non_integer_answer_id_fails_closed(
    tmp_path: Path, invalid_answer_id: object
) -> None:
    path = tmp_path / "mathoverflow.net.jsonl"
    row = _jsonl_rows(GOLDEN_DIR / "mathoverflow.net.jsonl")[1]
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta["answer_id"] = invalid_answer_id
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    with pytest.raises(ValueError, match=r"answer_id"):
        list(adapter.acquire(descriptor))


def test_unparseable_question_identity_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "math.stackexchange.com.jsonl"
    row = _jsonl_rows(GOLDEN_DIR / "math.stackexchange.com.jsonl")[0]
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta["url"] = "https://math.stackexchange.com/users/1001/not-a-question"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    with pytest.raises(ValueError, match=r"question identity"):
        list(adapter.acquire(descriptor))


def test_unrecognized_source_filename_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unknown.stackexchange.com.jsonl"
    row = _jsonl_rows(GOLDEN_DIR / "math.stackexchange.com.jsonl")[0]
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    adapter = StackMathQAAdapter()

    with pytest.raises(ValueError, match=r"official StackMathQA source file"):
        tuple(adapter.discover(_config(path)))


def test_stackmathqa_mapping_and_identity_are_stable_across_runs() -> None:
    path = GOLDEN_DIR / "math.stackexchange.com.jsonl"
    first_adapter = StackMathQAAdapter()
    second_adapter = StackMathQAAdapter()
    first_descriptor = tuple(first_adapter.discover(_config(path)))[0]
    second_descriptor = tuple(second_adapter.discover(_config(path)))[0]

    first = list(first_adapter.acquire(first_descriptor))
    second = list(second_adapter.acquire(second_descriptor))

    assert [record.model_dump(mode="json") for record in first] == [
        record.model_dump(mode="json") for record in second
    ]


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
