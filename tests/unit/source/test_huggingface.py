from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from college_builder.source.huggingface import HFDatasetAdapter

FIXTURE = Path("tests/fixtures/sources/hf_sample.jsonl")


def _config(*, revision: str = "abc123") -> dict[str, object]:
    return {
        "dataset": "demo/university-stem",
        "data_files": [str(FIXTURE)],
        "revision": revision,
        "field_mapping": {
            "source_id": "qid",
            "question": "prompt",
            "answer": "solution",
            "analysis": "explanation",
            "source_url": "original_url",
        },
        "license_metadata": {
            "declared": "CC-BY-4.0",
            "source": "fixture-card",
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


def test_hf_mapping_is_explicit_lossless_and_does_not_rewrite_content() -> None:
    adapter = HFDatasetAdapter()
    descriptor = tuple(adapter.discover(_config()))[0]

    records = list(adapter.acquire(descriptor))

    assert len(records) == 3
    first = records[0]
    dumped = first.model_dump(mode="json")
    assert first.source_dataset == "demo/university-stem"
    assert first.source_id == "101"
    assert first.source_url == "https://example.test/questions/101"
    assert first.raw_question == "  Compute 1 + 1.  "
    assert first.raw_answer == "2"
    assert first.raw_analysis == "Because one plus one equals two."
    assert dumped["raw_payload"]["unused_field"] == {
        "difficulty": "easy",
        "labels": ["arithmetic", "warmup"],
    }
    assert dumped["license_metadata"] == {
        "declared": "CC-BY-4.0",
        "source": "fixture-card",
    }
    acquisition = dumped["metadata"]["acquisition"]
    assert acquisition["adapter"] == "huggingface"
    assert acquisition["row_offset"] == 0
    assert acquisition["source_revision"] == "abc123"
    assert acquisition["shard"] == str(FIXTURE)
    assert first.raw_sha256 == _canonical_sha256(dumped["raw_payload"])
    assert "university_level" not in dumped["metadata"]
    assert "quality" not in dumped["metadata"]


def test_hf_records_are_stable_across_two_runs() -> None:
    first = HFDatasetAdapter()
    second = HFDatasetAdapter()
    first_descriptor = tuple(first.discover(_config()))[0]
    second_descriptor = tuple(second.discover(_config()))[0]

    first_records = list(first.acquire(first_descriptor))
    second_records = list(second.acquire(second_descriptor))

    assert [item.record_id for item in first_records] == [
        item.record_id for item in second_records
    ]
    assert [item.raw_sha256 for item in first_records] == [
        item.raw_sha256 for item in second_records
    ]
    assert [item.model_dump(mode="json") for item in first_records] == [
        item.model_dump(mode="json") for item in second_records
    ]


def test_hf_checkpoint_resume_skips_already_persisted_record_ids() -> None:
    adapter = HFDatasetAdapter()
    descriptor = tuple(adapter.discover(_config()))[0]
    iterator = iter(adapter.acquire(descriptor))

    first = next(iterator)
    assert adapter.checkpoint().next_row_offset == 1
    second = next(iterator)
    checkpoint = adapter.checkpoint()

    assert checkpoint.descriptor == descriptor
    assert checkpoint.next_row_offset == 2
    assert checkpoint.next_shard == str(FIXTURE)
    assert checkpoint.source_revision == "abc123"

    resumed = HFDatasetAdapter(resume_from=checkpoint)
    resumed_descriptors = tuple(resumed.discover(_config()))
    remaining = list(resumed.acquire(resumed_descriptors[0]))

    persisted_ids = {first.record_id, second.record_id}
    resumed_ids = {item.record_id for item in remaining}
    assert [item.source_id for item in remaining] == ["103"]
    assert persisted_ids.isdisjoint(resumed_ids)
    assert resumed.checkpoint().next_row_offset == 3
    assert resumed.checkpoint().next_shard is None


def test_hf_resume_fails_closed_when_source_revision_changes() -> None:
    adapter = HFDatasetAdapter()
    descriptor = tuple(adapter.discover(_config()))[0]
    iterator = iter(adapter.acquire(descriptor))
    next(iterator)
    checkpoint = adapter.checkpoint()

    resumed = HFDatasetAdapter(resume_from=checkpoint)

    with pytest.raises(ValueError, match="checkpoint.*revision"):
        tuple(resumed.discover(_config(revision="changed-revision")))


def test_hf_missing_explicitly_mapped_field_fails_closed() -> None:
    config = _config()
    mapping = dict(config["field_mapping"])  # type: ignore[arg-type]
    mapping["answer"] = "missing_answer_field"
    config["field_mapping"] = mapping
    adapter = HFDatasetAdapter()
    descriptor = tuple(adapter.discover(config))[0]

    with pytest.raises(ValueError, match="missing_answer_field"):
        list(adapter.acquire(descriptor))


def test_hf_nonlocal_data_file_resolves_to_huggingface_revision_url() -> None:
    config = _config()
    config["data_files"] = ["data/train.jsonl"]
    adapter = HFDatasetAdapter()

    descriptor = tuple(adapter.discover(config))[0]

    assert descriptor.source_url == "https://huggingface.co/datasets/demo/university-stem"
    assert descriptor.locator == (
        "https://huggingface.co/datasets/demo/university-stem/resolve/abc123/data/train.jsonl"
    )
    assert descriptor.source_revision == "abc123"
