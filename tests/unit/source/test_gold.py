from __future__ import annotations

import json
from pathlib import Path

import pytest

from college_builder.source.base import SourceAdapter
from college_builder.source.gold import GoldDatasetAdapter, GoldRole


@pytest.mark.parametrize(
    ("dataset_name", "gold_role", "expected_holdout"),
    [
        ("stemq", "CALIBRATION_GOLD", True),
        ("scibench", "TRUSTED_TRAINING", False),
        ("cfe", "HIGH_CONFIDENCE", False),
    ],
)
def test_gold_dataset_preserves_source_identity_content_and_role(
    tmp_path: Path,
    dataset_name: str,
    gold_role: str,
    expected_holdout: bool,
) -> None:
    row = {
        "id": f"{dataset_name}-record-7",
        "question": f"Original {dataset_name} question?",
        "answer": "source final answer",
        "solution": "Original source solution with all reasoning intact.",
        "url": f"https://example.test/{dataset_name}/record-7",
        "institution": "Example University",
        "textbook": "Example STEM Textbook",
        "extra_provenance": {"split": "train", "source_version": 3},
    }
    data_file = tmp_path / f"{dataset_name}.jsonl"
    data_file.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    config = {
        "dataset_name": dataset_name,
        "data_file": str(data_file),
        "source_url": f"https://example.test/{dataset_name}",
        "revision": "fixture-v1",
        "gold_role": gold_role,
        "field_mapping": {
            "source_id": "id",
            "question": "question",
            "answer": "answer",
            "solution": "solution",
            "source_url": "url",
            "institution": "institution",
            "textbook": "textbook",
        },
        "license_metadata": {
            "declared": "CC-BY-4.0",
            "status": "UNREVIEWED",
        },
    }

    adapter = GoldDatasetAdapter()
    assert isinstance(adapter, SourceAdapter)
    descriptor = tuple(adapter.discover(config))[0]
    record = list(adapter.acquire(descriptor))[0]
    dumped = record.model_dump(mode="json")

    assert descriptor.source_dataset == dataset_name
    assert record.source_dataset == dataset_name
    assert record.source_id == row["id"]
    assert record.source_url == row["url"]
    assert record.raw_question == row["question"]
    assert record.raw_answer == row["answer"]
    assert record.raw_analysis == row["solution"]
    assert dumped["raw_payload"] == row
    assert dumped["metadata"]["institution"] == row["institution"]
    assert dumped["metadata"]["textbook"] == row["textbook"]
    assert dumped["metadata"]["gold_role"] == gold_role
    assert dumped["metadata"]["calibration_holdout"] is expected_holdout
    assert dumped["metadata"]["quality_tier_candidate"] == "GOLD"
    assert "accepted" not in dumped["metadata"]
    assert "final_verdict" not in dumped["metadata"]
    assert dumped["license_metadata"] == config["license_metadata"]

    acquisition = dumped["metadata"]["acquisition"]
    assert acquisition["adapter"] == "gold_dataset"
    assert acquisition["source_revision"] == "fixture-v1"
    assert acquisition["source_descriptor"]["source_dataset"] == dataset_name


def test_calibration_gold_is_machine_readable_holdout_not_training_candidate(
    tmp_path: Path,
) -> None:
    row = {
        "id": "holdout-1",
        "question": "Original calibration question?",
        "answer": "A",
        "solution": "Original calibration solution.",
    }
    data_file = tmp_path / "calibration.jsonl"
    data_file.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = {
        "dataset_name": "stemq",
        "data_file": str(data_file),
        "source_url": "https://example.test/stemq",
        "gold_role": GoldRole.CALIBRATION_GOLD.value,
        "field_mapping": {
            "source_id": "id",
            "question": "question",
            "answer": "answer",
            "solution": "solution",
        },
        "license_metadata": {"declared": "CC-BY-4.0"},
    }

    adapter = GoldDatasetAdapter()
    descriptor = tuple(adapter.discover(config))[0]
    record = list(adapter.acquire(descriptor))[0]

    assert record.metadata["gold_role"] == "CALIBRATION_GOLD"
    assert record.metadata["calibration_holdout"] is True
    assert record.metadata["quality_tier_candidate"] == "GOLD"


def test_gold_record_identity_is_stable_and_resume_skips_persisted_rows(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "id": "gold-1",
            "question": "Q1",
            "answer": "A1",
            "solution": "S1",
        },
        {
            "id": "gold-2",
            "question": "Q2",
            "answer": "A2",
            "solution": "S2",
        },
    ]
    data_file = tmp_path / "gold.jsonl"
    data_file.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    config = {
        "dataset_name": "scibench",
        "data_file": str(data_file),
        "source_url": "https://example.test/scibench",
        "revision": "fixture-v2",
        "gold_role": "TRUSTED_TRAINING",
        "field_mapping": {
            "source_id": "id",
            "question": "question",
            "answer": "answer",
            "solution": "solution",
        },
        "license_metadata": {"declared": "CC-BY-4.0"},
    }

    first_adapter = GoldDatasetAdapter()
    first_descriptor = tuple(first_adapter.discover(config))[0]
    iterator = iter(first_adapter.acquire(first_descriptor))
    first = next(iterator)
    checkpoint = first_adapter.checkpoint()
    assert checkpoint.next_row_offset == 1

    resumed_adapter = GoldDatasetAdapter(resume_from=checkpoint)
    resumed_descriptor = tuple(resumed_adapter.discover(config))[0]
    resumed = list(resumed_adapter.acquire(resumed_descriptor))

    clean_adapter = GoldDatasetAdapter()
    clean_descriptor = tuple(clean_adapter.discover(config))[0]
    clean = list(clean_adapter.acquire(clean_descriptor))

    assert [record.source_id for record in resumed] == ["gold-2"]
    assert first.record_id == clean[0].record_id
    assert resumed[0].record_id == clean[1].record_id
