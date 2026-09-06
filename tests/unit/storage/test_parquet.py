from __future__ import annotations

import json

import duckdb
import pytest

from college_builder.domain.source import RawSourceRecord
from college_builder.storage import parquet as parquet_module
from college_builder.storage.parquet import ParquetStageStore


def _record(index: int) -> RawSourceRecord:
    return RawSourceRecord(
        record_id=f"raw-demo-{index}",
        source_type="dataset",
        source_dataset="demo",
        source_id=str(index),
        source_url=f"https://example.test/{index}",
        raw_question=f"Question {index}\n$x^{index}$",
        raw_answer=f"Answer {index}",
        raw_analysis=f"Analysis {index}",
        raw_payload={"z": index, "a": {"nested": [index, {"b": 2, "a": 1}]}},
        metadata={"tags": ["math", f"tag-{index}"], "score": index},
        license_metadata={"status": "UNREVIEWED", "name": f"license-{index}"},
        raw_sha256=f"{index:064x}",
    )


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_record_ids(partition) -> list[str]:
    connection = duckdb.connect()
    try:
        rows = (
            connection.read_parquet(str(partition / "*.parquet"))
            .project("record_id")
            .order("record_id")
            .fetchall()
        )
    finally:
        connection.close()
    return [str(row[0]) for row in rows]


def test_raw_source_records_round_trip_through_partitioned_parquet(tmp_path) -> None:
    records = [_record(1), _record(2), _record(3)]
    store = ParquetStageStore(tmp_path / "parquet")

    path = store.write_raw_records(
        "run-1", "ACQUIRED", "demo", "batch-001", records
    )

    assert path.name == "part-batch-001.parquet"
    assert path.parent.name == "source_dataset=demo"
    assert path.parent.parent.name == "stage=ACQUIRED"
    assert path.parent.parent.parent.name == "run_id=run-1"
    assert not list(path.parent.glob("*.tmp"))

    connection = duckdb.connect()
    try:
        rows = (
            connection.read_parquet(str(path))
            .project(
                "record_id, raw_question, raw_answer, raw_analysis, "
                "raw_payload, metadata, license_metadata, raw_sha256"
            )
            .order("record_id")
            .fetchall()
        )
    finally:
        connection.close()

    expected = []
    for record in records:
        payload = record.model_dump(mode="json")
        expected.append(
            (
                record.record_id,
                record.raw_question,
                record.raw_answer,
                record.raw_analysis,
                _canonical(payload["raw_payload"]),
                _canonical(payload["metadata"]),
                _canonical(payload["license_metadata"]),
                record.raw_sha256,
            )
        )

    assert rows == expected


def test_same_partition_keeps_records_from_multiple_batches(tmp_path) -> None:
    store = ParquetStageStore(tmp_path / "parquet")

    first = store.write_raw_records(
        "run-1", "ACQUIRED", "demo", "batch-a", [_record(1), _record(2)]
    )
    second = store.write_raw_records(
        "run-1", "ACQUIRED", "demo", "batch-b", [_record(3), _record(4)]
    )

    assert first != second
    assert first.exists()
    assert second.exists()
    assert _read_record_ids(first.parent) == [
        "raw-demo-1",
        "raw-demo-2",
        "raw-demo-3",
        "raw-demo-4",
    ]


def test_same_batch_and_same_content_is_idempotent(tmp_path) -> None:
    store = ParquetStageStore(tmp_path / "parquet")
    records = [_record(1), _record(2)]

    first = store.write_raw_records(
        "run-1", "ACQUIRED", "demo", "batch-a", records
    )
    second = store.write_raw_records(
        "run-1", "ACQUIRED", "demo", "batch-a", list(reversed(records))
    )

    assert second == first
    assert sorted(path.name for path in first.parent.glob("*.parquet")) == [
        "part-batch-a.parquet"
    ]
    assert _read_record_ids(first.parent) == ["raw-demo-1", "raw-demo-2"]


def test_same_batch_with_different_content_is_rejected(tmp_path) -> None:
    store = ParquetStageStore(tmp_path / "parquet")
    path = store.write_raw_records(
        "run-1", "ACQUIRED", "demo", "batch-a", [_record(1), _record(2)]
    )

    with pytest.raises(ValueError, match="different content"):
        store.write_raw_records(
            "run-1", "ACQUIRED", "demo", "batch-a", [_record(1), _record(3)]
        )

    assert _read_record_ids(path.parent) == ["raw-demo-1", "raw-demo-2"]


def test_reopened_store_preserves_existing_batches(tmp_path) -> None:
    root = tmp_path / "parquet"
    first_store = ParquetStageStore(root)
    first = first_store.write_raw_records(
        "run-1", "ACQUIRED", "demo", "batch-a", [_record(1), _record(2)]
    )

    reopened = ParquetStageStore(root)
    reopened.write_raw_records(
        "run-1", "ACQUIRED", "demo", "batch-b", [_record(3), _record(4)]
    )

    assert _read_record_ids(first.parent) == [
        "raw-demo-1",
        "raw-demo-2",
        "raw-demo-3",
        "raw-demo-4",
    ]


def test_failed_write_removes_temporary_file(tmp_path, monkeypatch) -> None:
    store = ParquetStageStore(tmp_path / "parquet")

    def fail_write(*args, **kwargs) -> None:
        raise RuntimeError("simulated parquet failure")

    monkeypatch.setattr(parquet_module.pq, "write_table", fail_write)

    with pytest.raises(RuntimeError, match="simulated parquet failure"):
        store.write_raw_records(
            "run-1", "ACQUIRED", "demo", "batch-a", [_record(1)]
        )

    partition = (
        tmp_path
        / "parquet"
        / "run_id=run-1"
        / "stage=ACQUIRED"
        / "source_dataset=demo"
    )
    assert not list(partition.glob("*.tmp"))
    assert not list(partition.glob("*.parquet"))
