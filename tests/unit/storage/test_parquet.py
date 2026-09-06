from __future__ import annotations

import json

import duckdb

from college_builder.domain.source import RawSourceRecord
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


def test_raw_source_records_round_trip_through_partitioned_parquet(tmp_path) -> None:
    records = [_record(1), _record(2), _record(3)]
    store = ParquetStageStore(tmp_path / "parquet")

    path = store.write_raw_records("run-1", "ACQUIRED", "demo", records)

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
