"""Atomic Parquet stage storage for immutable raw-source records."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from college_builder.domain.source import RawSourceRecord

_PARTITION_VALUE = re.compile(r"^[A-Za-z0-9._-]+$")


def _validate_partition_value(name: str, value: str) -> None:
    if not value or _PARTITION_VALUE.fullmatch(value) is None:
        raise ValueError(f"{name} must contain only letters, digits, dot, underscore, or hyphen")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _batch_fingerprint(rows: Sequence[object]) -> str:
    canonical_rows = sorted(_canonical_json(row) for row in rows)
    payload = _canonical_json(canonical_rows).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class ParquetStageStore:
    """Persist stage rows into deterministic Hive-style Parquet partitions."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def write_raw_records(
        self,
        run_id: str,
        stage: str,
        source_dataset: str,
        batch_id: str,
        records: Sequence[RawSourceRecord],
    ) -> Path:
        """Atomically persist one deterministic batch without overwriting other batches."""
        _validate_partition_value("run_id", run_id)
        _validate_partition_value("stage", stage)
        _validate_partition_value("source_dataset", source_dataset)
        _validate_partition_value("batch_id", batch_id)
        if not records:
            raise ValueError("records must be non-empty")

        rows: list[dict[str, str]] = []
        for record in records:
            if record.source_dataset != source_dataset:
                raise ValueError(
                    "record source_dataset must match the target source_dataset partition"
                )
            payload = record.model_dump(mode="json")
            rows.append(
                {
                    "record_id": record.record_id,
                    "source_type": record.source_type,
                    "source_dataset": record.source_dataset,
                    "source_id": record.source_id,
                    "source_url": record.source_url,
                    "raw_question": record.raw_question,
                    "raw_answer": record.raw_answer,
                    "raw_analysis": record.raw_analysis,
                    "raw_payload": _canonical_json(payload["raw_payload"]),
                    "metadata": _canonical_json(payload["metadata"]),
                    "license_metadata": _canonical_json(payload["license_metadata"]),
                    "raw_sha256": record.raw_sha256,
                }
            )

        partition = (
            self._root
            / f"run_id={run_id}"
            / f"stage={stage}"
            / f"source_dataset={source_dataset}"
        )
        partition.mkdir(parents=True, exist_ok=True)
        final_path = partition / f"part-{batch_id}.parquet"
        incoming_fingerprint = _batch_fingerprint(rows)

        if final_path.exists():
            existing_rows = pq.ParquetFile(str(final_path)).read().to_pylist()
            if _batch_fingerprint(existing_rows) == incoming_fingerprint:
                return final_path
            raise ValueError(f"batch_id {batch_id} already exists with different content")

        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".part-{batch_id}-",
            suffix=".tmp",
            dir=partition,
        )
        os.close(file_descriptor)
        temp_path = Path(temp_name)
        try:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, str(temp_path))
            os.replace(temp_path, final_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        return final_path
