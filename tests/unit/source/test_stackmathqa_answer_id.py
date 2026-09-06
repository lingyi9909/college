from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from college_builder.source.stackmathqa import StackMathQAAdapter

GOLDEN = Path("tests/golden/stackmathqa/mathoverflow.net.jsonl")


def _config(path: Path) -> dict[str, object]:
    return {
        "data_file": str(path),
        "revision": "stackmathqafull-1q1a-answer-id-fixture-v1",
        "dataset_url": "https://huggingface.co/datasets/math-ai/StackMathQA",
        "license_metadata": {"declared": "CC-BY-4.0", "status": "UNREVIEWED"},
    }


def _zero_answer_row() -> dict[str, object]:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    row = rows[1]
    meta = row["meta"]
    assert isinstance(meta, dict)
    assert meta["answer_id"] == 0
    return row


def _write_official_row(tmp_path: Path, row: dict[str, object]) -> Path:
    path = tmp_path / "mathoverflow.net.jsonl"
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def test_answer_id_zero_generates_stable_source_and_record_identity() -> None:
    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(GOLDEN)))[0]

    records = list(adapter.acquire(descriptor))
    zero_record = records[1]
    expected_source_id = "mathoverflow:1002:0"
    expected_digest = hashlib.sha256(
        f"stackmathqa\0{expected_source_id}".encode()
    ).hexdigest()

    assert zero_record.source_id == expected_source_id
    assert zero_record.record_id == f"raw_stackmathqa_{expected_digest}"
    assert zero_record.metadata["answer_id"] == 0


def test_negative_answer_id_fails_closed(tmp_path: Path) -> None:
    row = _zero_answer_row()
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta["answer_id"] = -1
    path = _write_official_row(tmp_path, row)
    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    with pytest.raises(ValueError, match=r"answer_id"):
        list(adapter.acquire(descriptor))


def test_missing_answer_id_fails_closed(tmp_path: Path) -> None:
    row = _zero_answer_row()
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta.pop("answer_id")
    path = _write_official_row(tmp_path, row)
    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    with pytest.raises(ValueError, match=r"answer_id"):
        list(adapter.acquire(descriptor))


@pytest.mark.parametrize("invalid_answer_id", [True, "0", 0.0])
def test_non_integer_answer_id_fails_closed(
    tmp_path: Path, invalid_answer_id: object
) -> None:
    row = _zero_answer_row()
    meta = row["meta"]
    assert isinstance(meta, dict)
    meta["answer_id"] = invalid_answer_id
    path = _write_official_row(tmp_path, row)
    adapter = StackMathQAAdapter()
    descriptor = tuple(adapter.discover(_config(path)))[0]

    with pytest.raises(ValueError, match=r"answer_id"):
        list(adapter.acquire(descriptor))
