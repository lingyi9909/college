from __future__ import annotations

import hashlib
import heapq
import json
import urllib.request
from pathlib import Path
from typing import Any

from college_builder.domain.source import RawSourceRecord
from college_builder.pipeline.pilot import FiveKPilotPlan, sample_stackmathqa_source_root
from college_builder.source.stackmathqa import OFFICIAL_SOURCE_FILES

SEED = 20260910
STACK_REVISION = "13239ec8c8078bc8dfb43e1383069eb9be000ff7"
STACK_REVISION_TOKEN = f"git:{STACK_REVISION}"
STACK_BASE_URL = (
    "https://huggingface.co/datasets/math-ai/StackMathQA/resolve/"
    + STACK_REVISION
    + "/preprocessed/stackexchange-math--1q1a"
)
STACK_QUOTAS = {
    "math": 20000,
    "mathoverflow": 5000,
    "statistics": 10000,
    "physics": 10000,
}
OPENSTAX_DATASET = "pranav-gupta/openstax-qa"
OPENSTAX_REVISION = "1eafca778505d699033c92554b5984621f11380d"
OPENSTAX_FILE = "openstax_processed_data.json"
OPENSTAX_FILE_SHA256 = "4df9ff9fcefc6c41d527618f80935f5ce4198011b438f70e811dff26051a9fd2"
OPENSTAX_URL = (
    "https://huggingface.co/datasets/"
    + OPENSTAX_DATASET
    + "/resolve/"
    + OPENSTAX_REVISION
    + "/"
    + OPENSTAX_FILE
    + "?download=true"
)
OPENSTAX_QUOTA = 5000
OUT_DIR = Path("/tmp/task16c-50k-preflight")
SOURCE_ROOT = Path("/tmp/task16c-stackmathqa-source")
OPENSTAX_PATH = Path("/tmp/task16c-openstax.json")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _iter_openstax_rows(node: object):
    if isinstance(node, dict):
        if {"id", "problem", "solution"}.issubset(node):
            yield node
            return
        for value in node.values():
            yield from _iter_openstax_rows(value)
    elif isinstance(node, list):
        for value in node:
            yield from _iter_openstax_rows(value)


def _openstax_record(row: dict[str, Any], row_offset: int) -> RawSourceRecord:
    source_id = str(row["id"])
    question = row["problem"]
    solution = row["solution"]
    if not isinstance(question, str) or not question.strip():
        raise ValueError("OpenStax problem must be non-empty text")
    if not isinstance(solution, str) or not solution.strip():
        raise ValueError("OpenStax solution must be non-empty text")
    identity = f"huggingface\0{OPENSTAX_DATASET}\0{source_id}".encode()
    return RawSourceRecord.model_validate(
        {
            "record_id": f"raw_hf_{hashlib.sha256(identity).hexdigest()}",
            "source_type": "dataset",
            "source_dataset": OPENSTAX_DATASET,
            "source_id": source_id,
            "source_url": f"https://huggingface.co/datasets/{OPENSTAX_DATASET}",
            "raw_question": question,
            "raw_answer": "",
            "raw_analysis": solution,
            "raw_payload": row,
            "metadata": {
                "language": row.get("language"),
                "book": row.get("book"),
                "chapter_number": row.get("chapter_number"),
                "acquisition": {
                    "adapter": "task16c_openstax_raw_snapshot",
                    "row_offset": row_offset,
                    "source_revision": f"git:{OPENSTAX_REVISION}",
                    "source_file_sha256": OPENSTAX_FILE_SHA256,
                },
            },
            "license_metadata": {"license": "cc-by-4.0"},
            "raw_sha256": _sha256_bytes(_canonical_json(row)),
        }
    )


def _sample_openstax() -> tuple[tuple[RawSourceRecord, ...], dict[str, object]]:
    urllib.request.urlretrieve(OPENSTAX_URL, OPENSTAX_PATH)
    actual_sha = _sha256_bytes(OPENSTAX_PATH.read_bytes())
    if actual_sha != OPENSTAX_FILE_SHA256:
        raise ValueError(f"OpenStax file SHA mismatch: {actual_sha}")
    raw = json.loads(OPENSTAX_PATH.read_text(encoding="utf-8"))
    heaps: list[tuple[int, str, RawSourceRecord]] = []
    seen: set[str] = set()
    eligible = 0
    for row_offset, row in enumerate(_iter_openstax_rows(raw)):
        source_id = str(row.get("id", ""))
        if not source_id or source_id in seen:
            if source_id in seen:
                raise ValueError(f"duplicate OpenStax id: {source_id}")
            continue
        seen.add(source_id)
        language = row.get("language")
        if language not in (None, "en"):
            continue
        record = _openstax_record(row, row_offset)
        eligible += 1
        priority = int(
            hashlib.sha256(
                f"{OPENSTAX_REVISION}\0{SEED}\0{record.source_id}\0{record.raw_sha256}".encode()
            ).hexdigest(),
            16,
        )
        entry = (-priority, record.record_id, record)
        if len(heaps) < OPENSTAX_QUOTA:
            heapq.heappush(heaps, entry)
        elif entry > heaps[0]:
            heapq.heapreplace(heaps, entry)
    if eligible < OPENSTAX_QUOTA:
        raise ValueError(f"insufficient English OpenStax rows: {eligible}")
    ordered = sorted(
        ((-priority, record_id, record) for priority, record_id, record in heaps),
        key=lambda item: (item[0], item[1]),
    )
    selected = tuple(item[2] for item in ordered)
    logical_sha = _sha256_bytes(
        _canonical_json(
            {
                "seed": SEED,
                "revision": OPENSTAX_REVISION,
                "records": [(r.record_id, r.raw_sha256) for r in selected],
            }
        )
    )
    return selected, {
        "dataset": OPENSTAX_DATASET,
        "revision": f"git:{OPENSTAX_REVISION}",
        "source_file_sha256": actual_sha,
        "eligible_english_rows": eligible,
        "quota": OPENSTAX_QUOTA,
        "logical_sample_sha256": logical_sha,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    for filename in OFFICIAL_SOURCE_FILES:
        urllib.request.urlretrieve(f"{STACK_BASE_URL}/{filename}", SOURCE_ROOT / filename)

    stack_sample = sample_stackmathqa_source_root(
        SOURCE_ROOT,
        source_revision=STACK_REVISION_TOKEN,
        plan=FiveKPilotPlan(seed=SEED, quotas=STACK_QUOTAS),
    )
    if stack_sample.manifest.total != 45000:
        raise AssertionError(stack_sample.manifest.total)
    openstax_records, openstax_manifest = _sample_openstax()
    records = tuple(stack_sample.records) + openstax_records
    if len(records) != 50000:
        raise AssertionError(len(records))
    if len({r.record_id for r in records}) != 50000:
        raise AssertionError("duplicate record_id in combined sample")

    payload = b"".join(
        _canonical_json(record.model_dump(mode="json")) + b"\n" for record in records
    )
    payload_sha = _sha256_bytes(payload)
    identity_sha = _sha256_bytes(
        _canonical_json([(r.record_id, r.raw_sha256) for r in records])
    )
    sample_path = OUT_DIR / "sample.jsonl"
    sample_path.write_bytes(payload)
    manifest = {
        "schema_version": "task16c-50k-sample-v1",
        "seed": SEED,
        "total": 50000,
        "strata": {
            "stackmathqa": STACK_QUOTAS,
            "openstax": OPENSTAX_QUOTA,
        },
        "sample_payload_sha256": payload_sha,
        "record_identity_sha256": identity_sha,
        "stackmathqa": stack_sample.manifest.model_dump(mode="json"),
        "openstax": openstax_manifest,
        "record_ids": [r.record_id for r in records],
        "raw_sha256": [r.raw_sha256 for r in records],
    }
    (OUT_DIR / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    compact = {
        key: value
        for key, value in manifest.items()
        if key not in {"record_ids", "raw_sha256"}
    }
    (OUT_DIR / "preflight_report.json").write_text(
        json.dumps(compact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("TASK16C_50K_SAMPLE_PREFLIGHT=PASS")
    print("TASK16C_50K_SAMPLE_PAYLOAD_SHA256=" + payload_sha)
    print("TASK16C_50K_RECORD_IDENTITY_SHA256=" + identity_sha)
    print("TASK16C_STACK45K_LOGICAL_SHA256=" + stack_sample.manifest.sample_sha256)
    print("TASK16C_OPENSTAX5K_LOGICAL_SHA256=" + str(openstax_manifest["logical_sample_sha256"]))


if __name__ == "__main__":
    main()
