from __future__ import annotations

import hashlib
import json
from pathlib import Path

PARENT_SAMPLE_SHA = "ae513933791c2ac27d3916a4149193c2477e00ce993c59b597b27ad02101ae8e"
FROZEN_SAMPLE_SHA = "50026b04d0d5a55a27c8a796f54f35eeea756410835a82c29bbd155013d08304"
FIELDS = (
    "task15_audit_index",
    "source_id",
    "record_id",
    "raw_sha256",
    "baseline_manual_verdict",
    "baseline_task15_reject_reason",
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _records(frozen: dict[str, object]) -> list[dict[str, object]]:
    assert frozen["schema_version"] == "task16-small-recert-sample-v2"
    assert frozen["record_tuple_fields"] == list(FIELDS)
    raw_records = frozen["records"]
    assert isinstance(raw_records, list) and len(raw_records) == 50
    records = [dict(zip(FIELDS, row, strict=True)) for row in raw_records]
    assert len({row["task15_audit_index"] for row in records}) == 50
    return records


def main() -> None:
    parent = Path("/tmp/upstream/100-sample.jsonl")
    parent_manifest = Path("/tmp/upstream/100-manifest.json")
    frozen_manifest_path = Path("artifacts/task16/re-certification/sample_manifest.json")
    assert parent.is_file()
    assert parent_manifest.is_file()
    assert frozen_manifest_path.is_file()

    parent_manifest_data = json.loads(parent_manifest.read_text(encoding="utf-8"))
    assert parent_manifest_data["sample_sha256"] == PARENT_SAMPLE_SHA
    parent_ids = parent_manifest_data["sampled_record_ids"]
    parent_hashes = parent_manifest_data["sampled_raw_sha256"]
    assert parent_manifest_data["total"] == 100
    assert len(parent_ids) == len(parent_hashes) == 100
    parent_identity = dict(zip(parent_ids, parent_hashes, strict=True))
    assert len(parent_identity) == 100

    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    assert frozen["selection_frozen_before_model_calls"] is True
    assert frozen["sample"]["total"] == 50
    assert frozen["sample"]["sample_payload_sha256"] == FROZEN_SAMPLE_SHA
    assert frozen["selection_policy"]["false_reject_count"] == 43
    assert frozen["selection_policy"]["precision_control_count"] == 7
    records = _records(frozen)

    source_lines: dict[str, tuple[str, dict[str, object]]] = {}
    for line in parent.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = row["record_id"]
        raw_sha = row["raw_sha256"]
        assert isinstance(record_id, str)
        assert isinstance(raw_sha, str)
        assert parent_identity.get(record_id) == raw_sha
        source_lines[record_id] = (line, row)
    assert len(source_lines) == 100
    assert set(source_lines) == set(parent_identity)

    selected_lines: list[str] = []
    seen: set[str] = set()
    for expected in records:
        record_id = expected["record_id"]
        assert isinstance(record_id, str)
        assert record_id not in seen
        seen.add(record_id)
        line, row = source_lines[record_id]
        assert row["source_id"] == expected["source_id"]
        assert row["raw_sha256"] == expected["raw_sha256"]
        selected_lines.append(line)

    payload = ("\n".join(selected_lines) + "\n").encode("utf-8")
    assert len(selected_lines) == 50
    assert _sha256(payload) == FROZEN_SAMPLE_SHA

    output = Path("/tmp/task16b-sample.jsonl")
    output.write_bytes(payload)
    evidence_dir = Path("/tmp/task16b-certification")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    (evidence_dir / "sample_manifest.json").write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence_dir / "sample_identity.json").write_text(
        json.dumps(
            {
                "parent_logical_sample_sha256": PARENT_SAMPLE_SHA,
                "sample_payload_sha256": FROZEN_SAMPLE_SHA,
                "record_count": 50,
                "record_ids": [row["record_id"] for row in records],
                "raw_sha256": [row["raw_sha256"] for row in records],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("TASK16B_FROZEN_SAMPLE_GUARD=PASS")
    print(f"TASK16B_SAMPLE_PAYLOAD_SHA256={FROZEN_SAMPLE_SHA}")


if __name__ == "__main__":
    main()
