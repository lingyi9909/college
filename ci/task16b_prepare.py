from __future__ import annotations

import hashlib
import json
from pathlib import Path

PARENT_SAMPLE_SHA = "ae513933791c2ac27d3916a4149193c2477e00ce993c59b597b27ad02101ae8e"
FROZEN_SAMPLE_SHA = "50026b04d0d5a55a27c8a796f54f35eeea756410835a82c29bbd155013d08304"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    parent = Path("/tmp/upstream/100-sample.jsonl")
    parent_manifest = Path("/tmp/upstream/100-manifest.json")
    frozen_manifest_path = Path("artifacts/task16/re-certification/sample_manifest.json")
    assert parent.is_file()
    assert parent_manifest.is_file()
    assert frozen_manifest_path.is_file()

    parent_manifest_data = json.loads(parent_manifest.read_text(encoding="utf-8"))
    assert parent_manifest_data["sample_sha256"] == PARENT_SAMPLE_SHA
    assert _sha256(parent.read_bytes()) == PARENT_SAMPLE_SHA

    frozen = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
    assert frozen["selection_frozen_before_model_calls"] is True
    assert frozen["sample"]["total"] == 50
    assert frozen["sample"]["sample_sha256"] == FROZEN_SAMPLE_SHA
    assert frozen["selection_policy"]["false_reject_count"] == 43
    assert frozen["selection_policy"]["precision_control_count"] == 7

    source_lines: dict[str, tuple[str, dict[str, object]]] = {}
    for line in parent.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = row["record_id"]
        assert isinstance(record_id, str)
        source_lines[record_id] = (line, row)
    assert len(source_lines) == 100

    selected_lines: list[str] = []
    seen: set[str] = set()
    for expected in frozen["records"]:
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
                "parent_sample_sha256": PARENT_SAMPLE_SHA,
                "sample_sha256": FROZEN_SAMPLE_SHA,
                "record_count": 50,
                "record_ids": [row["record_id"] for row in frozen["records"]],
                "raw_sha256": [row["raw_sha256"] for row in frozen["records"]],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print("TASK16B_FROZEN_SAMPLE_GUARD=PASS")
    print(f"TASK16B_SAMPLE_SHA256={FROZEN_SAMPLE_SHA}")


if __name__ == "__main__":
    main()
