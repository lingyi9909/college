from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

CODE_SHA = "01f453ef241659e673ddcb1e012f1f270f3316dc"
COUNT = 20
CONFIG_PATH = Path("config/task16-recertification.yaml")
SAMPLE_DIR = Path("/tmp/task16c-small200")
SAMPLE = SAMPLE_DIR / "sample.jsonl"
SAMPLE_MANIFEST = SAMPLE_DIR / "sample_manifest.json"
WORKSPACE = Path("/tmp/task16c-small200-workspace")
OUTPUT = Path("/tmp/task16c-small200-output")
CERT_DIR = Path("/tmp/task16c-small200-certification")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            assert isinstance(row, dict)
            rows.append(row)
    return rows


def _load_audits(run_dir: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in sorted((run_dir / "audits").glob("*.json")):
        row = _json(path)
        assert isinstance(row, dict)
        record_id = row.get("record_id")
        assert isinstance(record_id, str)
        rows[record_id] = row
    return rows


def main() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = _json(SAMPLE_MANIFEST)
    assert isinstance(manifest, dict)
    assert manifest["schema_version"] == "task16c-small20-regression-v1"
    assert manifest["selection_frozen_before_model_calls"] is True
    assert manifest["total"] == COUNT
    assert _sha256(SAMPLE) == manifest["sample_payload_sha256"]

    run_dirs = [path for path in (WORKSPACE / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    pipeline_report = _json(run_dir / "run_report.json")
    assert isinstance(pipeline_report, dict)
    assert pipeline_report["raw_count"] == COUNT
    assert pipeline_report["accepted_count"] + pipeline_report["rejected_count"] == COUNT

    audits = _load_audits(run_dir)
    assert len(audits) == COUNT
    expected_records = manifest["records"]
    assert isinstance(expected_records, list) and len(expected_records) == COUNT
    expected_ids = {row["record_id"] for row in expected_records if isinstance(row, dict)}
    assert set(audits) == expected_ids

    exports = _load_jsonl(OUTPUT / "questions.jsonl")
    assert len(exports) == pipeline_report["accepted_count"]
    assert all(len(row) == 19 for row in exports)

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["config_version"] == "task16-recertification-v2"
    assert config["providers"]["classifier"]["model"] == "deepseek-flash"
    assert config["providers"]["verifier"]["model"] == "deepseek-v4-pro"

    reject_reasons: Counter[str] = Counter()
    accepted_ids: list[str] = []
    for record_id, audit in audits.items():
        if bool(audit.get("accepted", False)):
            accepted_ids.append(record_id)
        else:
            reason = audit.get("reject_reason")
            reject_reasons[str(reason) if reason else "REJECTED_WITHOUT_REASON"] += 1

    report = {
        "certification": "Task16C fixed Small-20 regression validation",
        "scope": "20 high-value real records before any larger validation",
        "exact_production_code_sha": CODE_SHA,
        "sample": {
            "count": COUNT,
            "cohort": manifest["cohort"],
            "payload_sha256": manifest["sample_payload_sha256"],
            "record_identity_sha256": manifest["record_identity_sha256"],
            "parent_small200_payload_sha256": manifest["parent_small200_payload_sha256"],
            "selection_frozen_before_model_calls": True,
            "source_dataset_counts": manifest["source_dataset_counts"],
        },
        "config": {
            "version": config["config_version"],
            "sha256": _sha256(CONFIG_PATH),
            "classifier": config["providers"]["classifier"],
            "verifier": config["providers"]["verifier"],
            "prompt_versions": config["prompt_versions"],
        },
        "pipeline": {
            "run_id": pipeline_report["run_id"],
            "funnel": pipeline_report["funnel"],
            "accepted_count": pipeline_report["accepted_count"],
            "rejected_count": pipeline_report["rejected_count"],
            "reject_reason_counts": dict(sorted(reject_reasons.items())),
            "provider_call_counts": pipeline_report["provider_call_counts"],
            "provider_errors": pipeline_report["provider_errors"],
            "tokens": pipeline_report["tokens"],
            "tokens_available": pipeline_report["tokens_available"],
            "estimated_cost_usd": pipeline_report["estimated_cost_usd"],
            "estimated_cost_usd_available": pipeline_report["estimated_cost_usd_available"],
        },
        "export": {
            "count": len(exports),
            "exact_19_field_export_valid": all(len(row) == 19 for row in exports),
            "accepted_record_ids": sorted(accepted_ids),
        },
        "engineering_gate": {
            "all_20_terminal": len(audits) == COUNT,
            "provider_errors_zero": pipeline_report["provider_errors"] == 0,
            "export_count_matches_accepted": len(exports) == pipeline_report["accepted_count"],
            "exact_19_field_export_valid": all(len(row) == 19 for row in exports),
        },
        "manual_quality_audit_status": "PENDING_POST_RUN_REVIEW",
        "scale_decision": "BLOCKED_PENDING_SMALL20_AUDIT",
    }
    (CERT_DIR / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(SAMPLE, CERT_DIR / "sample.jsonl")
    shutil.copy2(SAMPLE_MANIFEST, CERT_DIR / "sample_manifest.json")
    if (OUTPUT / "questions.jsonl").is_file():
        shutil.copy2(OUTPUT / "questions.jsonl", CERT_DIR / "questions.jsonl")
    shutil.copytree(run_dir / "audits", CERT_DIR / "audits", dirs_exist_ok=True)
    if (run_dir / "rejected").is_dir():
        shutil.copytree(run_dir / "rejected", CERT_DIR / "rejected", dirs_exist_ok=True)

    print("TASK16C_SMALL20_REPORT=PASS")
    print(json.dumps(report["engineering_gate"], sort_keys=True))
    print(json.dumps(report["pipeline"]["reject_reason_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
