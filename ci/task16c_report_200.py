from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

CODE_SHA = "456174b0ed0497239733ddd2909eea39eda83ecb"
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
    rows: list[dict[str, object]] = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
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
    assert manifest["schema_version"] == "task16c-small-production-validation-v1"
    assert manifest["selection_frozen_before_model_calls"] is True
    assert manifest["total"] == 200
    assert _sha256(SAMPLE) == manifest["sample_payload_sha256"]

    run_dirs = [path for path in (WORKSPACE / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    pipeline_report = _json(run_dir / "run_report.json")
    assert isinstance(pipeline_report, dict)
    assert pipeline_report["raw_count"] == 200
    assert pipeline_report["accepted_count"] + pipeline_report["rejected_count"] == 200

    audits = _load_audits(run_dir)
    assert len(audits) == 200
    expected_records = manifest["records"]
    assert isinstance(expected_records, list) and len(expected_records) == 200
    expected_ids = {row["record_id"] for row in expected_records if isinstance(row, dict)}
    assert set(audits) == expected_ids

    exports = _load_jsonl(OUTPUT / "questions.jsonl")
    assert len(exports) == pipeline_report["accepted_count"]
    for row in exports:
        assert len(row) == 19

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["config_version"] == "task16-recertification-v2"
    assert config["providers"]["classifier"] == {
        "name": "openai_compatible",
        "model": "deepseek-flash",
        "timeout_seconds": 60.0,
        "max_attempts": 3,
        "retry_backoff_seconds": 1.0,
    }
    assert config["providers"]["verifier"] == {
        "name": "openai_compatible",
        "model": "deepseek-v4-pro",
        "timeout_seconds": 90.0,
        "max_attempts": 3,
        "retry_backoff_seconds": 1.0,
    }

    reject_reasons: Counter[str] = Counter()
    terminal_stage_counts: Counter[str] = Counter()
    accepted_ids: list[str] = []
    for record_id, audit in audits.items():
        accepted = bool(audit.get("accepted", False))
        reason = audit.get("reject_reason")
        if accepted:
            accepted_ids.append(record_id)
            terminal_stage_counts["ACCEPTED"] += 1
        else:
            if isinstance(reason, str) and reason:
                reject_reasons[reason] += 1
                terminal_stage_counts[reason] += 1
            else:
                terminal_stage_counts["REJECTED_WITHOUT_REASON"] += 1

    prompt_sha256: dict[str, str] = {}
    for task, version in config["prompt_versions"].items():
        path = Path("prompts") / task / f"{version}.txt"
        prompt_sha256[task] = _sha256(path)

    source_counts = Counter(
        str(row.get("source_dataset")) for row in expected_records if isinstance(row, dict)
    )

    report = {
        "certification": "Task16C small production validation",
        "scope": "200 real records before any 5K/50K scale-up",
        "exact_production_code_sha": CODE_SHA,
        "sample": {
            "count": 200,
            "payload_sha256": manifest["sample_payload_sha256"],
            "record_identity_sha256": manifest["record_identity_sha256"],
            "selection_frozen_before_model_calls": True,
            "strata": manifest["strata"],
            "source_dataset_counts": dict(sorted(source_counts.items())),
        },
        "config": {
            "version": config["config_version"],
            "sha256": _sha256(CONFIG_PATH),
            "thresholds": config["thresholds"],
            "classifier": config["providers"]["classifier"],
            "verifier": config["providers"]["verifier"],
            "prompt_versions": config["prompt_versions"],
            "prompt_sha256": prompt_sha256,
            "concurrency": config["concurrency"],
        },
        "pipeline": {
            "run_id": pipeline_report["run_id"],
            "funnel": pipeline_report["funnel"],
            "accepted_count": pipeline_report["accepted_count"],
            "rejected_count": pipeline_report["rejected_count"],
            "reject_reason_counts": dict(sorted(reject_reasons.items())),
            "terminal_stage_counts": dict(sorted(terminal_stage_counts.items())),
            "provider_call_counts": pipeline_report["provider_call_counts"],
            "cache_hits": pipeline_report["cache_hits"],
            "cache_misses": pipeline_report["cache_misses"],
            "provider_errors": pipeline_report["provider_errors"],
            "provider_latency_seconds": pipeline_report["provider_latency_seconds"],
            "tokens": pipeline_report["tokens"],
            "tokens_available": pipeline_report["tokens_available"],
            "estimated_cost_usd": pipeline_report["estimated_cost_usd"],
            "estimated_cost_usd_available": pipeline_report["estimated_cost_usd_available"],
        },
        "export": {
            "count": len(exports),
            "exact_19_field_export_valid": all(len(row) == 19 for row in exports),
            "questions_jsonl_sha256": _sha256(OUTPUT / "questions.jsonl")
            if (OUTPUT / "questions.jsonl").is_file()
            else None,
            "accepted_record_ids": sorted(accepted_ids),
        },
        "engineering_gate": {
            "all_200_terminal": len(audits) == 200,
            "provider_errors_zero": pipeline_report["provider_errors"] == 0,
            "export_count_matches_accepted": len(exports) == pipeline_report["accepted_count"],
            "exact_19_field_export_valid": all(len(row) == 19 for row in exports),
        },
        "manual_quality_audit_status": "PENDING_POST_RUN_REVIEW",
        "scale_decision": "BLOCKED_PENDING_MANUAL_QUALITY_AUDIT",
    }
    (CERT_DIR / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(SAMPLE, CERT_DIR / "sample.jsonl")
    shutil.copy2(SAMPLE_MANIFEST, CERT_DIR / "sample_manifest.json")
    if (OUTPUT / "questions.jsonl").is_file():
        shutil.copy2(OUTPUT / "questions.jsonl", CERT_DIR / "questions.jsonl")
    if (OUTPUT / "pilot_report.json").is_file():
        shutil.copy2(OUTPUT / "pilot_report.json", CERT_DIR / "pilot_report.json")
    shutil.copytree(run_dir / "audits", CERT_DIR / "audits", dirs_exist_ok=True)
    if (run_dir / "rejected").is_dir():
        shutil.copytree(run_dir / "rejected", CERT_DIR / "rejected", dirs_exist_ok=True)
    if (run_dir / "provider_usage.json").is_file():
        shutil.copy2(run_dir / "provider_usage.json", CERT_DIR / "provider_usage.json")

    print("TASK16C_SMALL200_REPORT=PASS")
    print(json.dumps(report["engineering_gate"], sort_keys=True))
    print(json.dumps(report["pipeline"]["reject_reason_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
