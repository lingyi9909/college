from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import yaml

CODE_SHA = "8e35645a016770d6db635a698617c03b9a07fc46"
FROZEN_SAMPLE_SHA = "50026b04d0d5a55a27c8a796f54f35eeea756410835a82c29bbd155013d08304"
CONFIG_PATH = Path("config/task16-recertification.yaml")
CERT_DIR = Path("/tmp/task16b-certification")
WORKSPACE = Path("/tmp/task16b-workspace")
OUTPUT = Path("/tmp/task16b-output")
SAMPLE = Path("/tmp/task16b-sample.jsonl")
MANIFEST = Path("artifacts/task16/re-certification/sample_manifest.json")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_raw() -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for line in SAMPLE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        record_id = row["record_id"]
        assert isinstance(record_id, str)
        rows[record_id] = row
    assert len(rows) == 50
    return rows


def _load_audits(run_dir: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for path in sorted((run_dir / "audits").glob("*.json")):
        row = _json(path)
        assert isinstance(row, dict)
        record_id = row.get("record_id")
        assert isinstance(record_id, str)
        rows[record_id] = row
    assert len(rows) == 50
    return rows


def _load_exports() -> list[dict[str, object]]:
    path = OUTPUT / "questions.jsonl"
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        assert isinstance(row, dict)
        assert len(row) == 19
        rows.append(row)
    return rows


def _accepted_source_bindings(
    exports: list[dict[str, object]],
    frozen_by_source: dict[str, dict[str, object]],
) -> list[dict[str, object]]:
    bindings: list[dict[str, object]] = []
    for row in exports:
        static_raw = row.get("static_info")
        answer = row.get("text_answer")
        assert isinstance(static_raw, str)
        assert isinstance(answer, str) and answer.strip()
        static = json.loads(static_raw)
        assert isinstance(static, dict)
        source_id = static.get("source_id")
        raw_sha = static.get("raw_sha256")
        span = static.get("answer_source_span")
        assert isinstance(source_id, str)
        assert isinstance(raw_sha, str)
        assert isinstance(span, str)
        expected = frozen_by_source[source_id]
        assert raw_sha == expected["raw_sha256"]
        bindings.append(
            {
                "source_id": source_id,
                "raw_sha256": raw_sha,
                "answer_source_span": span,
                "final_answer": answer,
                "task15_audit_index": expected["task15_audit_index"],
                "baseline_manual_verdict": expected["baseline_manual_verdict"],
                "baseline_task15_reject_reason": expected["baseline_task15_reject_reason"],
            }
        )
    return bindings


def main() -> None:
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    assert _sha256_file(SAMPLE) == FROZEN_SAMPLE_SHA
    frozen = _json(MANIFEST)
    assert isinstance(frozen, dict)
    records = frozen["records"]
    assert isinstance(records, list) and len(records) == 50
    frozen_by_id = {row["record_id"]: row for row in records}
    frozen_by_source = {row["source_id"]: row for row in records}

    run_dirs = [path for path in (WORKSPACE / "runs").iterdir() if path.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    pipeline_report = _json(run_dir / "run_report.json")
    assert isinstance(pipeline_report, dict)
    assert pipeline_report["raw_count"] == 50
    audits = _load_audits(run_dir)
    assert set(audits) == set(frozen_by_id)
    exports = _load_exports()
    assert len(exports) == pipeline_report["accepted_count"]

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["config_version"] == "task16-recertification-v2"
    prompt_versions = dict(config["prompt_versions"])
    prompt_sha256: dict[str, str] = {}
    for task, version in prompt_versions.items():
        path = Path("prompts") / task / f"{version}.txt"
        prompt_sha256[task] = _sha256_file(path)

    outcomes: list[dict[str, object]] = []
    baseline_false_rejects = 0
    recovered_false_rejects = 0
    retained_false_rejects = 0
    accepted_controls = 0
    new_reasons: Counter[str] = Counter()
    for expected in records:
        record_id = expected["record_id"]
        audit = audits[record_id]
        accepted = bool(audit.get("accepted", False))
        reject_reason = audit.get("reject_reason")
        if isinstance(reject_reason, str) and reject_reason:
            new_reasons[reject_reason] += 1
        if expected["baseline_manual_verdict"] == "FALSE_REJECT":
            baseline_false_rejects += 1
            if accepted:
                recovered_false_rejects += 1
            else:
                retained_false_rejects += 1
        elif accepted:
            accepted_controls += 1
        outcomes.append(
            {
                "record_id": record_id,
                "source_id": expected["source_id"],
                "raw_sha256": expected["raw_sha256"],
                "task15_audit_index": expected["task15_audit_index"],
                "coverage": expected["coverage"],
                "baseline_manual_verdict": expected["baseline_manual_verdict"],
                "baseline_task15_reject_reason": expected["baseline_task15_reject_reason"],
                "task16_accepted": accepted,
                "task16_reject_reason": reject_reason,
                "normalized": bool(audit.get("normalized", False)),
                "stem": bool(audit.get("stem", False)),
                "university": bool(audit.get("university", False)),
                "problem": bool(audit.get("problem", False)),
                "answer_valid": bool(audit.get("answer_valid", False)),
                "analysis_valid": bool(audit.get("analysis_valid", False)),
                "after_dedup": bool(audit.get("after_dedup", False)),
                "alignment_pass": bool(audit.get("alignment_pass", False)),
                "correctness_pass": bool(audit.get("correctness_pass", False)),
            }
        )

    accepted_bindings = _accepted_source_bindings(exports, frozen_by_source)
    report = {
        "certification": "Task16B small real-model re-certification",
        "exact_code_sha": CODE_SHA,
        "sample_sha256": FROZEN_SAMPLE_SHA,
        "sample_count": 50,
        "parent_task15": frozen["parent_task15"],
        "selection_policy": frozen["selection_policy"],
        "coverage_counts": frozen["coverage_counts"],
        "config_version": config["config_version"],
        "config_sha256": _sha256_file(CONFIG_PATH),
        "prompt_versions": prompt_versions,
        "prompt_sha256": prompt_sha256,
        "classifier": config["providers"]["classifier"],
        "verifier": config["providers"]["verifier"],
        "thresholds": config["thresholds"],
        "pipeline_run_id": pipeline_report["run_id"],
        "funnel": pipeline_report["funnel"],
        "accepted_count": pipeline_report["accepted_count"],
        "rejected_count": pipeline_report["rejected_count"],
        "reject_reason_counts": dict(sorted(new_reasons.items())),
        "provider_call_counts": pipeline_report["provider_call_counts"],
        "cache_hits": pipeline_report["cache_hits"],
        "cache_misses": pipeline_report["cache_misses"],
        "provider_errors": pipeline_report["provider_errors"],
        "provider_latency_seconds": pipeline_report["provider_latency_seconds"],
        "tokens": pipeline_report["tokens"],
        "tokens_available": pipeline_report["tokens_available"],
        "estimated_cost_usd": pipeline_report["estimated_cost_usd"],
        "estimated_cost_usd_available": pipeline_report["estimated_cost_usd_available"],
        "before_after": {
            "task15_known_false_rejects_in_sample": baseline_false_rejects,
            "task16_pipeline_accepted_from_known_false_rejects": recovered_false_rejects,
            "task16_pipeline_rejected_from_known_false_rejects": retained_false_rejects,
            "precision_controls": 7,
            "pipeline_accepted_precision_controls": accepted_controls,
        },
        "accepted_source_bindings": accepted_bindings,
        "exact_19_field_export_valid": len(exports) == pipeline_report["accepted_count"],
        "questions_jsonl_sha256": _sha256_file(OUTPUT / "questions.jsonl"),
        "records": outcomes,
        "manual_audit_status": "PENDING_POST_RUN_REVIEW",
        "acceptance_gate_status": "PENDING_MANUAL_AUDIT",
        "production_scale_claim": False,
    }
    (CERT_DIR / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(SAMPLE, CERT_DIR / "sample.jsonl")
    shutil.copy2(OUTPUT / "questions.jsonl", CERT_DIR / "questions.jsonl")
    shutil.copy2(OUTPUT / "pilot_report.json", CERT_DIR / "pilot_report.json")
    shutil.copytree(run_dir / "audits", CERT_DIR / "audits", dirs_exist_ok=True)
    if (run_dir / "rejected").is_dir():
        shutil.copytree(run_dir / "rejected", CERT_DIR / "rejected", dirs_exist_ok=True)
    if (run_dir / "stages").is_dir():
        shutil.copytree(run_dir / "stages", CERT_DIR / "stages", dirs_exist_ok=True)
    shutil.copy2(run_dir / "provider_usage.json", CERT_DIR / "provider_usage.json")
    print("TASK16B_REPORT_BUILT=PASS")
    print(json.dumps(report["before_after"], sort_keys=True))
    print(json.dumps(report["funnel"], sort_keys=True))


if __name__ == "__main__":
    main()
