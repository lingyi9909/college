from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from college_builder.config import PipelineConfig

CONFIG_PATH = Path("config/pilot-5k-frozen.yaml")
WORKSPACE = Path("/tmp/task15-real100-workspace")
OUTPUT = Path("/tmp/task15-real100-output")
CERT = Path("/tmp/task15-real100-certification")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    code_sha = os.environ["TASK15_CODE_SHA"]
    expected_100_sha = os.environ["TASK15_EXPECTED_100_SHA"]
    run_dirs = tuple((WORKSPACE / "runs").iterdir())
    assert len(run_dirs) == 1, run_dirs
    run_dir = run_dirs[0]

    pipeline_report = json.loads((run_dir / "run_report.json").read_text(encoding="utf-8"))
    manifest_5k = json.loads(
        Path("/tmp/task15-real100-5k-manifest.json").read_text(encoding="utf-8")
    )
    manifest_100 = json.loads(
        Path("/tmp/task15-real100-100-manifest.json").read_text(encoding="utf-8")
    )
    cfg = PipelineConfig.load(CONFIG_PATH)

    assert pipeline_report["raw_count"] == 100
    assert pipeline_report["provider_errors"] == 0
    assert manifest_100["total"] == 100
    assert manifest_100["sample_sha256"] == expected_100_sha
    assert manifest_100["parent_sample_sha256"] == manifest_5k["sample_sha256"]

    prompt_versions = cfg.prompt_versions.model_dump(mode="json")
    prompt_sha256 = {
        "university_classify": _sha256(
            Path("prompts/university_classify") / f"{cfg.prompt_versions.university_classify}.txt"
        ),
        "problem_classify": _sha256(
            Path("prompts/problem_classify") / f"{cfg.prompt_versions.problem_classify}.txt"
        ),
        "analysis_classify": _sha256(
            Path("prompts/analysis_classify") / f"{cfg.prompt_versions.analysis_classify}.txt"
        ),
        "qa_alignment": _sha256(
            Path("prompts/qa_alignment") / f"{cfg.prompt_versions.qa_alignment}.txt"
        ),
        "correctness_verify": _sha256(
            Path("prompts/correctness_verify") / f"{cfg.prompt_versions.correctness_verify}.txt"
        ),
    }

    report = {
        "certification_scope": "task15-100-real-model-e2e",
        "exact_code_sha": code_sha,
        "pipeline_version": cfg.pipeline_version,
        "source_dataset": "math-ai/StackMathQA",
        "source_revision": "git:13239ec8c8078bc8dfb43e1383069eb9be000ff7",
        "source_file_sha256": manifest_5k["source_file_sha256"],
        "five_k_sample_sha256": manifest_5k["sample_sha256"],
        "hundred_sample_sha256": manifest_100["sample_sha256"],
        "seed": manifest_100["seed"],
        "five_k_strata": manifest_5k["quotas"],
        "hundred_strata": manifest_100["quotas"],
        "config_sha256": _sha256(CONFIG_PATH),
        "config_version": cfg.config_version,
        "prompt_versions": prompt_versions,
        "prompt_sha256": prompt_sha256,
        "classifier": {
            "provider": cfg.providers.classifier.name,
            "model": cfg.providers.classifier.model,
        },
        "verifier": {
            "provider": cfg.providers.verifier.name,
            "model": cfg.providers.verifier.model,
        },
        "raw_count": pipeline_report["raw_count"],
        "accepted_count": pipeline_report["accepted_count"],
        "rejected_count": pipeline_report["rejected_count"],
        "funnel": pipeline_report["funnel"],
        "reject_reason_counts": pipeline_report["reject_reason_counts"],
        "acceptance_rate": pipeline_report["acceptance_rate"],
        "provider_call_counts": pipeline_report["provider_call_counts"],
        "cache_hits": pipeline_report["cache_hits"],
        "cache_misses": pipeline_report["cache_misses"],
        "provider_errors": pipeline_report["provider_errors"],
        "provider_latency_seconds": pipeline_report["provider_latency_seconds"],
        "wall_time_seconds": pipeline_report["wall_time_seconds"],
        "tokens": pipeline_report["tokens"],
        "tokens_available": pipeline_report["tokens_available"],
        "estimated_cost_usd": pipeline_report["estimated_cost_usd"],
        "estimated_cost_usd_available": pipeline_report[
            "estimated_cost_usd_available"
        ],
    }

    if not report["tokens_available"]:
        assert report["tokens"] is None
    if not report["estimated_cost_usd_available"]:
        assert report["estimated_cost_usd"] is None

    CERT.mkdir(parents=True, exist_ok=True)
    (CERT / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.copy2("/tmp/task15-real100-5k-manifest.json", CERT / "5k-manifest.json")
    shutil.copy2("/tmp/task15-real100-100-manifest.json", CERT / "100-manifest.json")
    shutil.copy2("/tmp/task15-real100-sample.jsonl", CERT / "100-sample.jsonl")
    shutil.copytree(run_dir, CERT / "run", dirs_exist_ok=True)
    shutil.copytree(OUTPUT, CERT / "output", dirs_exist_ok=True)
    print("TASK15_REAL100_REPORT=PASS")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
