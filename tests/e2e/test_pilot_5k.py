from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from college_builder.cli import app
from college_builder.config import PipelineConfig
from college_builder.pipeline.pilot import FIVE_K_QUOTAS, FiveKPilotPlan


def test_frozen_config_loads_and_preserves_precision_thresholds() -> None:
    config = PipelineConfig.load(Path("config/pilot-5k-frozen.yaml"))
    assert config.config_version == "pilot-5k-frozen-v1"
    assert config.thresholds.university == 0.98
    assert config.thresholds.problem == 0.98
    assert config.thresholds.answer_extract == 0.995
    assert config.thresholds.analysis == 0.98
    assert config.thresholds.qa_alignment == 0.995
    assert config.prompt_versions.model_dump() == {
        "university_classify": "v1",
        "problem_classify": "v1",
        "analysis_classify": "v1",
        "qa_alignment": "v1",
        "correctness_verify": "v1",
    }


def test_pilot_plan_command_emits_exact_5k_strata_and_seed() -> None:
    result = CliRunner().invoke(app, ["pilot", "plan", "--seed", "20260910"])
    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["seed"] == 20260910
    assert payload["total"] == 5000
    assert payload["quotas"] == FIVE_K_QUOTAS
    assert FiveKPilotPlan.model_validate(
        {"seed": payload["seed"], "quotas": payload["quotas"]}
    ).total == 5000


def test_task15_report_and_summary_are_non_raw_and_self_consistent() -> None:
    report = Path("docs/validation/5k-dry-run-report.md").read_text(encoding="utf-8")
    summary = json.loads(Path("artifacts/5k/run_report.json").read_text(encoding="utf-8"))
    assert "raw copyrighted dataset rows" not in summary
    assert summary["pilot_size"] == 5000
    assert summary["sampling"]["quotas"] == FIVE_K_QUOTAS
    assert isinstance(summary["sampling"]["seed"], int)
    assert len(summary["config_sha256"]) == 64
    assert summary["config_version"] == "pilot-5k-frozen-v1"
    assert summary["status"] in {"EXECUTED", "BLOCKED_EXTERNAL_PROVIDER"}
    assert summary["status"] in report
    assert summary["source_revision"] in report
    assert summary["config_sha256"] in report
