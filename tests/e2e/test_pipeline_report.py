from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from college_builder.cli import app
from college_builder.reporting.pilot_report import (
    ProviderUsage,
    RecordAudit,
    build_pilot_report,
)


def test_pilot_report_contains_full_funnel_splits_rejects_and_cost_fields() -> None:
    audits = (
        RecordAudit(
            record_id="1",
            source_dataset="source-a",
            subject="MATHEMATICS",
            normalized=True,
            university_stem=True,
            problem=True,
            answer_valid=True,
            analysis_valid=True,
            after_dedup=True,
            alignment_pass=True,
            correctness_pass=True,
            accepted=True,
        ),
        RecordAudit(
            record_id="2",
            source_dataset="source-a",
            subject="MATHEMATICS",
            normalized=True,
            university_stem=True,
            problem=True,
            answer_valid=True,
            analysis_valid=True,
            after_dedup=False,
            alignment_pass=False,
            correctness_pass=False,
            accepted=False,
            reject_reason="DUPLICATE",
        ),
        RecordAudit(
            record_id="3",
            source_dataset="source-b",
            subject="PHYSICS",
            normalized=True,
            university_stem=True,
            problem=False,
            answer_valid=False,
            analysis_valid=False,
            after_dedup=False,
            alignment_pass=False,
            correctness_pass=False,
            accepted=False,
            reject_reason="NOT_PROBLEM",
        ),
        RecordAudit(
            record_id="4",
            source_dataset="source-b",
            subject="UNKNOWN",
            normalized=True,
            university_stem=False,
            problem=False,
            answer_valid=False,
            analysis_valid=False,
            after_dedup=False,
            alignment_pass=False,
            correctness_pass=False,
            accepted=False,
            reject_reason="UNIVERSITY_LEVEL_UNCERTAIN",
        ),
    )
    usage = ProviderUsage(
        provider_call_counts={"gate_1_university_stem": 4, "gate_2_problem": 3},
        cache_hits=5,
        cache_misses=7,
        latency_seconds=1.25,
        tokens=0,
        estimated_cost_usd=0.0,
        fallback_count=0,
        provider_errors=0,
    )

    report = build_pilot_report(
        run_id="run-1",
        audits=audits,
        provider_usage=usage,
        wall_time_seconds=2.5,
    )

    assert report.funnel == {
        "raw": 4,
        "normalized": 4,
        "stem": 3,
        "university": 3,
        "problem": 2,
        "answer_valid": 2,
        "analysis_valid": 2,
        "alignment_pass": 1,
        "after_dedup": 1,
        "final_accepted": 1,
    }
    assert report.reject_reason_counts == {
        "DUPLICATE": 1,
        "NOT_PROBLEM": 1,
        "UNIVERSITY_LEVEL_UNCERTAIN": 1,
    }
    assert report.source_distribution["source-a"].raw == 2
    assert report.source_distribution["source-a"].accepted == 1
    assert report.subject_distribution["MATHEMATICS"].raw == 2
    assert report.subject_distribution["PHYSICS"].accepted == 0
    assert report.provider_call_counts["gate_1_university_stem"] == 4
    assert report.cache_hit_rate == 5 / 12
    assert report.estimated_cost_usd == 0.0
    assert report.provider_latency_seconds == 1.25
    assert report.wall_time_seconds == 2.5
    assert report.acceptance_rate == 0.25


def test_pilot_report_distinguishes_stem_from_university() -> None:
    audits = (
        RecordAudit(
            record_id="university",
            source_dataset="source-a",
            subject="MATHEMATICS",
            normalized=True,
            stem=True,
            university=True,
            university_stem=True,
        ),
        RecordAudit(
            record_id="non-university-stem",
            source_dataset="source-a",
            subject="MATHEMATICS",
            normalized=True,
            stem=True,
            university=False,
            university_stem=False,
            reject_reason="NON_UNIVERSITY_STEM",
        ),
    )
    report = build_pilot_report(
        run_id="run-stem-split",
        audits=audits,
        provider_usage=ProviderUsage(),
        wall_time_seconds=0.0,
    )
    assert report.funnel["stem"] == 2
    assert report.funnel["university"] == 1


def test_cli_exposes_run_resume_report_and_config_validate(tmp_path: Path) -> None:
    runner = CliRunner()
    help_result = runner.invoke(app, ["--help"])
    assert help_result.exit_code == 0
    assert "run" in help_result.output
    assert "resume" in help_result.output
    assert "report" in help_result.output
    assert "config" in help_result.output

    validate_result = runner.invoke(app, ["config", "validate", "config/pilot.yaml"])
    assert validate_result.exit_code == 0
    assert "pilot-v1" in validate_result.output
