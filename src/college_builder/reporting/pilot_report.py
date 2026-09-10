"""Deterministic Task 14 pilot funnel and run reporting."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RecordAudit(_FrozenModel):
    """Per-record reporting facts emitted by the pipeline runner."""

    record_id: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    normalized: bool = False
    stem: bool | None = None
    university: bool | None = None
    university_stem: bool = False
    problem: bool = False
    answer_valid: bool = False
    analysis_valid: bool = False
    after_dedup: bool = False
    alignment_pass: bool = False
    correctness_pass: bool = False
    accepted: bool = False
    reject_reason: str | None = None


class ProviderUsage(_FrozenModel):
    """Provider/cache usage accumulated for one run attempt."""

    provider_call_counts: dict[str, int] = Field(default_factory=dict)
    cache_hits: int = Field(default=0, ge=0)
    cache_misses: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    fallback_count: int = Field(default=0, ge=0)
    provider_errors: int = Field(default=0, ge=0)


class DistributionCount(_FrozenModel):
    raw: int = Field(ge=0)
    accepted: int = Field(ge=0)


class PilotReport(_FrozenModel):
    """Stable JSON report required by the Pilot implementation plan."""

    run_id: str = Field(min_length=1)
    funnel: dict[str, int]
    raw_count: int = Field(ge=0)
    normalized_count: int = Field(ge=0)
    accepted_count: int = Field(ge=0)
    rejected_count: int = Field(ge=0)
    acceptance_rate: float = Field(ge=0.0, le=1.0)
    reject_reason_counts: dict[str, int]
    source_distribution: dict[str, DistributionCount]
    subject_distribution: dict[str, DistributionCount]
    provider_call_counts: dict[str, int]
    cache_hit_rate: float = Field(ge=0.0, le=1.0)
    fallback_count: int = Field(ge=0)
    provider_errors: int = Field(ge=0)
    tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0.0)
    provider_latency_seconds: float = Field(ge=0.0)
    wall_time_seconds: float = Field(ge=0.0)


def build_pilot_report(
    *,
    run_id: str,
    audits: Iterable[RecordAudit],
    provider_usage: ProviderUsage,
    wall_time_seconds: float,
) -> PilotReport:
    """Aggregate a complete, deterministic funnel from per-record audit facts."""

    rows = tuple(audits)
    raw_count = len(rows)
    accepted_count = sum(row.accepted for row in rows)
    funnel = {
        "raw": raw_count,
        "normalized": sum(row.normalized for row in rows),
        "stem": sum(row.stem if row.stem is not None else row.university_stem for row in rows),
        "university": sum(
            row.university if row.university is not None else row.university_stem for row in rows
        ),
        "problem": sum(row.problem for row in rows),
        "answer_valid": sum(row.answer_valid for row in rows),
        "analysis_valid": sum(row.analysis_valid for row in rows),
        "alignment_pass": sum(row.alignment_pass for row in rows),
        "after_dedup": sum(row.after_dedup for row in rows),
        "final_accepted": accepted_count,
    }
    reject_reasons = Counter(row.reject_reason for row in rows if row.reject_reason is not None)
    total_cache = provider_usage.cache_hits + provider_usage.cache_misses

    return PilotReport(
        run_id=run_id,
        funnel=funnel,
        raw_count=raw_count,
        normalized_count=funnel["normalized"],
        accepted_count=accepted_count,
        rejected_count=raw_count - accepted_count,
        acceptance_rate=(accepted_count / raw_count) if raw_count else 0.0,
        reject_reason_counts=dict(sorted(reject_reasons.items())),
        source_distribution=_distribution(rows, "source_dataset"),
        subject_distribution=_distribution(rows, "subject"),
        provider_call_counts=dict(sorted(provider_usage.provider_call_counts.items())),
        cache_hit_rate=(provider_usage.cache_hits / total_cache) if total_cache else 0.0,
        fallback_count=provider_usage.fallback_count,
        provider_errors=provider_usage.provider_errors,
        tokens=provider_usage.tokens,
        estimated_cost_usd=provider_usage.estimated_cost_usd,
        provider_latency_seconds=provider_usage.latency_seconds,
        wall_time_seconds=wall_time_seconds,
    )


def write_pilot_report(report: PilotReport, path: Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        report.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    temp = destination.with_name(f".{destination.name}.tmp")
    temp.write_text(payload + "\n", encoding="utf-8", newline="\n")
    temp.replace(destination)
    return destination


def load_pilot_report(path: Path) -> PilotReport:
    return PilotReport.model_validate_json(Path(path).read_text(encoding="utf-8"))


def _distribution(
    rows: tuple[RecordAudit, ...],
    attribute: str,
) -> dict[str, DistributionCount]:
    raw_counts: Counter[str] = Counter()
    accepted_counts: Counter[str] = Counter()
    for row in rows:
        value = getattr(row, attribute)
        if not isinstance(value, str):
            raise TypeError(f"{attribute} must resolve to a string")
        raw_counts[value] += 1
        if row.accepted:
            accepted_counts[value] += 1
    return {
        key: DistributionCount(raw=raw_counts[key], accepted=accepted_counts[key])
        for key in sorted(raw_counts)
    }
