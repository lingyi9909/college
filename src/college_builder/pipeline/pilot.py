"""Deterministic Task 15 pilot sampling, evidence, and configuration freeze helpers."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Annotated

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from college_builder.config import PipelineConfig
from college_builder.domain.source import RawSourceRecord
from college_builder.source.stackmathqa import stratified_sample

PositiveInt = Annotated[int, Field(ge=1)]
SHA256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
FIVE_K_TOTAL = 5000
FIVE_K_QUOTAS: dict[str, int] = {
    "math": 2000,
    "physics": 1000,
    "statistics": 1000,
    "mathoverflow": 1000,
}
DEFAULT_FIVE_K_SEED = 20260910
_IMMUTABLE_REVISION = re.compile(
    r"^(?:sha256:[0-9a-fA-F]{64}|git:(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64}))$"
)


class FiveKPilotPlan(BaseModel):
    """Frozen deterministic sampling plan; defaults are the approved 5K strata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = DEFAULT_FIVE_K_SEED
    quotas: dict[str, PositiveInt] = Field(default_factory=lambda: dict(FIVE_K_QUOTAS))

    @model_validator(mode="after")
    def _validate_sites(self) -> FiveKPilotPlan:
        if not set(self.quotas).issubset(FIVE_K_QUOTAS):
            raise ValueError("pilot quotas contain an unsupported StackMathQA stratum")
        return self

    @property
    def total(self) -> int:
        return sum(self.quotas.values())

    def as_dict(self) -> dict[str, object]:
        return {"seed": self.seed, "quotas": dict(self.quotas), "total": self.total}


class PilotSampleManifest(BaseModel):
    """Source-traceable sampling metadata committed without copyrighted raw rows."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int
    quotas: dict[str, PositiveInt]
    total: PositiveInt
    sample_sha256: SHA256Hex
    sampled_record_ids: tuple[str, ...]
    source_revisions: dict[str, str]

    @model_validator(mode="after")
    def _validate_manifest(self) -> PilotSampleManifest:
        if self.total != sum(self.quotas.values()):
            raise ValueError("pilot manifest total does not equal quota sum")
        if self.total != len(self.sampled_record_ids):
            raise ValueError("pilot manifest total does not equal sampled record count")
        for revision in self.source_revisions.values():
            if not _IMMUTABLE_REVISION.fullmatch(revision):
                raise ValueError("pilot source revision must be explicitly immutable")
        return self


class PilotSample(BaseModel):
    """Runtime-only sample plus a safe commit-ready manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    records: tuple[RawSourceRecord, ...]
    manifest: PilotSampleManifest


def build_pilot_sample(
    records: Iterable[RawSourceRecord],
    *,
    plan: FiveKPilotPlan,
    source_revisions: Mapping[str, str],
) -> PilotSample:
    """Select deterministic exact strata and derive a stable non-content sample identity."""
    sampled = tuple(stratified_sample(records, plan.quotas, plan.seed))
    record_ids = tuple(record.record_id for record in sampled)
    digest = hashlib.sha256(
        json.dumps(
            {"seed": plan.seed, "quotas": plan.quotas, "record_ids": record_ids},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = PilotSampleManifest(
        seed=plan.seed,
        quotas=plan.quotas,
        total=len(sampled),
        sample_sha256=digest,
        sampled_record_ids=record_ids,
        source_revisions=dict(source_revisions),
    )
    return PilotSample(records=sampled, manifest=manifest)


def freeze_pipeline_config(
    source: Path,
    target: Path,
    *,
    frozen_config_version: str,
) -> str:
    """Freeze only an already-valid pipeline config; thresholds/prompts/models are not altered."""
    config = PipelineConfig.load(source)
    payload = config.model_dump(mode="json")
    payload["config_version"] = frozen_config_version
    frozen = PipelineConfig.model_validate(payload)
    text = yaml.safe_dump(frozen.model_dump(mode="json"), allow_unicode=True, sort_keys=False)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return hashlib.sha256(target.read_bytes()).hexdigest()
