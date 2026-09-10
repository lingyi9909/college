"""Deterministic Task 15 pilot sampling, evidence, and configuration freeze helpers."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Annotated

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from college_builder.config import PipelineConfig
from college_builder.domain.source import RawSourceRecord
from college_builder.source.stackmathqa import OFFICIAL_SOURCE_FILES, StackMathQAAdapter

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
        if not self.quotas:
            raise ValueError("pilot quotas must not be empty")
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
    sampled_raw_sha256: tuple[SHA256Hex, ...]
    source_revisions: dict[str, str]
    source_file_sha256: dict[str, SHA256Hex] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_manifest(self) -> PilotSampleManifest:
        if self.total != sum(self.quotas.values()):
            raise ValueError("pilot manifest total does not equal quota sum")
        if self.total != len(self.sampled_record_ids):
            raise ValueError("pilot manifest total does not equal sampled record count")
        if self.total != len(self.sampled_raw_sha256):
            raise ValueError("pilot manifest total does not equal sampled hash count")
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
    """Select exact deterministic strata with memory bounded by the requested quotas."""
    sampled = _bounded_priority_sample(records, plan)
    record_ids = tuple(record.record_id for record in sampled)
    raw_hashes = tuple(record.raw_sha256 for record in sampled)
    digest = hashlib.sha256(
        json.dumps(
            {
                "seed": plan.seed,
                "quotas": plan.quotas,
                "records": list(zip(record_ids, raw_hashes, strict=True)),
            },
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
        sampled_raw_sha256=raw_hashes,
        source_revisions=dict(source_revisions),
    )
    return PilotSample(records=sampled, manifest=manifest)


def sample_stackmathqa_source_root(
    source_root: Path,
    *,
    source_revision: str,
    plan: FiveKPilotPlan,
) -> PilotSample:
    """Stream the four official StackMathQA files and materialize the approved sample."""
    if not _IMMUTABLE_REVISION.fullmatch(source_revision):
        raise ValueError("pilot source revision must be explicitly immutable")
    root = Path(source_root)
    if not root.is_dir():
        raise ValueError(f"StackMathQA source root does not exist: {root}")

    file_hashes: dict[str, str] = {}
    for filename in OFFICIAL_SOURCE_FILES:
        source_file = root / filename
        if not source_file.is_file():
            raise ValueError(f"required StackMathQA source file is missing: {filename}")
        file_hashes[filename] = _sha256_file(source_file)

    sample = build_pilot_sample(
        _iter_stackmathqa_records(root, source_revision),
        plan=plan,
        source_revisions={"stackmathqa": source_revision},
    )
    manifest = sample.manifest.model_copy(update={"source_file_sha256": file_hashes})
    return PilotSample(records=sample.records, manifest=manifest)


def write_pilot_sample(
    sample: PilotSample,
    *,
    output_path: Path,
    manifest_path: Path,
) -> None:
    """Materialize runtime raw rows and a safe manifest without committing either implicitly."""
    output = Path(output_path)
    manifest = Path(manifest_path)
    if output.resolve() == manifest.resolve():
        raise ValueError("pilot sample output and manifest paths must differ")
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    output_tmp = output.with_suffix(output.suffix + ".tmp")
    manifest_tmp = manifest.with_suffix(manifest.suffix + ".tmp")
    output_text = "".join(
        json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
        for record in sample.records
    )
    manifest_text = (
        json.dumps(
            sample.manifest.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    output_tmp.write_text(output_text, encoding="utf-8")
    manifest_tmp.write_text(manifest_text, encoding="utf-8")
    output_tmp.replace(output)
    manifest_tmp.replace(manifest)


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


def _bounded_priority_sample(
    records: Iterable[RawSourceRecord], plan: FiveKPilotPlan
) -> tuple[RawSourceRecord, ...]:
    heaps: dict[str, list[tuple[int, str, str, RawSourceRecord]]] = {
        site: [] for site in plan.quotas
    }
    counts = {site: 0 for site in plan.quotas}
    seen_ids: set[str] = set()
    for record in records:
        site = record.metadata.get("source_site")
        if not isinstance(site, str) or site not in plan.quotas:
            continue
        if record.record_id in seen_ids:
            raise ValueError(f"duplicate pilot source record_id: {record.record_id}")
        seen_ids.add(record.record_id)
        counts[site] += 1
        priority = int(
            hashlib.sha256(f"{plan.seed}\0{site}\0{record.record_id}".encode()).hexdigest(),
            16,
        )
        entry = (-priority, record.record_id, record.raw_sha256, record)
        heap = heaps[site]
        quota = plan.quotas[site]
        if len(heap) < quota:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    for site, quota in plan.quotas.items():
        if counts[site] < quota:
            raise ValueError(
                f"insufficient records for {site}: required {quota}, found {counts[site]}"
            )

    selected: list[RawSourceRecord] = []
    for site in plan.quotas:
        candidates = (
            (-priority, record_id, raw_hash, record)
            for priority, record_id, raw_hash, record in heaps[site]
        )
        ordered = sorted(candidates, key=lambda item: (item[0], item[1], item[2]))
        selected.extend(item[3] for item in ordered)
    return tuple(selected)


def _iter_stackmathqa_records(root: Path, source_revision: str) -> Iterator[RawSourceRecord]:
    for filename in OFFICIAL_SOURCE_FILES:
        source_file = root / filename
        adapter = StackMathQAAdapter()
        descriptor = tuple(
            adapter.discover(
                {
                    "data_file": str(source_file),
                    "revision": source_revision,
                    "license_metadata": {
                        "declared": "CC-BY-4.0",
                        "status": "UNREVIEWED",
                    },
                }
            )
        )[0]
        yield from adapter.acquire(descriptor)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
