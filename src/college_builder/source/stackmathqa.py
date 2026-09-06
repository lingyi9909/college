"""StackMathQA source mapping and deterministic source-site stratified sampling."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Annotated, Literal, cast
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from college_builder.domain.source import JsonLike, RawSourceRecord
from college_builder.source.base import AdapterCheckpoint, SourceDescriptor

NonEmptyStr = Annotated[str, Field(min_length=1)]
StackMathQASite = Literal["math", "mathoverflow", "statistics", "physics"]
PILOT_STRATA: tuple[StackMathQASite, ...] = (
    "math",
    "mathoverflow",
    "statistics",
    "physics",
)
OFFICIAL_SOURCE_FILES: dict[str, tuple[StackMathQASite, str]] = {
    "math.stackexchange.com.jsonl": ("math", "math.stackexchange.com"),
    "mathoverflow.net.jsonl": ("mathoverflow", "mathoverflow.net"),
    "stats.stackexchange.com.jsonl": ("statistics", "stats.stackexchange.com"),
    "physics.stackexchange.com.jsonl": ("physics", "physics.stackexchange.com"),
}


class StackMathQAConfig(BaseModel):
    """Source-only acquisition settings for a StackMathQA export."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    data_file: NonEmptyStr
    revision: NonEmptyStr | None = None
    dataset_url: NonEmptyStr = "https://huggingface.co/datasets/math-ai/StackMathQA"
    license_metadata: dict[str, JsonLike] = Field(default_factory=dict)


class StackMathQARow(BaseModel):
    """Official stackmathqafull-1q1a row at the provider boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    Q: str
    A: str
    meta: dict[str, JsonLike]


class StackMathQAAdapter:
    """Acquire StackMathQA pairs without university, course, or quality decisions."""

    def __init__(self, *, resume_from: AdapterCheckpoint | None = None) -> None:
        self._resume_from = resume_from
        self._config: StackMathQAConfig | None = None
        self._descriptor: SourceDescriptor | None = None
        self._checkpoint: AdapterCheckpoint | None = resume_from

    def discover(self, config: object) -> Iterable[SourceDescriptor]:
        validated = StackMathQAConfig.model_validate(config).model_copy(deep=True)
        self._official_source_identity(validated.data_file)
        descriptor = SourceDescriptor(
            source_type="dataset",
            source_dataset="stackmathqa",
            source_url=validated.dataset_url,
            locator=validated.data_file,
            shard=validated.data_file,
            source_revision=validated.revision,
        )
        self._config = validated
        self._descriptor = descriptor

        if self._resume_from is not None:
            checkpoint = self._resume_from
            if checkpoint.descriptor != descriptor:
                raise ValueError("checkpoint descriptor does not match StackMathQA source")
            if checkpoint.source_revision != descriptor.source_revision:
                raise ValueError("checkpoint source revision does not match StackMathQA source")
            if checkpoint.next_shard not in {descriptor.shard, None}:
                raise ValueError("checkpoint shard does not match StackMathQA source")
        else:
            self._checkpoint = AdapterCheckpoint(
                descriptor=descriptor,
                next_row_offset=0,
                next_shard=descriptor.shard,
                source_revision=descriptor.source_revision,
            )

        return (descriptor,)

    def acquire(self, descriptor: SourceDescriptor) -> Iterable[RawSourceRecord]:
        config = self._require_config()
        if descriptor != self._descriptor:
            raise ValueError("descriptor was not produced by the current discover call")

        start_offset = 0
        if self._resume_from is not None:
            start_offset = self._resume_from.next_row_offset

        total_rows = 0
        for row_offset, raw_row in enumerate(self._load_rows(descriptor.locator)):
            total_rows = row_offset + 1
            if row_offset < start_offset:
                continue
            record = self._map_row(config, descriptor, row_offset, raw_row)
            self._checkpoint = AdapterCheckpoint(
                descriptor=descriptor,
                next_row_offset=row_offset + 1,
                next_shard=descriptor.shard,
                source_revision=descriptor.source_revision,
            )
            yield record

        if start_offset > total_rows:
            raise ValueError("checkpoint row offset exceeds current StackMathQA source length")
        self._checkpoint = AdapterCheckpoint(
            descriptor=descriptor,
            next_row_offset=total_rows,
            next_shard=None,
            source_revision=descriptor.source_revision,
        )

    def checkpoint(self) -> AdapterCheckpoint:
        if self._checkpoint is None:
            raise RuntimeError("adapter has not discovered a source")
        return self._checkpoint

    def _map_row(
        self,
        config: StackMathQAConfig,
        descriptor: SourceDescriptor,
        row_offset: int,
        raw_row: Mapping[str, JsonLike],
    ) -> RawSourceRecord:
        row = StackMathQARow.model_validate(raw_row)
        source_site, expected_host = self._official_source_identity(descriptor.locator)
        source_url = self._required_source_url(row.meta)
        self._validate_source_host(source_site, expected_host, source_url)
        question_id = self._question_identity(source_url)
        answer_id = self._answer_identity(row.meta)
        source_id = f"{source_site}:{question_id}:{answer_id}"
        raw_sha256 = self._raw_sha256(raw_row)
        metadata = self._metadata(row.meta, source_site, descriptor, row_offset)

        return RawSourceRecord.model_validate(
            {
                "record_id": self._record_id(source_id),
                "source_type": "dataset",
                "source_dataset": "stackmathqa",
                "source_id": source_id,
                "source_url": source_url,
                "raw_question": row.Q,
                "raw_answer": "",
                "raw_analysis": row.A,
                "raw_payload": raw_row,
                "metadata": metadata,
                "license_metadata": config.license_metadata,
                "raw_sha256": raw_sha256,
            }
        )

    def _load_rows(self, locator: str) -> Iterator[Mapping[str, JsonLike]]:
        for line_number, line in enumerate(self._iter_lines(locator), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line, parse_constant=self._reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(
                    f"invalid StackMathQA JSONL at line {line_number}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"StackMathQA JSONL line {line_number} must be an object"
                )
            yield cast(dict[str, JsonLike], parsed)

    @staticmethod
    def _iter_lines(locator: str) -> Iterator[str]:
        parsed = urlparse(locator)
        if parsed.scheme in {"http", "https"}:
            with httpx.stream(
                "GET", locator, timeout=30.0, follow_redirects=True
            ) as response:
                response.raise_for_status()
                yield from response.iter_lines()
            return
        if parsed.scheme:
            raise ValueError(f"unsupported StackMathQA locator scheme: {parsed.scheme}")
        with Path(locator).open("r", encoding="utf-8") as source_file:
            yield from source_file

    @staticmethod
    def _official_source_identity(locator: str) -> tuple[StackMathQASite, str]:
        filename = Path(urlparse(locator).path).name
        identity = OFFICIAL_SOURCE_FILES.get(filename)
        if identity is None:
            raise ValueError(
                f"unrecognized official StackMathQA source file: {filename or locator}"
            )
        return identity

    @staticmethod
    def _required_source_url(meta: Mapping[str, JsonLike]) -> str:
        source_url = meta.get("url")
        if not isinstance(source_url, str) or not source_url:
            raise ValueError("StackMathQA meta.url is required for source identity")
        return source_url

    @staticmethod
    def _validate_source_host(
        source_site: StackMathQASite,
        expected_host: str,
        source_url: str,
    ) -> None:
        actual_host = (urlparse(source_url).hostname or "").lower()
        if actual_host != expected_host:
            raise ValueError(
                f"source site {source_site} does not match StackMathQA meta.url host "
                f"{actual_host or '<missing>'}"
            )

    @staticmethod
    def _question_identity(source_url: str) -> str:
        path_parts = [part for part in urlparse(source_url).path.split("/") if part]
        for marker in ("questions", "q"):
            if marker not in path_parts:
                continue
            marker_index = path_parts.index(marker)
            if marker_index + 1 >= len(path_parts):
                break
            question_id = path_parts[marker_index + 1]
            if question_id.isdigit() and int(question_id) > 0:
                return question_id
            break
        raise ValueError(
            f"cannot derive stable StackMathQA question identity from meta.url: {source_url}"
        )

    @staticmethod
    def _answer_identity(meta: Mapping[str, JsonLike]) -> int:
        answer_id = meta.get("answer_id")
        if isinstance(answer_id, bool) or not isinstance(answer_id, int) or answer_id <= 0:
            raise ValueError("StackMathQA meta.answer_id must be a positive integer")
        return answer_id

    @staticmethod
    def _metadata(
        source_meta: Mapping[str, JsonLike],
        source_site: StackMathQASite,
        descriptor: SourceDescriptor,
        row_offset: int,
    ) -> dict[str, JsonLike]:
        for reserved_key in ("source_site", "acquisition"):
            if reserved_key in source_meta:
                raise ValueError(
                    f"StackMathQA meta contains reserved provenance key: {reserved_key}"
                )
        metadata = dict(source_meta)
        metadata["source_site"] = source_site
        metadata["acquisition"] = {
            "adapter": "stackmathqa",
            "row_offset": row_offset,
            "source_revision": descriptor.source_revision,
            "source_descriptor": descriptor.model_dump(mode="json"),
        }
        return metadata

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is not allowed: {value}")

    @staticmethod
    def _record_id(source_id: str) -> str:
        digest = hashlib.sha256(f"stackmathqa\0{source_id}".encode()).hexdigest()
        return f"raw_stackmathqa_{digest}"

    @staticmethod
    def _raw_sha256(row: Mapping[str, JsonLike]) -> str:
        canonical = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
        return hashlib.sha256(canonical).hexdigest()

    def _require_config(self) -> StackMathQAConfig:
        if self._config is None:
            raise RuntimeError("discover must be called before acquire")
        return self._config


def stratified_sample(
    records: Iterable[RawSourceRecord],
    quotas: Mapping[str, int],
    seed: int,
) -> list[RawSourceRecord]:
    """Sample exact StackMathQA source-site quotas reproducibly and fail closed."""
    requested = _validate_quotas(quotas)
    groups: dict[StackMathQASite, list[RawSourceRecord]] = {
        stratum: [] for stratum in PILOT_STRATA
    }
    seen_record_ids: set[str] = set()

    for record in records:
        if record.source_dataset != "stackmathqa":
            raise ValueError(f"non-StackMathQA record supplied to sampler: {record.record_id}")
        if record.record_id in seen_record_ids:
            raise ValueError(f"duplicate record_id supplied to sampler: {record.record_id}")
        seen_record_ids.add(record.record_id)

        source_site = record.metadata.get("source_site")
        if not isinstance(source_site, str) or source_site not in PILOT_STRATA:
            raise ValueError(f"invalid or missing StackMathQA source_site: {record.record_id}")
        groups[source_site].append(record)

    sampled: list[RawSourceRecord] = []
    for stratum in PILOT_STRATA:
        if stratum not in requested:
            continue
        count = requested[stratum]
        candidates = sorted(groups[stratum], key=lambda item: item.record_id)
        if len(candidates) < count:
            raise ValueError(
                f"stratum {stratum} requested {count} records but only available "
                f"{len(candidates)}"
            )
        rng = random.Random(_stratum_seed(seed, stratum))
        chosen = rng.sample(candidates, count)
        sampled.extend(sorted(chosen, key=lambda item: item.record_id))

    return sampled


def _validate_quotas(quotas: Mapping[str, int]) -> dict[StackMathQASite, int]:
    validated: dict[StackMathQASite, int] = {}
    for stratum, count in quotas.items():
        if stratum not in PILOT_STRATA:
            raise ValueError(f"unsupported StackMathQA stratum: {stratum}")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"quota for {stratum} must be a non-negative integer")
        validated[stratum] = count
    return validated


def _stratum_seed(seed: int, stratum: StackMathQASite) -> int:
    material = f"{seed}\0{stratum}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")
