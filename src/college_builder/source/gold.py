"""Trusted Gold dataset acquisition with explicit, non-bypassable source roles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from enum import StrEnum
from pathlib import Path
from typing import Annotated, cast
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from college_builder.domain.source import JsonLike, RawSourceRecord
from college_builder.source.base import AdapterCheckpoint, SourceDescriptor

NonEmptyStr = Annotated[str, Field(min_length=1)]


class GoldRole(StrEnum):
    """Machine-readable source role; none of these values implies final acceptance."""

    CALIBRATION_GOLD = "CALIBRATION_GOLD"
    TRUSTED_TRAINING = "TRUSTED_TRAINING"
    HIGH_CONFIDENCE = "HIGH_CONFIDENCE"


class GoldFieldMapping(BaseModel):
    """Explicit provider-row mapping for heterogeneous trusted datasets."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: NonEmptyStr
    question: NonEmptyStr
    solution: NonEmptyStr
    answer: NonEmptyStr | None = None
    source_url: NonEmptyStr | None = None
    institution: NonEmptyStr | None = None
    textbook: NonEmptyStr | None = None


class GoldDatasetConfig(BaseModel):
    """Source-only configuration for STEMQ, SciBench, CFE, and similar Gold data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_name: NonEmptyStr
    data_file: NonEmptyStr
    source_url: NonEmptyStr
    gold_role: GoldRole
    field_mapping: GoldFieldMapping
    revision: NonEmptyStr | None = None
    license_metadata: dict[str, JsonLike] = Field(default_factory=dict)


class GoldDatasetAdapter:
    """Acquire trusted-source rows without granting any downstream gate bypass."""

    def __init__(
        self,
        *,
        resume_from: AdapterCheckpoint | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._resume_from = resume_from
        self._timeout_seconds = timeout_seconds
        self._config: GoldDatasetConfig | None = None
        self._descriptor: SourceDescriptor | None = None
        self._checkpoint: AdapterCheckpoint | None = resume_from

    def discover(self, config: object) -> Iterable[SourceDescriptor]:
        validated = GoldDatasetConfig.model_validate(config).model_copy(deep=True)
        descriptor = SourceDescriptor(
            source_type="dataset",
            source_dataset=validated.dataset_name,
            source_url=validated.source_url,
            locator=validated.data_file,
            shard=validated.data_file,
            source_revision=validated.revision,
        )
        self._config = validated
        self._descriptor = descriptor
        self._validate_resume(descriptor)
        if self._resume_from is None:
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

        start_offset = self._resume_from.next_row_offset if self._resume_from else 0
        total_rows = 0
        seen_source_ids: set[str] = set()
        for row_offset, row in enumerate(self._load_rows(descriptor.locator)):
            total_rows = row_offset + 1
            source_id = self._source_id(row, config.field_mapping.source_id)
            if source_id in seen_source_ids:
                raise ValueError(f"duplicate Gold source_id: {source_id}")
            seen_source_ids.add(source_id)
            if row_offset < start_offset:
                continue
            record = self._map_row(config, descriptor, row_offset, row, source_id)
            self._checkpoint = AdapterCheckpoint(
                descriptor=descriptor,
                next_row_offset=row_offset + 1,
                next_shard=descriptor.shard,
                source_revision=descriptor.source_revision,
            )
            yield record

        if start_offset > total_rows:
            raise ValueError("checkpoint row offset exceeds current Gold source length")
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

    def _load_rows(self, locator: str) -> Iterator[Mapping[str, JsonLike]]:
        text = self._load_text(locator)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line, parse_constant=self._reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid Gold JSONL row at line {line_number}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"Gold JSONL row at line {line_number} must be an object")
            yield cast(dict[str, JsonLike], parsed)

    def _load_text(self, locator: str) -> str:
        parsed = urlparse(locator)
        if parsed.scheme in {"http", "https"}:
            response = httpx.get(
                locator,
                timeout=self._timeout_seconds,
                follow_redirects=True,
            )
            response.raise_for_status()
            return response.text
        if parsed.scheme:
            raise ValueError(f"unsupported Gold source locator scheme: {parsed.scheme}")
        return Path(locator).read_text(encoding="utf-8")

    def _map_row(
        self,
        config: GoldDatasetConfig,
        descriptor: SourceDescriptor,
        row_offset: int,
        row: Mapping[str, JsonLike],
        source_id: str,
    ) -> RawSourceRecord:
        mapping = config.field_mapping
        question = self._mapped_text(row, mapping.question)
        answer = self._mapped_text(row, mapping.answer) if mapping.answer else ""
        solution = self._mapped_text(row, mapping.solution)
        source_url = descriptor.source_url
        if mapping.source_url is not None:
            source_url = self._mapped_nonempty_text(row, mapping.source_url)

        metadata = gold_role_metadata(config.gold_role)
        if mapping.institution is not None:
            metadata["institution"] = self._mapped_text(row, mapping.institution)
        if mapping.textbook is not None:
            metadata["textbook"] = self._mapped_text(row, mapping.textbook)
        metadata["acquisition"] = {
            "adapter": "gold_dataset",
            "row_offset": row_offset,
            "shard": descriptor.shard,
            "source_revision": descriptor.source_revision,
            "source_descriptor": descriptor.model_dump(mode="json"),
        }

        return RawSourceRecord.model_validate(
            {
                "record_id": self._record_id(descriptor.source_dataset, source_id),
                "source_type": "dataset",
                "source_dataset": descriptor.source_dataset,
                "source_id": source_id,
                "source_url": source_url,
                "raw_question": question,
                "raw_answer": answer,
                "raw_analysis": solution,
                "raw_payload": row,
                "metadata": metadata,
                "license_metadata": config.license_metadata,
                "raw_sha256": self._raw_sha256(row),
            }
        )

    def _validate_resume(self, descriptor: SourceDescriptor) -> None:
        if self._resume_from is None:
            return
        checkpoint = self._resume_from
        if checkpoint.descriptor != descriptor:
            raise ValueError("checkpoint descriptor does not match Gold source")
        if checkpoint.source_revision != descriptor.source_revision:
            raise ValueError("checkpoint source revision does not match Gold source")
        if checkpoint.next_shard not in {descriptor.shard, None}:
            raise ValueError("checkpoint shard does not match Gold source")

    @staticmethod
    def _mapped_text(row: Mapping[str, JsonLike], field_name: str) -> str:
        if field_name not in row:
            raise ValueError(f"mapped field is missing from Gold source row: {field_name}")
        value = row[field_name]
        if not isinstance(value, str):
            raise ValueError(f"mapped Gold field must be a string: {field_name}")
        return value

    @classmethod
    def _mapped_nonempty_text(cls, row: Mapping[str, JsonLike], field_name: str) -> str:
        value = cls._mapped_text(row, field_name)
        if not value:
            raise ValueError(f"mapped Gold field must be non-empty: {field_name}")
        return value

    @staticmethod
    def _source_id(row: Mapping[str, JsonLike], field_name: str) -> str:
        if field_name not in row:
            raise ValueError(f"mapped field is missing from Gold source row: {field_name}")
        value = row[field_name]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"mapped Gold source_id must be string or integer: {field_name}")
        source_id = str(value)
        if not source_id:
            raise ValueError(f"mapped Gold source_id must be non-empty: {field_name}")
        return source_id

    @staticmethod
    def _record_id(dataset_name: str, source_id: str) -> str:
        identity = f"gold\0{dataset_name}\0{source_id}".encode()
        return f"raw_gold_{hashlib.sha256(identity).hexdigest()}"

    @staticmethod
    def _raw_sha256(row: Mapping[str, JsonLike]) -> str:
        encoded = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is not allowed: {value}")

    def _require_config(self) -> GoldDatasetConfig:
        if self._config is None:
            raise RuntimeError("discover must be called before acquire")
        return self._config


def gold_role_metadata(role: GoldRole) -> dict[str, JsonLike]:
    """Encode Gold provenance without implying schema/content acceptance."""
    return {
        "gold_role": role.value,
        "calibration_holdout": role is GoldRole.CALIBRATION_GOLD,
        "quality_tier_candidate": "GOLD",
    }
