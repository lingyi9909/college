"""Generic Hugging Face JSONL acquisition with explicit field mapping."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path
from typing import Annotated, cast
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field

from college_builder.domain.source import JsonLike, RawSourceRecord
from college_builder.source.base import AdapterCheckpoint, SourceDescriptor

NonEmptyStr = Annotated[str, Field(min_length=1)]
DataFiles = Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]


class HFFieldMapping(BaseModel):
    """Explicit mapping from source row fields to raw-domain fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_id: NonEmptyStr
    question: NonEmptyStr
    answer: NonEmptyStr
    analysis: NonEmptyStr
    source_url: NonEmptyStr | None = None


class HFDatasetConfig(BaseModel):
    """Dataset-agnostic Hugging Face acquisition configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: NonEmptyStr
    data_files: DataFiles
    revision: NonEmptyStr | None = None
    source_url: NonEmptyStr | None = None
    field_mapping: HFFieldMapping
    license_metadata: dict[str, JsonLike] = Field(default_factory=dict)


class HFDatasetAdapter:
    """Acquire Hugging Face JSONL rows without classification or rewriting."""

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
        self._config: HFDatasetConfig | None = None
        self._descriptors: tuple[SourceDescriptor, ...] = ()
        self._checkpoint: AdapterCheckpoint | None = resume_from

    def discover(self, config: object) -> Iterable[SourceDescriptor]:
        """Validate config and discover deterministic source descriptors."""
        validated = HFDatasetConfig.model_validate(config).model_copy(deep=True)
        self._config = validated
        descriptors = tuple(
            SourceDescriptor(
                source_type="dataset",
                source_dataset=validated.dataset,
                source_url=(
                    validated.source_url
                    or f"https://huggingface.co/datasets/{validated.dataset}"
                ),
                locator=self._resolve_locator(data_file, validated),
                shard=data_file,
                source_revision=validated.revision,
            )
            for data_file in validated.data_files
        )
        self._descriptors = descriptors

        if self._resume_from is None:
            first = descriptors[0]
            self._checkpoint = AdapterCheckpoint(
                descriptor=first,
                next_row_offset=0,
                next_shard=first.shard,
                source_revision=first.source_revision,
            )
            return descriptors

        checkpoint = self._resume_from
        if checkpoint.source_revision != checkpoint.descriptor.source_revision:
            raise ValueError("checkpoint source revision is internally inconsistent")
        for index, descriptor in enumerate(descriptors):
            if descriptor == checkpoint.descriptor:
                if checkpoint.source_revision != descriptor.source_revision:
                    raise ValueError(
                        "checkpoint source revision does not match descriptor revision"
                    )
                return descriptors[index:]
        raise ValueError("checkpoint descriptor or revision does not match discovered source")

    def acquire(self, descriptor: SourceDescriptor) -> Iterable[RawSourceRecord]:
        """Yield raw records exactly as mapped from one source descriptor."""
        config = self._require_config()
        if descriptor not in self._descriptors:
            raise ValueError("descriptor was not produced by the current discover call")

        start_offset = 0
        if self._resume_from is not None and descriptor == self._resume_from.descriptor:
            if self._resume_from.source_revision != descriptor.source_revision:
                raise ValueError("checkpoint source revision does not match descriptor revision")
            start_offset = self._resume_from.next_row_offset

        total_rows = 0
        for row_offset, row in enumerate(self._load_rows(descriptor.locator)):
            total_rows = row_offset + 1
            if row_offset < start_offset:
                continue
            record = self._map_row(config, descriptor, row_offset, row)
            self._checkpoint = AdapterCheckpoint(
                descriptor=descriptor,
                next_row_offset=row_offset + 1,
                next_shard=descriptor.shard,
                source_revision=descriptor.source_revision,
            )
            yield record

        if start_offset > total_rows:
            raise ValueError("checkpoint row offset exceeds current source length")
        self._checkpoint_after_shard(descriptor, total_rows)

    def checkpoint(self) -> AdapterCheckpoint:
        """Return the current deterministic resume cursor."""
        if self._checkpoint is None:
            raise RuntimeError("adapter has not discovered a source")
        return self._checkpoint

    @staticmethod
    def _resolve_locator(data_file: str, config: HFDatasetConfig) -> str:
        parsed = urlparse(data_file)
        if parsed.scheme in {"http", "https"}:
            return data_file
        if Path(data_file).is_file():
            return data_file
        revision = config.revision or "main"
        relative = data_file.lstrip("/")
        return (
            f"https://huggingface.co/datasets/{config.dataset}/resolve/"
            f"{revision}/{relative}"
        )

    def _load_rows(self, locator: str) -> Iterator[Mapping[str, JsonLike]]:
        text = self._load_text(locator)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                parsed = json.loads(line, parse_constant=self._reject_json_constant)
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"invalid JSONL row at line {line_number}") from exc
            if not isinstance(parsed, dict):
                raise ValueError(f"JSONL row at line {line_number} must be an object")
            yield cast(dict[str, JsonLike], parsed)

    def _load_text(self, locator: str) -> str:
        """Single isolation point for local reads and all network access."""
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
            raise ValueError(f"unsupported source locator scheme: {parsed.scheme}")
        return Path(locator).read_text(encoding="utf-8")

    @staticmethod
    def _reject_json_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant is not allowed: {value}")

    def _map_row(
        self,
        config: HFDatasetConfig,
        descriptor: SourceDescriptor,
        row_offset: int,
        row: Mapping[str, JsonLike],
    ) -> RawSourceRecord:
        mapping = config.field_mapping
        source_id = self._source_id(row, mapping.source_id)
        question = self._mapped_text(row, mapping.question)
        answer = self._mapped_text(row, mapping.answer)
        analysis = self._mapped_text(row, mapping.analysis)
        source_url = descriptor.source_url
        if mapping.source_url is not None:
            source_url = self._mapped_nonempty_text(row, mapping.source_url)

        raw_sha256 = self._raw_sha256(row)
        return RawSourceRecord.model_validate(
            {
                "record_id": self._record_id(descriptor.source_dataset, source_id),
                "source_type": "dataset",
                "source_dataset": descriptor.source_dataset,
                "source_id": source_id,
                "source_url": source_url,
                "raw_question": question,
                "raw_answer": answer,
                "raw_analysis": analysis,
                "raw_payload": row,
                "metadata": {
                    "acquisition": {
                        "adapter": "huggingface",
                        "row_offset": row_offset,
                        "shard": descriptor.shard,
                        "source_revision": descriptor.source_revision,
                        "source_descriptor": descriptor.model_dump(mode="json"),
                    }
                },
                "license_metadata": config.license_metadata,
                "raw_sha256": raw_sha256,
            }
        )

    @staticmethod
    def _mapped_text(row: Mapping[str, JsonLike], field_name: str) -> str:
        if field_name not in row:
            raise ValueError(f"mapped field is missing from source row: {field_name}")
        value = row[field_name]
        if not isinstance(value, str):
            raise ValueError(f"mapped field must be a string: {field_name}")
        return value

    @classmethod
    def _mapped_nonempty_text(cls, row: Mapping[str, JsonLike], field_name: str) -> str:
        value = cls._mapped_text(row, field_name)
        if not value:
            raise ValueError(f"mapped field must be non-empty: {field_name}")
        return value

    @staticmethod
    def _source_id(row: Mapping[str, JsonLike], field_name: str) -> str:
        if field_name not in row:
            raise ValueError(f"mapped field is missing from source row: {field_name}")
        value = row[field_name]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError(f"mapped source_id must be a string or integer: {field_name}")
        source_id = str(value)
        if not source_id:
            raise ValueError(f"mapped source_id must be non-empty: {field_name}")
        return source_id

    @staticmethod
    def _record_id(source_dataset: str, source_id: str) -> str:
        identity = f"huggingface\0{source_dataset}\0{source_id}".encode()
        return f"raw_hf_{hashlib.sha256(identity).hexdigest()}"

    @staticmethod
    def _raw_sha256(row: Mapping[str, JsonLike]) -> str:
        canonical = json.dumps(
            row,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def _checkpoint_after_shard(self, descriptor: SourceDescriptor, total_rows: int) -> None:
        index = self._descriptors.index(descriptor)
        if index + 1 < len(self._descriptors):
            next_descriptor = self._descriptors[index + 1]
            self._checkpoint = AdapterCheckpoint(
                descriptor=next_descriptor,
                next_row_offset=0,
                next_shard=next_descriptor.shard,
                source_revision=next_descriptor.source_revision,
            )
            return
        self._checkpoint = AdapterCheckpoint(
            descriptor=descriptor,
            next_row_offset=total_rows,
            next_shard=None,
            source_revision=descriptor.source_revision,
        )

    def _require_config(self) -> HFDatasetConfig:
        if self._config is None:
            raise RuntimeError("discover must be called before acquire")
        return self._config
