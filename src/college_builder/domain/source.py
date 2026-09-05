"""Immutable raw-source domain contracts."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

type JsonPrimitive = str | int | float | bool | None
type JsonLike = JsonPrimitive | Mapping[str, JsonLike] | list[JsonLike] | tuple[JsonLike, ...]
type JsonValue = JsonPrimitive | list[JsonValue] | dict[str, JsonValue]
type FrozenJsonValue = JsonPrimitive | tuple[FrozenJsonValue, ...] | Mapping[str, FrozenJsonValue]
type FrozenJsonObject = Mapping[str, FrozenJsonValue]

NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


def _freeze_json(value: JsonLike) -> FrozenJsonValue:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_object(value: Mapping[str, JsonLike]) -> FrozenJsonObject:
    return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})


def _thaw_json(value: FrozenJsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


class RawSourceRecord(BaseModel):
    """Lossless, deeply immutable snapshot of one source record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: NonEmptyStr
    source_type: NonEmptyStr
    source_dataset: NonEmptyStr
    source_id: NonEmptyStr
    source_url: NonEmptyStr
    raw_question: str
    raw_answer: str
    raw_analysis: str
    raw_payload: FrozenJsonObject
    metadata: FrozenJsonObject
    license_metadata: FrozenJsonObject
    raw_sha256: Sha256Hex

    @field_validator("raw_payload", "metadata", "license_metadata", mode="after")
    @classmethod
    def freeze_json_objects(cls, value: Mapping[str, JsonLike]) -> FrozenJsonObject:
        """Defensively copy nested JSON so caller mutation cannot alter provenance."""
        return _freeze_object(value)

    @field_serializer("raw_payload", "metadata", "license_metadata")
    def serialize_frozen_json(self, value: FrozenJsonObject) -> JsonValue:
        """Serialize immutable provenance back to ordinary JSON-compatible values."""
        return _thaw_json(value)


class NormalizedQA(BaseModel):
    """Normalized content linked to, but never replacing, an immutable raw record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    record_id: NonEmptyStr
    source_record_id: NonEmptyStr
    question: str
    answer: str
    analysis: str
    subject_candidates: tuple[str, ...]
    images: tuple[str, ...]
    metadata: FrozenJsonObject
    normalization_evidence: FrozenJsonObject

    @field_validator("metadata", "normalization_evidence", mode="after")
    @classmethod
    def freeze_json_objects(cls, value: Mapping[str, JsonLike]) -> FrozenJsonObject:
        """Keep normalized metadata/evidence typed and isolated from caller mutation."""
        return _freeze_object(value)

    @field_serializer("metadata", "normalization_evidence")
    def serialize_frozen_json(self, value: FrozenJsonObject) -> JsonValue:
        """Serialize immutable JSON metadata to ordinary JSON-compatible values."""
        return _thaw_json(value)
