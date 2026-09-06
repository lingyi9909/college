"""Typed source-adapter contracts for lossless acquisition."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from college_builder.domain.source import RawSourceRecord

NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]


class SourceDescriptor(BaseModel):
    """Stable, provider-neutral description of one source shard or object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_type: NonEmptyStr
    source_dataset: NonEmptyStr
    source_url: NonEmptyStr
    locator: NonEmptyStr
    shard: str | None = None
    source_revision: str | None = None


class AdapterCheckpoint(BaseModel):
    """Resume cursor tied to the exact source descriptor and revision."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    descriptor: SourceDescriptor
    next_row_offset: NonNegativeInt
    next_shard: str | None
    source_revision: str | None


@runtime_checkable
class SourceAdapter(Protocol):
    """Source-only contract; adapters acquire raw data without quality decisions."""

    def discover(self, config: object) -> Iterable[SourceDescriptor]: ...

    def acquire(self, descriptor: SourceDescriptor) -> Iterable[RawSourceRecord]: ...

    def checkpoint(self) -> AdapterCheckpoint: ...
