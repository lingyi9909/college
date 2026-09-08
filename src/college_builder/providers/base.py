"""Provider-neutral structured classification contracts."""

from __future__ import annotations

from typing import Annotated, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from college_builder.domain.source import JsonValue

NonEmptyStr = Annotated[str, Field(min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]
NonEmptyLabels = Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]
NonEmptyEvidenceReferences = Annotated[tuple[NonEmptyStr, ...], Field(min_length=1)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ModelClassificationRequest(_StrictModel):
    """Domain request sent to any structured classification provider."""

    task: NonEmptyStr
    prompt: NonEmptyStr
    inputs: dict[str, JsonValue]
    allowed_labels: NonEmptyLabels

    @field_validator("task", "prompt")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("classification request strings must be non-blank")
        return value

    @field_validator("allowed_labels")
    @classmethod
    def reject_blank_or_duplicate_labels(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("allowed labels must be non-blank")
        if len(set(values)) != len(values):
            raise ValueError("allowed labels must be unique")
        return values


class ModelDecision(_StrictModel):
    """Strict model output: classification only, never rewritten source content."""

    label: NonEmptyStr
    score: Score
    evidence_references: NonEmptyEvidenceReferences
    reason_code: NonEmptyStr

    @field_validator("label", "reason_code")
    @classmethod
    def reject_blank_strings(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("model decision strings must be non-blank")
        return value

    @field_validator("evidence_references")
    @classmethod
    def reject_blank_evidence_references(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("evidence references must be non-blank")
        return values


@runtime_checkable
class StructuredModelProvider(Protocol):
    """Provider-neutral boundary used by quality gates."""

    provider: str
    model: str

    def classify(self, request: ModelClassificationRequest) -> ModelDecision: ...
