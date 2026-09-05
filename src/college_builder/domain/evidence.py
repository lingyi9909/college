"""Versioned quality-evidence contracts."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]


class GateVerdict(StrEnum):
    PASS = "PASS"
    REJECT = "REJECT"
    VERIFY = "VERIFY"


class GateEvidence(BaseModel):
    """Auditable result emitted by one quality gate invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    gate_name: NonEmptyStr
    verdict: GateVerdict
    score: Score
    provider: NonEmptyStr
    model: NonEmptyStr
    prompt_version: NonEmptyStr
    config_version: NonEmptyStr
