"""Fail-closed typed configuration loading."""

from pathlib import Path
from typing import Annotated, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

NonEmptyStr = Annotated[str, Field(min_length=1)]
Score = Annotated[float, Field(ge=0.0, le=1.0)]
PositiveConcurrency = Annotated[int, Field(ge=1)]
ProviderName = Literal["fake", "openai_compatible"]


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ThresholdConfig(ConfigModel):
    university: Score
    problem: Score
    answer_extract: Score
    analysis: Score
    qa_alignment: Score


class ProviderConfig(ConfigModel):
    name: ProviderName
    model: NonEmptyStr


class ProviderSet(ConfigModel):
    classifier: ProviderConfig
    verifier: ProviderConfig


class PromptVersionConfig(ConfigModel):
    university_classify: NonEmptyStr
    problem_classify: NonEmptyStr
    analysis_classify: NonEmptyStr
    qa_alignment: NonEmptyStr
    correctness_verify: NonEmptyStr


class ConcurrencyConfig(ConfigModel):
    acquisition: PositiveConcurrency
    normalization: PositiveConcurrency
    rule_gate: PositiveConcurrency
    classifier: PositiveConcurrency
    verifier: PositiveConcurrency
    embedding_batch_workers: PositiveConcurrency


class PipelineConfig(ConfigModel):
    pipeline_version: Annotated[
        str,
        Field(min_length=1, pattern=r"^university_dataset_builder_\d+\.\d+\.\d+$"),
    ]
    config_version: NonEmptyStr
    profile: Literal["university_stem_v1"]
    thresholds: ThresholdConfig
    providers: ProviderSet
    prompt_versions: PromptVersionConfig
    concurrency: ConcurrencyConfig

    @classmethod
    def load(cls, path: Path) -> "PipelineConfig":
        """Load YAML and reject missing, malformed, unknown, or invalid configuration."""
        text = path.read_text(encoding="utf-8")
        loaded: object = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError("pipeline configuration must be a YAML mapping")
        return cls.model_validate(loaded)
