from pathlib import Path

import pytest
from pydantic import ValidationError

from college_builder.config import PipelineConfig

VALID_CONFIG = """
pipeline_version: university_dataset_builder_0.1.0
config_version: pilot_v1
profile: university_stem_v1
thresholds:
  university: 0.98
  problem: 0.98
  answer_extract: 0.995
  analysis: 0.98
  qa_alignment: 0.995
providers:
  classifier:
    name: openai_compatible
    model: classifier-v1
  verifier:
    name: openai_compatible
    model: verifier-v1
prompt_versions:
  university_classify: university_v1
  problem_classify: problem_v1
  analysis_classify: analysis_v1
  qa_alignment: qa_alignment_v1
  correctness_verify: correctness_v1
concurrency:
  acquisition: 8
  normalization: 16
  rule_gate: 32
  classifier: 16
  verifier: 8
  embedding_batch_workers: 4
"""


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "pilot.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_pipeline_config_loads_typed_valid_yaml(tmp_path: Path) -> None:
    config = PipelineConfig.load(_write(tmp_path, VALID_CONFIG))

    assert config.pipeline_version == "university_dataset_builder_0.1.0"
    assert config.profile == "university_stem_v1"
    assert config.thresholds.university == 0.98
    assert config.providers.verifier.model == "verifier-v1"
    assert config.concurrency.classifier == 16


@pytest.mark.parametrize(
    "invalid_fragment,replacement",
    [
        ("pipeline_version: university_dataset_builder_0.1.0", "pipeline_version: ''"),
        ("profile: university_stem_v1", "profile: ''"),
        ("university: 0.98", "university: 1.01"),
        ("name: openai_compatible", "name: ''"),
        ("university_classify: university_v1", "university_classify: ''"),
        ("classifier: 16", "classifier: 0"),
    ],
)
def test_pipeline_config_fails_closed_on_invalid_values(
    tmp_path: Path, invalid_fragment: str, replacement: str
) -> None:
    path = _write(tmp_path, VALID_CONFIG.replace(invalid_fragment, replacement, 1))

    with pytest.raises((ValidationError, ValueError)):
        PipelineConfig.load(path)


def test_pipeline_config_fails_closed_on_unknown_top_level_key(tmp_path: Path) -> None:
    path = _write(tmp_path, VALID_CONFIG + "unexpected: true\n")

    with pytest.raises(ValidationError):
        PipelineConfig.load(path)


def test_pipeline_config_fails_closed_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PipelineConfig.load(tmp_path / "missing.yaml")


def test_pipeline_config_fails_closed_on_non_mapping_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        PipelineConfig.load(_write(tmp_path, "- not\n- a\n- mapping\n"))
