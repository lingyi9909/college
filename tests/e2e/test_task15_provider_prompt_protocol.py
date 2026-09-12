from pathlib import Path

from college_builder.config import PipelineConfig

_TASK15_CONFIG = Path("config/pilot-5k-frozen.yaml")
_TASK16_CONFIG = Path("config/task16-recertification.yaml")


def _analysis_prompt(config_path: Path) -> tuple[PipelineConfig, str]:
    config = PipelineConfig.load(config_path)
    version = config.prompt_versions.analysis_classify
    prompt = Path("prompts/analysis_classify") / f"{version}.txt"
    return config, prompt.read_text(encoding="utf-8")


def test_task15_analysis_prompt_declares_json_mode_contract() -> None:
    _, text = _analysis_prompt(_TASK15_CONFIG)

    assert "json" in text.lower(), (
        "OpenAI-compatible JSON mode requires the prompt to explicitly declare JSON output"
    )


def test_task15_analysis_prompt_preserves_legacy_ambiguous_span_contract() -> None:
    config, text = _analysis_prompt(_TASK15_CONFIG)
    lowered = text.lower()

    assert config.config_version == "pilot-5k-frozen-v1"
    assert config.prompt_versions.analysis_classify == "v2"
    assert "analysis:<span>" in text
    assert "0-based" not in lowered
    assert "end-exclusive" not in lowered
    assert "[start,end)" not in lowered


def test_task16_analysis_prompt_defines_strict_half_open_span_contract() -> None:
    config, text = _analysis_prompt(_TASK16_CONFIG)
    lowered = text.lower()

    assert config.config_version == "task16-recertification-v1"
    assert config.prompt_versions.analysis_classify == "v3"
    assert "analysis:<start>-<end>" in text
    assert "0-based" in lowered
    assert "end-exclusive" in lowered
    assert "[start,end)" in lowered
    assert "python string offsets" in lowered
    assert "analysis[start:end]" in text
    assert "never generate" in lowered


def test_task16_config_does_not_mutate_task15_prompt_identity() -> None:
    task15 = PipelineConfig.load(_TASK15_CONFIG)
    task16 = PipelineConfig.load(_TASK16_CONFIG)

    assert task15.config_version == "pilot-5k-frozen-v1"
    assert task15.prompt_versions.analysis_classify == "v2"
    assert task16.config_version != task15.config_version
    assert task16.prompt_versions.analysis_classify == "v3"
