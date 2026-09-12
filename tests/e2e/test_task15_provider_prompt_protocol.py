from pathlib import Path

from college_builder.config import PipelineConfig

_TASK15_CONFIG = Path("config/pilot-5k-frozen.yaml")
_TASK16_CONFIG = Path("config/task16-recertification.yaml")


def _analysis_prompt(config_path: Path) -> tuple[PipelineConfig, str]:
    config = PipelineConfig.load(config_path)
    version = config.prompt_versions.analysis_classify
    prompt = Path("prompts/analysis_classify") / f"{version}.txt"
    return config, prompt.read_text(encoding="utf-8")


def _downstream_prompts(config_path: Path) -> tuple[PipelineConfig, str, str]:
    config = PipelineConfig.load(config_path)
    alignment = (
        Path("prompts/qa_alignment") / f"{config.prompt_versions.qa_alignment}.txt"
    ).read_text(encoding="utf-8")
    correctness = (
        Path("prompts/correctness_verify") / f"{config.prompt_versions.correctness_verify}.txt"
    ).read_text(encoding="utf-8")
    return config, alignment, correctness


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

    assert config.config_version == "task16-recertification-v2"
    assert config.prompt_versions.analysis_classify == "v3"
    assert "analysis:<start>-<end>" in text
    assert "0-based" in lowered
    assert "end-exclusive" in lowered
    assert "[start,end)" in lowered
    assert "python string offsets" in lowered
    assert "analysis[start:end]" in text
    assert "never generate" in lowered


def test_task15_downstream_prompts_preserve_legacy_ambiguous_span_contract() -> None:
    config, alignment, correctness = _downstream_prompts(_TASK15_CONFIG)

    assert config.config_version == "pilot-5k-frozen-v1"
    assert config.prompt_versions.qa_alignment == "v1"
    assert config.prompt_versions.correctness_verify == "v1"
    for text in (alignment, correctness):
        lowered = text.lower()
        assert "question:<start>-<end>" in text
        assert "answer:<start>-<end>" in text
        assert "analysis:<start>-<end>" in text
        assert "0-based" not in lowered
        assert "end-exclusive" not in lowered
        assert "[start,end)" not in lowered


def test_task16_downstream_prompts_define_strict_half_open_span_contract() -> None:
    config, alignment, correctness = _downstream_prompts(_TASK16_CONFIG)

    assert config.config_version == "task16-recertification-v2"
    assert config.prompt_versions.qa_alignment == "v2"
    assert config.prompt_versions.correctness_verify == "v2"
    for text in (alignment, correctness):
        lowered = text.lower()
        assert "question:<start>-<end>" in text
        assert "answer:<start>-<end>" in text
        assert "analysis:<start>-<end>" in text
        assert "0-based" in lowered
        assert "end-exclusive" in lowered
        assert "[start,end)" in lowered
        assert "python string offsets" in lowered
        assert "question[start:end]" in text
        assert "answer[start:end]" in text
        assert "analysis[start:end]" in text
        assert "inclusive end" in lowered
        assert "never generate" in lowered


def test_task16_config_versions_downstream_without_mutating_task15_identity() -> None:
    task15 = PipelineConfig.load(_TASK15_CONFIG)
    task16 = PipelineConfig.load(_TASK16_CONFIG)

    assert task15.config_version == "pilot-5k-frozen-v1"
    assert task15.prompt_versions.analysis_classify == "v2"
    assert task15.prompt_versions.qa_alignment == "v1"
    assert task15.prompt_versions.correctness_verify == "v1"

    assert task16.config_version == "task16-recertification-v2"
    assert task16.prompt_versions.analysis_classify == "v3"
    assert task16.prompt_versions.qa_alignment == "v2"
    assert task16.prompt_versions.correctness_verify == "v2"
