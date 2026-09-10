from pathlib import Path

from college_builder.config import PipelineConfig


def test_task15_analysis_prompt_declares_json_mode_contract() -> None:
    config = PipelineConfig.load(Path("config/pilot-5k-frozen.yaml"))
    version = config.prompt_versions.analysis_classify
    prompt = Path("prompts/analysis_classify") / f"{version}.txt"
    text = prompt.read_text(encoding="utf-8")

    assert "json" in text.lower(), (
        "OpenAI-compatible JSON mode requires the prompt to explicitly declare JSON output"
    )
