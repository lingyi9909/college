from __future__ import annotations

from pathlib import Path

import yaml


def test_gate5_v3_prompts_require_formal_answer_completeness_not_analysis_substitution() -> None:
    for path in (
        Path("prompts/qa_alignment/v3.txt"),
        Path("prompts/correctness_verify/v3.txt"),
    ):
        prompt = path.read_text(encoding="utf-8").lower()
        assert "formal answer must" in prompt
        assert "all requested" in prompt
        assert "analysis must not substitute" in prompt
        assert "proof" in prompt
        assert "conclusion fragment" in prompt


def test_correctness_v3_explicitly_rejects_unfinished_required_operations() -> None:
    prompt = Path("prompts/correctness_verify/v3.txt").read_text(encoding="utf-8").lower()

    assert "orthonormal" in prompt
    assert "orthogonal" in prompt
    assert "normalization" in prompt
    assert "must fail" in prompt
    assert "can be done later" in prompt


def test_task16_recertification_selects_completeness_v3_prompts() -> None:
    config = yaml.safe_load(
        Path("config/task16-recertification.yaml").read_text(encoding="utf-8")
    )

    assert config["prompt_versions"]["qa_alignment"] == "v3"
    assert config["prompt_versions"]["correctness_verify"] == "v3"
