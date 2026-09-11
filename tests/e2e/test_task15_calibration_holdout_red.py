from __future__ import annotations

import json
from pathlib import Path

from college_builder.config import PipelineConfig
from college_builder.pipeline.runner import GatePipelineProcessor, PipelineRunner
from college_builder.providers.base import ModelClassificationRequest, ModelDecision
from college_builder.source.gold import GoldDatasetAdapter


class PassProvider:
    provider = "holdout-red"

    def __init__(self, model: str) -> None:
        self.model = model
        self.tasks: list[str] = []

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        self.tasks.append(request.task)
        question = str(request.inputs.get("question", ""))
        answer = str(request.inputs.get("answer", ""))
        analysis = str(request.inputs.get("analysis", ""))
        if request.task == "gate_1_university_stem":
            return ModelDecision(
                label="UNIVERSITY_STEM",
                score=0.999,
                evidence_references=("question:0-1",),
                reason_code="PASS",
            )
        if request.task == "gate_2_problem":
            return ModelDecision(
                label="CALCULATION",
                score=0.999,
                evidence_references=("question:0-1",),
                reason_code="PASS",
            )
        if request.task == "gate_4_original_analysis":
            return ModelDecision(
                label="STEP_BY_STEP",
                score=0.999,
                evidence_references=(f"analysis:0-{len(analysis)}",),
                reason_code="PASS",
            )
        if request.task in {
            "gate_5_qa_alignment",
            "independent_correctness_verification",
        }:
            return ModelDecision(
                label="PASS",
                score=0.999,
                evidence_references=(
                    f"question:0-{len(question)}",
                    f"answer:0-{len(answer)}",
                    f"analysis:0-{len(analysis)}",
                ),
                reason_code="PASS",
            )
        raise AssertionError(request.task)


def _config() -> PipelineConfig:
    return PipelineConfig.load(Path("config/pilot-5k-frozen.yaml"))


def test_calibration_gold_passes_quality_but_is_not_training_exported(tmp_path: Path) -> None:
    source = tmp_path / "gold.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "cal-1",
                "question": "Compute 2 + 2.",
                "answer": "4",
                "solution": "Add the two values. Therefore 4",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_config = {
        "dataset_name": "stemq",
        "data_file": str(source),
        "source_url": "https://gold.invalid/stemq",
        "revision": "git:" + "d" * 40,
        "gold_role": "CALIBRATION_GOLD",
        "language": "en",
        "field_mapping": {
            "source_id": "id",
            "question": "question",
            "answer": "answer",
            "solution": "solution",
        },
        "license_metadata": {"declared": "CC-BY-4.0"},
    }
    primary = PassProvider("primary")
    verifier = PassProvider("verifier")
    workspace = tmp_path / "workspace"
    processor = GatePipelineProcessor(
        config=_config(),
        cache_path=workspace / "cache" / "calls.sqlite3",
        primary_provider=primary,
        verifier_provider=verifier,
        project_root=Path("."),
    )
    result = PipelineRunner(config=_config(), workspace=workspace, processor=processor).run(
        adapter=GoldDatasetAdapter(), source_config=source_config, output_dir=tmp_path / "out"
    )

    expected_tasks = {
        "gate_1_university_stem",
        "gate_2_problem",
        "gate_4_original_analysis",
        "gate_5_qa_alignment",
        "independent_correctness_verification",
    }
    assert expected_tasks.issubset(set(primary.tasks + verifier.tasks))
    assert result.accepted_count == 0
    assert result.output_path.read_text(encoding="utf-8").strip() == ""
