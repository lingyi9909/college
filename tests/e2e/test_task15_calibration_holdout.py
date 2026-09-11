from __future__ import annotations

import json
from pathlib import Path

from college_builder.config import PipelineConfig
from college_builder.pipeline.runner import GatePipelineProcessor, PipelineRunner
from college_builder.providers.base import ModelClassificationRequest, ModelDecision
from college_builder.source.gold import GoldDatasetAdapter


class PassProvider:
    provider = "holdout-regression"

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
        if request.task in {"gate_5_qa_alignment", "independent_correctness_verification"}:
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


def _source_config(path: Path, role: str) -> dict[str, object]:
    return {
        "dataset_name": "stemq",
        "data_file": str(path),
        "source_url": "https://gold.invalid/stemq",
        "revision": "git:" + "d" * 40,
        "gold_role": role,
        "language": "en",
        "field_mapping": {
            "source_id": "id",
            "question": "question",
            "answer": "answer",
            "solution": "solution",
        },
        "license_metadata": {"declared": "CC-BY-4.0"},
    }


def _run(tmp_path: Path, role: str):
    source = tmp_path / f"{role}.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": role,
                "question": "Compute 2 + 2.",
                "answer": "4",
                "solution": "Add the two values. Therefore 4",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    config = PipelineConfig.load(Path("config/pilot-5k-frozen.yaml"))
    primary = PassProvider("primary")
    verifier = PassProvider("verifier")
    workspace = tmp_path / f"workspace-{role}"
    processor = GatePipelineProcessor(
        config=config,
        cache_path=workspace / "cache" / "calls.sqlite3",
        primary_provider=primary,
        verifier_provider=verifier,
        project_root=Path("."),
    )
    result = PipelineRunner(config=config, workspace=workspace, processor=processor).run(
        adapter=GoldDatasetAdapter(),
        source_config=_source_config(source, role),
        output_dir=tmp_path / f"out-{role}",
    )
    return result, primary, verifier


def test_calibration_gold_passes_all_quality_gates_but_is_not_training_exported(
    tmp_path: Path,
) -> None:
    result, primary, verifier = _run(tmp_path, "CALIBRATION_GOLD")
    expected = {
        "gate_1_university_stem",
        "gate_2_problem",
        "gate_4_original_analysis",
        "gate_5_qa_alignment",
        "independent_correctness_verification",
    }
    assert expected.issubset(set(primary.tasks + verifier.tasks))
    assert result.accepted_count == 0
    assert result.output_path.read_text(encoding="utf-8").strip() == ""


def test_trusted_and_high_confidence_gold_receive_no_holdout_bypass(tmp_path: Path) -> None:
    for role in ("TRUSTED_TRAINING", "HIGH_CONFIDENCE"):
        result, primary, verifier = _run(tmp_path, role)
        expected = {
            "gate_1_university_stem",
            "gate_2_problem",
            "gate_4_original_analysis",
            "gate_5_qa_alignment",
            "independent_correctness_verification",
        }
        assert expected.issubset(set(primary.tasks + verifier.tasks))
        assert result.accepted_count == 1
        assert result.output_path.read_text(encoding="utf-8").strip()
