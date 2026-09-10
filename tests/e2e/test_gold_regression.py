from __future__ import annotations

import json
from pathlib import Path

from college_builder.config import PipelineConfig
from college_builder.pipeline.runner import GatePipelineProcessor, PipelineRunner
from college_builder.providers.base import ModelClassificationRequest, ModelDecision
from college_builder.source.gold import GoldDatasetAdapter


class GoldRegressionProvider:
    provider = "gold-regression"

    def __init__(self, model: str) -> None:
        self.model = model

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        question = str(request.inputs.get("question", ""))
        answer = str(request.inputs.get("answer", ""))
        analysis = str(request.inputs.get("analysis", ""))
        if request.task == "gate_1_university_stem":
            return ModelDecision(
                label="UNIVERSITY_STEM",
                score=0.999,
                evidence_references=("question:gold",),
                reason_code="GOLD_UNIVERSITY_STEM",
            )
        if request.task == "gate_2_problem":
            return ModelDecision(
                label="CALCULATION",
                score=0.999,
                evidence_references=("question:gold",),
                reason_code="GOLD_PROBLEM",
            )
        if request.task == "gate_4_original_analysis":
            return ModelDecision(
                label="STEP_BY_STEP",
                score=0.999,
                evidence_references=(f"analysis:0-{len(analysis)}",),
                reason_code="GOLD_ANALYSIS",
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
                reason_code="GOLD_VERIFIED",
            )
        raise AssertionError(request.task)


def _config() -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "pipeline_version": "university_dataset_builder_0.1.0",
            "config_version": "pilot-5k-frozen-v1",
            "profile": "university_stem_v1",
            "thresholds": {
                "university": 0.98,
                "problem": 0.98,
                "answer_extract": 0.995,
                "analysis": 0.98,
                "qa_alignment": 0.995,
            },
            "providers": {
                "classifier": {"name": "fake", "model": "gold-primary"},
                "verifier": {"name": "fake", "model": "gold-verifier"},
            },
            "prompt_versions": {
                "university_classify": "v1",
                "problem_classify": "v1",
                "analysis_classify": "v1",
                "qa_alignment": "v1",
                "correctness_verify": "v1",
            },
            "concurrency": {
                "acquisition": 1,
                "normalization": 1,
                "rule_gate": 1,
                "classifier": 1,
                "verifier": 1,
                "embedding_batch_workers": 1,
            },
        }
    )


def test_stemq_scibench_cfe_gold_records_survive_complete_pipeline(tmp_path: Path) -> None:
    for dataset, role in (
        ("stemq", "CALIBRATION_GOLD"),
        ("scibench", "TRUSTED_TRAINING"),
        ("cfe", "HIGH_CONFIDENCE"),
    ):
        data = tmp_path / f"{dataset}.jsonl"
        row = {
            "id": f"{dataset}-gold-1",
            "question": "Solve 2*x + 3 = 11 for x.",
            "answer": "x = 4",
            "solution": "Subtract 3 to get 2*x = 8, then divide by 2. Therefore x = 4.",
        }
        data.write_text(json.dumps(row) + "\n", encoding="utf-8")
        source_config = {
            "dataset_name": dataset,
            "data_file": str(data),
            "source_url": f"https://gold.invalid/{dataset}",
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
        primary = GoldRegressionProvider("gold-primary")
        verifier = GoldRegressionProvider("gold-verifier")
        workspace = tmp_path / f"workspace-{dataset}"
        processor = GatePipelineProcessor(
            config=_config(),
            cache_path=workspace / "cache" / "calls.sqlite3",
            primary_provider=primary,
            verifier_provider=verifier,
            project_root=Path("."),
        )
        result = PipelineRunner(config=_config(), workspace=workspace, processor=processor).run(
            adapter=GoldDatasetAdapter(),
            source_config=source_config,
            output_dir=tmp_path / f"output-{dataset}",
        )
        assert result.accepted_count == 1
        exported = json.loads(result.output_path.read_text(encoding="utf-8"))
        assert exported["text_answer"] == "x = 4"
        assert "Therefore x = 4" in exported["answer_analysis"]
