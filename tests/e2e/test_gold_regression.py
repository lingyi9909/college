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
                evidence_references=("question:gold",),
                reason_code="GOLD_UNIVERSITY_STEM",
            )
        if request.task == "gate_2_problem":
            label = "CALCULATION"
            lowered = question.lower()
            if lowered.startswith("explain"):
                label = "CONCEPTUAL"
            elif lowered.startswith("derive"):
                label = "DERIVATION"
            elif lowered.startswith("prove"):
                label = "PROOF"
            return ModelDecision(
                label=label,
                score=0.999,
                evidence_references=("question:gold",),
                reason_code="GOLD_PROBLEM",
            )
        if request.task == "gate_4_original_analysis":
            lowered = question.lower()
            label = "STEP_BY_STEP"
            if lowered.startswith("explain"):
                label = "CONCEPTUAL_REASONING"
            elif lowered.startswith("derive"):
                label = "DERIVATION"
            elif lowered.startswith("prove"):
                label = "PROOF"
            return ModelDecision(
                label=label,
                score=0.999,
                evidence_references=(f"analysis:0-{len(analysis)}",),
                reason_code="GOLD_ANALYSIS",
            )
        if request.task == "gate_5_qa_alignment":
            if "unrelated integral" in analysis.lower():
                return ModelDecision(
                    label="FAIL",
                    score=0.999,
                    evidence_references=(
                        f"question:0-{len(question)}",
                        f"answer:0-{len(answer)}",
                        f"analysis:0-{len(analysis)}",
                    ),
                    reason_code="GOLD_KNOWN_MISMATCH",
                )
            return _pass_verification(question, answer, analysis)
        if request.task == "independent_correctness_verification":
            return _pass_verification(question, answer, analysis)
        raise AssertionError(request.task)


def _pass_verification(question: str, answer: str, analysis: str) -> ModelDecision:
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


def _config() -> PipelineConfig:
    return PipelineConfig.load(Path("config/pilot-5k-frozen.yaml"))


def _source_config(path: Path, dataset: str, role: str) -> dict[str, object]:
    return {
        "dataset_name": dataset,
        "data_file": str(path),
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


def _run(
    tmp_path: Path,
    *,
    case_id: str,
    dataset: str,
    role: str,
    question: str,
    answer: str,
    solution: str,
):
    data = tmp_path / f"{case_id}.jsonl"
    data.write_text(
        json.dumps(
            {
                "id": case_id,
                "question": question,
                "answer": answer,
                "solution": solution,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    primary = GoldRegressionProvider(f"{case_id}-primary")
    verifier = GoldRegressionProvider(f"{case_id}-verifier")
    workspace = tmp_path / f"workspace-{case_id}"
    processor = GatePipelineProcessor(
        config=_config(),
        cache_path=workspace / "cache" / "calls.sqlite3",
        primary_provider=primary,
        verifier_provider=verifier,
        project_root=Path("."),
    )
    result = PipelineRunner(config=_config(), workspace=workspace, processor=processor).run(
        adapter=GoldDatasetAdapter(),
        source_config=_source_config(data, dataset, role),
        output_dir=tmp_path / f"output-{case_id}",
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    return result, report, primary, verifier


def _single_export(result) -> dict[str, object]:
    lines = [line for line in result.output_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert len(row) == 19
    return row


def test_stemq_calibration_gold_runs_all_quality_gates_but_never_exports(tmp_path: Path) -> None:
    result, report, primary, verifier = _run(
        tmp_path,
        case_id="stemq-calculation",
        dataset="stemq",
        role="CALIBRATION_GOLD",
        question="Compute 2 + 2.",
        answer="4",
        solution="Add two and two. Therefore 4",
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
    assert report["funnel"]["final_accepted"] == 0
    assert result.output_path.read_text(encoding="utf-8").strip() == ""


def test_scibench_conceptual_gold_preserves_source_and_exact_19_fields(tmp_path: Path) -> None:
    question = "Explain why the acceleration of an object is zero when its velocity is constant."
    answer = "Constant velocity has zero time derivative."
    solution = (
        "Acceleration is the time derivative of velocity. A constant has derivative zero, "
        "so the acceleration is zero."
    )
    result, _, _, _ = _run(
        tmp_path,
        case_id="scibench-conceptual",
        dataset="scibench",
        role="TRUSTED_TRAINING",
        question=question,
        answer=answer,
        solution=solution,
    )
    assert result.accepted_count == 1
    exported = _single_export(result)
    assert exported["text_question"] == question
    assert exported["text_answer"] == answer
    assert exported["answer_analysis"] == solution


def test_cfe_derivation_gold_preserves_source_and_exact_19_fields(tmp_path: Path) -> None:
    question = "Derive the constant-acceleration relation between final and initial velocity."
    answer = "v = u + a*t"
    solution = "Integrate dv/dt = a over time t from initial velocity u. Therefore v = u + a*t"
    result, _, _, _ = _run(
        tmp_path,
        case_id="cfe-derivation",
        dataset="cfe",
        role="HIGH_CONFIDENCE",
        question=question,
        answer=answer,
        solution=solution,
    )
    assert result.accepted_count == 1
    exported = _single_export(result)
    assert exported["text_question"] == question
    assert exported["text_answer"] == answer
    assert exported["answer_analysis"] == solution


def test_gold_extracts_final_answer_from_original_analysis_without_generation(
    tmp_path: Path,
) -> None:
    question = "Compute 3 + 3."
    solution = "Adding the two integers gives the sum. Therefore 6"
    result, _, _, _ = _run(
        tmp_path,
        case_id="analysis-answer-extraction",
        dataset="scibench",
        role="TRUSTED_TRAINING",
        question=question,
        answer="",
        solution=solution,
    )
    assert result.accepted_count == 1
    exported = _single_export(result)
    assert exported["text_answer"] == "6"
    assert exported["answer_analysis"] == solution


def test_gold_rejects_answer_only_analysis_as_too_shallow(tmp_path: Path) -> None:
    result, report, _, _ = _run(
        tmp_path,
        case_id="shallow-analysis",
        dataset="scibench",
        role="TRUSTED_TRAINING",
        question="Compute 2 + 2.",
        answer="4",
        solution="4",
    )
    assert result.accepted_count == 0
    assert report["reject_reason_counts"] == {"ANALYSIS_TOO_SHALLOW": 1}
    assert result.output_path.read_text(encoding="utf-8").strip() == ""


def test_gold_rejects_known_question_analysis_mismatch(tmp_path: Path) -> None:
    result, report, _, _ = _run(
        tmp_path,
        case_id="qa-mismatch",
        dataset="cfe",
        role="HIGH_CONFIDENCE",
        question="Differentiate x^2 with respect to x.",
        answer="2*x",
        solution="This is an unrelated integral discussion. Therefore 7",
    )
    assert result.accepted_count == 0
    assert report["reject_reason_counts"] == {"QA_ALIGNMENT_MISMATCH": 1}
    assert result.output_path.read_text(encoding="utf-8").strip() == ""


def test_gold_rejects_deterministically_wrong_calculation_answer(tmp_path: Path) -> None:
    result, report, _, _ = _run(
        tmp_path,
        case_id="wrong-calculation",
        dataset="stemq",
        role="TRUSTED_TRAINING",
        question="Compute 2 + 2.",
        answer="5",
        solution="Add the two values. Therefore 5",
    )
    assert result.accepted_count == 0
    assert report["reject_reason_counts"] == {"DETERMINISTIC_CORRECTNESS_MISMATCH": 1}
    assert result.output_path.read_text(encoding="utf-8").strip() == ""
