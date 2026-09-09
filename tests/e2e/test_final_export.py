from __future__ import annotations

import json
from pathlib import Path

import pytest

from college_builder.domain.evidence import GateEvidence, GateVerdict
from college_builder.domain.final_record import FinalQuestionRecord, slim_question_md5_v1
from college_builder.domain.question import (
    AnalysisContent,
    AnalysisType,
    AnswerContent,
    Classification,
    DedupState,
    Discipline,
    ProblemType,
    QualityState,
    QuestionContent,
    QuestionMetadata,
    QuestionProvenance,
    UniversityLevel,
    UniversityQuestionIR,
)
from college_builder.export.jsonl import ExportRecord, export_jsonl
from college_builder.storage.state import RunStage


def _ir(
    *,
    candidate_id: str = "candidate-1",
    question: str = "Compute 2 + 3.",
    assets: tuple[str, ...] = (),
) -> UniversityQuestionIR:
    return UniversityQuestionIR(
        candidate_id=candidate_id,
        source_record_id=f"raw-{candidate_id}",
        question=QuestionContent(raw=question, normalized=question, assets=assets),
        answer=AnswerContent(raw="5", final_answer="5", source_span="answer:0:1"),
        analysis=AnalysisContent(
            raw="Adding two and three gives five.",
            type=AnalysisType.STEP_BY_STEP,
        ),
        classification=Classification(
            discipline=Discipline.MATHEMATICS,
            course="Calculus I",
            level=UniversityLevel.UNDERGRADUATE,
            problem_type=ProblemType.CALCULATION,
        ),
        metadata=QuestionMetadata(
            knowledge_points=("arithmetic",),
            exam_points=("calculation",),
            language="en",
        ),
        quality=QualityState(
            gates=(
                GateEvidence(
                    gate_name="independent_correctness_verification",
                    verdict=GateVerdict.PASS,
                    score=0.999,
                    provider="provider-secret",
                    model="model-secret",
                    prompt_version="v1",
                    config_version="pilot-v1",
                ),
            )
        ),
        provenance=QuestionProvenance(
            source_dataset="gold",
            source_id=candidate_id,
            source_url=f"https://example.edu/{candidate_id}",
            raw_sha256="b" * 64,
        ),
        dedup=DedupState(exact_hash=slim_question_md5_v1(question)),
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _land_image(output: Path, relative: str) -> None:
    target = output / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"fake-image")


def test_export_jsonl_writes_only_exact_19_field_accepted_records(tmp_path: Path) -> None:
    records = [ExportRecord(ir=_ir(), stage=RunStage.ACCEPTED)]

    path = export_jsonl(records, tmp_path)

    assert path == tmp_path / "questions.jsonl"
    rows = _read_jsonl(path)
    assert len(rows) == 1
    assert list(rows[0]) == list(FinalQuestionRecord.model_fields)
    assert len(rows[0]) == 19
    assert rows[0]["text_question"] == "Compute 2 + 3."
    assert rows[0]["text_answer"] == "5"
    assert "provider-secret" not in path.read_text(encoding="utf-8")
    assert "model-secret" not in path.read_text(encoding="utf-8")


def test_export_is_deterministic_for_same_records(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    records = [
        ExportRecord(ir=_ir(candidate_id="a"), stage=RunStage.ACCEPTED),
        ExportRecord(ir=_ir(candidate_id="b"), stage=RunStage.ACCEPTED),
    ]

    first = export_jsonl(records, first_dir)
    second = export_jsonl(records, second_dir)

    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize(
    "stage",
    [
        RunStage.FINAL_DEDUPED,
        RunStage.REJECTED,
        RunStage.VERIFIED,
    ],
)
def test_non_accepted_terminal_or_intermediate_record_fails_closed_without_partial_file(
    tmp_path: Path,
    stage: RunStage,
) -> None:
    records = [ExportRecord(ir=_ir(), stage=stage)]

    with pytest.raises(ValueError, match="ACCEPTED"):
        export_jsonl(records, tmp_path)

    assert not (tmp_path / "questions.jsonl").exists()


def test_one_declared_asset_same_img_and_landed_file_exports(tmp_path: Path) -> None:
    _land_image(tmp_path, "image/diagram.png")
    ir = _ir(
        question='Use the figure. <img src="image/diagram.png">',
        assets=("image/diagram.png",),
    )

    path = export_jsonl(
        [ExportRecord(ir=ir, stage=RunStage.ACCEPTED)],
        tmp_path,
    )

    row = _read_jsonl(path)[0]
    assert row["is_pic_included"] == 1


def test_declared_asset_without_question_img_placement_rejects(tmp_path: Path) -> None:
    _land_image(tmp_path, "image/diagram.png")
    ir = _ir(
        question="Use the diagram to solve the problem.",
        assets=("image/diagram.png",),
    )

    with pytest.raises(ValueError, match="image"):
        export_jsonl([ExportRecord(ir=ir, stage=RunStage.ACCEPTED)], tmp_path)

    assert not (tmp_path / "questions.jsonl").exists()


def test_declared_asset_with_img_but_missing_file_rejects(tmp_path: Path) -> None:
    ir = _ir(
        question='Use the figure. <img src="image/missing.png">',
        assets=("image/missing.png",),
    )

    with pytest.raises(ValueError, match="image/missing.png"):
        export_jsonl([ExportRecord(ir=ir, stage=RunStage.ACCEPTED)], tmp_path)

    assert not (tmp_path / "questions.jsonl").exists()


def test_question_img_reference_not_declared_by_ir_asset_inventory_rejects(
    tmp_path: Path,
) -> None:
    _land_image(tmp_path, "image/diagram.png")
    ir = _ir(
        question='Use the figure. <img src="image/diagram.png">',
        assets=(),
    )

    with pytest.raises(ValueError, match="image"):
        export_jsonl([ExportRecord(ir=ir, stage=RunStage.ACCEPTED)], tmp_path)

    assert not (tmp_path / "questions.jsonl").exists()


def test_declared_asset_and_final_reference_mismatch_rejects(tmp_path: Path) -> None:
    _land_image(tmp_path, "image/declared.png")
    _land_image(tmp_path, "image/placed.png")
    ir = _ir(
        question='Use the figure. <img src="image/placed.png">',
        assets=("image/declared.png",),
    )

    with pytest.raises(ValueError, match="image"):
        export_jsonl([ExportRecord(ir=ir, stage=RunStage.ACCEPTED)], tmp_path)

    assert not (tmp_path / "questions.jsonl").exists()


@pytest.mark.parametrize(
    ("question", "assets"),
    [
        ('Use it. <img src="image/../escape.png">', ("image/../escape.png",)),
        ('Use it. <img src="/tmp/escape.png">', ("/tmp/escape.png",)),
    ],
)
def test_traversal_or_absolute_image_path_continues_to_reject(
    tmp_path: Path,
    question: str,
    assets: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="image"):
        export_jsonl(
            [ExportRecord(ir=_ir(question=question, assets=assets), stage=RunStage.ACCEPTED)],
            tmp_path,
        )

    assert not (tmp_path / "questions.jsonl").exists()


def test_pure_text_empty_assets_and_no_img_exports_with_picture_flag_zero(
    tmp_path: Path,
) -> None:
    path = export_jsonl(
        [ExportRecord(ir=_ir(assets=()), stage=RunStage.ACCEPTED)],
        tmp_path,
    )

    assert _read_jsonl(path)[0]["is_pic_included"] == 0


def test_batch_validation_is_atomic_when_later_record_is_invalid(tmp_path: Path) -> None:
    records = [
        ExportRecord(ir=_ir(candidate_id="valid"), stage=RunStage.ACCEPTED),
        ExportRecord(ir=_ir(candidate_id="invalid"), stage=RunStage.REJECTED),
    ]

    with pytest.raises(ValueError, match="ACCEPTED"):
        export_jsonl(records, tmp_path)

    assert not (tmp_path / "questions.jsonl").exists()
