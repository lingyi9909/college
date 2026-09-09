"""Deterministic accepted-only JSONL export for University STEM records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from college_builder.domain.final_record import FinalQuestionRecord
from college_builder.domain.question import UniversityQuestionIR
from college_builder.export.profile import image_references, to_final_record
from college_builder.storage.state import RunStage


@dataclass(frozen=True, slots=True)
class ExportRecord:
    """Typed export boundary carrying the terminal pipeline state with its IR."""

    ir: UniversityQuestionIR
    stage: RunStage


def export_jsonl(records: Iterable[ExportRecord], output_dir: Path) -> Path:
    """Validate an entire accepted batch, then atomically write questions.jsonl."""
    destination = Path(output_dir)
    prepared: list[str] = []

    for item in records:
        if item.stage is not RunStage.ACCEPTED:
            raise ValueError(
                f"record {item.ir.candidate_id} must be terminal ACCEPTED before export"
            )
        final_record = to_final_record(item.ir)
        _validate_image_targets(final_record, destination)
        prepared.append(_serialize(final_record))

    destination.mkdir(parents=True, exist_ok=True)
    output_path = destination / "questions.jsonl"
    temp_path = destination / ".questions.jsonl.tmp"
    payload = "".join(f"{line}\n" for line in prepared)
    temp_path.write_text(payload, encoding="utf-8", newline="\n")
    temp_path.replace(output_path)
    return output_path


def _serialize(record: FinalQuestionRecord) -> str:
    ordered = {
        field_name: getattr(record, field_name)
        for field_name in FinalQuestionRecord.model_fields
    }
    return json.dumps(ordered, ensure_ascii=False, separators=(",", ":"))


def _validate_image_targets(record: FinalQuestionRecord, output_dir: Path) -> None:
    for reference in image_references(record.text_question):
        relative = Path(reference)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"invalid image reference: {reference}")
        target = output_dir / relative
        if not target.is_file():
            raise ValueError(f"referenced image is missing from export output: {reference}")
