from __future__ import annotations

import json

from college_builder.domain.question import (
    AnalysisContent,
    AnalysisType,
    AnswerContent,
    Classification,
    Discipline,
    ProblemType,
    UniversityLevel,
)
from college_builder.domain.source import NormalizedQA, RawSourceRecord
from college_builder.export.profile import to_final_record
from college_builder.pipeline.runner import PipelineRunner


def _raw(license_metadata: dict[str, str]) -> RawSourceRecord:
    return RawSourceRecord(
        record_id="raw-openstax-1",
        source_type="openstax",
        source_dataset="openstax-university-physics-volume-3",
        source_id="openstax-1",
        source_url="https://openstax.org/books/university-physics-volume-3/pages/1-introduction",
        raw_question="Why does geometric optics apply to a microscope image?",
        raw_answer="Because the produced image is macroscopic.",
        raw_analysis="Microscopes create images of macroscopic size, so geometric optics applies.",
        raw_payload={},
        metadata={"language": "en"},
        license_metadata=license_metadata,
        raw_sha256="a" * 64,
    )


def _candidate(raw: RawSourceRecord) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-openstax-1",
        source_record_id=raw.record_id,
        question=raw.raw_question,
        answer=raw.raw_answer,
        analysis=raw.raw_analysis,
        subject_candidates=("physics",),
        images=(),
        metadata={"language": "en", "course": "University Physics"},
        normalization_evidence={},
    )


def _exported_source_license(raw: RawSourceRecord) -> str:
    candidate = _candidate(raw)
    runner = object.__new__(PipelineRunner)
    ir = runner._build_ir(
        raw,
        candidate,
        Classification(
            discipline=Discipline.PHYSICS,
            course="University Physics",
            level=UniversityLevel.UNDERGRADUATE,
            problem_type=ProblemType.CONCEPTUAL,
        ),
        AnswerContent(
            raw=candidate.answer,
            final_answer=candidate.answer,
            source_span=f"answer:0:{len(candidate.answer)}",
        ),
        AnalysisContent(raw=candidate.analysis, type=AnalysisType.CONCEPTUAL_REASONING),
        (),
    )
    static_info = json.loads(to_final_record(ir).static_info)
    return str(static_info["source_license"])


def test_openstax_license_metadata_is_preserved_exactly_through_ir_and_export() -> None:
    assert _exported_source_license(_raw({"license": "cc-by-4.0"})) == "cc-by-4.0"


def test_missing_source_license_remains_unknown_without_guessing() -> None:
    assert _exported_source_license(_raw({})) == "UNKNOWN"
