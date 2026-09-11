from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from college_builder.domain.evidence import GateVerdict
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
from college_builder.domain.source import RawSourceRecord
from college_builder.export.jsonl import ExportRecord, export_jsonl
from college_builder.normalize.normalizer import normalize
from college_builder.providers.openai_compatible import OpenAICompatibleStructuredModelProvider
from college_builder.quality.dedup import DedupItem, ExactDeduper
from college_builder.quality.engine import GateContext, GateEngine
from college_builder.quality.verify import AlignmentGate, CorrectnessVerifier
from college_builder.storage.state import RunStage

FEATURE_SHA = "01ab7f86b62d614239a3d5ceae6b83821cc9f664"
REAL100_RUN_ID = 34558426996
REAL100_ARTIFACT_ID = 10183643305
HUNDRED_SAMPLE_SHA = "ae513933791c2ac27d3916a4149193c2477e00ce993c59b597b27ad02101ae8e"
CONFIG_VERSION = "pilot-5k-frozen-v1"

SELECTED = {
    "raw_stackmathqa_1c4381e616889f6b78d899c8520c891cc6416683e764225410193b1962f9b0c8": {
        "raw_sha256": "1c8d1511d6e8b04e3fbe0e9dcc043f6e1cafefb680e63073d7a4f2b2c00ccd19",
        "answer": r"\frac{1}{2}",
        "start": 352,
        "end": 363,
        "discipline": Discipline.STATISTICS,
        "problem_type": ProblemType.PROOF,
        "analysis_type": AnalysisType.STEP_BY_STEP,
    },
    "raw_stackmathqa_02710c86aaba8c5134a75a6a9932a022057f16b6da36955041ca4e053f3912d7": {
        "raw_sha256": "2d394cda5cdc5a0cd3d014c047df0840acc9ed7873c05d424bff79d47ac1d16a",
        "answer": r"\lim_{n}a_{n}=L",
        "start": 508,
        "end": 523,
        "discipline": Discipline.MATHEMATICS,
        "problem_type": ProblemType.PROOF,
        "analysis_type": AnalysisType.PROOF,
    },
    "raw_stackmathqa_a90f524eb66bd0b1f923e9b2bf418e4a505ecc26c5076a4c30a34056eabbfd10": {
        "raw_sha256": "c7a236ccf808703e6856d5311f49bb65dfd01ff51e8b25eb87069091847fa039",
        "answer": r"\lfloor \frac{p-c}{2}\rfloor",
        "start": 165,
        "end": 193,
        "discipline": Discipline.MATHEMATICS,
        "problem_type": ProblemType.PROOF,
        "analysis_type": AnalysisType.PROOF,
    },
}


def _load_source_records(path: Path) -> dict[str, RawSourceRecord]:
    rows: dict[str, RawSourceRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = RawSourceRecord.model_validate_json(line)
        rows[record.record_id] = record
    return rows


def _selected_candidates(records: dict[str, RawSourceRecord]):
    candidates = []
    bindings = []
    for record_id, spec in SELECTED.items():
        raw = records[record_id]
        assert raw.raw_sha256 == spec["raw_sha256"]
        start = int(spec["start"])
        end = int(spec["end"])
        answer = str(spec["answer"])
        assert raw.raw_analysis[start:end] == answer
        normalized = normalize(raw)
        assert normalized.analysis == raw.raw_analysis
        candidate = normalized.model_copy(update={"answer": answer})
        candidates.append((raw, candidate, spec))
        bindings.append(
            {
                "record_id": record_id,
                "source_id": raw.source_id,
                "source_url": raw.source_url,
                "raw_sha256": raw.raw_sha256,
                "answer": answer,
                "authority_source": "raw_analysis",
                "authority_span": f"analysis:{start}:{end}",
            }
        )
    return candidates, bindings


def main() -> None:
    source_root = Path(os.environ.get("TASK15_UPSTREAM_DIR", "/tmp/upstream"))
    sample_path = source_root / "100-sample.jsonl"
    manifest_path = source_root / "100-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["sample_sha256"] == HUNDRED_SAMPLE_SHA
    records = _load_source_records(sample_path)
    assert len(records) == 100

    base_url = os.environ["OPENAI_COMPATIBLE_BASE_URL"]
    api_key = os.environ["OPENAI_COMPATIBLE_API_KEY"]
    verifier = OpenAICompatibleStructuredModelProvider(
        base_url=base_url,
        api_key=api_key,
        model="deepseek-v4-pro",
        timeout_seconds=90.0,
        max_attempts=3,
        retry_backoff_seconds=1.0,
    )
    alignment = AlignmentGate(
        provider=verifier,
        prompt=Path("prompts/qa_alignment/v1.txt").read_text(encoding="utf-8"),
        prompt_version="v1",
        pass_threshold=0.995,
    )
    correctness = CorrectnessVerifier(
        provider=verifier,
        prompt=Path("prompts/correctness_verify/v1.txt").read_text(encoding="utf-8"),
        prompt_version="v1",
    )
    context = GateContext(config_version=CONFIG_VERSION)

    chosen, bindings = _selected_candidates(records)
    dedup = ExactDeduper().deduplicate(
        tuple(DedupItem(candidate=candidate, provenance_score=1.0, quality_score=1.0) for _, candidate, _ in chosen)
    )
    assert len(dedup.kept_record_ids) == 3
    assert not dedup.dropped_record_ids

    irs: list[UniversityQuestionIR] = []
    evidence_rows: list[dict[str, object]] = []
    for raw, candidate, spec in chosen:
        alignment_result = GateEngine(gates=(alignment,), context=context).run(candidate)
        correctness_result = GateEngine(gates=(correctness,), context=context).run(candidate)
        assert alignment_result.verdict is GateVerdict.PASS, alignment_result.model_dump() if hasattr(alignment_result, "model_dump") else alignment_result
        assert correctness_result.verdict is GateVerdict.PASS, correctness_result.model_dump() if hasattr(correctness_result, "model_dump") else correctness_result

        start = int(spec["start"])
        end = int(spec["end"])
        answer = str(spec["answer"])
        ir = UniversityQuestionIR(
            candidate_id=candidate.record_id,
            source_record_id=raw.record_id,
            question=QuestionContent(raw=raw.raw_question, normalized=candidate.question, assets=()),
            answer=AnswerContent(raw="", final_answer=answer, source_span=f"analysis:{start}:{end}"),
            analysis=AnalysisContent(raw=raw.raw_analysis, type=spec["analysis_type"]),
            classification=Classification(
                discipline=spec["discipline"],
                course=None,
                level=UniversityLevel.UNIVERSITY_UNKNOWN,
                problem_type=spec["problem_type"],
            ),
            metadata=QuestionMetadata(language="en"),
            quality=QualityState(gates=tuple((*alignment_result.evidence, *correctness_result.evidence))),
            provenance=QuestionProvenance(
                source_dataset=raw.source_dataset,
                source_id=raw.source_id,
                source_url=raw.source_url,
                raw_sha256=raw.raw_sha256,
            ),
            dedup=DedupState(exact_hash=ExactDeduper().fingerprint(candidate)),
        )
        irs.append(ir)
        evidence_rows.append(
            {
                "record_id": raw.record_id,
                "alignment": alignment_result.evidence[-1].model_dump(mode="json"),
                "correctness": correctness_result.evidence[-1].model_dump(mode="json"),
            }
        )

    out = Path("/tmp/task15-downstream-micro")
    out.mkdir(parents=True, exist_ok=True)
    export_path = export_jsonl(
        [ExportRecord(ir=ir, stage=RunStage.ACCEPTED) for ir in irs],
        out,
    )
    exported = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 3
    assert all(len(row) == 19 for row in exported)

    Path(out / "evidence.json").write_text(
        json.dumps(evidence_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = {
        "certification": "Task15 downstream real-model micro",
        "feature_sha": FEATURE_SHA,
        "source_real100_run_id": REAL100_RUN_ID,
        "source_real100_artifact_id": REAL100_ARTIFACT_ID,
        "hundred_sample_sha256": HUNDRED_SAMPLE_SHA,
        "record_count": 3,
        "records": bindings,
        "scope": "Gate5 alignment -> independent correctness -> exact dedup -> exact 19-field export",
        "does_not_certify": ["Gate1", "Gate2", "Gate3", "Gate4"],
        "verifier": {"provider": "openai_compatible", "model": "deepseek-v4-pro"},
        "gate5_pass_count": 3,
        "correctness_pass_count": 3,
        "final_dedup_kept_count": len(dedup.kept_record_ids),
        "final_export_count": len(exported),
        "exact_19_field_valid": all(len(row) == 19 for row in exported),
        "tokens": verifier.total_tokens,
        "tokens_available": verifier.total_tokens is not None,
        "questions_jsonl_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
    }
    Path(out / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("TASK15_DOWNSTREAM_REAL_MODEL_MICRO=PASS")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
