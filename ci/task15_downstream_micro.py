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
    "raw_stackmathqa_36383322c62efa98ac513621c69ff25f05bdb72a2c74dadc9870c33f8b9c030d": {
        "raw_sha256": "e32e233508cb23c63e5eee7024e7ea15eaa6a667cf636cd2826108205868c1ea",
        "answer": "$B$ is an open set.",
        "start": 305,
        "end": 324,
        "discipline": Discipline.MATHEMATICS,
        "problem_type": ProblemType.PROOF,
        "analysis_type": AnalysisType.PROOF,
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
    "raw_stackmathqa_acad4b62acbb4a6e4284ac9777d02fe98b6180808a3e923e2ef17a9709c0f5e6": {
        "raw_sha256": "4deb3f4f77b724cc0da2e6722eba372272ba86e2ff6be089c89bf450375e9406",
        "answer": "logically equivalent",
        "start": 80,
        "end": 100,
        "discipline": Discipline.MATHEMATICS,
        "problem_type": ProblemType.CONCEPTUAL,
        "analysis_type": AnalysisType.PROOF,
    },
}


def _load_source_records(path: Path) -> dict[str, RawSourceRecord]:
    rows: dict[str, RawSourceRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
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
        assert normalized.source_record_id == raw.record_id
        assert answer in normalized.analysis
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
                "normalized_analysis_sha256": hashlib.sha256(
                    normalized.analysis.encode("utf-8")
                ).hexdigest(),
            }
        )
    return candidates, bindings


def main() -> None:
    source_root = Path(os.environ.get("TASK15_UPSTREAM_DIR", "/tmp/upstream"))
    manifest = json.loads((source_root / "100-manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_sha256"] == HUNDRED_SAMPLE_SHA
    records = _load_source_records(source_root / "100-sample.jsonl")
    assert len(records) == 100

    verifier = OpenAICompatibleStructuredModelProvider(
        base_url=os.environ["OPENAI_COMPATIBLE_BASE_URL"],
        api_key=os.environ["OPENAI_COMPATIBLE_API_KEY"],
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
        tuple(
            DedupItem(candidate=candidate, provenance_score=1.0, quality_score=1.0)
            for _, candidate, _ in chosen
        )
    )
    assert len(dedup.kept_record_ids) == 3
    assert not dedup.dropped_record_ids

    results = []
    for raw, candidate, spec in chosen:
        alignment_result = GateEngine(gates=(alignment,), context=context).run(candidate)
        correctness_result = GateEngine(gates=(correctness,), context=context).run(candidate)
        results.append((raw, candidate, spec, alignment_result, correctness_result))

    out = Path("/tmp/task15-downstream-micro")
    out.mkdir(parents=True, exist_ok=True)
    evidence_rows = [
        {
            "record_id": raw.record_id,
            "alignment": alignment_result.evidence[-1].model_dump(mode="json"),
            "correctness": correctness_result.evidence[-1].model_dump(mode="json"),
        }
        for raw, _, _, alignment_result, correctness_result in results
    ]
    (out / "evidence.json").write_text(
        json.dumps(evidence_rows, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    failures = [
        row
        for row in evidence_rows
        if row["alignment"]["verdict"] != GateVerdict.PASS.value
        or row["correctness"]["verdict"] != GateVerdict.PASS.value
    ]
    if failures:
        raise AssertionError(json.dumps(failures, ensure_ascii=False, sort_keys=True))

    irs: list[UniversityQuestionIR] = []
    for raw, candidate, spec, alignment_result, correctness_result in results:
        start = int(spec["start"])
        end = int(spec["end"])
        answer = str(spec["answer"])
        irs.append(
            UniversityQuestionIR(
                candidate_id=candidate.record_id,
                source_record_id=raw.record_id,
                question=QuestionContent(
                    raw=raw.raw_question,
                    normalized=candidate.question,
                    assets=(),
                ),
                answer=AnswerContent(
                    raw="",
                    final_answer=answer,
                    source_span=f"analysis:{start}:{end}",
                ),
                analysis=AnalysisContent(
                    raw=raw.raw_analysis,
                    type=spec["analysis_type"],
                ),
                classification=Classification(
                    discipline=spec["discipline"],
                    course=None,
                    level=UniversityLevel.UNIVERSITY_UNKNOWN,
                    problem_type=spec["problem_type"],
                ),
                metadata=QuestionMetadata(language="en"),
                quality=QualityState(
                    gates=tuple((*alignment_result.evidence, *correctness_result.evidence))
                ),
                provenance=QuestionProvenance(
                    source_dataset=raw.source_dataset,
                    source_id=raw.source_id,
                    source_url=raw.source_url,
                    raw_sha256=raw.raw_sha256,
                ),
                dedup=DedupState(exact_hash=ExactDeduper().fingerprint(candidate)),
            )
        )

    export_path = export_jsonl(
        [ExportRecord(ir=ir, stage=RunStage.ACCEPTED) for ir in irs],
        out,
    )
    exported = [
        json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines()
    ]
    assert len(exported) == 3
    assert all(len(row) == 19 for row in exported)

    report = {
        "certification": "Task15 downstream real-model micro",
        "feature_sha": FEATURE_SHA,
        "source_real100_run_id": REAL100_RUN_ID,
        "source_real100_artifact_id": REAL100_ARTIFACT_ID,
        "hundred_sample_sha256": HUNDRED_SAMPLE_SHA,
        "record_count": 3,
        "records": bindings,
        "scope": (
            "Gate5 alignment -> independent correctness -> exact dedup -> "
            "exact 19-field export"
        ),
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
    (out / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("TASK15_DOWNSTREAM_REAL_MODEL_MICRO=PASS")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
