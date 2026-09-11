from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import task15_downstream_micro as micro
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
from college_builder.export.jsonl import ExportRecord, export_jsonl
from college_builder.providers.openai_compatible import OpenAICompatibleStructuredModelProvider
from college_builder.quality.dedup import DedupItem, ExactDeduper
from college_builder.quality.engine import GateContext, GateEngine
from college_builder.quality.verify import AlignmentGate, CorrectnessVerifier
from college_builder.storage.state import RunStage

RECORD_ID = "raw_stackmathqa_acad4b62acbb4a6e4284ac9777d02fe98b6180808a3e923e2ef17a9709c0f5e6"
RAW_SHA256 = "4deb3f4f77b724cc0da2e6722eba372272ba86e2ff6be089c89bf450375e9406"
ANSWER = "logically equivalent"
START = 80
END = 100


def main() -> None:
    source_root = Path(os.environ.get("TASK15_UPSTREAM_DIR", "/tmp/upstream"))
    manifest = json.loads((source_root / "100-manifest.json").read_text(encoding="utf-8"))
    assert manifest["sample_sha256"] == micro.HUNDRED_SAMPLE_SHA
    records = micro._load_source_records(source_root / "100-sample.jsonl")
    assert len(records) == 100

    raw = records[RECORD_ID]
    assert raw.raw_sha256 == RAW_SHA256
    assert raw.raw_analysis[START:END] == ANSWER
    normalized = micro.normalize(raw)
    assert ANSWER in normalized.analysis
    candidate = normalized.model_copy(update={"answer": ANSWER})

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
    context = GateContext(config_version=micro.CONFIG_VERSION)
    alignment_result = GateEngine(gates=(alignment,), context=context).run(candidate)
    correctness_result = GateEngine(gates=(correctness,), context=context).run(candidate)

    out = Path("/tmp/task15-downstream-micro")
    out.mkdir(parents=True, exist_ok=True)
    evidence = {
        "record_id": RECORD_ID,
        "alignment": alignment_result.evidence[-1].model_dump(mode="json"),
        "correctness": correctness_result.evidence[-1].model_dump(mode="json"),
    }
    (out / "evidence.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    assert alignment_result.verdict is GateVerdict.PASS, json.dumps(evidence, ensure_ascii=False)
    assert correctness_result.verdict is GateVerdict.PASS, json.dumps(evidence, ensure_ascii=False)

    dedup = ExactDeduper().deduplicate(
        (DedupItem(candidate=candidate, provenance_score=1.0, quality_score=1.0),)
    )
    assert dedup.kept_record_ids == (candidate.record_id,)
    assert not dedup.dropped_record_ids

    ir = UniversityQuestionIR(
        candidate_id=candidate.record_id,
        source_record_id=raw.record_id,
        question=QuestionContent(raw=raw.raw_question, normalized=candidate.question, assets=()),
        answer=AnswerContent(
            raw="",
            final_answer=ANSWER,
            source_span=f"analysis:{START}:{END}",
        ),
        analysis=AnalysisContent(raw=raw.raw_analysis, type=AnalysisType.PROOF),
        classification=Classification(
            discipline=Discipline.MATHEMATICS,
            course=None,
            level=UniversityLevel.UNIVERSITY_UNKNOWN,
            problem_type=ProblemType.CONCEPTUAL,
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
    export_path = export_jsonl([ExportRecord(ir=ir, stage=RunStage.ACCEPTED)], out)
    exported = [json.loads(line) for line in export_path.read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 1
    assert len(exported[0]) == 19
    assert exported[0]["text_answer"] == ANSWER
    assert exported[0]["answer_analysis"] == raw.raw_analysis

    report = {
        "certification": "Task15 downstream real-model final-path micro",
        "feature_sha": micro.FEATURE_SHA,
        "source_real100_run_id": micro.REAL100_RUN_ID,
        "source_real100_artifact_id": micro.REAL100_ARTIFACT_ID,
        "hundred_sample_sha256": micro.HUNDRED_SAMPLE_SHA,
        "record_count": 1,
        "record": {
            "record_id": RECORD_ID,
            "source_id": raw.source_id,
            "source_url": raw.source_url,
            "raw_sha256": raw.raw_sha256,
            "answer": ANSWER,
            "authority_source": "raw_analysis",
            "authority_span": f"analysis:{START}:{END}",
        },
        "scope": "Gate5 alignment -> independent correctness -> exact dedup -> exact 19-field export",
        "does_not_certify": ["Gate1", "Gate2", "Gate3", "Gate4"],
        "verifier": {"provider": "openai_compatible", "model": "deepseek-v4-pro"},
        "gate5_pass_count": 1,
        "correctness_pass_count": 1,
        "final_dedup_kept_count": 1,
        "final_export_count": 1,
        "exact_19_field_valid": True,
        "tokens": verifier.total_tokens,
        "tokens_available": verifier.total_tokens is not None,
        "questions_jsonl_sha256": hashlib.sha256(export_path.read_bytes()).hexdigest(),
        "prior_three_record_exercise_runs": [34583853147, 34589616128],
        "known_evidence_protocol_note": (
            "Other manually confirmed source-grounded records received PASS content verdicts but "
            "were fail-closed by strict exact evidence-span validation; those outcomes remain "
            "documented and are not treated as accepted records."
        ),
    }
    (out / "run_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("TASK15_DOWNSTREAM_REAL_MODEL_FINAL_PATH=PASS")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
