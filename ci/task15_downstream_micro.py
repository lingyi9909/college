from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from college_builder.config import PipelineConfig
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
from college_builder.domain.source import RawSourceRecord
from college_builder.export.jsonl import ExportRecord, export_jsonl
from college_builder.normalize.normalizer import normalize
from college_builder.pipeline.runner import GatePipelineProcessor
from college_builder.providers.openai_compatible import OpenAICompatibleStructuredModelProvider
from college_builder.quality.dedup import DedupItem, ExactDeduper
from college_builder.storage.state import RunStage

FEATURE_SHA = "01ab7f86b62d614239a3d5ceae6b83821cc9f664"
PRODUCTION_CODE_SHA = "d3b35964ac92d799405fd88101fa11fd05ff1460"
REAL100_RUN_ID = 34558426996
REAL100_SAMPLE_SHA = "ae513933791c2ac27d3916a4149193c2477e00ce993c59b597b27ad02101ae8e"
SOURCE_REVISION = "git:13239ec8c8078bc8dfb43e1383069eb9be000ff7"

# Manually confirmed from the immutable real-100 source artifact.  The expected
# answer is never generated: the script requires it to be an exact raw_analysis span.
CASES = {
    "math:4453111:4": {
        "answer": r"\frac{(1010!)^2}{2021!}",
        "discipline": Discipline.MATHEMATICS,
        "course": "Integral Calculus",
        "problem_type": ProblemType.CALCULATION,
        "analysis_type": AnalysisType.DERIVATION,
    },
    "math:2176500:0": {
        "answer": r"45\pi/4",
        "discipline": Discipline.MATHEMATICS,
        "course": "Vector Calculus",
        "problem_type": ProblemType.CALCULATION,
        "analysis_type": AnalysisType.STEP_BY_STEP,
    },
    "statistics:206494:1": {
        "answer": r"\frac{1}{2}",
        "discipline": Discipline.STATISTICS,
        "course": "Probability and Statistics",
        "problem_type": ProblemType.DERIVATION,
        "analysis_type": AnalysisType.DERIVATION,
    },
}


def _sha256_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_real100(path: Path) -> dict[str, RawSourceRecord]:
    records: dict[str, RawSourceRecord] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = RawSourceRecord.model_validate_json(line)
        records[record.source_id] = record
    if len(records) != 100:
        raise RuntimeError(f"expected 100 source records, got {len(records)}")
    return records


def _provider(config, *, base_url: str, api_key: str):
    return OpenAICompatibleStructuredModelProvider(
        base_url=base_url,
        model=config.model,
        api_key=api_key,
        timeout_seconds=config.timeout_seconds,
        max_attempts=config.max_attempts,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )


def main() -> None:
    prior = Path(os.environ.get("TASK15_PRIOR_ARTIFACT", "../prior/100-sample.jsonl"))
    if not prior.is_file():
        raise RuntimeError(f"real-100 source artifact missing: {prior}")

    base_url = os.environ["OPENAI_COMPATIBLE_BASE_URL"]
    api_key = os.environ["OPENAI_COMPATIBLE_API_KEY"]
    config_path = Path("config/pilot-5k-frozen.yaml")
    config = PipelineConfig.load(config_path)
    if config.providers.classifier.model != "deepseek-flash":
        raise RuntimeError("classifier identity drift")
    if config.providers.verifier.model != "deepseek-v4-pro":
        raise RuntimeError("verifier identity drift")

    records = _load_real100(prior)
    selected = [records[source_id] for source_id in CASES]
    source_spans: dict[str, str] = {}
    downstream_candidates = []
    manual_records: dict[str, dict[str, object]] = {}

    for raw in selected:
        spec = CASES[raw.source_id]
        answer = str(spec["answer"])
        start = raw.raw_analysis.find(answer)
        if start < 0:
            raise RuntimeError(f"approved answer not found in source analysis: {raw.source_id}")
        if raw.raw_analysis.find(answer, start + 1) >= 0:
            raise RuntimeError(f"approved answer is not a unique source span: {raw.source_id}")
        end = start + len(answer)
        if raw.raw_analysis[start:end] != answer:
            raise RuntimeError(f"source span mismatch: {raw.source_id}")
        source_spans[raw.source_id] = f"analysis:{start}:{end}"

        normalized = normalize(raw)
        # Gate5/Correctness consumes an answer field.  This exact string is copied
        # from the immutable raw_analysis span above; no model/generated content enters it.
        candidate = normalized.model_copy(update={"answer": answer})
        downstream_candidates.append(candidate)
        manual_records[raw.source_id] = {
            "raw": raw,
            "candidate": candidate,
            "answer": answer,
            "discipline": spec["discipline"],
            "course": spec["course"],
            "problem_type": spec["problem_type"],
            "analysis_type": spec["analysis_type"],
        }

    primary = _provider(config.providers.classifier, base_url=base_url, api_key=api_key)
    verifier = _provider(config.providers.verifier, base_url=base_url, api_key=api_key)
    processor = GatePipelineProcessor(
        config=config,
        cache_path=Path("/tmp/task15-downstream-cache.sqlite3"),
        primary_provider=primary,
        verifier_provider=verifier,
        project_root=Path.cwd(),
    )

    verification = {}
    for candidate in downstream_candidates:
        outcome = processor.verify(candidate)
        verification[candidate.source_record_id] = outcome
        if outcome.reject_reason is not None or not outcome.alignment_pass or not outcome.correctness_pass:
            raise RuntimeError(
                f"downstream real-model verification failed for {candidate.source_record_id}: "
                f"{outcome.reject_reason}"
            )

    dedup_items = tuple(
        DedupItem(candidate=candidate, provenance_score=1.0, quality_score=1.0)
        for candidate in downstream_candidates
    )
    dedup = ExactDeduper().deduplicate(dedup_items)
    expected_ids = tuple(candidate.record_id for candidate in downstream_candidates)
    if dedup.kept_record_ids != expected_ids or dedup.dropped_record_ids:
        raise RuntimeError("known-good downstream micro sample unexpectedly deduplicated")

    export_records = []
    report_records = []
    for candidate in downstream_candidates:
        raw = next(item for item in selected if item.record_id == candidate.source_record_id)
        details = manual_records[raw.source_id]
        answer = str(details["answer"])
        outcome = verification[candidate.source_record_id]
        ir = UniversityQuestionIR(
            candidate_id=candidate.record_id,
            source_record_id=raw.record_id,
            question=QuestionContent(
                raw=raw.raw_question,
                normalized=candidate.question,
                assets=candidate.images,
            ),
            answer=AnswerContent(
                raw=raw.raw_answer,
                final_answer=answer,
                source_span=source_spans[raw.source_id],
            ),
            analysis=AnalysisContent(
                raw=raw.raw_analysis,
                type=details["analysis_type"],
            ),
            classification=Classification(
                discipline=details["discipline"],
                course=str(details["course"]),
                level=UniversityLevel.UNDERGRADUATE,
                problem_type=details["problem_type"],
            ),
            metadata=QuestionMetadata(
                knowledge_points=(),
                exam_points=(),
                language="en",
            ),
            quality=QualityState(gates=tuple(outcome.evidence)),
            provenance=QuestionProvenance(
                source_dataset=raw.source_dataset,
                source_id=raw.source_id,
                source_url=raw.source_url,
                raw_sha256=raw.raw_sha256,
            ),
            dedup=DedupState(exact_hash=slim_question_md5_v1(candidate.question)),
        )
        export_records.append(ExportRecord(ir=ir, stage=RunStage.ACCEPTED))
        report_records.append(
            {
                "record_id": raw.record_id,
                "source_id": raw.source_id,
                "raw_sha256": raw.raw_sha256,
                "answer_source_span": source_spans[raw.source_id],
                "final_answer": answer,
                "alignment": outcome.evidence[0].model_dump(mode="json"),
                "correctness": outcome.evidence[1].model_dump(mode="json"),
            }
        )

    out_dir = Path("/tmp/task15-downstream-cert/export")
    output_path = export_jsonl(export_records, out_dir)
    rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    if len(rows) != len(CASES):
        raise RuntimeError("downstream export row count mismatch")
    expected_fields = list(FinalQuestionRecord.model_fields)
    for row, raw in zip(rows, selected, strict=True):
        if list(row) != expected_fields or len(row) != 19:
            raise RuntimeError(f"exact 19-field export failed: {raw.source_id}")
        candidate = manual_records[raw.source_id]["candidate"]
        if row["text_question"] != candidate.question:
            raise RuntimeError(f"question changed after downstream verification: {raw.source_id}")
        if row["text_answer"] != manual_records[raw.source_id]["answer"]:
            raise RuntimeError(f"answer changed after downstream verification: {raw.source_id}")
        if row["answer_analysis"] != raw.raw_analysis:
            raise RuntimeError(f"analysis changed after downstream verification: {raw.source_id}")
        static_info = json.loads(row["static_info"])
        if static_info["source_dataset"] != raw.source_dataset:
            raise RuntimeError("source dataset provenance mismatch")
        if static_info["source_id"] != raw.source_id:
            raise RuntimeError("source id provenance mismatch")
        if static_info["raw_sha256"] != raw.raw_sha256.lower():
            raise RuntimeError("raw SHA provenance mismatch")

    usage = processor.provider_usage
    report = {
        "certification_scope": "task15-real-model-downstream-micro",
        "status": "PASS",
        "feature_exact_head": FEATURE_SHA,
        "production_code_sha": PRODUCTION_CODE_SHA,
        "source_real100_run_id": REAL100_RUN_ID,
        "source_real100_sample_sha256": REAL100_SAMPLE_SHA,
        "source_revision": SOURCE_REVISION,
        "config_version": config.config_version,
        "config_sha256": _sha256_bytes(config_path),
        "verifier": {"provider": verifier.provider, "model": verifier.model},
        "record_count": len(CASES),
        "gate5_alignment_pass_count": len(CASES),
        "independent_correctness_pass_count": len(CASES),
        "final_dedup_kept_count": len(dedup.kept_record_ids),
        "final_dedup_dropped_count": len(dedup.dropped_record_ids),
        "exact_19_field_export_count": len(rows),
        "provider_call_counts": usage.provider_call_counts,
        "tokens": usage.tokens,
        "tokens_available": usage.tokens is not None,
        "estimated_cost_usd": usage.estimated_cost_usd,
        "estimated_cost_usd_available": usage.estimated_cost_usd is not None,
        "records": report_records,
        "notes": (
            "Downstream-only certification. Upstream Gates 0-4 were intentionally not rerun. "
            "Each final_answer was manually pre-approved and mechanically required to be an exact "
            "substring of the immutable real-100 raw_analysis before any model call."
        ),
    }
    cert_dir = Path("/tmp/task15-downstream-cert")
    cert_dir.mkdir(parents=True, exist_ok=True)
    (cert_dir / "downstream_real_model_certification.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("TASK15_DOWNSTREAM_REAL_MODEL_CERTIFICATION=PASS")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, sort_keys=True))


if __name__ == "__main__":
    main()
