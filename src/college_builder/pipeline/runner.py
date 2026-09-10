"""Task 14 resumable acquisition-to-export pipeline orchestration."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from college_builder.config import PipelineConfig
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
from college_builder.domain.source import JsonValue, NormalizedQA, RawSourceRecord
from college_builder.export.jsonl import ExportRecord, export_jsonl
from college_builder.export.profile import to_final_record
from college_builder.normalize.normalizer import NORMALIZATION_VERSION, normalize
from college_builder.providers.base import (
    ModelClassificationRequest,
    ModelDecision,
    StructuredModelProvider,
)
from college_builder.quality.analysis import AnalysisGate
from college_builder.quality.answer import AnswerGate
from college_builder.quality.classify import ProblemGate, UniversityStemGate
from college_builder.quality.dedup import DedupItem, ExactDeduper
from college_builder.quality.engine import GateContext, GateEngine, GateResultEvidence
from college_builder.quality.integrity import IntegrityGate
from college_builder.quality.verify import AlignmentGate, CorrectnessVerifier
from college_builder.reporting.pilot_report import (
    ProviderUsage,
    RecordAudit,
    build_pilot_report,
    load_pilot_report,
    write_pilot_report,
)
from college_builder.source.base import SourceAdapter, SourceDescriptor
from college_builder.storage.cache import CallCache
from college_builder.storage.state import RunStage, RunStateStore

_NORMALIZER = Callable[[RawSourceRecord], NormalizedQA]
_PROMPT_LOADER = Callable[[str, str], str]


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    run_id: str
    run_dir: Path
    output_path: Path
    report_path: Path
    accepted_count: int
    rejected_count: int


class PipelineInterrupted(RuntimeError):
    """Intentional test/operations interruption after one completed stage."""

    def __init__(self, run_id: str, stage: RunStage) -> None:
        super().__init__(f"pipeline interrupted after {stage.value}: {run_id}")
        self.run_id = run_id
        self.stage = stage


@dataclass(frozen=True, slots=True)
class _ClassificationOutcome:
    classification: Classification | None
    evidence: tuple[GateResultEvidence, ...]
    reject_reason: str | None


@dataclass(frozen=True, slots=True)
class _AnswerOutcome:
    answer: AnswerContent | None
    evidence: tuple[GateResultEvidence, ...]
    reject_reason: str | None


@dataclass(frozen=True, slots=True)
class _AnalysisOutcome:
    analysis: AnalysisContent | None
    evidence: tuple[GateResultEvidence, ...]
    reject_reason: str | None


@dataclass(frozen=True, slots=True)
class _VerificationOutcome:
    evidence: tuple[GateResultEvidence, ...]
    reject_reason: str | None
    alignment_pass: bool
    correctness_pass: bool


@dataclass(slots=True)
class _UsageAccumulator:
    provider_call_counts: Counter[str]
    cache_hits: int = 0
    cache_misses: int = 0
    latency_seconds: float = 0.0
    provider_errors: int = 0


class _CachedStructuredProvider:
    """Cache model decisions with the Task 3 canonical cache key builder."""

    def __init__(
        self,
        *,
        provider: StructuredModelProvider,
        cache: CallCache,
        prompt_versions: Mapping[str, str],
        gate_config_fingerprints: Mapping[str, str],
        usage: _UsageAccumulator,
    ) -> None:
        self._provider = provider
        self.provider = provider.provider
        self.model = provider.model
        self._cache = cache
        self._prompt_versions = dict(prompt_versions)
        self._gate_config_fingerprints = dict(gate_config_fingerprints)
        self._usage = usage

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        prompt_version = self._prompt_versions.get(request.task)
        config_fingerprint = self._gate_config_fingerprints.get(request.task)
        if prompt_version is None or config_fingerprint is None:
            raise ValueError(f"cache identity missing for model task {request.task}")
        content_hash = _hash_json(request.inputs)
        cache_key = CallCache.key(
            content_hash,
            request.task,
            self.provider,
            self.model,
            prompt_version,
            config_fingerprint,
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self._usage.cache_hits += 1
            return ModelDecision.model_validate(cached)

        self._usage.cache_misses += 1
        self._usage.provider_call_counts[request.task] += 1
        started = time.perf_counter()
        try:
            raw: object = self._provider.classify(request)
            if not isinstance(raw, ModelDecision):
                raise TypeError("provider returned non-ModelDecision output")
            validated = ModelDecision.model_validate(raw.model_dump(mode="python", warnings=False))
        except Exception:
            self._usage.provider_errors += 1
            raise
        finally:
            self._usage.latency_seconds += time.perf_counter() - started
        self._cache.put(cache_key, validated.model_dump(mode="json"))
        return validated


class GatePipelineProcessor:
    """Thin adapter that applies already-approved Tasks 8-12 gates."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        cache_path: Path,
        primary_provider: StructuredModelProvider,
        verifier_provider: StructuredModelProvider,
        project_root: Path,
        prompt_loader: _PROMPT_LOADER | None = None,
    ) -> None:
        self.config = config
        self._project_root = Path(project_root)
        self._prompt_loader = prompt_loader
        cache = CallCache(cache_path)
        self._usage = _UsageAccumulator(provider_call_counts=Counter())
        prompt_versions = {
            "gate_1_university_stem": config.prompt_versions.university_classify,
            "gate_2_problem": config.prompt_versions.problem_classify,
            "gate_4_original_analysis": config.prompt_versions.analysis_classify,
            "gate_5_qa_alignment": config.prompt_versions.qa_alignment,
            "independent_correctness_verification": config.prompt_versions.correctness_verify,
        }
        gate_config_fingerprints = {
            "gate_1_university_stem": _hash_json(
                {"pass_threshold": config.thresholds.university, "verify_threshold": 0.90}
            ),
            "gate_2_problem": _hash_json(
                {"pass_threshold": config.thresholds.problem, "verify_threshold": 0.90}
            ),
            "gate_4_original_analysis": _hash_json(
                {"pass_threshold": config.thresholds.analysis, "verify_threshold": 0.90}
            ),
            "gate_5_qa_alignment": _hash_json({"pass_threshold": config.thresholds.qa_alignment}),
            "independent_correctness_verification": _hash_json(
                {"verification_contract": "correctness_v1"}
            ),
        }
        primary = _CachedStructuredProvider(
            provider=primary_provider,
            cache=cache,
            prompt_versions=prompt_versions,
            gate_config_fingerprints=gate_config_fingerprints,
            usage=self._usage,
        )
        verifier = _CachedStructuredProvider(
            provider=verifier_provider,
            cache=cache,
            prompt_versions=prompt_versions,
            gate_config_fingerprints=gate_config_fingerprints,
            usage=self._usage,
        )
        self._integrity = IntegrityGate()
        self._university = UniversityStemGate(
            primary=primary,
            verifier=verifier,
            prompt=self._prompt("university_classify", config.prompt_versions.university_classify),
            prompt_version=config.prompt_versions.university_classify,
            pass_threshold=config.thresholds.university,
        )
        self._problem = ProblemGate(
            primary=primary,
            verifier=verifier,
            prompt=self._prompt("problem_classify", config.prompt_versions.problem_classify),
            prompt_version=config.prompt_versions.problem_classify,
            pass_threshold=config.thresholds.problem,
        )
        self._answer = AnswerGate(extract_threshold=config.thresholds.answer_extract)
        self._analysis = AnalysisGate(
            primary=primary,
            verifier=verifier,
            prompt=self._prompt("analysis_classify", config.prompt_versions.analysis_classify),
            prompt_version=config.prompt_versions.analysis_classify,
            pass_threshold=config.thresholds.analysis,
        )
        self._alignment = AlignmentGate(
            provider=verifier,
            prompt=self._prompt("qa_alignment", config.prompt_versions.qa_alignment),
            prompt_version=config.prompt_versions.qa_alignment,
            pass_threshold=config.thresholds.qa_alignment,
        )
        self._correctness = CorrectnessVerifier(
            provider=verifier,
            prompt=self._prompt("correctness_verify", config.prompt_versions.correctness_verify),
            prompt_version=config.prompt_versions.correctness_verify,
        )

    @property
    def provider_usage(self) -> ProviderUsage:
        return ProviderUsage(
            provider_call_counts=dict(self._usage.provider_call_counts),
            cache_hits=self._usage.cache_hits,
            cache_misses=self._usage.cache_misses,
            latency_seconds=self._usage.latency_seconds,
            tokens=0,
            estimated_cost_usd=0.0,
            fallback_count=0,
            provider_errors=self._usage.provider_errors,
        )

    def integrity(self, candidate: NormalizedQA, raw: RawSourceRecord) -> GateResultEvidence:
        context = GateContext(
            config_version=self.config.config_version,
            raw_records={raw.record_id: raw},
        )
        result = GateEngine(gates=(self._integrity,), context=context).run(candidate)
        return result.evidence[-1]

    def classify(self, candidate: NormalizedQA) -> _ClassificationOutcome:
        context = GateContext(config_version=self.config.config_version)
        university = GateEngine(gates=(self._university,), context=context).run(candidate)
        university_evidence = university.evidence[-1]
        if university.verdict is GateVerdict.REJECT:
            return _ClassificationOutcome(
                classification=None,
                evidence=university.evidence,
                reject_reason=university_evidence.reason_code,
            )
        problem = GateEngine(gates=(self._problem,), context=context).run(candidate)
        evidence = (*university.evidence, *problem.evidence)
        problem_evidence = problem.evidence[-1]
        if problem.verdict is GateVerdict.REJECT:
            return _ClassificationOutcome(
                classification=None,
                evidence=evidence,
                reject_reason=problem_evidence.reason_code,
            )
        problem_label = _decision_label(problem_evidence)
        if problem_label is None:
            return _ClassificationOutcome(None, evidence, "PROBLEM_TYPE_UNCERTAIN")
        try:
            problem_type = ProblemType(problem_label)
        except ValueError:
            return _ClassificationOutcome(None, evidence, "PROBLEM_TYPE_UNCERTAIN")
        classification = Classification(
            discipline=_discipline_from_metadata(candidate),
            course=_metadata_text(candidate, "course"),
            level=_university_level_from_metadata(candidate),
            problem_type=problem_type,
        )
        return _ClassificationOutcome(classification, evidence, None)

    def answer(self, candidate: NormalizedQA) -> _AnswerOutcome:
        context = GateContext(config_version=self.config.config_version)
        result = GateEngine(gates=(self._answer,), context=context).run(candidate)
        evidence = result.evidence[-1]
        if result.verdict is GateVerdict.REJECT:
            return _AnswerOutcome(None, result.evidence, evidence.reason_code)
        payload = evidence.evidence_payload
        final_answer = payload.get("final_answer")
        source_span = payload.get("source_span")
        if not isinstance(final_answer, str) or not isinstance(source_span, str):
            return _AnswerOutcome(None, result.evidence, "ANSWER_SOURCE_UNTRACEABLE")
        return _AnswerOutcome(
            AnswerContent(raw=candidate.answer, final_answer=final_answer, source_span=source_span),
            result.evidence,
            None,
        )

    def analysis(self, candidate: NormalizedQA) -> _AnalysisOutcome:
        context = GateContext(config_version=self.config.config_version)
        result = GateEngine(gates=(self._analysis,), context=context).run(candidate)
        evidence = result.evidence[-1]
        if result.verdict is GateVerdict.REJECT:
            return _AnalysisOutcome(None, result.evidence, evidence.reason_code)
        analysis_type = evidence.evidence_payload.get("analysis_type")
        if not isinstance(analysis_type, str):
            return _AnalysisOutcome(None, result.evidence, "ANALYSIS_UNCERTAIN")
        try:
            parsed = AnalysisType(analysis_type)
        except ValueError:
            return _AnalysisOutcome(None, result.evidence, "ANALYSIS_UNCERTAIN")
        return _AnalysisOutcome(
            AnalysisContent(raw=candidate.analysis, type=parsed), result.evidence, None
        )

    def verify(self, candidate: NormalizedQA) -> _VerificationOutcome:
        context = GateContext(config_version=self.config.config_version)
        alignment = GateEngine(gates=(self._alignment,), context=context).run(candidate)
        alignment_evidence = alignment.evidence[-1]
        if alignment.verdict is GateVerdict.REJECT:
            return _VerificationOutcome(
                alignment.evidence,
                alignment_evidence.reason_code,
                False,
                False,
            )
        correctness = GateEngine(gates=(self._correctness,), context=context).run(candidate)
        evidence = (*alignment.evidence, *correctness.evidence)
        correctness_evidence = correctness.evidence[-1]
        if correctness.verdict is GateVerdict.REJECT:
            return _VerificationOutcome(evidence, correctness_evidence.reason_code, True, False)
        return _VerificationOutcome(evidence, None, True, True)

    def _prompt(self, task: str, version: str) -> str:
        if self._prompt_loader is not None:
            value = self._prompt_loader(task, version)
        else:
            path = self._project_root / "prompts" / task / f"{version}.txt"
            value = path.read_text(encoding="utf-8")
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"prompt {task}/{version} must be non-empty")
        return value


class PipelineRunner:
    """Persist, resume, deduplicate, verify, export, and report one source snapshot."""

    def __init__(
        self,
        *,
        config: PipelineConfig,
        workspace: Path,
        processor: GatePipelineProcessor,
        normalizer: _NORMALIZER = normalize,
    ) -> None:
        self.config = config
        self.workspace = Path(workspace)
        self.processor = processor
        self._normalizer = normalizer
        self.workspace.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        *,
        adapter: SourceAdapter,
        source_config: object,
        output_dir: Path,
        interrupt_after: RunStage | None = None,
    ) -> PipelineRunResult:
        descriptors = tuple(adapter.discover(source_config))
        if not descriptors:
            raise ValueError("source adapter discovered no inputs")
        source_snapshot_hash = _source_snapshot_hash(descriptors)
        raw_records = self._load_or_acquire(adapter, descriptors, source_snapshot_hash)
        config_hash = _hash_json(self.config.model_dump(mode="json"))
        run_id = _run_id(source_snapshot_hash, config_hash, self.config.pipeline_version)
        run_dir = self.workspace / "runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._write_manifest(
            run_dir,
            {
                "run_id": run_id,
                "source_snapshot_hash": source_snapshot_hash,
                "config_hash": config_hash,
                "pipeline_version": self.config.pipeline_version,
            },
        )
        return self._execute(
            run_id=run_id,
            run_dir=run_dir,
            raw_records=raw_records,
            output_dir=Path(output_dir),
            interrupt_after=interrupt_after,
        )

    def resume(self, *, run_id: str, output_dir: Path) -> PipelineRunResult:
        run_dir = self.workspace / "runs" / run_id
        manifest = _load_json_object(run_dir / "manifest.json")
        config_hash = _hash_json(self.config.model_dump(mode="json"))
        if manifest.get("config_hash") != config_hash:
            raise ValueError("resume configuration does not match the original run")
        source_snapshot_hash = manifest.get("source_snapshot_hash")
        if not isinstance(source_snapshot_hash, str) or not source_snapshot_hash:
            raise ValueError("run manifest source snapshot is invalid")

        destination = Path(output_dir)
        completed_report = run_dir / "run_report.json"
        completed_output = destination / "questions.jsonl"
        if completed_report.is_file() and completed_output.is_file():
            report = load_pilot_report(completed_report)
            write_pilot_report(report, destination / "pilot_report.json")
            return PipelineRunResult(
                run_id=run_id,
                run_dir=run_dir,
                output_path=completed_output,
                report_path=completed_report,
                accepted_count=report.accepted_count,
                rejected_count=report.rejected_count,
            )

        raw_records = self._load_source_cache(source_snapshot_hash)
        return self._execute(
            run_id=run_id,
            run_dir=run_dir,
            raw_records=raw_records,
            output_dir=destination,
            interrupt_after=None,
        )

    def _execute(
        self,
        *,
        run_id: str,
        run_dir: Path,
        raw_records: tuple[RawSourceRecord, ...],
        output_dir: Path,
        interrupt_after: RunStage | None,
    ) -> PipelineRunResult:
        started = time.perf_counter()
        state = RunStateStore(run_dir / "state.sqlite3")
        normalized: dict[str, NormalizedQA] = {}
        classifications: dict[str, Classification] = {}
        answers: dict[str, AnswerContent] = {}
        analyses: dict[str, AnalysisContent] = {}
        evidence: dict[str, list[GateResultEvidence]] = {row.record_id: [] for row in raw_records}
        audits: dict[str, dict[str, Any]] = {}
        rejected: set[str] = set()
        terminal_accepted: dict[str, UniversityQuestionIR] = {}

        for raw in raw_records:
            current = state.current_stage(run_id, raw.record_id)
            saved_audit = self._load_audit(run_dir, raw.record_id)
            audits[raw.record_id] = saved_audit or _audit_facts(raw)
            if current is RunStage.REJECTED:
                rejected.add(raw.record_id)
                continue
            if current is RunStage.ACCEPTED:
                terminal_accepted[raw.record_id] = self._load_terminal_ir(run_dir, raw.record_id)
                continue

            acquired_fp = _hash_json(
                {"source_snapshot": _raw_identity(raw), "raw_sha256": raw.raw_sha256}
            )
            if current is None:
                state.advance(run_id, raw.record_id, RunStage.ACQUIRED, acquired_fp)

            candidate = self._load_or_normalize(raw)
            normalized[raw.record_id] = candidate
            integrity = self.processor.integrity(candidate, raw)
            evidence[raw.record_id].append(integrity)
            if integrity.verdict is GateVerdict.REJECT:
                self._mark_rejected(
                    state,
                    run_id,
                    run_dir,
                    raw.record_id,
                    integrity.reason_code,
                    rejected,
                    audits,
                )
                continue

            normalized_fp = _hash_json(
                {"raw_sha256": raw.raw_sha256, "normalization_version": NORMALIZATION_VERSION}
            )
            if state.current_stage(run_id, raw.record_id) is RunStage.ACQUIRED:
                state.advance(run_id, raw.record_id, RunStage.NORMALIZED, normalized_fp)
            audits[raw.record_id]["normalized"] = True
            self._save_audit(run_dir, raw.record_id, audits[raw.record_id])

        self._interrupt_if_requested(run_id, interrupt_after, RunStage.NORMALIZED)

        for raw in raw_records:
            if raw.record_id in rejected or raw.record_id in terminal_accepted:
                continue
            candidate = normalized[raw.record_id]
            outcome = self._stage_classification(run_dir, raw.record_id, candidate)
            evidence[raw.record_id].extend(outcome.evidence)
            if outcome.reject_reason is not None or outcome.classification is None:
                reason = outcome.reject_reason or "UNIVERSITY_LEVEL_UNCERTAIN"
                self._mark_rejected(state, run_id, run_dir, raw.record_id, reason, rejected, audits)
                continue
            classifications[raw.record_id] = outcome.classification
            fp = self._classification_fingerprint(candidate)
            if state.current_stage(run_id, raw.record_id) is RunStage.NORMALIZED:
                state.advance(run_id, raw.record_id, RunStage.CLASSIFIED, fp)
            audits[raw.record_id]["university_stem"] = True
            audits[raw.record_id]["problem"] = True
            audits[raw.record_id]["subject"] = outcome.classification.discipline.value
            self._save_audit(run_dir, raw.record_id, audits[raw.record_id])

        self._interrupt_if_requested(run_id, interrupt_after, RunStage.CLASSIFIED)

        for raw in raw_records:
            if (
                raw.record_id in rejected
                or raw.record_id in terminal_accepted
                or raw.record_id not in classifications
            ):
                continue
            candidate = normalized[raw.record_id]
            outcome = self._stage_answer(run_dir, raw.record_id, candidate)
            evidence[raw.record_id].extend(outcome.evidence)
            if outcome.reject_reason is not None or outcome.answer is None:
                reason = outcome.reject_reason or "ANSWER_NOT_EXTRACTABLE"
                self._mark_rejected(state, run_id, run_dir, raw.record_id, reason, rejected, audits)
                continue
            answers[raw.record_id] = outcome.answer
            fp = _hash_json(
                {
                    "content": _candidate_hash(candidate),
                    "threshold": self.config.thresholds.answer_extract,
                }
            )
            if state.current_stage(run_id, raw.record_id) is RunStage.CLASSIFIED:
                state.advance(run_id, raw.record_id, RunStage.ANSWER_VALIDATED, fp)
            audits[raw.record_id]["answer_valid"] = True
            self._save_audit(run_dir, raw.record_id, audits[raw.record_id])

        self._interrupt_if_requested(run_id, interrupt_after, RunStage.ANSWER_VALIDATED)

        for raw in raw_records:
            if (
                raw.record_id in rejected
                or raw.record_id in terminal_accepted
                or raw.record_id not in answers
            ):
                continue
            candidate = normalized[raw.record_id]
            outcome = self._stage_analysis(run_dir, raw.record_id, candidate)
            evidence[raw.record_id].extend(outcome.evidence)
            if outcome.reject_reason is not None or outcome.analysis is None:
                reason = outcome.reject_reason or "ANALYSIS_UNCERTAIN"
                self._mark_rejected(state, run_id, run_dir, raw.record_id, reason, rejected, audits)
                continue
            analyses[raw.record_id] = outcome.analysis
            fp = self._analysis_fingerprint(candidate)
            if state.current_stage(run_id, raw.record_id) is RunStage.ANSWER_VALIDATED:
                state.advance(run_id, raw.record_id, RunStage.ANALYSIS_VALIDATED, fp)
            audits[raw.record_id]["analysis_valid"] = True
            self._save_audit(run_dir, raw.record_id, audits[raw.record_id])

        self._interrupt_if_requested(run_id, interrupt_after, RunStage.ANALYSIS_VALIDATED)

        early_candidates = tuple(
            DedupItem(
                candidate=normalized[raw.record_id],
                provenance_score=1.0,
                quality_score=_quality_score(evidence[raw.record_id]),
            )
            for raw in raw_records
            if raw.record_id in analyses
            and raw.record_id not in rejected
            and raw.record_id not in terminal_accepted
        )
        early = ExactDeduper().deduplicate(early_candidates) if early_candidates else None
        dropped_normalized = set(early.dropped_record_ids if early is not None else ())
        for raw in raw_records:
            if (
                raw.record_id in rejected
                or raw.record_id in terminal_accepted
                or raw.record_id not in analyses
            ):
                continue
            candidate = normalized[raw.record_id]
            if candidate.record_id in dropped_normalized:
                self._mark_rejected(
                    state, run_id, run_dir, raw.record_id, "DUPLICATE", rejected, audits
                )
                continue
            fp = _hash_json({"exact_hash": ExactDeduper().fingerprint(candidate)})
            if state.current_stage(run_id, raw.record_id) is RunStage.ANALYSIS_VALIDATED:
                state.advance(run_id, raw.record_id, RunStage.EARLY_DEDUPED, fp)
            audits[raw.record_id]["after_dedup"] = True
            self._save_audit(run_dir, raw.record_id, audits[raw.record_id])

        self._interrupt_if_requested(run_id, interrupt_after, RunStage.EARLY_DEDUPED)

        verified: set[str] = set()
        for raw in raw_records:
            if raw.record_id in rejected or raw.record_id in terminal_accepted:
                continue
            current = state.current_stage(run_id, raw.record_id)
            if current not in {RunStage.EARLY_DEDUPED, RunStage.VERIFIED, RunStage.FINAL_DEDUPED}:
                continue
            candidate = normalized[raw.record_id]
            outcome = self._stage_verify(run_dir, raw.record_id, candidate)
            evidence[raw.record_id].extend(outcome.evidence)
            audits[raw.record_id]["alignment_pass"] = outcome.alignment_pass
            audits[raw.record_id]["correctness_pass"] = outcome.correctness_pass
            if outcome.reject_reason is not None:
                self._mark_rejected(
                    state,
                    run_id,
                    run_dir,
                    raw.record_id,
                    outcome.reject_reason,
                    rejected,
                    audits,
                )
                continue
            if current is RunStage.EARLY_DEDUPED:
                state.advance(
                    run_id,
                    raw.record_id,
                    RunStage.VERIFIED,
                    self._verification_fingerprint(candidate),
                )
            verified.add(raw.record_id)
            self._save_audit(run_dir, raw.record_id, audits[raw.record_id])

        self._interrupt_if_requested(run_id, interrupt_after, RunStage.VERIFIED)

        final_items = tuple(
            DedupItem(
                candidate=normalized[record_id],
                provenance_score=1.0,
                quality_score=_quality_score(evidence[record_id]),
            )
            for record_id in verified
        )
        final = ExactDeduper().deduplicate(final_items) if final_items else None
        final_dropped = set(final.dropped_record_ids if final is not None else ())
        accepted_by_id = dict(terminal_accepted)

        for raw in raw_records:
            if raw.record_id not in verified or raw.record_id in rejected:
                continue
            candidate = normalized[raw.record_id]
            if candidate.record_id in final_dropped:
                self._mark_rejected(
                    state, run_id, run_dir, raw.record_id, "DUPLICATE", rejected, audits
                )
                continue
            current = state.current_stage(run_id, raw.record_id)
            if current is RunStage.VERIFIED:
                state.advance(
                    run_id,
                    raw.record_id,
                    RunStage.FINAL_DEDUPED,
                    _hash_json({"final_exact_hash": ExactDeduper().fingerprint(candidate)}),
                )
            try:
                ir = self._build_ir(
                    raw,
                    candidate,
                    classifications[raw.record_id],
                    answers[raw.record_id],
                    analyses[raw.record_id],
                    tuple(evidence[raw.record_id]),
                )
                to_final_record(ir)
            except ValueError as exc:
                reason = (
                    "LANGUAGE_UNRESOLVED"
                    if str(exc) == "LANGUAGE_UNRESOLVED"
                    else "SCHEMA_VALIDATION_FAILED"
                )
                self._mark_rejected(state, run_id, run_dir, raw.record_id, reason, rejected, audits)
                continue

            accepted_fp = _hash_json({"ir": ir.model_dump(mode="json")})
            self._write_stage_artifact(
                run_dir,
                raw.record_id,
                "accepted",
                accepted_fp,
                {"ir": ir.model_dump(mode="json")},
            )
            if state.current_stage(run_id, raw.record_id) is RunStage.FINAL_DEDUPED:
                state.advance(run_id, raw.record_id, RunStage.ACCEPTED, accepted_fp)
            audits[raw.record_id]["accepted"] = True
            self._save_audit(run_dir, raw.record_id, audits[raw.record_id])
            accepted_by_id[raw.record_id] = ir

        self._interrupt_if_requested(run_id, interrupt_after, RunStage.ACCEPTED)

        accepted_irs = [
            accepted_by_id[raw.record_id] for raw in raw_records if raw.record_id in accepted_by_id
        ]
        output_path = export_jsonl(
            (ExportRecord(ir=ir, stage=RunStage.ACCEPTED) for ir in accepted_irs),
            output_dir,
        )
        audit_rows = tuple(RecordAudit.model_validate(audits[row.record_id]) for row in raw_records)
        report = build_pilot_report(
            run_id=run_id,
            audits=audit_rows,
            provider_usage=self.processor.provider_usage,
            wall_time_seconds=time.perf_counter() - started,
        )
        report_path = write_pilot_report(report, run_dir / "run_report.json")
        write_pilot_report(report, output_dir / "pilot_report.json")
        return PipelineRunResult(
            run_id=run_id,
            run_dir=run_dir,
            output_path=output_path,
            report_path=report_path,
            accepted_count=report.accepted_count,
            rejected_count=report.rejected_count,
        )

    def _interrupt_if_requested(
        self,
        run_id: str,
        requested: RunStage | None,
        completed: RunStage,
    ) -> None:
        if requested is completed:
            raise PipelineInterrupted(run_id, completed)

    def _stage_classification(
        self,
        run_dir: Path,
        record_id: str,
        candidate: NormalizedQA,
    ) -> _ClassificationOutcome:
        fingerprint = self._classification_fingerprint(candidate)
        artifact = self._load_stage_artifact(run_dir, record_id, "classification", fingerprint)
        if artifact is not None:
            classification_raw = artifact.get("classification")
            classification = (
                _classification_from_json(classification_raw)
                if classification_raw is not None
                else None
            )
            return _ClassificationOutcome(
                classification,
                _evidence_tuple(artifact.get("evidence")),
                _optional_text(artifact.get("reject_reason")),
            )
        outcome = self.processor.classify(candidate)
        self._write_stage_artifact(
            run_dir,
            record_id,
            "classification",
            fingerprint,
            {
                "classification": (
                    outcome.classification.model_dump(mode="json")
                    if outcome.classification is not None
                    else None
                ),
                "evidence": [item.model_dump(mode="json") for item in outcome.evidence],
                "reject_reason": outcome.reject_reason,
            },
        )
        return outcome

    def _stage_answer(
        self,
        run_dir: Path,
        record_id: str,
        candidate: NormalizedQA,
    ) -> _AnswerOutcome:
        fingerprint = _hash_json(
            {
                "content": _candidate_hash(candidate),
                "threshold": self.config.thresholds.answer_extract,
            }
        )
        artifact = self._load_stage_artifact(run_dir, record_id, "answer", fingerprint)
        if artifact is not None:
            value = artifact.get("answer")
            answer = _answer_from_json(value) if value is not None else None
            return _AnswerOutcome(
                answer,
                _evidence_tuple(artifact.get("evidence")),
                _optional_text(artifact.get("reject_reason")),
            )
        outcome = self.processor.answer(candidate)
        self._write_stage_artifact(
            run_dir,
            record_id,
            "answer",
            fingerprint,
            {
                "answer": outcome.answer.model_dump(mode="json") if outcome.answer else None,
                "evidence": [item.model_dump(mode="json") for item in outcome.evidence],
                "reject_reason": outcome.reject_reason,
            },
        )
        return outcome

    def _stage_analysis(
        self,
        run_dir: Path,
        record_id: str,
        candidate: NormalizedQA,
    ) -> _AnalysisOutcome:
        fingerprint = self._analysis_fingerprint(candidate)
        artifact = self._load_stage_artifact(run_dir, record_id, "analysis", fingerprint)
        if artifact is not None:
            value = artifact.get("analysis")
            analysis = _analysis_from_json(value) if value is not None else None
            return _AnalysisOutcome(
                analysis,
                _evidence_tuple(artifact.get("evidence")),
                _optional_text(artifact.get("reject_reason")),
            )
        outcome = self.processor.analysis(candidate)
        self._write_stage_artifact(
            run_dir,
            record_id,
            "analysis",
            fingerprint,
            {
                "analysis": outcome.analysis.model_dump(mode="json") if outcome.analysis else None,
                "evidence": [item.model_dump(mode="json") for item in outcome.evidence],
                "reject_reason": outcome.reject_reason,
            },
        )
        return outcome

    def _stage_verify(
        self,
        run_dir: Path,
        record_id: str,
        candidate: NormalizedQA,
    ) -> _VerificationOutcome:
        fingerprint = self._verification_fingerprint(candidate)
        artifact = self._load_stage_artifact(run_dir, record_id, "verify", fingerprint)
        if artifact is not None:
            return _VerificationOutcome(
                _evidence_tuple(artifact.get("evidence")),
                _optional_text(artifact.get("reject_reason")),
                bool(artifact.get("alignment_pass", False)),
                bool(artifact.get("correctness_pass", False)),
            )
        outcome = self.processor.verify(candidate)
        self._write_stage_artifact(
            run_dir,
            record_id,
            "verify",
            fingerprint,
            {
                "evidence": [item.model_dump(mode="json") for item in outcome.evidence],
                "reject_reason": outcome.reject_reason,
                "alignment_pass": outcome.alignment_pass,
                "correctness_pass": outcome.correctness_pass,
            },
        )
        return outcome

    def _classification_fingerprint(self, candidate: NormalizedQA) -> str:
        return _hash_json(
            {
                "content": _candidate_hash(candidate),
                "university_threshold": self.config.thresholds.university,
                "problem_threshold": self.config.thresholds.problem,
                "university_prompt": self.config.prompt_versions.university_classify,
                "problem_prompt": self.config.prompt_versions.problem_classify,
                "classifier": self.config.providers.classifier.model,
                "verifier": self.config.providers.verifier.model,
            }
        )

    def _analysis_fingerprint(self, candidate: NormalizedQA) -> str:
        return _hash_json(
            {
                "content": _candidate_hash(candidate),
                "threshold": self.config.thresholds.analysis,
                "prompt": self.config.prompt_versions.analysis_classify,
                "classifier": self.config.providers.classifier.model,
                "verifier": self.config.providers.verifier.model,
            }
        )

    def _verification_fingerprint(self, candidate: NormalizedQA) -> str:
        return _hash_json(
            {
                "content": _candidate_hash(candidate),
                "alignment_threshold": self.config.thresholds.qa_alignment,
                "alignment_prompt": self.config.prompt_versions.qa_alignment,
                "correctness_prompt": self.config.prompt_versions.correctness_verify,
                "verifier": self.config.providers.verifier.model,
            }
        )

    def _build_ir(
        self,
        raw: RawSourceRecord,
        candidate: NormalizedQA,
        classification: Classification,
        answer: AnswerContent,
        analysis: AnalysisContent,
        evidence: tuple[GateResultEvidence, ...],
    ) -> UniversityQuestionIR:
        if candidate.images and any(not image.startswith("image/") for image in candidate.images):
            raise ValueError("unresolved source image cannot enter formal Task 14 export")
        language = _metadata_text(candidate, "language")
        if language is None or len(language) != 2 or language.lower() != language:
            raise ValueError("LANGUAGE_UNRESOLVED")
        return UniversityQuestionIR(
            candidate_id=candidate.record_id,
            source_record_id=raw.record_id,
            question=QuestionContent(
                raw=raw.raw_question,
                normalized=candidate.question,
                assets=tuple(candidate.images),
            ),
            answer=answer,
            analysis=analysis,
            classification=classification,
            metadata=QuestionMetadata(
                knowledge_points=_metadata_text_tuple(candidate, "knowledge_points"),
                exam_points=_metadata_text_tuple(candidate, "exam_points"),
                language=language,
            ),
            quality=QualityState(gates=tuple(evidence)),
            provenance=QuestionProvenance(
                source_dataset=raw.source_dataset,
                source_id=raw.source_id,
                source_url=raw.source_url,
                raw_sha256=raw.raw_sha256,
            ),
            dedup=DedupState(
                exact_hash=ExactDeduper().fingerprint(candidate),
                duplicate_of=None,
                candidate_ids=(),
            ),
        )

    def _load_or_acquire(
        self,
        adapter: SourceAdapter,
        descriptors: tuple[SourceDescriptor, ...],
        source_snapshot_hash: str,
    ) -> tuple[RawSourceRecord, ...]:
        cache_path = self._source_cache_path(source_snapshot_hash)
        if cache_path.is_file():
            return self._load_source_cache(source_snapshot_hash)
        records: list[RawSourceRecord] = []
        seen: set[str] = set()
        for descriptor in descriptors:
            for record in adapter.acquire(descriptor):
                if record.record_id in seen:
                    raise ValueError(f"duplicate acquired record_id: {record.record_id}")
                seen.add(record.record_id)
                records.append(record)
        if not records:
            raise ValueError("source acquisition returned no records")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in records
        )
        _atomic_write_text(cache_path, payload)
        return tuple(records)

    def _load_source_cache(self, source_snapshot_hash: str) -> tuple[RawSourceRecord, ...]:
        path = self._source_cache_path(source_snapshot_hash)
        if not path.is_file():
            raise ValueError("source snapshot cache is missing")
        rows = tuple(
            RawSourceRecord.model_validate_json(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if not rows:
            raise ValueError("source snapshot cache is empty")
        return rows

    def _source_cache_path(self, source_snapshot_hash: str) -> Path:
        return self.workspace / "cache" / "sources" / f"{source_snapshot_hash}.jsonl"

    def _load_or_normalize(self, raw: RawSourceRecord) -> NormalizedQA:
        fingerprint = _hash_json(
            {"raw_sha256": raw.raw_sha256, "normalization_version": NORMALIZATION_VERSION}
        )
        path = self.workspace / "cache" / "normalized" / f"{fingerprint}.json"
        if path.is_file():
            return NormalizedQA.model_validate_json(path.read_text(encoding="utf-8"))
        candidate = self._normalizer(raw)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            path,
            json.dumps(
                candidate.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )
        return candidate

    def _write_manifest(self, run_dir: Path, manifest: Mapping[str, JsonValue]) -> None:
        path = run_dir / "manifest.json"
        if path.exists():
            existing = _load_json_object(path)
            if existing != dict(manifest):
                raise ValueError("existing run manifest differs from computed fingerprint")
            return
        _atomic_write_text(
            path,
            json.dumps(dict(manifest), sort_keys=True, separators=(",", ":")) + "\n",
        )

    def _mark_rejected(
        self,
        state: RunStateStore,
        run_id: str,
        run_dir: Path,
        record_id: str,
        reason: str,
        rejected: set[str],
        audits: dict[str, dict[str, Any]],
    ) -> None:
        current = state.current_stage(run_id, record_id)
        if current is not RunStage.REJECTED:
            state.advance(run_id, record_id, RunStage.REJECTED, _hash_json({"reason": reason}))
        rejected.add(record_id)
        audits[record_id]["reject_reason"] = reason
        audits[record_id]["accepted"] = False
        self._save_audit(run_dir, record_id, audits[record_id])

    def _stage_artifact_path(self, run_dir: Path, record_id: str, stage: str) -> Path:
        safe_id = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        return run_dir / "stages" / safe_id / f"{stage}.json"

    def _audit_path(self, run_dir: Path, record_id: str) -> Path:
        safe_id = hashlib.sha256(record_id.encode("utf-8")).hexdigest()
        return run_dir / "audits" / f"{safe_id}.json"

    def _save_audit(self, run_dir: Path, record_id: str, value: Mapping[str, Any]) -> None:
        _atomic_write_text(
            self._audit_path(run_dir, record_id),
            json.dumps(dict(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + "\n",
        )

    def _load_audit(self, run_dir: Path, record_id: str) -> dict[str, Any] | None:
        path = self._audit_path(run_dir, record_id)
        return _load_json_object(path) if path.is_file() else None

    def _load_terminal_ir(self, run_dir: Path, record_id: str) -> UniversityQuestionIR:
        path = self._stage_artifact_path(run_dir, record_id, "accepted")
        if not path.is_file():
            raise RuntimeError(f"accepted record {record_id} is missing its terminal IR artifact")
        payload = _load_json_object(path)
        value = payload.get("value")
        if not isinstance(value, dict) or "ir" not in value:
            raise RuntimeError(f"accepted record {record_id} has malformed terminal IR artifact")
        return _ir_from_json(value["ir"])

    def _load_stage_artifact(
        self,
        run_dir: Path,
        record_id: str,
        stage: str,
        fingerprint: str,
    ) -> dict[str, Any] | None:
        path = self._stage_artifact_path(run_dir, record_id, stage)
        if not path.is_file():
            return None
        payload = _load_json_object(path)
        if payload.get("fingerprint") != fingerprint:
            return None
        value = payload.get("value")
        return cast(dict[str, Any], value) if isinstance(value, dict) else None

    def _write_stage_artifact(
        self,
        run_dir: Path,
        record_id: str,
        stage: str,
        fingerprint: str,
        value: Mapping[str, Any],
    ) -> None:
        path = self._stage_artifact_path(run_dir, record_id, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(
            path,
            json.dumps(
                {"fingerprint": fingerprint, "value": dict(value)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
        )


def _source_snapshot_hash(descriptors: tuple[SourceDescriptor, ...]) -> str:
    values = [descriptor.model_dump(mode="json") for descriptor in descriptors]
    values.sort(key=lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")))
    return _hash_json(values)


def _run_id(source_snapshot_hash: str, config_hash: str, pipeline_version: str) -> str:
    fingerprint = _hash_json(
        {
            "source_snapshot_hash": source_snapshot_hash,
            "config_hash": config_hash,
            "pipeline_version": pipeline_version,
        }
    )
    return f"run-{fingerprint}"


def _hash_json(value: object) -> str:
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _candidate_hash(candidate: NormalizedQA) -> str:
    return _hash_json(
        {
            "question": candidate.question,
            "answer": candidate.answer,
            "analysis": candidate.analysis,
            "metadata": candidate.model_dump(mode="json")["metadata"],
        }
    )


def _raw_identity(raw: RawSourceRecord) -> dict[str, str]:
    return {
        "record_id": raw.record_id,
        "source_dataset": raw.source_dataset,
        "source_id": raw.source_id,
    }


def _audit_facts(raw: RawSourceRecord) -> dict[str, Any]:
    return {
        "record_id": raw.record_id,
        "source_dataset": raw.source_dataset,
        "subject": "UNKNOWN",
        "normalized": False,
        "university_stem": False,
        "problem": False,
        "answer_valid": False,
        "analysis_valid": False,
        "after_dedup": False,
        "alignment_pass": False,
        "correctness_pass": False,
        "accepted": False,
        "reject_reason": None,
    }


def _decision_label(evidence: GateResultEvidence) -> str | None:
    primary = evidence.evidence_payload.get("primary")
    if not isinstance(primary, dict):
        return None
    label = primary.get("label")
    return label if isinstance(label, str) else None


def _discipline_from_metadata(candidate: NormalizedQA) -> Discipline:
    value = _metadata_text(candidate, "discipline")
    if value is not None:
        try:
            discipline = Discipline(value)
            if discipline is not Discipline.NON_STEM:
                return discipline
        except ValueError:
            pass
    return Discipline.OTHER_STEM


def _university_level_from_metadata(candidate: NormalizedQA) -> UniversityLevel:
    value = _metadata_text(candidate, "university_level")
    if value is not None:
        try:
            level = UniversityLevel(value)
            if level is not UniversityLevel.NON_UNIVERSITY:
                return level
        except ValueError:
            pass
    return UniversityLevel.UNIVERSITY_UNKNOWN


def _metadata_text(candidate: NormalizedQA, key: str) -> str | None:
    metadata = candidate.model_dump(mode="json")["metadata"]
    value = metadata.get(key) if isinstance(metadata, dict) else None
    return value if isinstance(value, str) and value.strip() else None


def _metadata_text_tuple(candidate: NormalizedQA, key: str) -> tuple[str, ...]:
    metadata = candidate.model_dump(mode="json")["metadata"]
    value = metadata.get(key) if isinstance(metadata, dict) else None
    if not isinstance(value, list):
        return ()
    if not all(isinstance(item, str) and item.strip() for item in value):
        return ()
    return tuple(cast(list[str], value))


def _quality_score(evidence: Iterable[GateResultEvidence]) -> float:
    scores = [item.score for item in evidence if item.verdict is not GateVerdict.REJECT]
    return sum(scores) / len(scores) if scores else 0.0


def _classification_from_json(value: object) -> Classification:
    if not isinstance(value, dict):
        raise ValueError("classification artifact must be a JSON object")
    discipline = value.get("discipline")
    level = value.get("level")
    problem_type = value.get("problem_type")
    course = value.get("course")
    if (
        not isinstance(discipline, str)
        or not isinstance(level, str)
        or not isinstance(problem_type, str)
    ):
        raise ValueError("classification artifact enum values must be strings")
    if course is not None and not isinstance(course, str):
        raise ValueError("classification artifact course must be text or null")
    return Classification(
        discipline=Discipline(discipline),
        course=course,
        level=UniversityLevel(level),
        problem_type=ProblemType(problem_type),
    )


def _answer_from_json(value: object) -> AnswerContent:
    if not isinstance(value, dict):
        raise ValueError("answer artifact must be a JSON object")
    raw = value.get("raw")
    final_answer = value.get("final_answer")
    source_span = value.get("source_span")
    if not isinstance(raw, str):
        raise ValueError("answer artifact raw must be text")
    if final_answer is not None and not isinstance(final_answer, str):
        raise ValueError("answer artifact final_answer must be text or null")
    if source_span is not None and not isinstance(source_span, str):
        raise ValueError("answer artifact source_span must be text or null")
    return AnswerContent(raw=raw, final_answer=final_answer, source_span=source_span)


def _analysis_from_json(value: object) -> AnalysisContent:
    if not isinstance(value, dict):
        raise ValueError("analysis artifact must be a JSON object")
    raw = value.get("raw")
    type_value = value.get("type")
    if not isinstance(raw, str):
        raise ValueError("analysis artifact raw must be text")
    parsed_type = None if type_value is None else AnalysisType(str(type_value))
    return AnalysisContent(raw=raw, type=parsed_type)


def _ir_from_json(value: object) -> UniversityQuestionIR:
    if not isinstance(value, dict):
        raise ValueError("terminal IR artifact must be a JSON object")
    serialized = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return UniversityQuestionIR.model_validate_json(serialized)


def _evidence_tuple(value: object) -> tuple[GateResultEvidence, ...]:
    if not isinstance(value, list):
        return ()
    result: list[GateResultEvidence] = []
    for item in value:
        serialized = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        result.append(GateResultEvidence.model_validate_json(serialized))
    return tuple(result)


def _optional_text(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _load_json_object(path: Path) -> dict[str, Any]:
    parsed: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(parsed, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], parsed)


def _atomic_write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(payload, encoding="utf-8", newline="\n")
    temp.replace(path)
