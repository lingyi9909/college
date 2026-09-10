from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import pytest

from college_builder.config import PipelineConfig
from college_builder.domain.source import RawSourceRecord
from college_builder.normalize.normalizer import normalize
from college_builder.pipeline.runner import (
    GatePipelineProcessor,
    PipelineInterrupted,
    PipelineRunner,
)
from college_builder.providers.base import ModelClassificationRequest, ModelDecision
from college_builder.source.base import AdapterCheckpoint, SourceDescriptor
from college_builder.storage.state import RunStage


class CountingAdapter:
    def __init__(
        self, records: tuple[RawSourceRecord, ...], *, revision: str | None = "fixture-v1"
    ) -> None:
        self.records = records
        self.revision = revision
        self.discover_calls = 0
        self.acquire_calls = 0
        self._descriptor = SourceDescriptor(
            source_type="dataset",
            source_dataset="task14-fixture",
            source_url="https://fixture.invalid/task14",
            locator="local-task14-fixture",
            source_revision=revision,
        )

    def discover(self, config: object) -> Iterable[SourceDescriptor]:
        self.discover_calls += 1
        return (self._descriptor,)

    def acquire(self, descriptor: SourceDescriptor) -> Iterable[RawSourceRecord]:
        assert descriptor == self._descriptor
        self.acquire_calls += 1
        return self.records

    def checkpoint(self) -> AdapterCheckpoint:
        return AdapterCheckpoint(
            descriptor=self._descriptor,
            next_row_offset=len(self.records),
            next_shard=None,
            source_revision=self.revision,
        )


class RuleProvider:
    def __init__(self, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model
        self.calls: Counter[str] = Counter()

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        self.calls[request.task] += 1
        inputs = request.inputs
        question = str(inputs.get("question", ""))

        if request.task == "gate_1_university_stem":
            if "NONUNI" in question:
                return _decision("NON_UNIVERSITY_STEM", 0.999, ("question:classification",))
            if "AMBIG" in question:
                return _decision("UNIVERSITY_STEM", 0.50, ("question:classification",))
            return _decision("UNIVERSITY_STEM", 0.999, ("question:classification",))

        if request.task == "gate_2_problem":
            if "NONPROBLEM" in question:
                return _decision("DISCUSSION", 0.999, ("question:problem",))
            return _decision("CALCULATION", 0.999, ("question:problem",))

        if request.task == "gate_4_original_analysis":
            analysis = str(inputs["analysis"])
            return _decision(
                "STEP_BY_STEP",
                0.999,
                (f"analysis:0-{len(analysis)}",),
            )

        if request.task in {"gate_5_qa_alignment", "independent_correctness_verification"}:
            answer = str(inputs["answer"])
            analysis = str(inputs["analysis"])
            refs = (
                f"question:0-{len(question)}",
                f"answer:0-{len(answer)}",
                f"analysis:0-{len(analysis)}",
            )
            if request.task == "gate_5_qa_alignment" and "MISMATCH" in question:
                return _decision("FAIL", 0.999, refs)
            return _decision("PASS", 0.999, refs)

        raise AssertionError(f"unexpected task: {request.task}")


def _decision(label: str, score: float, refs: tuple[str, ...]) -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        evidence_references=refs,
        reason_code=f"FIXTURE_{label}",
    )


def _config(*, analysis_prompt: str = "v1", problem_threshold: float = 0.98) -> PipelineConfig:
    return PipelineConfig.model_validate(
        {
            "pipeline_version": "university_dataset_builder_0.1.0",
            "config_version": "pilot-task14-v1",
            "profile": "university_stem_v1",
            "thresholds": {
                "university": 0.98,
                "problem": problem_threshold,
                "answer_extract": 0.995,
                "analysis": 0.98,
                "qa_alignment": 0.995,
            },
            "providers": {
                "classifier": {"name": "fake", "model": "fixture-primary"},
                "verifier": {"name": "fake", "model": "fixture-verifier"},
            },
            "prompt_versions": {
                "university_classify": "v1",
                "problem_classify": "v1",
                "analysis_classify": analysis_prompt,
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


def _raw_payload_hash(payload: dict[str, object]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _record(index: int, kind: str, *, question_override: str | None = None) -> RawSourceRecord:
    question = question_override or f"{kind} Q{index}: Calculate {index} + 1"
    answer = str(index + 1)
    analysis = f"Add one to {index}; therefore the source answer is {index + 1}."
    if kind == "MISSING_ANSWER":
        answer = ""
        analysis = "The source discusses the method but does not state a final result."
    elif kind == "MISSING_ANALYSIS":
        analysis = "略"
    payload = {"id": index, "kind": kind, "question": question}
    source_dataset = "fixture-a" if index <= 10 else "fixture-b"
    return RawSourceRecord(
        record_id=f"raw-{index}",
        source_type="dataset",
        source_dataset=source_dataset,
        source_id=str(index),
        source_url=f"https://fixture.invalid/{index}",
        raw_question=question,
        raw_answer=answer,
        raw_analysis=analysis,
        raw_payload=payload,
        metadata={
            "discipline": "MATHEMATICS",
            "university_level": "UNDERGRADUATE",
            "language": "en",
            "knowledge_points": ["arithmetic fixture"],
            "exam_points": ["calculation"],
        },
        license_metadata={"declared": "UNREVIEWED"},
        raw_sha256=_raw_payload_hash(payload),
    )


def _twenty_records() -> tuple[RawSourceRecord, ...]:
    rows: list[RawSourceRecord] = []
    for index in range(1, 9):
        rows.append(_record(index, "VALID"))
    rows.extend((_record(9, "NONUNI"), _record(10, "NONUNI")))
    rows.extend((_record(11, "NONPROBLEM"), _record(12, "NONPROBLEM")))
    rows.append(_record(13, "MISSING_ANSWER"))
    rows.append(_record(14, "MISSING_ANALYSIS"))
    rows.append(_record(15, "MISMATCH"))
    duplicate_question = "DUPLICATE: Calculate 2 + 2"
    rows.append(_record(16, "DUPLICATE", question_override=duplicate_question))
    rows.append(_record(17, "DUPLICATE", question_override=duplicate_question))
    rows.extend((_record(18, "AMBIG"), _record(19, "AMBIG"), _record(20, "AMBIG")))
    return tuple(rows)


def _processor(
    *,
    config: PipelineConfig,
    workspace: Path,
    primary: RuleProvider,
    verifier: RuleProvider,
) -> GatePipelineProcessor:
    def prompt_loader(task: str, version: str) -> str:
        base = Path("prompts") / task / "v1.txt"
        return f"{base.read_text(encoding='utf-8')}\nfixture_prompt_version={version}\n"

    return GatePipelineProcessor(
        config=config,
        cache_path=workspace / "cache" / "calls.sqlite3",
        primary_provider=primary,
        verifier_provider=verifier,
        project_root=Path("."),
        prompt_loader=prompt_loader,
    )


def _runner(
    *,
    config: PipelineConfig,
    workspace: Path,
    primary: RuleProvider | None = None,
    verifier: RuleProvider | None = None,
) -> PipelineRunner:
    primary = primary or RuleProvider(provider="fixture-primary", model="primary-v1")
    verifier = verifier or RuleProvider(provider="fixture-verifier", model="verifier-v1")
    return PipelineRunner(
        config=config,
        workspace=workspace,
        processor=_processor(
            config=config,
            workspace=workspace,
            primary=primary,
            verifier=verifier,
        ),
    )


def _single_rejection(run_dir: Path) -> dict[str, object]:
    paths = list((run_dir / "rejected").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_resume_reuses_acquisition_normalization_and_cached_decisions(tmp_path: Path) -> None:
    records = _twenty_records()
    adapter = CountingAdapter(records)
    config = _config()
    workspace = tmp_path / "workspace"
    primary = RuleProvider(provider="fixture-primary", model="primary-v1")
    verifier = RuleProvider(provider="fixture-verifier", model="verifier-v1")
    normalization_calls = 0

    def counting_normalizer(record: RawSourceRecord):  # type: ignore[no-untyped-def]
        nonlocal normalization_calls
        normalization_calls += 1
        return normalize(record)

    runner = PipelineRunner(
        config=config,
        workspace=workspace,
        processor=_processor(
            config=config,
            workspace=workspace,
            primary=primary,
            verifier=verifier,
        ),
        normalizer=counting_normalizer,
    )

    with pytest.raises(PipelineInterrupted) as interrupted:
        runner.run(
            adapter=adapter,
            source_config={},
            output_dir=tmp_path / "output",
            interrupt_after=RunStage.NORMALIZED,
        )

    run_id = interrupted.value.run_id
    assert adapter.acquire_calls == 1
    assert normalization_calls == 20
    assert sum(primary.calls.values()) == 0
    assert sum(verifier.calls.values()) == 0

    result = runner.resume(run_id=run_id, output_dir=tmp_path / "output")
    assert result.accepted_count == 9
    assert adapter.acquire_calls == 1
    assert normalization_calls == 20

    calls_after_resume = primary.calls + verifier.calls
    runner.resume(run_id=run_id, output_dir=tmp_path / "output")
    assert primary.calls + verifier.calls == calls_after_resume


def test_prompt_or_gate_config_change_invalidates_only_affected_decisions(tmp_path: Path) -> None:
    records = tuple(_record(index, "VALID") for index in range(1, 4))
    adapter = CountingAdapter(records)
    workspace = tmp_path / "workspace"

    first_config = _config()
    first_primary = RuleProvider(provider="fixture-primary", model="primary-v1")
    first_verifier = RuleProvider(provider="fixture-verifier", model="verifier-v1")
    first_runner = PipelineRunner(
        config=first_config,
        workspace=workspace,
        processor=_processor(
            config=first_config,
            workspace=workspace,
            primary=first_primary,
            verifier=first_verifier,
        ),
    )
    first_runner.run(adapter=adapter, source_config={}, output_dir=tmp_path / "first")
    assert adapter.acquire_calls == 1

    second_config = _config(analysis_prompt="v2")
    second_primary = RuleProvider(provider="fixture-primary", model="primary-v1")
    second_verifier = RuleProvider(provider="fixture-verifier", model="verifier-v1")
    second_runner = PipelineRunner(
        config=second_config,
        workspace=workspace,
        processor=_processor(
            config=second_config,
            workspace=workspace,
            primary=second_primary,
            verifier=second_verifier,
        ),
    )
    second_runner.run(adapter=adapter, source_config={}, output_dir=tmp_path / "second")

    assert adapter.acquire_calls == 1
    assert second_primary.calls["gate_1_university_stem"] == 0
    assert second_primary.calls["gate_2_problem"] == 0
    assert second_primary.calls["gate_4_original_analysis"] == 3
    assert second_verifier.calls["gate_5_qa_alignment"] == 0
    assert second_verifier.calls["independent_correctness_verification"] == 0

    third_config = _config(analysis_prompt="v2", problem_threshold=0.99)
    third_primary = RuleProvider(provider="fixture-primary", model="primary-v1")
    third_verifier = RuleProvider(provider="fixture-verifier", model="verifier-v1")
    third_runner = PipelineRunner(
        config=third_config,
        workspace=workspace,
        processor=_processor(
            config=third_config,
            workspace=workspace,
            primary=third_primary,
            verifier=third_verifier,
        ),
    )
    third_runner.run(adapter=adapter, source_config={}, output_dir=tmp_path / "third")

    assert adapter.acquire_calls == 1
    assert third_primary.calls["gate_1_university_stem"] == 0
    assert third_primary.calls["gate_2_problem"] == 3
    assert third_primary.calls["gate_4_original_analysis"] == 0
    assert sum(third_verifier.calls.values()) == 0


def test_resume_report_preserves_usage_from_interrupted_attempt(tmp_path: Path) -> None:
    records = tuple(_record(index, "VALID") for index in range(1, 4))
    adapter = CountingAdapter(records)
    config = _config()
    workspace = tmp_path / "workspace"
    first_primary = RuleProvider(provider="fixture-primary", model="primary-v1")
    first_verifier = RuleProvider(provider="fixture-verifier", model="verifier-v1")
    first_runner = PipelineRunner(
        config=config,
        workspace=workspace,
        processor=_processor(
            config=config,
            workspace=workspace,
            primary=first_primary,
            verifier=first_verifier,
        ),
    )

    with pytest.raises(PipelineInterrupted) as interrupted:
        first_runner.run(
            adapter=adapter,
            source_config={},
            output_dir=tmp_path / "first",
            interrupt_after=RunStage.ANALYSIS_VALIDATED,
        )

    first_calls = first_primary.calls + first_verifier.calls
    assert sum(first_calls.values()) > 0

    second_primary = RuleProvider(provider="fixture-primary", model="primary-v1")
    second_verifier = RuleProvider(provider="fixture-verifier", model="verifier-v1")
    second_runner = PipelineRunner(
        config=config,
        workspace=workspace,
        processor=_processor(
            config=config,
            workspace=workspace,
            primary=second_primary,
            verifier=second_verifier,
        ),
    )
    result = second_runner.resume(run_id=interrupted.value.run_id, output_dir=tmp_path / "second")
    report = json.loads(result.report_path.read_text(encoding="utf-8"))
    second_calls = second_primary.calls + second_verifier.calls
    expected = first_calls + second_calls

    assert report["provider_call_counts"] == dict(sorted(expected.items()))


def test_problem_reject_keeps_reliable_subject_from_source_metadata(tmp_path: Path) -> None:
    config = _config()
    workspace = tmp_path / "workspace"
    result = _runner(config=config, workspace=workspace).run(
        adapter=CountingAdapter((_record(11, "NONPROBLEM"),)),
        source_config={},
        output_dir=tmp_path / "output",
    )
    report = json.loads(result.report_path.read_text(encoding="utf-8"))

    assert report["subject_distribution"]["MATHEMATICS"]["raw"] == 1
    assert "UNKNOWN" not in report["subject_distribution"]


def test_revisionless_source_content_change_creates_new_snapshot_and_run(tmp_path: Path) -> None:
    config = _config()
    workspace = tmp_path / "workspace"
    first_adapter = CountingAdapter(
        (_record(1, "VALID", question_override="Q1: Calculate 1 + 1"),),
        revision=None,
    )
    first = _runner(config=config, workspace=workspace).run(
        adapter=first_adapter,
        source_config={},
        output_dir=tmp_path / "first",
    )

    second_adapter = CountingAdapter(
        (_record(1, "VALID", question_override="Q2: Calculate 2 + 2"),),
        revision=None,
    )
    second = _runner(config=config, workspace=workspace).run(
        adapter=second_adapter,
        source_config={},
        output_dir=tmp_path / "second",
    )
    output = json.loads(second.output_path.read_text(encoding="utf-8").splitlines()[0])

    assert first_adapter.acquire_calls == 1
    assert second_adapter.acquire_calls == 1
    assert first.run_id != second.run_id
    assert "Q2" in output["text_question"]
    assert "Q1" not in output["text_question"]


def test_revisionless_identical_content_keeps_identity_and_reuses_downstream(
    tmp_path: Path,
) -> None:
    config = _config()
    workspace = tmp_path / "workspace"
    records = (_record(1, "VALID"),)
    first_adapter = CountingAdapter(records, revision=None)
    first = _runner(config=config, workspace=workspace).run(
        adapter=first_adapter,
        source_config={},
        output_dir=tmp_path / "first",
    )

    second_primary = RuleProvider(provider="fixture-primary", model="primary-v1")
    second_verifier = RuleProvider(provider="fixture-verifier", model="verifier-v1")
    second_adapter = CountingAdapter(records, revision=None)
    second = _runner(
        config=config,
        workspace=workspace,
        primary=second_primary,
        verifier=second_verifier,
    ).run(
        adapter=second_adapter,
        source_config={},
        output_dir=tmp_path / "second",
    )

    assert first.run_id == second.run_id
    assert first_adapter.acquire_calls == 1
    assert second_adapter.acquire_calls == 1
    assert sum(second_primary.calls.values()) == 0
    assert sum(second_verifier.calls.values()) == 0


def test_integrity_reject_persists_full_evidence_and_resume_uses_it(tmp_path: Path) -> None:
    config = _config()
    workspace = tmp_path / "workspace"
    corrupt = _record(1, "VALID").model_copy(update={"raw_sha256": "0" * 64})
    first = _runner(config=config, workspace=workspace).run(
        adapter=CountingAdapter((corrupt,)),
        source_config={},
        output_dir=tmp_path / "first",
    )
    rejection = _single_rejection(first.run_dir)

    assert rejection["source_record_id"] == corrupt.record_id
    assert rejection["stage"] == "integrity"
    assert rejection["reason_code"] == "SOURCE_CORRUPTED"
    assert rejection["pipeline_version"] == config.pipeline_version
    assert rejection["config_version"] == config.config_version
    assert rejection["prompt_version"] == "not_applicable"
    assert rejection["provider"] == "deterministic"
    assert rejection["model"] == "integrity_v1"
    assert rejection["details"]
    evidence = rejection["evidence"]
    assert isinstance(evidence, list)
    assert evidence[0]["gate_name"] == "gate_0_integrity_provenance"
    assert evidence[0]["reason_code"] == "SOURCE_CORRUPTED"
    assert evidence[0]["evidence_payload"]["expected_raw_sha256"] == "0" * 64

    for audit_path in (first.run_dir / "audits").glob("*.json"):
        audit_path.unlink()

    resumed = _runner(config=config, workspace=workspace).resume(
        run_id=first.run_id,
        output_dir=tmp_path / "resumed",
    )
    resumed_report = json.loads(resumed.report_path.read_text(encoding="utf-8"))
    assert resumed_report["reject_reason_counts"] == {"SOURCE_CORRUPTED": 1}
    assert _single_rejection(resumed.run_dir) == rejection


def test_model_and_duplicate_rejects_persist_identity_and_context(tmp_path: Path) -> None:
    config = _config()

    model_workspace = tmp_path / "model-workspace"
    model_result = _runner(config=config, workspace=model_workspace).run(
        adapter=CountingAdapter((_record(11, "NONPROBLEM"),)),
        source_config={},
        output_dir=tmp_path / "model-output",
    )
    model_rejection = _single_rejection(model_result.run_dir)
    assert model_rejection["stage"] == "classification"
    assert model_rejection["reason_code"] == "NOT_PROBLEM"
    assert model_rejection["provider"] == "fixture-primary"
    assert model_rejection["model"] == "primary-v1"
    assert model_rejection["prompt_version"] == "v1"
    assert model_rejection["config_version"] == config.config_version
    assert any(
        item["gate_name"] == "gate_2_problem" and item["reason_code"] == "NOT_PROBLEM"
        for item in model_rejection["evidence"]
    )

    duplicate_workspace = tmp_path / "duplicate-workspace"
    duplicate_question = "DUPLICATE REVIEW: Calculate 2 + 2"
    duplicate_result = _runner(config=config, workspace=duplicate_workspace).run(
        adapter=CountingAdapter(
            (
                _record(1, "VALID", question_override=duplicate_question),
                _record(2, "VALID", question_override=duplicate_question),
            )
        ),
        source_config={},
        output_dir=tmp_path / "duplicate-output",
    )
    rejection_paths = list((duplicate_result.run_dir / "rejected").glob("*.json"))
    assert len(rejection_paths) == 1
    duplicate_rejection = json.loads(rejection_paths[0].read_text(encoding="utf-8"))
    assert duplicate_rejection["stage"] == "early_dedup"
    assert duplicate_rejection["reason_code"] == "DUPLICATE"
    assert duplicate_rejection["provider"] == "deterministic"
    assert duplicate_rejection["model"] == "pipeline_runner_v1"
    assert duplicate_rejection["prompt_version"] == "not_applicable"
    assert duplicate_rejection["details"]
    assert duplicate_rejection["evidence"][0]["verdict"] == "REJECT"


def test_language_reject_persists_deterministic_context(tmp_path: Path) -> None:
    config = _config()
    workspace = tmp_path / "language-workspace"
    payload = _record(1, "VALID").model_dump(mode="json")
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["language"] = "EN"
    bad_language = RawSourceRecord.model_validate(payload)

    result = _runner(config=config, workspace=workspace).run(
        adapter=CountingAdapter((bad_language,)),
        source_config={},
        output_dir=tmp_path / "language-output",
    )
    rejection = _single_rejection(result.run_dir)

    assert rejection["stage"] == "final_validation"
    assert rejection["reason_code"] == "LANGUAGE_UNRESOLVED"
    assert rejection["provider"] == "deterministic"
    assert rejection["model"] == "pipeline_runner_v1"
    assert rejection["prompt_version"] == "not_applicable"
    assert rejection["details"]
    assert rejection["evidence"][0]["reason_code"] == "LANGUAGE_UNRESOLVED"


@pytest.mark.parametrize("revision", ["main", "master", "latest", "dev", "release-v1"])
def test_mutable_symbolic_source_revision_does_not_authorize_snapshot_cache(
    tmp_path: Path, revision: str
) -> None:
    config = _config()
    workspace = tmp_path / revision
    first_adapter = CountingAdapter(
        (_record(1, "VALID", question_override="Q1: Calculate 1 + 1"),),
        revision=revision,
    )
    first = _runner(config=config, workspace=workspace).run(
        adapter=first_adapter,
        source_config={},
        output_dir=tmp_path / f"{revision}-first",
    )

    second_adapter = CountingAdapter(
        (_record(1, "VALID", question_override="Q2: Calculate 2 + 2"),),
        revision=revision,
    )
    second = _runner(config=config, workspace=workspace).run(
        adapter=second_adapter,
        source_config={},
        output_dir=tmp_path / f"{revision}-second",
    )
    output = json.loads(second.output_path.read_text(encoding="utf-8").splitlines()[0])

    assert first_adapter.acquire_calls == 1
    assert second_adapter.acquire_calls == 1
    assert first.run_id != second.run_id
    assert "Q2" in output["text_question"]
    assert "Q1" not in output["text_question"]


def test_explicit_immutable_revision_can_reuse_acquisition_snapshot(tmp_path: Path) -> None:
    config = _config()
    workspace = tmp_path / "immutable-same"
    revision = "sha256:" + "a" * 64
    records = (_record(1, "VALID"),)
    first_adapter = CountingAdapter(records, revision=revision)
    first = _runner(config=config, workspace=workspace).run(
        adapter=first_adapter,
        source_config={},
        output_dir=tmp_path / "immutable-first",
    )

    second_adapter = CountingAdapter(records, revision=revision)
    second = _runner(config=config, workspace=workspace).run(
        adapter=second_adapter,
        source_config={},
        output_dir=tmp_path / "immutable-second",
    )

    assert first.run_id == second.run_id
    assert first_adapter.acquire_calls == 1
    assert second_adapter.acquire_calls == 0


def test_explicit_immutable_revision_change_creates_new_snapshot_and_run(tmp_path: Path) -> None:
    config = _config()
    workspace = tmp_path / "immutable-change"
    first_adapter = CountingAdapter(
        (_record(1, "VALID", question_override="Q1: Calculate 1 + 1"),),
        revision="sha256:" + "a" * 64,
    )
    first = _runner(config=config, workspace=workspace).run(
        adapter=first_adapter,
        source_config={},
        output_dir=tmp_path / "immutable-change-first",
    )

    second_adapter = CountingAdapter(
        (_record(1, "VALID", question_override="Q2: Calculate 2 + 2"),),
        revision="sha256:" + "b" * 64,
    )
    second = _runner(config=config, workspace=workspace).run(
        adapter=second_adapter,
        source_config={},
        output_dir=tmp_path / "immutable-change-second",
    )
    output = json.loads(second.output_path.read_text(encoding="utf-8").splitlines()[0])

    assert first_adapter.acquire_calls == 1
    assert second_adapter.acquire_calls == 1
    assert first.run_id != second.run_id
    assert "Q2" in output["text_question"]
