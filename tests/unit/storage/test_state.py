from __future__ import annotations

import pytest

from college_builder.storage.state import RunStage, RunStateStore

PIPELINE = (
    RunStage.ACQUIRED,
    RunStage.NORMALIZED,
    RunStage.CLASSIFIED,
    RunStage.ANSWER_VALIDATED,
    RunStage.ANALYSIS_VALIDATED,
    RunStage.EARLY_DEDUPED,
    RunStage.VERIFIED,
    RunStage.FINAL_DEDUPED,
)


def _advance_to(
    store: RunStateStore,
    run_id: str,
    record_id: str,
    target: RunStage,
) -> None:
    for stage in PIPELINE:
        assert store.advance(run_id, record_id, stage, f"fp-{stage.value}") is True
        if stage is target:
            return
    raise AssertionError(f"target stage {target.value} is not in the pipeline")


def _advance_to_accepted(store: RunStateStore, run_id: str, record_id: str) -> None:
    _advance_to(store, run_id, record_id, RunStage.FINAL_DEDUPED)
    assert store.advance(run_id, record_id, RunStage.ACCEPTED, "fp-ACCEPTED") is True


def test_run_state_follows_approved_two_dedup_pipeline(tmp_path) -> None:
    store = RunStateStore(tmp_path / "state.sqlite3")

    for stage in PIPELINE:
        assert store.advance("run-1", "record-1", stage, f"fp-{stage.value}") is True

    assert store.advance("run-1", "record-1", RunStage.ACCEPTED, "fp-ACCEPTED") is True
    assert store.current_stage("run-1", "record-1") is RunStage.ACCEPTED


def test_run_state_allows_only_next_approved_non_rejection_transition(tmp_path) -> None:
    store = RunStateStore(tmp_path / "state.sqlite3")

    assert store.advance("run-1", "record-1", RunStage.ACQUIRED, "fp-acquired") is True

    with pytest.raises(ValueError, match="expected NORMALIZED"):
        store.advance("run-1", "record-1", RunStage.CLASSIFIED, "fp-classified")

    assert store.advance("run-1", "record-1", RunStage.NORMALIZED, "fp-normalized") is True
    assert store.current_stage("run-1", "record-1") is RunStage.NORMALIZED


def test_reopen_after_verified_resumes_at_final_deduped(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    store = RunStateStore(path)
    _advance_to(store, "run-1", "record-1", RunStage.VERIFIED)

    reopened = RunStateStore(path)

    assert reopened.current_stage("run-1", "record-1") is RunStage.VERIFIED
    assert reopened.resume_stage("run-1", "record-1") is RunStage.FINAL_DEDUPED


@pytest.mark.parametrize("rejection_point", PIPELINE)
def test_any_non_terminal_gate_can_enter_rejected(
    tmp_path,
    rejection_point: RunStage,
) -> None:
    store = RunStateStore(tmp_path / f"{rejection_point.value}.sqlite3")
    _advance_to(store, "run-1", "record-1", rejection_point)

    assert store.advance("run-1", "record-1", RunStage.REJECTED, "fp-REJECTED") is True
    assert store.current_stage("run-1", "record-1") is RunStage.REJECTED
    assert store.resume_stage("run-1", "record-1") is None


def test_accepted_and_rejected_are_terminal(tmp_path) -> None:
    accepted = RunStateStore(tmp_path / "accepted.sqlite3")
    _advance_to_accepted(accepted, "run-1", "accepted-record")

    with pytest.raises(ValueError, match="terminal"):
        accepted.advance(
            "run-1",
            "accepted-record",
            RunStage.REJECTED,
            "fp-rejected-after-accepted",
        )

    rejected = RunStateStore(tmp_path / "rejected.sqlite3")
    _advance_to(rejected, "run-1", "rejected-record", RunStage.CLASSIFIED)
    assert rejected.advance(
        "run-1", "rejected-record", RunStage.REJECTED, "fp-REJECTED"
    ) is True

    with pytest.raises(ValueError, match="terminal"):
        rejected.advance(
            "run-1",
            "rejected-record",
            RunStage.ANSWER_VALIDATED,
            "fp-after-rejected",
        )


def test_same_completed_stage_and_fingerprint_is_idempotent(tmp_path) -> None:
    store = RunStateStore(tmp_path / "state.sqlite3")

    assert store.advance("run-1", "record-1", RunStage.ACQUIRED, "fp-1") is True
    assert store.advance("run-1", "record-1", RunStage.ACQUIRED, "fp-1") is False
    assert store.is_completed("run-1", "record-1", RunStage.ACQUIRED, "fp-1") is True

    with pytest.raises(ValueError, match="fingerprint"):
        store.advance("run-1", "record-1", RunStage.ACQUIRED, "fp-2")


def test_reopen_resumes_after_normalized_without_repeating_completed_stages(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    store = RunStateStore(path)
    assert store.advance("run-1", "record-1", RunStage.ACQUIRED, "fp-acquired") is True
    assert store.advance("run-1", "record-1", RunStage.NORMALIZED, "fp-normalized") is True

    reopened = RunStateStore(path)

    assert reopened.resume_stage("run-1", "record-1") is RunStage.CLASSIFIED
    assert reopened.is_completed("run-1", "record-1", RunStage.ACQUIRED, "fp-acquired")
    assert reopened.is_completed("run-1", "record-1", RunStage.NORMALIZED, "fp-normalized")
    assert reopened.advance(
        "run-1", "record-1", RunStage.NORMALIZED, "fp-normalized"
    ) is False
