from __future__ import annotations

import pytest

from college_builder.storage.state import RunStage, RunStateStore

PIPELINE = (
    RunStage.ACQUIRED,
    RunStage.NORMALIZED,
    RunStage.CLASSIFIED,
    RunStage.ANSWER_VALIDATED,
    RunStage.ANALYSIS_VALIDATED,
    RunStage.DEDUPED,
    RunStage.VERIFIED,
)


def _advance_to_verified(store: RunStateStore, run_id: str, record_id: str) -> None:
    for stage in PIPELINE:
        assert store.advance(run_id, record_id, stage, f"fp-{stage.value}") is True


def test_run_state_allows_only_approved_transitions(tmp_path) -> None:
    store = RunStateStore(tmp_path / "state.sqlite3")

    assert store.advance("run-1", "record-1", RunStage.ACQUIRED, "fp-acquired") is True

    with pytest.raises(ValueError, match="expected NORMALIZED"):
        store.advance("run-1", "record-1", RunStage.CLASSIFIED, "fp-classified")

    assert store.advance("run-1", "record-1", RunStage.NORMALIZED, "fp-normalized") is True
    assert store.current_stage("run-1", "record-1") is RunStage.NORMALIZED


@pytest.mark.parametrize("terminal", [RunStage.ACCEPTED, RunStage.REJECTED])
def test_verified_record_can_enter_only_terminal_state(tmp_path, terminal: RunStage) -> None:
    store = RunStateStore(tmp_path / f"{terminal.value}.sqlite3")
    _advance_to_verified(store, "run-1", "record-1")

    assert store.advance("run-1", "record-1", terminal, f"fp-{terminal.value}") is True
    assert store.current_stage("run-1", "record-1") is terminal

    with pytest.raises(ValueError, match="terminal"):
        store.advance("run-1", "record-1", RunStage.VERIFIED, "another")


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
