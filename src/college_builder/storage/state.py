"""Crash-resilient SQLite run/stage state."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from enum import StrEnum
from pathlib import Path


class RunStage(StrEnum):
    """Approved Task 3 pipeline stages."""

    ACQUIRED = "ACQUIRED"
    NORMALIZED = "NORMALIZED"
    CLASSIFIED = "CLASSIFIED"
    ANSWER_VALIDATED = "ANSWER_VALIDATED"
    ANALYSIS_VALIDATED = "ANALYSIS_VALIDATED"
    EARLY_DEDUPED = "EARLY_DEDUPED"
    VERIFIED = "VERIFIED"
    FINAL_DEDUPED = "FINAL_DEDUPED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"


_LINEAR_STAGES = (
    RunStage.ACQUIRED,
    RunStage.NORMALIZED,
    RunStage.CLASSIFIED,
    RunStage.ANSWER_VALIDATED,
    RunStage.ANALYSIS_VALIDATED,
    RunStage.EARLY_DEDUPED,
    RunStage.VERIFIED,
    RunStage.FINAL_DEDUPED,
)
_TERMINAL_STAGES = frozenset({RunStage.ACCEPTED, RunStage.REJECTED})
_STAGE_ORDER = {
    stage: index
    for index, stage in enumerate((*_LINEAR_STAGES, RunStage.ACCEPTED, RunStage.REJECTED))
}
_NEXT_STAGE = {
    stage: _LINEAR_STAGES[index + 1]
    for index, stage in enumerate(_LINEAR_STAGES[:-1])
}


def _require_text(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


class RunStateStore:
    """Persist completed per-record stages and compute crash-resume position."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._path, timeout=30.0)

    def _initialize(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS record_stage_state (
                    run_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    PRIMARY KEY (run_id, record_id, stage)
                )
                """
            )
            connection.commit()

    def _current_stage(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        record_id: str,
    ) -> RunStage | None:
        rows = connection.execute(
            """
            SELECT stage
            FROM record_stage_state
            WHERE run_id = ? AND record_id = ?
            """,
            (run_id, record_id),
        ).fetchall()
        if not rows:
            return None
        stages = [RunStage(str(row[0])) for row in rows]
        return max(stages, key=_STAGE_ORDER.__getitem__)

    def current_stage(self, run_id: str, record_id: str) -> RunStage | None:
        """Return the highest completed stage for a record."""
        _require_text("run_id", run_id)
        _require_text("record_id", record_id)
        with closing(self._connect()) as connection:
            return self._current_stage(connection, run_id, record_id)

    def resume_stage(self, run_id: str, record_id: str) -> RunStage | None:
        """Return the next deterministic stage after a crash."""
        current = self.current_stage(run_id, record_id)
        if current is None:
            return RunStage.ACQUIRED
        if current in _TERMINAL_STAGES:
            return None
        if current is RunStage.FINAL_DEDUPED:
            return RunStage.ACCEPTED
        return _NEXT_STAGE[current]

    def is_completed(
        self,
        run_id: str,
        record_id: str,
        stage: RunStage,
        fingerprint: str,
    ) -> bool:
        """Check whether the exact stage fingerprint was already persisted."""
        _require_text("run_id", run_id)
        _require_text("record_id", record_id)
        _require_text("fingerprint", fingerprint)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT 1
                FROM record_stage_state
                WHERE run_id = ? AND record_id = ? AND stage = ? AND fingerprint = ?
                """,
                (run_id, record_id, stage.value, fingerprint),
            ).fetchone()
        return row is not None

    def advance(
        self,
        run_id: str,
        record_id: str,
        stage: RunStage,
        fingerprint: str,
    ) -> bool:
        """Persist one approved transition; return False for an exact idempotent replay."""
        _require_text("run_id", run_id)
        _require_text("record_id", record_id)
        _require_text("fingerprint", fingerprint)

        with closing(self._connect()) as connection:
            with connection:
                existing = connection.execute(
                    """
                    SELECT fingerprint
                    FROM record_stage_state
                    WHERE run_id = ? AND record_id = ? AND stage = ?
                    """,
                    (run_id, record_id, stage.value),
                ).fetchone()
                if existing is not None and str(existing[0]) == fingerprint:
                    return False

                current = self._current_stage(connection, run_id, record_id)
                if current in _TERMINAL_STAGES:
                    raise ValueError(f"{current.value} is terminal")
                if existing is not None:
                    raise ValueError(
                        f"{stage.value} already completed with a different fingerprint"
                    )

                if stage is RunStage.REJECTED:
                    if current is None:
                        raise ValueError("REJECTED requires a completed non-terminal gate")
                else:
                    if current is None:
                        expected = RunStage.ACQUIRED
                    elif current is RunStage.FINAL_DEDUPED:
                        expected = RunStage.ACCEPTED
                    else:
                        expected = _NEXT_STAGE[current]
                    if stage is not expected:
                        raise ValueError(
                            f"invalid transition: expected {expected.value}, got {stage.value}"
                        )

                connection.execute(
                    """
                    INSERT INTO record_stage_state (run_id, record_id, stage, fingerprint)
                    VALUES (?, ?, ?, ?)
                    """,
                    (run_id, record_id, stage.value, fingerprint),
                )
        return True
