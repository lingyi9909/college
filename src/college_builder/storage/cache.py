"""Deterministic persistent cache for expensive provider calls."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path
from typing import cast

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list[JsonValue] | dict[str, JsonValue]


def _require_text(name: str, value: str) -> None:
    if not value:
        raise ValueError(f"{name} must be non-empty")


class CallCache:
    """SQLite-backed call cache keyed only through the canonical key builder."""

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
                CREATE TABLE IF NOT EXISTS call_cache (
                    cache_key TEXT PRIMARY KEY,
                    status TEXT NOT NULL CHECK (status = 'COMPLETED'),
                    value_json TEXT NOT NULL
                )
                """
            )
            connection.commit()

    @staticmethod
    def key(
        content_hash: str,
        gate_name: str,
        provider: str,
        model: str,
        prompt_version: str,
        config_version: str,
    ) -> str:
        """Build the sole canonical cache key for an expensive call."""
        parts = {
            "content_hash": content_hash,
            "gate_name": gate_name,
            "provider": provider,
            "model": model,
            "prompt_version": prompt_version,
            "config_version": config_version,
        }
        for name, value in parts.items():
            _require_text(name, value)
        canonical = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> dict[str, JsonValue] | None:
        """Return a completed cached result, if present."""
        _require_text("cache_key", cache_key)
        with closing(self._connect()) as connection:
            row = connection.execute(
                """
                SELECT value_json
                FROM call_cache
                WHERE cache_key = ? AND status = 'COMPLETED'
                """,
                (cache_key,),
            ).fetchone()
        if row is None:
            return None
        parsed = json.loads(str(row[0]))
        if not isinstance(parsed, dict):
            raise RuntimeError("cached call value must be a JSON object")
        return cast(dict[str, JsonValue], parsed)

    def put(self, cache_key: str, value: Mapping[str, JsonValue]) -> None:
        """Persist a completed result as deterministic sorted-key JSON."""
        _require_text("cache_key", cache_key)
        value_json = json.dumps(
            dict(value),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO call_cache (cache_key, status, value_json)
                    VALUES (?, 'COMPLETED', ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        status = excluded.status,
                        value_json = excluded.value_json
                    """,
                    (cache_key, value_json),
                )

    def get_or_compute(
        self,
        cache_key: str,
        compute: Callable[[], Mapping[str, JsonValue]],
    ) -> dict[str, JsonValue]:
        """Compute once across normal restarts when a completed cache entry exists."""
        cached = self.get(cache_key)
        if cached is not None:
            return cached
        computed = dict(compute())
        self.put(cache_key, computed)
        return computed
