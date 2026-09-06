from __future__ import annotations

from college_builder.storage.cache import CallCache


def _key(*, model: str = "model-a", prompt_version: str = "v1") -> str:
    return CallCache.key(
        content_hash="a" * 64,
        gate_name="university",
        provider="provider-a",
        model=model,
        prompt_version=prompt_version,
        config_version="pilot-v1",
    )


def test_cache_key_is_deterministic_and_version_sensitive() -> None:
    baseline = _key()

    assert _key() == baseline
    assert _key(model="model-b") != baseline
    assert _key(prompt_version="v2") != baseline


def test_cached_expensive_call_survives_reopen_and_is_not_repeated(tmp_path) -> None:
    path = tmp_path / "state.sqlite3"
    calls = 0

    def compute() -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"verdict": "PASS", "score": 0.998, "issues": []}

    cache = CallCache(path)
    first = cache.get_or_compute(_key(), compute)

    reopened = CallCache(path)
    second = reopened.get_or_compute(_key(), compute)

    assert first == {"verdict": "PASS", "score": 0.998, "issues": []}
    assert second == first
    assert calls == 1
