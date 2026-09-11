from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one patch context, got {count}")
    return text.replace(old, new, 1)


provider = Path("src/college_builder/providers/openai_compatible.py")
text = provider.read_text(encoding="utf-8")
text = replace_once(
    text,
    "import time\nfrom typing import Any\n",
    "import time\nfrom threading import Lock\nfrom typing import Any\n",
    "provider imports",
)
text = replace_once(
    text,
    '''        self.retry_backoff_seconds = float(retry_backoff_seconds)\n        self._client = client or httpx.Client(timeout=self.timeout_seconds)\n''',
    '''        self.retry_backoff_seconds = float(retry_backoff_seconds)\n        self._client = client or httpx.Client(timeout=self.timeout_seconds)\n        self._usage_lock = Lock()\n        self._total_tokens = 0\n        self._usage_complete = True\n\n    @property\n    def total_tokens(self) -> int | None:\n        """Return consumed tokens only when every successful response exposed usage."""\n        with self._usage_lock:\n            if not self._usage_complete:\n                return None\n            return self._total_tokens\n''',
    "provider usage state",
)
text = replace_once(
    text,
    '''        response.raise_for_status()\n        content = _message_content(response.json())\n        parsed: object = json.loads(content)\n''',
    '''        response.raise_for_status()\n        try:\n            response_payload = response.json()\n        except (json.JSONDecodeError, ValueError):\n            self._mark_usage_unavailable()\n            raise\n        self._record_usage(response_payload)\n        content = _message_content(response_payload)\n        parsed: object = json.loads(content)\n''',
    "provider response parsing",
)
text = replace_once(
    text,
    '''        return ModelDecision.model_validate(parsed)\n\n\ndef _is_retryable(error: Exception) -> bool:\n''',
    '''        return ModelDecision.model_validate(parsed)\n\n    def _record_usage(self, payload: Any) -> None:\n        usage = payload.get("usage") if isinstance(payload, dict) else None\n        total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None\n        valid = (\n            isinstance(total_tokens, int)\n            and not isinstance(total_tokens, bool)\n            and total_tokens >= 0\n        )\n        with self._usage_lock:\n            if valid:\n                self._total_tokens += total_tokens\n            else:\n                self._usage_complete = False\n\n    def _mark_usage_unavailable(self) -> None:\n        with self._usage_lock:\n            self._usage_complete = False\n\n\ndef _is_retryable(error: Exception) -> bool:\n''',
    "provider usage recorder",
)
provider.write_text(text, encoding="utf-8", newline="\n")

reporting = Path("src/college_builder/reporting/pilot_report.py")
text = reporting.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    latency_seconds: float = Field(default=0.0, ge=0.0)\n    tokens: int = Field(default=0, ge=0)\n    estimated_cost_usd: float = Field(default=0.0, ge=0.0)\n''',
    '''    latency_seconds: float = Field(default=0.0, ge=0.0)\n    tokens: int | None = Field(default=None, ge=0)\n    estimated_cost_usd: float | None = Field(default=None, ge=0.0)\n''',
    "provider usage schema",
)
text = replace_once(
    text,
    '''    provider_call_counts: dict[str, int]\n    cache_hit_rate: float = Field(ge=0.0, le=1.0)\n    fallback_count: int = Field(ge=0)\n    provider_errors: int = Field(ge=0)\n    tokens: int = Field(ge=0)\n    estimated_cost_usd: float = Field(ge=0.0)\n''',
    '''    provider_call_counts: dict[str, int]\n    cache_hits: int = Field(default=0, ge=0)\n    cache_misses: int = Field(default=0, ge=0)\n    cache_hit_rate: float = Field(ge=0.0, le=1.0)\n    fallback_count: int = Field(ge=0)\n    provider_errors: int = Field(ge=0)\n    tokens: int | None = Field(default=None, ge=0)\n    tokens_available: bool = False\n    estimated_cost_usd: float | None = Field(default=None, ge=0.0)\n    estimated_cost_usd_available: bool = False\n''',
    "pilot report usage schema",
)
text = replace_once(
    text,
    '''        provider_call_counts=dict(sorted(provider_usage.provider_call_counts.items())),\n        cache_hit_rate=(provider_usage.cache_hits / total_cache) if total_cache else 0.0,\n        fallback_count=provider_usage.fallback_count,\n        provider_errors=provider_usage.provider_errors,\n        tokens=provider_usage.tokens,\n        estimated_cost_usd=provider_usage.estimated_cost_usd,\n''',
    '''        provider_call_counts=dict(sorted(provider_usage.provider_call_counts.items())),\n        cache_hits=provider_usage.cache_hits,\n        cache_misses=provider_usage.cache_misses,\n        cache_hit_rate=(provider_usage.cache_hits / total_cache) if total_cache else 0.0,\n        fallback_count=provider_usage.fallback_count,\n        provider_errors=provider_usage.provider_errors,\n        tokens=provider_usage.tokens,\n        tokens_available=provider_usage.tokens is not None,\n        estimated_cost_usd=provider_usage.estimated_cost_usd,\n        estimated_cost_usd_available=provider_usage.estimated_cost_usd is not None,\n''',
    "pilot report usage values",
)
reporting.write_text(text, encoding="utf-8", newline="\n")

runner = Path("src/college_builder/pipeline/runner.py")
text = runner.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''        cache = CallCache(cache_path)\n        self._usage = _UsageAccumulator(provider_call_counts=Counter())\n''',
    '''        cache = CallCache(cache_path)\n        self._usage = _UsageAccumulator(provider_call_counts=Counter())\n        self._usage_providers = (primary_provider, verifier_provider)\n''',
    "runner usage providers",
)
text = replace_once(
    text,
    '''                latency_seconds=self._usage.latency_seconds,\n                tokens=0,\n                estimated_cost_usd=0.0,\n                fallback_count=0,\n''',
    '''                latency_seconds=self._usage.latency_seconds,\n                tokens=_combined_provider_tokens(self._usage_providers),\n                estimated_cost_usd=None,\n                fallback_count=0,\n''',
    "runner usage report",
)
text = replace_once(
    text,
    '''def _hash_json(value: object) -> str:\n''',
    '''def _combined_provider_tokens(providers: tuple[StructuredModelProvider, ...]) -> int | None:\n    totals: list[int] = []\n    for provider in providers:\n        value = getattr(provider, "total_tokens", None)\n        if not isinstance(value, int) or isinstance(value, bool) or value < 0:\n            return None\n        totals.append(value)\n    return sum(totals)\n\n\ndef _hash_json(value: object) -> str:\n''',
    "runner token combiner",
)
runner.write_text(text, encoding="utf-8", newline="\n")

provider_tests = Path("tests/unit/providers/test_openai_compatible_resilience.py")
text = provider_tests.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''def _valid_response() -> httpx.Response:\n    return httpx.Response(\n        200,\n        json={\n''',
    '''def _valid_response(*, total_tokens: int | None = None) -> httpx.Response:\n    payload: dict[str, object] = {\n''',
    "provider test helper start",
)
text = replace_once(
    text,
    '''            "choices": [\n                {\n                    "message": {\n                        "content": json.dumps(\n                            {\n                                "label": "UNIVERSITY_STEM",\n                                "score": 0.99,\n                                "evidence_references": ["question:0-32"],\n                                "reason_code": "COLLEGE_LEVEL_STEM",\n                            }\n                        )\n                    }\n                }\n            ]\n        },\n    )\n''',
    '''        "choices": [\n            {\n                "message": {\n                    "content": json.dumps(\n                        {\n                            "label": "UNIVERSITY_STEM",\n                            "score": 0.99,\n                            "evidence_references": ["question:0-32"],\n                            "reason_code": "COLLEGE_LEVEL_STEM",\n                        }\n                    )\n                }\n            }\n        ]\n    }\n    if total_tokens is not None:\n        payload["usage"] = {"total_tokens": total_tokens}\n    return httpx.Response(200, json=payload)\n''',
    "provider test helper body",
)
text += '''\n\ndef test_provider_accumulates_real_total_tokens_from_successful_responses() -> None:\n    calls = 0\n\n    def handler(request: httpx.Request) -> httpx.Response:\n        nonlocal calls\n        calls += 1\n        return _valid_response(total_tokens=11 if calls == 1 else 13)\n\n    provider = OpenAICompatibleStructuredModelProvider(\n        base_url="https://model.local/v1",\n        model="classifier-v1",\n        client=httpx.Client(transport=httpx.MockTransport(handler)),\n        max_attempts=1,\n    )\n    provider.classify(_request())\n    provider.classify(_request())\n    assert provider.total_tokens == 24\n\n\ndef test_provider_marks_token_usage_unavailable_if_any_success_omits_usage() -> None:\n    provider = OpenAICompatibleStructuredModelProvider(\n        base_url="https://model.local/v1",\n        model="classifier-v1",\n        client=httpx.Client(transport=httpx.MockTransport(lambda request: _valid_response())),\n        max_attempts=1,\n    )\n    provider.classify(_request())\n    assert provider.total_tokens is None\n\n\ndef test_provider_counts_tokens_from_malformed_success_before_retry() -> None:\n    calls = 0\n\n    def handler(request: httpx.Request) -> httpx.Response:\n        nonlocal calls\n        calls += 1\n        if calls == 1:\n            return httpx.Response(\n                200,\n                request=request,\n                json={\n                    "choices": [{"message": {"content": "not-json"}}],\n                    "usage": {"total_tokens": 7},\n                },\n            )\n        return _valid_response(total_tokens=9)\n\n    provider = OpenAICompatibleStructuredModelProvider(\n        base_url="https://model.local/v1",\n        model="classifier-v1",\n        client=httpx.Client(transport=httpx.MockTransport(handler)),\n        max_attempts=2,\n        retry_backoff_seconds=0.0,\n    )\n    assert provider.classify(_request()).label == "UNIVERSITY_STEM"\n    assert provider.total_tokens == 16\n'''
provider_tests.write_text(text, encoding="utf-8", newline="\n")

report_tests = Path("tests/e2e/test_pipeline_report.py")
text = report_tests.read_text(encoding="utf-8")
text = replace_once(
    text,
    '''    assert report.provider_call_counts["gate_1_university_stem"] == 4\n    assert report.cache_hit_rate == 5 / 12\n    assert report.estimated_cost_usd == 0.0\n''',
    '''    assert report.provider_call_counts["gate_1_university_stem"] == 4\n    assert report.cache_hits == 5\n    assert report.cache_misses == 7\n    assert report.cache_hit_rate == 5 / 12\n    assert report.tokens == 0\n    assert report.tokens_available is True\n    assert report.estimated_cost_usd == 0.0\n    assert report.estimated_cost_usd_available is True\n''',
    "report explicit usage assertions",
)
text += '''\n\ndef test_pilot_report_marks_unknown_tokens_and_cost_unavailable() -> None:\n    report = build_pilot_report(\n        run_id="run-unknown-usage",\n        audits=(),\n        provider_usage=ProviderUsage(),\n        wall_time_seconds=0.0,\n    )\n    assert report.tokens is None\n    assert report.tokens_available is False\n    assert report.estimated_cost_usd is None\n    assert report.estimated_cost_usd_available is False\n'''
report_tests.write_text(text, encoding="utf-8", newline="\n")
