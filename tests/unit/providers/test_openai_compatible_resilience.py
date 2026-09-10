from __future__ import annotations

import json

import httpx
import pytest

from college_builder.providers.base import ModelClassificationRequest
from college_builder.providers.openai_compatible import OpenAICompatibleStructuredModelProvider


def _request() -> ModelClassificationRequest:
    return ModelClassificationRequest(
        task="gate_1_university_stem",
        prompt="Return classification JSON only.",
        inputs={"question": "Compute the eigenvalues of A."},
        allowed_labels=("UNIVERSITY_STEM", "K12"),
    )


def _valid_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "label": "UNIVERSITY_STEM",
                                "score": 0.99,
                                "evidence_references": ["question:0-32"],
                                "reason_code": "COLLEGE_LEVEL_STEM",
                            }
                        )
                    }
                }
            ]
        },
    )


def test_provider_retries_read_timeout_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("transient", request=request)
        return _valid_response()

    provider = OpenAICompatibleStructuredModelProvider(
        base_url="https://model.local/v1",
        model="classifier-v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_attempts=2,
        retry_backoff_seconds=0.0,
    )

    assert provider.classify(_request()).label == "UNIVERSITY_STEM"
    assert calls == 2


def test_provider_retries_retryable_http_status_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, request=request)
        return _valid_response()

    provider = OpenAICompatibleStructuredModelProvider(
        base_url="https://model.local/v1",
        model="classifier-v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_attempts=2,
        retry_backoff_seconds=0.0,
    )

    assert provider.classify(_request()).label == "UNIVERSITY_STEM"
    assert calls == 2


def test_provider_does_not_retry_permanent_http_400() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(400, request=request)

    provider = OpenAICompatibleStructuredModelProvider(
        base_url="https://model.local/v1",
        model="classifier-v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_attempts=3,
        retry_backoff_seconds=0.0,
    )

    with pytest.raises(httpx.HTTPStatusError):
        provider.classify(_request())
    assert calls == 1


def test_provider_retries_malformed_model_decision_then_succeeds() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                request=request,
                json={
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "label": "UNIVERSITY_STEM",
                                        "score": 0.99,
                                        "reason_code": "COLLEGE_LEVEL_STEM",
                                    }
                                )
                            }
                        }
                    ]
                },
            )
        return _valid_response()

    provider = OpenAICompatibleStructuredModelProvider(
        base_url="https://model.local/v1",
        model="classifier-v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        max_attempts=2,
        retry_backoff_seconds=0.0,
    )

    assert provider.classify(_request()).label == "UNIVERSITY_STEM"
    assert calls == 2
