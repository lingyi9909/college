from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from college_builder.providers.base import ModelClassificationRequest
from college_builder.providers.openai_compatible import (
    OpenAICompatibleStructuredModelProvider,
)


def _request() -> ModelClassificationRequest:
    return ModelClassificationRequest(
        task="gate_1_university_stem",
        prompt="Return classification JSON only.",
        inputs={"question": "Compute the eigenvalues of A."},
        allowed_labels=("UNIVERSITY_STEM", "K12"),
    )


def test_openai_compatible_adapter_returns_domain_model_decision_without_vendor_dto() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == httpx.URL("https://model.local/v1/chat/completions")
        payload = json.loads(request.content)
        assert payload["model"] == "classifier-v1"
        assert payload["temperature"] == 0
        assert payload["response_format"] == {"type": "json_object"}
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

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleStructuredModelProvider(
        base_url="https://model.local/v1",
        model="classifier-v1",
        api_key="secret",
        client=client,
    )

    decision = provider.classify(_request())

    assert decision.label == "UNIVERSITY_STEM"
    assert decision.score == 0.99
    assert provider.provider == "openai_compatible"
    assert provider.model == "classifier-v1"


def test_openai_compatible_adapter_rejects_malformed_model_decision() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "label": "UNIVERSITY_STEM",
                                    "score": 1.5,
                                    "evidence_references": ["question:0-32"],
                                    "reason_code": "INVALID",
                                }
                            )
                        }
                    }
                ]
            },
        )

    provider = OpenAICompatibleStructuredModelProvider(
        base_url="https://model.local/v1",
        model="classifier-v1",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(ValidationError):
        provider.classify(_request())


def test_openai_compatible_adapter_requires_endpoint_and_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_COMPATIBLE_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPATIBLE_MODEL", raising=False)

    with pytest.raises(ValueError, match="base URL"):
        OpenAICompatibleStructuredModelProvider(model="classifier-v1")

    with pytest.raises(ValueError, match="model"):
        OpenAICompatibleStructuredModelProvider(base_url="https://model.local/v1")
