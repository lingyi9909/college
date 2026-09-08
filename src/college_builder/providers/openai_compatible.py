"""OpenAI-compatible HTTP adapter for structured classification decisions."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from college_builder.providers.base import ModelClassificationRequest, ModelDecision


class OpenAICompatibleStructuredModelProvider:
    """Translate provider-neutral classification requests to an OpenAI-compatible API."""

    provider = "openai_compatible"

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        resolved_base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        resolved_model = model or os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not isinstance(resolved_base_url, str) or not resolved_base_url.strip():
            raise ValueError("OpenAI-compatible base URL must be configured")
        if not isinstance(resolved_model, str) or not resolved_model.strip():
            raise ValueError("OpenAI-compatible model must be configured")

        self.base_url = resolved_base_url.rstrip("/")
        self.model = resolved_model
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        self._client = client or httpx.Client(timeout=30.0)

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json={
                "model": self.model,
                "temperature": 0,
                "response_format": {"type": "json_object"},
                "messages": [
                    {"role": "system", "content": request.prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "task": request.task,
                                "inputs": request.inputs,
                                "allowed_labels": list(request.allowed_labels),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    },
                ],
            },
        )
        response.raise_for_status()
        content = _message_content(response.json())
        parsed: object = json.loads(content)
        return ModelDecision.model_validate(parsed)


def _message_content(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI-compatible response is missing message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI-compatible response message content must be non-empty")
    return content
