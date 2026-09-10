"""OpenAI-compatible HTTP adapter for structured classification decisions."""

from __future__ import annotations

import json
import os
import random
import time
from typing import Any

import httpx
from pydantic import ValidationError

from college_builder.providers.base import ModelClassificationRequest, ModelDecision

_MAX_RETRY_DELAY_SECONDS = 30.0
_RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429})


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
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        resolved_base_url = base_url or os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        resolved_model = model or os.getenv("OPENAI_COMPATIBLE_MODEL")
        if not isinstance(resolved_base_url, str) or not resolved_base_url.strip():
            raise ValueError("OpenAI-compatible base URL must be configured")
        if not isinstance(resolved_model, str) or not resolved_model.strip():
            raise ValueError("OpenAI-compatible model must be configured")
        if timeout_seconds <= 0.0:
            raise ValueError("OpenAI-compatible timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("OpenAI-compatible max_attempts must be at least 1")
        if retry_backoff_seconds < 0.0:
            raise ValueError("OpenAI-compatible retry_backoff_seconds must be non-negative")

        self.base_url = resolved_base_url.rstrip("/")
        self.model = resolved_model
        self.api_key = api_key or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        self.timeout_seconds = float(timeout_seconds)
        self.max_attempts = int(max_attempts)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self._client = client or httpx.Client(timeout=self.timeout_seconds)

    def classify(self, request: ModelClassificationRequest) -> ModelDecision:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {
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
        }

        for attempt in range(1, self.max_attempts + 1):
            try:
                return self._classify_once(headers=headers, payload=payload)
            except Exception as exc:
                if attempt >= self.max_attempts or not _is_retryable(exc):
                    raise
                _sleep_before_retry(
                    attempt=attempt,
                    error=exc,
                    base_backoff_seconds=self.retry_backoff_seconds,
                )
        raise RuntimeError("OpenAI-compatible retry loop exited unexpectedly")

    def _classify_once(self, *, headers: dict[str, str], payload: dict[str, Any]) -> ModelDecision:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        content = _message_content(response.json())
        parsed: object = json.loads(content)
        return ModelDecision.model_validate(parsed)


def _is_retryable(error: Exception) -> bool:
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        return status in _RETRYABLE_STATUS_CODES or 500 <= status <= 599
    if isinstance(error, httpx.TransportError):
        return True
    return isinstance(error, (ValidationError, json.JSONDecodeError, ValueError))


def _sleep_before_retry(*, attempt: int, error: Exception, base_backoff_seconds: float) -> None:
    delay = base_backoff_seconds * (2 ** (attempt - 1))
    if isinstance(error, httpx.HTTPStatusError):
        retry_after = _retry_after_seconds(error.response.headers.get("Retry-After"))
        if retry_after is not None:
            delay = max(delay, retry_after)
    delay = min(delay, _MAX_RETRY_DELAY_SECONDS)
    if delay <= 0.0:
        return
    jitter = random.random() * min(delay * 0.25, 1.0)
    time.sleep(delay + jitter)


def _retry_after_seconds(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, seconds)


def _message_content(payload: Any) -> str:
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("OpenAI-compatible response is missing message content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("OpenAI-compatible response message content must be non-empty")
    return content
