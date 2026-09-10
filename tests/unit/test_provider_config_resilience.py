from __future__ import annotations

from college_builder.config import ProviderConfig


def test_provider_config_versions_timeout_and_retry_policy() -> None:
    config = ProviderConfig.model_validate(
        {
            "name": "openai_compatible",
            "model": "deepseek-v4-pro",
            "timeout_seconds": 120.0,
            "max_attempts": 3,
            "retry_backoff_seconds": 1.0,
        }
    )

    assert config.timeout_seconds == 120.0
    assert config.max_attempts == 3
    assert config.retry_backoff_seconds == 1.0
