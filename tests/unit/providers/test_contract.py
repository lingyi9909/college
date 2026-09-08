from __future__ import annotations

import pytest
from pydantic import ValidationError

from college_builder.providers.base import (
    ModelClassificationRequest,
    ModelDecision,
    StructuredModelProvider,
)
from college_builder.providers.fake import FakeStructuredModelProvider


def test_model_decision_allows_only_classification_evidence_fields() -> None:
    decision = ModelDecision(
        label="UNIVERSITY_STEM",
        score=0.99,
        evidence_references=("question:0-32",),
        reason_code="COLLEGE_LEVEL_STEM_PROBLEM",
    )

    assert decision.label == "UNIVERSITY_STEM"
    assert decision.score == 0.99
    assert decision.evidence_references == ("question:0-32",)

    with pytest.raises(ValidationError):
        ModelDecision.model_validate(
            {
                "label": "UNIVERSITY_STEM",
                "score": 0.99,
                "evidence_references": ["question:0-32"],
                "reason_code": "INVALID_REWRITE",
                "question": "rewritten question",
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", 1.01),
        ("score", -0.01),
        ("evidence_references", []),
        ("reason_code", ""),
        ("label", ""),
    ],
)
def test_model_decision_rejects_malformed_evidence(field: str, value: object) -> None:
    payload: dict[str, object] = {
        "label": "UNIVERSITY_STEM",
        "score": 0.99,
        "evidence_references": ["question:0-32"],
        "reason_code": "VALID",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        ModelDecision.model_validate(payload)


def test_fake_provider_implements_provider_neutral_contract_and_records_request() -> None:
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="classifier-v1",
        decisions=(
            ModelDecision(
                label="UNIVERSITY_STEM",
                score=0.99,
                evidence_references=("question:0-10",),
                reason_code="TEST",
            ),
        ),
    )
    request = ModelClassificationRequest(
        task="gate_1_university_stem",
        prompt="Classify only; never rewrite source content.",
        inputs={"question": "Find the eigenvalues of A."},
        allowed_labels=("UNIVERSITY_STEM", "NON_UNIVERSITY_STEM"),
    )

    assert isinstance(provider, StructuredModelProvider)
    assert provider.classify(request).label == "UNIVERSITY_STEM"
    assert provider.requests == (request,)


def test_fake_provider_exhaustion_fails_explicitly() -> None:
    provider = FakeStructuredModelProvider(
        provider="fake",
        model="classifier-v1",
        decisions=(),
    )
    request = ModelClassificationRequest(
        task="gate_1_university_stem",
        prompt="Classify.",
        inputs={"question": "Q"},
        allowed_labels=("UNIVERSITY_STEM",),
    )

    with pytest.raises(RuntimeError, match="no scripted decision"):
        provider.classify(request)
