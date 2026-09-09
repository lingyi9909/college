from __future__ import annotations

from typing import Any

import pytest

from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelClassificationRequest, ModelDecision
from college_builder.quality.dedup import DedupItem, DuplicateDecision, NearDuplicateIndex


class StaticOutputProvider:
    provider = "boundary-test"
    model = "duplicate-verifier-boundary-v1"

    def __init__(self, output: object) -> None:
        self.output = output

    def classify(self, request: ModelClassificationRequest) -> Any:
        return self.output


def _item(record_id: str, question: str) -> DedupItem:
    return DedupItem(
        candidate=NormalizedQA(
            record_id=record_id,
            source_record_id=f"raw-{record_id}",
            question=question,
            answer="4",
            analysis="The source analysis derives 4.",
            subject_candidates=("mathematics",),
            images=(),
            metadata={},
            normalization_evidence={},
        ),
        provenance_score=0.8,
        quality_score=0.8,
    )


def _decide(output: object):
    index = NearDuplicateIndex(
        retrieval_threshold=0.20,
        verifier=StaticOutputProvider(output),
        prompt="Classify pair relation only.",
        prompt_version="v1",
        num_perm=64,
    )
    left = _item("left", "Evaluate the limit of sin(x)/x as x approaches 0.")
    right = _item("right", "Find lim sin(x)/x for x tending to zero.")
    return index.decide_pair(left, right, semantic_similarity=0.99)


@pytest.mark.parametrize(
    "output",
    [
        ModelDecision.model_construct(
            label="SAME_PROBLEM",
            score=0.99,
            evidence_references=(),
            reason_code="BYPASS",
        ),
        ModelDecision.model_construct(
            label="SAME_PROBLEM",
            score=2.0,
            evidence_references=("left:question", "right:question"),
            reason_code="BYPASS",
        ),
        ModelDecision.model_construct(
            label="SAME_PROBLEM",
            score=-0.01,
            evidence_references=("left:question", "right:question"),
            reason_code="BYPASS",
        ),
        ModelDecision.model_construct(
            label="   ",
            score=0.99,
            evidence_references=("left:question", "right:question"),
            reason_code="BYPASS",
        ),
        ModelDecision.model_construct(
            label="SAME_PROBLEM",
            score=0.99,
            evidence_references=("left:question", "right:question"),
            reason_code="   ",
        ),
        {"label": "SAME_PROBLEM", "score": 0.99},
        None,
        object(),
    ],
    ids=(
        "empty-evidence-references",
        "score-above-one",
        "score-below-zero",
        "blank-label",
        "blank-reason-code",
        "dict-output",
        "none-output",
        "arbitrary-object-output",
    ),
)
def test_malformed_semantic_verifier_output_keeps_both_without_crashing(output: object) -> None:
    evidence = _decide(output)

    assert evidence.decision is DuplicateDecision.DIFFERENT
    assert evidence.decision_source == "semantic_verifier_malformed"
    assert evidence.kept_record_ids == ("left", "right")
    assert evidence.dropped_record_ids == ()
    assert evidence.verifier_provider == "boundary-test"
    assert evidence.verifier_model == "duplicate-verifier-boundary-v1"
    assert evidence.prompt_version == "v1"
