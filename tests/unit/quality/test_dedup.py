from __future__ import annotations

import json
from pathlib import Path

import pytest

from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.dedup import (
    DedupItem,
    DuplicateDecision,
    ExactDeduper,
    NearDuplicateIndex,
)


def _qa(record_id: str, question: str) -> NormalizedQA:
    return NormalizedQA(
        record_id=record_id,
        source_record_id=f"raw-{record_id}",
        question=question,
        answer="x = 4",
        analysis="Subtract 3 from both sides and divide by 2.",
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _item(
    record_id: str,
    question: str,
    *,
    provenance_score: float = 0.8,
    quality_score: float = 0.8,
) -> DedupItem:
    return DedupItem(
        candidate=_qa(record_id, question),
        provenance_score=provenance_score,
        quality_score=quality_score,
    )


def _verifier(*decisions: ModelDecision) -> FakeStructuredModelProvider:
    return FakeStructuredModelProvider(
        provider="fake",
        model="duplicate-verifier-v1",
        decisions=tuple(decisions),
    )


def _decision(label: str, *, score: float = 0.99) -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        evidence_references=("left:question", "right:question"),
        reason_code="DEDUP_TEST",
    )


def test_exact_dedup_collides_only_on_approved_question_canonicalization() -> None:
    left = _item(
        "norm-a",
        "Compute café\r\n\r\n\r\nthen solve x = 4",
        provenance_score=0.7,
        quality_score=0.95,
    )
    right = _item(
        "norm-b",
        "Compute cafe\u0301\n\nthen solve x = 4",
        provenance_score=0.9,
        quality_score=0.90,
    )

    result = ExactDeduper().deduplicate((left, right))

    assert result.kept_record_ids == ("norm-b",)
    assert result.dropped_record_ids == ("norm-a",)
    assert len(result.evidence) == 1
    evidence = result.evidence[0]
    assert evidence.decision is DuplicateDecision.SAME_PROBLEM
    assert evidence.decision_source == "exact_hash"
    assert evidence.candidate_ids == ("norm-a", "norm-b")
    assert evidence.exact_hashes[0] == evidence.exact_hashes[1]
    assert evidence.kept_record_ids == ("norm-b",)
    assert evidence.dropped_record_ids == ("norm-a",)
    assert all(evidence.near_signatures)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Solve x = 4", "Solve x = 5"),
        ("Compute x^2", "Compute x^3"),
        ("Solve x + 1 = 0", "Solve x - 1 = 0"),
        ("Choose A. 1 B. 2", "Choose A. 1 B. 3"),
        ("Use ![diagram](image/a.png)", "Use ![diagram](image/b.png)"),
    ],
)
def test_exact_hash_preserves_semantic_digits_signs_options_and_images(
    left: str,
    right: str,
) -> None:
    deduper = ExactDeduper()

    assert deduper.fingerprint(_qa("left", left)) != deduper.fingerprint(_qa("right", right))


def test_near_duplicate_index_only_recalls_candidates_without_deleting() -> None:
    verifier = _verifier(_decision("SAME_PROBLEM"))
    index = NearDuplicateIndex(
        retrieval_threshold=0.20,
        verifier=verifier,
        prompt="Classify duplicate relation only.",
        prompt_version="v1",
        num_perm=64,
    )
    left = _item("left", "Evaluate the indefinite integral of x with respect to x.")
    right = _item("right", "Evaluate the indefinite integral of 2x with respect to x.")
    index.add(left)

    matches = index.find_candidates(right)

    assert any(match.record_id == "left" for match in matches)
    assert verifier.requests == ()


def test_math_variant_is_preserved_even_when_similarity_is_high() -> None:
    verifier = _verifier(_decision("VARIANT"))
    index = NearDuplicateIndex(
        retrieval_threshold=0.20,
        verifier=verifier,
        prompt="Classify duplicate relation only.",
        prompt_version="v1",
        num_perm=64,
    )
    left = _item("left", "Evaluate the indefinite integral ∫x dx.")
    right = _item("right", "Evaluate the indefinite integral ∫2x dx.")

    evidence = index.decide_pair(left, right)

    assert evidence.decision is DuplicateDecision.VARIANT
    assert evidence.decision_source == "semantic_verifier"
    assert evidence.kept_record_ids == ("left", "right")
    assert evidence.dropped_record_ids == ()
    assert evidence.retrieval_score >= 0.20
    assert len(verifier.requests) == 1
    request = verifier.requests[0]
    assert request.allowed_labels == ("SAME_PROBLEM", "VARIANT", "DIFFERENT", "UNCERTAIN")
    assert request.inputs["left_question"] == left.candidate.question
    assert request.inputs["right_question"] == right.candidate.question


def test_same_problem_keeps_higher_provenance_then_quality_and_preserves_lineage() -> None:
    verifier = _verifier(_decision("SAME_PROBLEM"))
    index = NearDuplicateIndex(
        retrieval_threshold=0.20,
        verifier=verifier,
        prompt="Classify duplicate relation only.",
        prompt_version="v1",
        num_perm=64,
    )
    left = _item(
        "mirror-a",
        "Solve the linear equation 2x + 3 = 11.",
        provenance_score=0.70,
        quality_score=0.99,
    )
    right = _item(
        "mirror-b",
        "Find x in the linear equation 2x + 3 = 11.",
        provenance_score=0.95,
        quality_score=0.90,
    )

    evidence = index.decide_pair(left, right)

    assert evidence.decision is DuplicateDecision.SAME_PROBLEM
    assert evidence.kept_record_ids == ("mirror-b",)
    assert evidence.dropped_record_ids == ("mirror-a",)
    assert evidence.candidate_ids == ("mirror-a", "mirror-b")
    assert evidence.verifier_provider == "fake"
    assert evidence.verifier_model == "duplicate-verifier-v1"
    assert evidence.prompt_version == "v1"
    assert evidence.verifier_evidence_references == ("left:question", "right:question")


def test_different_high_similarity_problem_is_preserved() -> None:
    verifier = _verifier(_decision("DIFFERENT"))
    index = NearDuplicateIndex(
        retrieval_threshold=0.20,
        verifier=verifier,
        prompt="Classify duplicate relation only.",
        prompt_version="v1",
        num_perm=64,
    )
    left = _item("left", "Compute the derivative of x^2 + 3x.")
    right = _item("right", "Compute the derivative of x^2 - 3x.")

    evidence = index.decide_pair(left, right)

    assert evidence.decision is DuplicateDecision.DIFFERENT
    assert evidence.kept_record_ids == ("left", "right")
    assert evidence.dropped_record_ids == ()
    assert len(verifier.requests) == 1


def test_below_retrieval_threshold_never_calls_semantic_verifier() -> None:
    verifier = _verifier()
    index = NearDuplicateIndex(
        retrieval_threshold=0.95,
        verifier=verifier,
        prompt="Classify duplicate relation only.",
        prompt_version="v1",
        num_perm=64,
    )
    left = _item("left", "Differentiate x^2.")
    right = _item("right", "State Newton's second law.")

    evidence = index.decide_pair(left, right)

    assert evidence.decision is DuplicateDecision.DIFFERENT
    assert evidence.decision_source == "below_retrieval_threshold"
    assert evidence.kept_record_ids == ("left", "right")
    assert evidence.dropped_record_ids == ()
    assert verifier.requests == ()


def test_semantic_candidate_score_can_trigger_verifier_but_cannot_delete_by_itself() -> None:
    verifier = _verifier(_decision("VARIANT"))
    index = NearDuplicateIndex(
        retrieval_threshold=0.90,
        verifier=verifier,
        prompt="Classify duplicate relation only.",
        prompt_version="v1",
        num_perm=64,
    )
    left = _item("left", "Show that this graph is connected.")
    right = _item("right", "Prove the network has one connected component.")

    evidence = index.decide_pair(left, right, semantic_similarity=0.96)

    assert evidence.decision is DuplicateDecision.VARIANT
    assert evidence.retrieval_score == 0.96
    assert evidence.dropped_record_ids == ()
    assert len(verifier.requests) == 1


@pytest.mark.parametrize("label", ["UNCERTAIN", "NOT_A_VALID_DECISION"])
def test_uncertain_or_invalid_verifier_output_fails_closed_to_keep_both(label: str) -> None:
    verifier = _verifier(_decision(label))
    index = NearDuplicateIndex(
        retrieval_threshold=0.20,
        verifier=verifier,
        prompt="Classify duplicate relation only.",
        prompt_version="v1",
        num_perm=64,
    )
    left = _item("left", "Evaluate the limit of sin(x)/x as x approaches 0.")
    right = _item("right", "Find lim sin(x)/x for x tending to zero.")

    evidence = index.decide_pair(left, right, semantic_similarity=0.99)

    assert evidence.decision is DuplicateDecision.DIFFERENT
    assert evidence.decision_source == "semantic_verifier_uncertain"
    assert evidence.kept_record_ids == ("left", "right")
    assert evidence.dropped_record_ids == ()


def test_dedup_golden_cases() -> None:
    cases_path = Path(__file__).parents[2] / "golden" / "dedup_cases.json"
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    assert cases

    for case in cases:
        left = _item(
            f"{case['id']}-left",
            case["left_question"],
            provenance_score=case.get("left_provenance_score", 0.8),
            quality_score=case.get("left_quality_score", 0.8),
        )
        right = _item(
            f"{case['id']}-right",
            case["right_question"],
            provenance_score=case.get("right_provenance_score", 0.8),
            quality_score=case.get("right_quality_score", 0.8),
        )

        if case["mode"] == "exact":
            result = ExactDeduper().deduplicate((left, right))
            assert result.evidence[0].decision.value == case["expected_decision"]
            continue

        verifier = _verifier(_decision(case["verifier_label"]))
        index = NearDuplicateIndex(
            retrieval_threshold=case["retrieval_threshold"],
            verifier=verifier,
            prompt="Classify duplicate relation only.",
            prompt_version="v1",
            num_perm=64,
        )
        evidence = index.decide_pair(
            left,
            right,
            semantic_similarity=case.get("semantic_similarity"),
        )
        assert evidence.decision.value == case["expected_decision"]
        assert bool(evidence.dropped_record_ids) is case["expect_drop"]
