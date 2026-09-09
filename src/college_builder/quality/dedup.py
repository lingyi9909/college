"""Precision-first exact, near, and semantic deduplication."""

from __future__ import annotations

import re
import unicodedata
from enum import StrEnum
from typing import Annotated, Any, cast

from datasketch import MinHash, MinHashLSH  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

from college_builder.domain.final_record import slim_question_md5_v1
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelClassificationRequest, StructuredModelProvider

UnitScore = Annotated[float, Field(ge=0.0, le=1.0)]


class DuplicateDecision(StrEnum):
    """Final pair relation used by deduplication."""

    SAME_PROBLEM = "SAME_PROBLEM"
    VARIANT = "VARIANT"
    DIFFERENT = "DIFFERENT"


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class DedupItem(_FrozenModel):
    """Normalized record plus explicit ranking inputs for deterministic retention."""

    candidate: NormalizedQA
    provenance_score: UnitScore
    quality_score: UnitScore


class NearDuplicateMatch(_FrozenModel):
    """Candidate recalled by the MinHash/LSH index; recall alone never deletes it."""

    record_id: str = Field(min_length=1)
    similarity: UnitScore
    exact_hash: str = Field(pattern=r"^[0-9a-f]{32}$")
    near_signature: tuple[int, ...] = Field(min_length=1)


class DedupEvidence(_FrozenModel):
    """Auditable evidence for one duplicate decision."""

    candidate_ids: tuple[str, str]
    decision: DuplicateDecision
    decision_source: str = Field(min_length=1)
    exact_hashes: tuple[str, str]
    near_signatures: tuple[tuple[int, ...], tuple[int, ...]]
    retrieval_score: UnitScore
    kept_record_ids: tuple[str, ...]
    dropped_record_ids: tuple[str, ...]
    verifier_provider: str | None = None
    verifier_model: str | None = None
    verifier_score: UnitScore | None = None
    prompt_version: str
    verifier_evidence_references: tuple[str, ...] = ()


class DedupBatchResult(_FrozenModel):
    """Result of exact deduplication over a deterministic input batch."""

    kept_record_ids: tuple[str, ...]
    dropped_record_ids: tuple[str, ...]
    evidence: tuple[DedupEvidence, ...]


class ExactDeduper:
    """Remove only exact normalized-question duplicates using the approved hash."""

    def __init__(self, *, num_perm: int = 64) -> None:
        self._num_perm = _positive_num_perm(num_perm)

    def fingerprint(self, candidate: NormalizedQA) -> str:
        return slim_question_md5_v1(candidate.question)

    def deduplicate(self, items: tuple[DedupItem, ...]) -> DedupBatchResult:
        seen_ids: set[str] = set()
        kept_by_hash: dict[str, DedupItem] = {}
        dropped_ids: list[str] = []
        evidence: list[DedupEvidence] = []

        for item in items:
            record_id = item.candidate.record_id
            if record_id in seen_ids:
                raise ValueError(f"duplicate dedup record_id: {record_id}")
            seen_ids.add(record_id)

            exact_hash = self.fingerprint(item.candidate)
            current = kept_by_hash.get(exact_hash)
            if current is None:
                kept_by_hash[exact_hash] = item
                continue

            kept, dropped = _choose_kept(current, item)
            kept_by_hash[exact_hash] = kept
            dropped_ids.append(dropped.candidate.record_id)
            evidence.append(
                _pair_evidence(
                    current,
                    item,
                    decision=DuplicateDecision.SAME_PROBLEM,
                    decision_source="exact_hash",
                    retrieval_score=1.0,
                    num_perm=self._num_perm,
                    prompt_version="not_applicable",
                    kept=(kept.candidate.record_id,),
                    dropped=(dropped.candidate.record_id,),
                )
            )

        final_kept = {item.candidate.record_id for item in kept_by_hash.values()}
        kept_ids = tuple(
            item.candidate.record_id for item in items if item.candidate.record_id in final_kept
        )
        dropped_id_set = set(dropped_ids)
        ordered_dropped = tuple(
            item.candidate.record_id
            for item in items
            if item.candidate.record_id in dropped_id_set
        )
        return DedupBatchResult(
            kept_record_ids=kept_ids,
            dropped_record_ids=ordered_dropped,
            evidence=tuple(evidence),
        )


class NearDuplicateIndex:
    """Recall near candidates and require a semantic verifier before any deletion."""

    def __init__(
        self,
        *,
        retrieval_threshold: float,
        verifier: StructuredModelProvider,
        prompt: str,
        prompt_version: str,
        num_perm: int = 64,
        verifier_threshold: float = 0.95,
    ) -> None:
        self._retrieval_threshold = _unit_interval(retrieval_threshold, "retrieval_threshold")
        self._verifier_threshold = _unit_interval(verifier_threshold, "verifier_threshold")
        self._num_perm = _positive_num_perm(num_perm)
        self._verifier = verifier
        self._provider = _nonblank(verifier.provider, "verifier.provider")
        self._model = _nonblank(verifier.model, "verifier.model")
        self._prompt = _nonblank(prompt, "prompt")
        self._prompt_version = _nonblank(prompt_version, "prompt_version")
        self._lsh: Any = MinHashLSH(
            threshold=self._retrieval_threshold,
            num_perm=self._num_perm,
        )
        self._items: dict[str, DedupItem] = {}
        self._minhashes: dict[str, Any] = {}

    def add(self, item: DedupItem) -> None:
        record_id = item.candidate.record_id
        if record_id in self._items:
            raise ValueError(f"record already indexed: {record_id}")
        minhash = _build_minhash(item.candidate.question, self._num_perm)
        self._lsh.insert(record_id, minhash)
        self._items[record_id] = item
        self._minhashes[record_id] = minhash

    def find_candidates(self, item: DedupItem) -> tuple[NearDuplicateMatch, ...]:
        query_hash = _build_minhash(item.candidate.question, self._num_perm)
        candidate_ids = cast(list[str], self._lsh.query(query_hash))
        matches: list[NearDuplicateMatch] = []
        for record_id in candidate_ids:
            stored_hash = self._minhashes[record_id]
            similarity = float(stored_hash.jaccard(query_hash))
            if similarity < self._retrieval_threshold:
                continue
            stored = self._items[record_id]
            matches.append(
                NearDuplicateMatch(
                    record_id=record_id,
                    similarity=similarity,
                    exact_hash=slim_question_md5_v1(stored.candidate.question),
                    near_signature=_signature_tuple(stored_hash),
                )
            )
        return tuple(sorted(matches, key=lambda match: (-match.similarity, match.record_id)))

    def decide_pair(
        self,
        left: DedupItem,
        right: DedupItem,
        *,
        semantic_similarity: float | None = None,
    ) -> DedupEvidence:
        if left.candidate.record_id == right.candidate.record_id:
            raise ValueError("dedup pair must contain two distinct record ids")

        left_exact = slim_question_md5_v1(left.candidate.question)
        right_exact = slim_question_md5_v1(right.candidate.question)
        left_minhash = _build_minhash(left.candidate.question, self._num_perm)
        right_minhash = _build_minhash(right.candidate.question, self._num_perm)
        near_similarity = float(left_minhash.jaccard(right_minhash))

        if left_exact == right_exact:
            kept, dropped = _choose_kept(left, right)
            return _pair_evidence(
                left,
                right,
                decision=DuplicateDecision.SAME_PROBLEM,
                decision_source="exact_hash",
                retrieval_score=1.0,
                num_perm=self._num_perm,
                prompt_version="not_applicable",
                kept=(kept.candidate.record_id,),
                dropped=(dropped.candidate.record_id,),
                precomputed=(left_exact, right_exact, left_minhash, right_minhash),
            )

        semantic_score = 0.0
        if semantic_similarity is not None:
            semantic_score = _unit_interval(semantic_similarity, "semantic_similarity")
        retrieval_score = max(near_similarity, semantic_score)

        if retrieval_score < self._retrieval_threshold:
            return _pair_evidence(
                left,
                right,
                decision=DuplicateDecision.DIFFERENT,
                decision_source="below_retrieval_threshold",
                retrieval_score=retrieval_score,
                num_perm=self._num_perm,
                prompt_version="not_applicable",
                kept=(left.candidate.record_id, right.candidate.record_id),
                dropped=(),
                precomputed=(left_exact, right_exact, left_minhash, right_minhash),
            )

        request = ModelClassificationRequest(
            task="duplicate_verification",
            prompt=self._prompt,
            inputs={
                "left_record_id": left.candidate.record_id,
                "left_question": left.candidate.question,
                "right_record_id": right.candidate.record_id,
                "right_question": right.candidate.question,
                "left_exact_hash": left_exact,
                "right_exact_hash": right_exact,
                "near_similarity": near_similarity,
                "semantic_similarity": semantic_similarity,
                "retrieval_score": retrieval_score,
            },
            allowed_labels=("SAME_PROBLEM", "VARIANT", "DIFFERENT", "UNCERTAIN"),
        )

        try:
            verifier_decision = self._verifier.classify(request)
        except Exception:
            return _pair_evidence(
                left,
                right,
                decision=DuplicateDecision.DIFFERENT,
                decision_source="semantic_verifier_error",
                retrieval_score=retrieval_score,
                num_perm=self._num_perm,
                prompt_version=self._prompt_version,
                kept=(left.candidate.record_id, right.candidate.record_id),
                dropped=(),
                verifier_provider=self._provider,
                verifier_model=self._model,
                precomputed=(left_exact, right_exact, left_minhash, right_minhash),
            )

        label = verifier_decision.label
        if label not in {decision.value for decision in DuplicateDecision} or (
            verifier_decision.score < self._verifier_threshold
        ):
            return _pair_evidence(
                left,
                right,
                decision=DuplicateDecision.DIFFERENT,
                decision_source="semantic_verifier_uncertain",
                retrieval_score=retrieval_score,
                num_perm=self._num_perm,
                prompt_version=self._prompt_version,
                kept=(left.candidate.record_id, right.candidate.record_id),
                dropped=(),
                verifier_provider=self._provider,
                verifier_model=self._model,
                verifier_score=verifier_decision.score,
                verifier_evidence_references=verifier_decision.evidence_references,
                precomputed=(left_exact, right_exact, left_minhash, right_minhash),
            )

        decision = DuplicateDecision(label)
        kept_ids: tuple[str, ...]
        dropped_ids: tuple[str, ...]
        if decision is DuplicateDecision.SAME_PROBLEM:
            kept, dropped = _choose_kept(left, right)
            kept_ids = (kept.candidate.record_id,)
            dropped_ids = (dropped.candidate.record_id,)
        else:
            kept_ids = (left.candidate.record_id, right.candidate.record_id)
            dropped_ids = ()

        return _pair_evidence(
            left,
            right,
            decision=decision,
            decision_source="semantic_verifier",
            retrieval_score=retrieval_score,
            num_perm=self._num_perm,
            prompt_version=self._prompt_version,
            kept=kept_ids,
            dropped=dropped_ids,
            verifier_provider=self._provider,
            verifier_model=self._model,
            verifier_score=verifier_decision.score,
            verifier_evidence_references=verifier_decision.evidence_references,
            precomputed=(left_exact, right_exact, left_minhash, right_minhash),
        )


def _pair_evidence(
    left: DedupItem,
    right: DedupItem,
    *,
    decision: DuplicateDecision,
    decision_source: str,
    retrieval_score: float,
    num_perm: int,
    prompt_version: str,
    kept: tuple[str, ...],
    dropped: tuple[str, ...],
    verifier_provider: str | None = None,
    verifier_model: str | None = None,
    verifier_score: float | None = None,
    verifier_evidence_references: tuple[str, ...] = (),
    precomputed: tuple[str, str, Any, Any] | None = None,
) -> DedupEvidence:
    if precomputed is None:
        left_exact = slim_question_md5_v1(left.candidate.question)
        right_exact = slim_question_md5_v1(right.candidate.question)
        left_minhash = _build_minhash(left.candidate.question, num_perm)
        right_minhash = _build_minhash(right.candidate.question, num_perm)
    else:
        left_exact, right_exact, left_minhash, right_minhash = precomputed

    return DedupEvidence(
        candidate_ids=(left.candidate.record_id, right.candidate.record_id),
        decision=decision,
        decision_source=decision_source,
        exact_hashes=(left_exact, right_exact),
        near_signatures=(
            _signature_tuple(left_minhash),
            _signature_tuple(right_minhash),
        ),
        retrieval_score=retrieval_score,
        kept_record_ids=kept,
        dropped_record_ids=dropped,
        verifier_provider=verifier_provider,
        verifier_model=verifier_model,
        verifier_score=verifier_score,
        prompt_version=prompt_version,
        verifier_evidence_references=verifier_evidence_references,
    )


def _choose_kept(left: DedupItem, right: DedupItem) -> tuple[DedupItem, DedupItem]:
    left_priority = (left.provenance_score, left.quality_score)
    right_priority = (right.provenance_score, right.quality_score)
    if left_priority > right_priority:
        return left, right
    if right_priority > left_priority:
        return right, left
    if left.candidate.record_id <= right.candidate.record_id:
        return left, right
    return right, left


def _build_minhash(text: str, num_perm: int) -> Any:
    minhash = MinHash(num_perm=num_perm, seed=1)
    for token in _near_tokens(text):
        minhash.update(token.encode("utf-8"))
    return minhash


def _near_tokens(text: str) -> tuple[str, ...]:
    canonical = unicodedata.normalize("NFC", text)
    canonical = canonical.replace("\r\n", "\n").replace("\r", "\n")
    canonical = re.sub(r"\s+", " ", canonical).strip().casefold()
    if not canonical:
        raise ValueError("question must be non-empty for deduplication")
    if len(canonical) < 3:
        return (canonical,)
    return tuple(sorted({canonical[index : index + 3] for index in range(len(canonical) - 2)}))


def _signature_tuple(minhash: Any) -> tuple[int, ...]:
    return tuple(int(value) for value in minhash.hashvalues)


def _positive_num_perm(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("num_perm must be a positive integer")
    return value


def _unit_interval(value: float, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    numeric = float(value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} must be within [0, 1]")
    return numeric


def _nonblank(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value
