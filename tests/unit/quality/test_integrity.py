from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import JsonLike, NormalizedQA, RawSourceRecord
from college_builder.normalize.normalizer import normalize
from college_builder.quality.engine import GateContext
from college_builder.quality.integrity import IntegrityGate


def _raw_hash(payload: Mapping[str, JsonLike]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _raw_record(
    *,
    question: str = "Compute 2+2.",
    answer: str = "",
    analysis: str = "The source solution concludes 4.",
    source_id: str = "fixture-8",
    source_dataset: str = "stackmathqa",
    source_url: str = "https://example.test/questions/8",
    payload: dict[str, JsonLike] | None = None,
    metadata: dict[str, JsonLike] | None = None,
    license_metadata: dict[str, JsonLike] | None = None,
    raw_sha256: str | None = None,
) -> RawSourceRecord:
    raw_payload = payload or {
        "id": source_id,
        "question": question,
        "answer": answer,
        "analysis": analysis,
    }
    return RawSourceRecord.model_validate(
        {
            "record_id": "raw_task8_fixture",
            "source_type": "dataset",
            "source_dataset": source_dataset,
            "source_id": source_id,
            "source_url": source_url,
            "raw_question": question,
            "raw_answer": answer,
            "raw_analysis": analysis,
            "raw_payload": raw_payload,
            "metadata": metadata or {"acquisition": {"adapter": "fixture"}},
            "license_metadata": (
                {"declared": "UNREVIEWED"}
                if license_metadata is None
                else license_metadata
            ),
            "raw_sha256": raw_sha256 or _raw_hash(raw_payload),
        }
    )


def _context(
    raw: RawSourceRecord | None,
    *,
    missing_image_sources: frozenset[str] = frozenset(),
) -> GateContext:
    records = {} if raw is None else {raw.record_id: raw}
    return GateContext(
        config_version="pilot-v1",
        raw_records=records,
        missing_image_sources=missing_image_sources,
    )


def _evaluate(raw: RawSourceRecord) -> tuple[NormalizedQA, object]:
    candidate = normalize(raw)
    return candidate, IntegrityGate().evaluate(candidate, _context(raw))


def test_gate0_accepts_stack_style_answer_evidence_and_unreviewed_license() -> None:
    raw = _raw_record(answer="", analysis="Answer: 4. Because 2+2=4.")
    candidate, evidence = _evaluate(raw)

    assert candidate.answer == ""
    assert evidence.verdict is GateVerdict.PASS
    assert evidence.score == 1.0
    assert evidence.reason_code == "INTEGRITY_OK"
    assert evidence.evidence_payload["source_record_id"] == raw.record_id
    assert evidence.evidence_payload["raw_sha256"] == raw.raw_sha256


def test_gate0_rejects_when_raw_record_cannot_be_found() -> None:
    raw = _raw_record()
    candidate = normalize(raw)

    evidence = IntegrityGate().evaluate(candidate, _context(None))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "PROVENANCE_MISSING"


def test_gate0_rejects_raw_payload_hash_mismatch() -> None:
    raw = _raw_record(raw_sha256="b" * 64)
    candidate = normalize(raw)

    evidence = IntegrityGate().evaluate(candidate, _context(raw))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "SOURCE_CORRUPTED"
    assert evidence.evidence_payload["expected_raw_sha256"] == raw.raw_sha256


def test_gate0_rejects_missing_source_identity_even_for_corrupt_stored_model() -> None:
    raw = _raw_record()
    corrupt = raw.model_copy(update={"source_id": "", "source_dataset": "", "source_url": ""})
    candidate = normalize(raw).model_copy(update={"source_record_id": corrupt.record_id})

    evidence = IntegrityGate().evaluate(candidate, _context(corrupt))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "PROVENANCE_MISSING"


def test_gate0_rejects_empty_source_question() -> None:
    raw = _raw_record(question="   ")
    candidate = normalize(raw)

    evidence = IntegrityGate().evaluate(candidate, _context(raw))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "SOURCE_INCOMPLETE"


def test_gate0_rejects_missing_answer_and_analysis_source_evidence() -> None:
    raw = _raw_record(answer="", analysis="")
    candidate = normalize(raw)

    evidence = IntegrityGate().evaluate(candidate, _context(raw))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "SOURCE_INCOMPLETE"


def test_gate0_rejects_missing_analysis_even_when_answer_exists() -> None:
    raw = _raw_record(answer="4", analysis="")
    candidate = normalize(raw)

    evidence = IntegrityGate().evaluate(candidate, _context(raw))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "SOURCE_INCOMPLETE"


def test_gate0_rejects_missing_license_metadata_but_allows_unreviewed_status() -> None:
    raw = _raw_record(license_metadata={})
    candidate = normalize(raw)

    evidence = IntegrityGate().evaluate(candidate, _context(raw))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "PROVENANCE_MISSING"


def test_gate0_rejects_malformed_provenance_json() -> None:
    raw = _raw_record(metadata={"bad": float("nan")})
    candidate = normalize(raw)

    evidence = IntegrityGate().evaluate(candidate, _context(raw))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "PROVENANCE_MISSING"


def test_gate0_rejects_normalization_provenance_raw_hash_mismatch() -> None:
    raw = _raw_record()
    candidate = normalize(raw)
    dumped = candidate.model_dump(mode="json")
    normalization_evidence = dict(dumped["normalization_evidence"])
    normalization_evidence["source_raw_sha256"] = "c" * 64
    tampered = candidate.model_copy(
        update={"normalization_evidence": normalization_evidence}
    )

    evidence = IntegrityGate().evaluate(tampered, _context(raw))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "SOURCE_CORRUPTED"


def test_gate0_rejects_tampered_normalized_content_hashes() -> None:
    raw = _raw_record()
    candidate = normalize(raw).model_copy(update={"question": "changed question"})

    evidence = IntegrityGate().evaluate(candidate, _context(raw))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "SOURCE_CORRUPTED"


def test_gate0_rejects_explicitly_missing_image_source() -> None:
    image = "https://example.test/diagram.png"
    raw = _raw_record(question=f"Inspect ![diagram]({image}).")
    candidate = normalize(raw)
    assert candidate.images == (image,)

    evidence = IntegrityGate().evaluate(
        candidate,
        _context(raw, missing_image_sources=frozenset({image})),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "IMAGE_SOURCE_MISSING"
    assert evidence.evidence_payload["missing_image_sources"] == [image]
