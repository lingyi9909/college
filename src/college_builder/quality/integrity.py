"""Gate 0 deterministic integrity and provenance validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import cast

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import JsonValue, NormalizedQA, RawSourceRecord
from college_builder.quality.engine import GateContext, GateResultEvidence


class IntegrityGate:
    """Reject incomplete, corrupted, or untraceable normalized source records."""

    name = "gate_0_integrity_provenance"
    provider = "deterministic"
    model = "integrity_v1"
    prompt_version = "not_applicable"

    def evaluate(
        self,
        candidate: NormalizedQA,
        context: GateContext,
    ) -> GateResultEvidence:
        raw = context.raw_records.get(candidate.source_record_id)
        if raw is None:
            return self._reject(
                context,
                "PROVENANCE_MISSING",
                {"source_record_id": candidate.source_record_id},
            )
        if raw.record_id != candidate.source_record_id:
            return self._reject(
                context,
                "PROVENANCE_MISSING",
                {
                    "source_record_id": candidate.source_record_id,
                    "raw_record_id": raw.record_id,
                },
            )
        if not _nonempty(raw.source_id) or not (
            _nonempty(raw.source_dataset) or _nonempty(raw.source_url)
        ):
            return self._reject(context, "PROVENANCE_MISSING", {})

        raw_dump = raw.model_dump(mode="json")
        raw_payload = _json_object(raw_dump.get("raw_payload"))
        if raw_payload is None:
            return self._reject(context, "SOURCE_CORRUPTED", {"field": "raw_payload"})
        try:
            actual_raw_sha256 = hashlib.sha256(_canonical_json(raw_payload)).hexdigest()
        except (TypeError, ValueError):
            return self._reject(context, "SOURCE_CORRUPTED", {"field": "raw_payload"})
        if actual_raw_sha256 != raw.raw_sha256:
            return self._reject(
                context,
                "SOURCE_CORRUPTED",
                {
                    "expected_raw_sha256": raw.raw_sha256,
                    "actual_raw_sha256": actual_raw_sha256,
                },
            )

        if not _nonempty(raw.raw_question) or not _nonempty(candidate.question):
            return self._reject(context, "SOURCE_INCOMPLETE", {"field": "question"})
        if not _nonempty(raw.raw_analysis) or not _nonempty(candidate.analysis):
            return self._reject(context, "SOURCE_INCOMPLETE", {"field": "analysis"})
        if not (_nonempty(raw.raw_answer) or _nonempty(raw.raw_analysis)):
            return self._reject(
                context,
                "SOURCE_INCOMPLETE",
                {"field": "answer_source"},
            )

        metadata = _json_object(raw_dump.get("metadata"))
        license_metadata = _json_object(raw_dump.get("license_metadata"))
        if metadata is None or license_metadata is None or not license_metadata:
            return self._reject(context, "PROVENANCE_MISSING", {})
        if not _is_canonical_json(metadata) or not _is_canonical_json(license_metadata):
            return self._reject(context, "PROVENANCE_MISSING", {})

        candidate_dump = candidate.model_dump(mode="json")
        candidate_metadata = _json_object(candidate_dump.get("metadata"))
        normalization_evidence = _json_object(candidate_dump.get("normalization_evidence"))
        if candidate_metadata is None or normalization_evidence is None:
            return self._reject(context, "PROVENANCE_MISSING", {})
        if not _is_canonical_json(candidate_metadata) or not _is_canonical_json(
            normalization_evidence
        ):
            return self._reject(context, "PROVENANCE_MISSING", {})

        provenance_failure = self._validate_normalization_provenance(
            raw,
            candidate,
            normalization_evidence,
            context,
        )
        if provenance_failure is not None:
            return provenance_failure

        missing_images = sorted(set(candidate.images) & context.missing_image_sources)
        if missing_images:
            return self._reject(
                context,
                "IMAGE_SOURCE_MISSING",
                {"missing_image_sources": cast(JsonValue, missing_images)},
            )

        return GateResultEvidence(
            gate_name=self.name,
            verdict=GateVerdict.PASS,
            score=1.0,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code="INTEGRITY_OK",
            evidence_payload={
                "source_record_id": raw.record_id,
                "source_dataset": raw.source_dataset,
                "source_id": raw.source_id,
                "raw_sha256": raw.raw_sha256,
                "license_metadata_keys": cast(JsonValue, sorted(license_metadata)),
            },
            timestamp=datetime.now(UTC),
        )

    def _validate_normalization_provenance(
        self,
        raw: RawSourceRecord,
        candidate: NormalizedQA,
        normalization_evidence: Mapping[str, JsonValue],
        context: GateContext,
    ) -> GateResultEvidence | None:
        source_record_id = normalization_evidence.get("source_record_id")
        source_raw_sha256 = normalization_evidence.get("source_raw_sha256")
        if source_record_id != raw.record_id:
            return self._reject(context, "PROVENANCE_MISSING", {})
        if not isinstance(source_raw_sha256, str):
            return self._reject(context, "PROVENANCE_MISSING", {})
        if source_raw_sha256 != raw.raw_sha256:
            return self._reject(
                context,
                "SOURCE_CORRUPTED",
                {
                    "expected_raw_sha256": raw.raw_sha256,
                    "normalization_raw_sha256": source_raw_sha256,
                },
            )

        input_hashes = _string_mapping(normalization_evidence.get("input_hashes"))
        output_hashes = _string_mapping(normalization_evidence.get("output_hashes"))
        if input_hashes is None or output_hashes is None:
            return self._reject(context, "PROVENANCE_MISSING", {})
        if input_hashes != _content_hashes(
            raw.raw_question,
            raw.raw_answer,
            raw.raw_analysis,
        ):
            return self._reject(context, "SOURCE_CORRUPTED", {"field": "input_hashes"})
        if output_hashes != _content_hashes(
            candidate.question,
            candidate.answer,
            candidate.analysis,
        ):
            return self._reject(context, "SOURCE_CORRUPTED", {"field": "output_hashes"})
        return None

    def _reject(
        self,
        context: GateContext,
        reason_code: str,
        evidence_payload: dict[str, JsonValue],
    ) -> GateResultEvidence:
        return GateResultEvidence(
            gate_name=self.name,
            verdict=GateVerdict.REJECT,
            score=0.0,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code=reason_code,
            evidence_payload=evidence_payload,
            timestamp=datetime.now(UTC),
        )


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_json(value: JsonValue) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _is_canonical_json(value: JsonValue) -> bool:
    try:
        _canonical_json(value)
    except (TypeError, ValueError):
        return False
    return True


def _json_object(value: object) -> dict[str, JsonValue] | None:
    if not isinstance(value, dict):
        return None
    return cast(dict[str, JsonValue], value)


def _string_mapping(value: JsonValue | None) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    if not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
        return None
    return cast(dict[str, str], value)


def _content_hashes(question: str, answer: str, analysis: str) -> dict[str, str]:
    return {
        "question": _sha256(question),
        "answer": _sha256(answer),
        "analysis": _sha256(analysis),
        "content": _sha256("\0".join((question, answer, analysis))),
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()
