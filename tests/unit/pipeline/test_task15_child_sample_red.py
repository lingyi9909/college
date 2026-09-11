from __future__ import annotations

import hashlib
import json

from college_builder.domain.source import RawSourceRecord
from college_builder.pipeline.pilot import (
    CertificationSamplePlan,
    build_certification_sample,
    build_pilot_sample,
    FiveKPilotPlan,
)


def _record(site: str, index: int) -> RawSourceRecord:
    payload = {"site": site, "index": index}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return RawSourceRecord(
        record_id=f"{site}-{index:03d}",
        source_type="dataset",
        source_dataset="stackmathqa",
        source_id=f"{site}:{index}",
        source_url=f"https://example.invalid/{site}/{index}",
        raw_question=f"Question {site} {index}",
        raw_answer="4",
        raw_analysis="Reasoning. Therefore 4",
        raw_payload=payload,
        metadata={"source_site": site, "language": "en"},
        license_metadata={"declared": "CC-BY-SA-4.0"},
        raw_sha256=digest,
    )


def test_certification_sample_is_deterministic_child_of_parent_sample() -> None:
    records = [
        _record(site, index)
        for site, size in {"math": 20, "physics": 10, "statistics": 10, "mathoverflow": 10}.items()
        for index in range(size)
    ]
    revision = "git:" + "a" * 40
    parent = build_pilot_sample(
        records,
        plan=FiveKPilotPlan(
            seed=20260910,
            quotas={"math": 20, "physics": 10, "statistics": 10, "mathoverflow": 10},
        ),
        source_revisions={"stackmathqa": revision},
    )
    child_plan = CertificationSamplePlan(
        seed=20260910,
        quotas={"math": 4, "physics": 2, "statistics": 2, "mathoverflow": 2},
    )
    first = build_certification_sample(parent, plan=child_plan)
    second = build_certification_sample(parent, plan=child_plan)

    assert first.manifest.parent_sample_sha256 == parent.manifest.sample_sha256
    assert first.manifest == second.manifest
    assert len(first.records) == 10
    parent_pairs = dict(
        zip(parent.manifest.sampled_record_ids, parent.manifest.sampled_raw_sha256, strict=True)
    )
    for record_id, raw_sha in zip(
        first.manifest.sampled_record_ids,
        first.manifest.sampled_raw_sha256,
        strict=True,
    ):
        assert parent_pairs[record_id] == raw_sha
