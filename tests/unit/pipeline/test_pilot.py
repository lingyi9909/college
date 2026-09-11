from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from college_builder.domain.source import RawSourceRecord
from college_builder.pipeline.pilot import (
    CERTIFICATION_QUOTAS,
    CERTIFICATION_TOTAL,
    FIVE_K_QUOTAS,
    FIVE_K_TOTAL,
    CertificationSamplePlan,
    FiveKPilotPlan,
    PilotSampleManifest,
    build_certification_sample,
    build_pilot_sample,
    freeze_pipeline_config,
)


def _record(site: str, index: int) -> RawSourceRecord:
    payload = {"site": site, "index": index}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return RawSourceRecord(
        record_id=f"{site}-{index:05d}",
        source_type="dataset",
        source_dataset="stackmathqa",
        source_id=f"{site}:{index}",
        source_url=f"https://example.invalid/{site}/{index}",
        raw_question=f"Question {site} {index}",
        raw_answer="",
        raw_analysis=f"Answer analysis {site} {index}",
        raw_payload=payload,
        metadata={"source_site": site, "language": "en"},
        license_metadata={"declared": "CC-BY-SA-4.0"},
        raw_sha256=digest,
    )


def _records() -> list[RawSourceRecord]:
    sizes = {"math": 20, "physics": 10, "statistics": 10, "mathoverflow": 10}
    return [_record(site, index) for site, size in sizes.items() for index in range(size)]


def test_five_k_plan_is_exact_and_frozen() -> None:
    assert FIVE_K_TOTAL == 5000
    assert FIVE_K_QUOTAS == {
        "math": 2000,
        "physics": 1000,
        "statistics": 1000,
        "mathoverflow": 1000,
    }
    plan = FiveKPilotPlan(seed=20260910)
    assert plan.total == 5000
    assert plan.quotas == FIVE_K_QUOTAS


def test_build_pilot_sample_is_deterministic_and_records_seed() -> None:
    records = _records()
    plan = FiveKPilotPlan(
        seed=42,
        quotas={"math": 4, "physics": 2, "statistics": 2, "mathoverflow": 2},
    )
    revision = "git:" + "a" * 40
    first = build_pilot_sample(records, plan=plan, source_revisions={"stackmathqa": revision})
    second = build_pilot_sample(
        reversed(records), plan=plan, source_revisions={"stackmathqa": revision}
    )

    assert [row.record_id for row in first.records] == [row.record_id for row in second.records]
    assert first.manifest == second.manifest
    assert first.manifest.seed == 42
    assert first.manifest.total == 10
    assert Counter(str(row.metadata["source_site"]) for row in first.records) == plan.quotas
    assert len(first.manifest.sample_sha256) == 64
    assert first.manifest.source_revisions == {"stackmathqa": revision}


def test_pilot_manifest_rejects_non_immutable_source_revision() -> None:
    with pytest.raises(ValueError, match="immutable"):
        PilotSampleManifest(
            seed=1,
            quotas={"math": 1},
            total=1,
            sample_sha256="a" * 64,
            sampled_record_ids=("math-00001",),
            sampled_raw_sha256=("b" * 64,),
            source_revisions={"stackmathqa": "main"},
        )


def test_freeze_pipeline_config_is_byte_stable_and_changes_version(tmp_path: Path) -> None:
    source = Path("config/pilot.yaml")
    target = tmp_path / "frozen.yaml"
    first_hash = freeze_pipeline_config(source, target, frozen_config_version="pilot-5k-frozen-v1")
    first_bytes = target.read_bytes()
    second_hash = freeze_pipeline_config(source, target, frozen_config_version="pilot-5k-frozen-v1")

    assert target.read_bytes() == first_bytes
    assert first_hash == second_hash == hashlib.sha256(first_bytes).hexdigest()
    assert b"config_version: pilot-5k-frozen-v1" in first_bytes


def test_sample_identity_changes_when_same_record_id_content_hash_changes() -> None:
    plan = FiveKPilotPlan(
        seed=9,
        quotas={"math": 1, "physics": 1, "statistics": 1, "mathoverflow": 1},
    )
    records = [_record(site, 1) for site in plan.quotas]
    revision = "git:" + "b" * 40
    first = build_pilot_sample(records, plan=plan, source_revisions={"stackmathqa": revision})
    changed = records[0].model_copy(update={"raw_sha256": "f" * 64})
    second = build_pilot_sample(
        [changed, *records[1:]],
        plan=plan,
        source_revisions={"stackmathqa": revision},
    )

    assert first.manifest.sampled_record_ids == second.manifest.sampled_record_ids
    assert first.manifest.sample_sha256 != second.manifest.sample_sha256
    assert first.manifest.sampled_raw_sha256 != second.manifest.sampled_raw_sha256


def test_certification_plan_defaults_to_exact_100_strata() -> None:
    plan = CertificationSamplePlan()
    assert CERTIFICATION_TOTAL == 100
    assert CERTIFICATION_QUOTAS == {"math": 40, "physics": 20, "statistics": 20, "mathoverflow": 20}
    assert plan.total == 100
    assert plan.quotas == CERTIFICATION_QUOTAS


def test_certification_sample_is_deterministic_child_of_parent_sample() -> None:
    records = _records()
    revision = "git:" + "c" * 40
    parent = build_pilot_sample(
        records,
        plan=FiveKPilotPlan(
            seed=20260910, quotas={"math": 20, "physics": 10, "statistics": 10, "mathoverflow": 10}
        ),
        source_revisions={"stackmathqa": revision},
    )
    plan = CertificationSamplePlan(
        seed=20260910, quotas={"math": 4, "physics": 2, "statistics": 2, "mathoverflow": 2}
    )
    first = build_certification_sample(parent, plan=plan)
    second = build_certification_sample(parent, plan=plan)
    assert first.manifest == second.manifest
    assert first.manifest.parent_sample_sha256 == parent.manifest.sample_sha256
    assert Counter(str(row.metadata["source_site"]) for row in first.records) == plan.quotas
    parent_pairs = dict(
        zip(parent.manifest.sampled_record_ids, parent.manifest.sampled_raw_sha256, strict=True)
    )
    for record_id, raw_sha in zip(
        first.manifest.sampled_record_ids, first.manifest.sampled_raw_sha256, strict=True
    ):
        assert parent_pairs[record_id] == raw_sha


def test_certification_sample_identity_binds_parent_sample() -> None:
    records = _records()
    revision = "git:" + "d" * 40
    parent = build_pilot_sample(
        records,
        plan=FiveKPilotPlan(
            seed=1, quotas={"math": 4, "physics": 2, "statistics": 2, "mathoverflow": 2}
        ),
        source_revisions={"stackmathqa": revision},
    )
    plan = CertificationSamplePlan(
        seed=9, quotas={"math": 1, "physics": 1, "statistics": 1, "mathoverflow": 1}
    )
    first = build_certification_sample(parent, plan=plan)
    rebound_parent = parent.model_copy(
        update={"manifest": parent.manifest.model_copy(update={"sample_sha256": "f" * 64})}
    )
    second = build_certification_sample(rebound_parent, plan=plan)
    assert first.manifest.sample_sha256 != second.manifest.sample_sha256


def test_certification_sample_rejects_parent_record_hash_mismatch() -> None:
    records = _records()
    revision = "git:" + "e" * 40
    parent = build_pilot_sample(
        records,
        plan=FiveKPilotPlan(
            seed=1, quotas={"math": 4, "physics": 2, "statistics": 2, "mathoverflow": 2}
        ),
        source_revisions={"stackmathqa": revision},
    )
    bad = parent.records[0].model_copy(update={"raw_sha256": "f" * 64})
    tampered = parent.model_copy(update={"records": (bad, *parent.records[1:])})
    with pytest.raises(ValueError, match="identity"):
        build_certification_sample(
            tampered, plan=CertificationSamplePlan(seed=1, quotas={"math": 1})
        )


def test_certification_sample_fails_closed_on_insufficient_parent_stratum() -> None:
    records = _records()
    revision = "git:" + "f" * 40
    parent = build_pilot_sample(
        records,
        plan=FiveKPilotPlan(seed=1, quotas={"math": 1}),
        source_revisions={"stackmathqa": revision},
    )
    with pytest.raises(ValueError, match="insufficient parent records"):
        build_certification_sample(parent, plan=CertificationSamplePlan(seed=1, quotas={"math": 2}))
