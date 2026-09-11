from __future__ import annotations

import hashlib
import json
import urllib.request
from collections import Counter
from pathlib import Path

from college_builder.pipeline.pilot import (
    CERTIFICATION_QUOTAS,
    FIVE_K_QUOTAS,
    CertificationSamplePlan,
    FiveKPilotPlan,
    build_certification_sample,
    build_pilot_sample,
)
from college_builder.source.stackmathqa import OFFICIAL_SOURCE_FILES, StackMathQAAdapter

REVISION = "13239ec8c8078bc8dfb43e1383069eb9be000ff7"
REVISION_TOKEN = f"git:{REVISION}"
SEED = 20260910
EXPECTED_5K_SHA = "e9d7b957a092114da35dbb839319a08ab555692a8a06863faf2b1980c2a681c5"
BASE_URL = (
    "https://huggingface.co/datasets/math-ai/StackMathQA/resolve/"
    + REVISION
    + "/preprocessed/stackexchange-math--1q1a"
)
SOURCE_ROOT = Path("/tmp/task15-stackmathqa-pinned")


def _download_sources() -> dict[str, str]:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for filename in OFFICIAL_SOURCE_FILES:
        destination = SOURCE_ROOT / filename
        if not destination.exists():
            urllib.request.urlretrieve(f"{BASE_URL}/{filename}", destination)
        hashes[filename] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return hashes


def _records():
    for filename in OFFICIAL_SOURCE_FILES:
        adapter = StackMathQAAdapter()
        descriptor = tuple(
            adapter.discover(
                {
                    "data_file": str(SOURCE_ROOT / filename),
                    "revision": REVISION_TOKEN,
                    "license_metadata": {
                        "declared": "CC-BY-4.0",
                        "status": "UPSTREAM_DECLARED",
                    },
                }
            )
        )[0]
        yield from adapter.acquire(descriptor)


def _build_parent():
    return build_pilot_sample(
        _records(),
        plan=FiveKPilotPlan(seed=SEED),
        source_revisions={"stackmathqa": REVISION_TOKEN},
    )


def _counts(records) -> dict[str, int]:
    return dict(Counter(str(record.metadata["source_site"]) for record in records))


def main() -> None:
    downloaded_hashes = _download_sources()

    first = _build_parent()
    second = _build_parent()

    assert first.manifest.total == 5000
    assert second.manifest.total == 5000
    assert _counts(first.records) == FIVE_K_QUOTAS
    assert _counts(second.records) == FIVE_K_QUOTAS
    assert first.manifest.source_revisions == {"stackmathqa": REVISION_TOKEN}
    assert second.manifest.source_revisions == {"stackmathqa": REVISION_TOKEN}
    assert first.manifest.sample_sha256 == EXPECTED_5K_SHA
    assert second.manifest.sample_sha256 == EXPECTED_5K_SHA
    assert first.manifest.sampled_record_ids == second.manifest.sampled_record_ids
    assert first.manifest.sampled_raw_sha256 == second.manifest.sampled_raw_sha256
    assert first.manifest.source_file_sha256 == second.manifest.source_file_sha256
    assert first.manifest.source_file_sha256 == downloaded_hashes

    child_plan = CertificationSamplePlan(seed=SEED)
    child_first = build_certification_sample(first, plan=child_plan)
    child_second = build_certification_sample(second, plan=child_plan)

    assert child_first.manifest.total == 100
    assert child_second.manifest.total == 100
    assert _counts(child_first.records) == CERTIFICATION_QUOTAS
    assert _counts(child_second.records) == CERTIFICATION_QUOTAS
    assert child_first.manifest.parent_sample_sha256 == EXPECTED_5K_SHA
    assert child_second.manifest.parent_sample_sha256 == EXPECTED_5K_SHA
    assert child_first.manifest.sample_sha256 == child_second.manifest.sample_sha256
    assert child_first.manifest.sampled_record_ids == child_second.manifest.sampled_record_ids
    assert child_first.manifest.sampled_raw_sha256 == child_second.manifest.sampled_raw_sha256

    Path("/tmp/task15-5k-manifest.json").write_text(
        json.dumps(first.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("/tmp/task15-100-manifest.json").write_text(
        json.dumps(child_first.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("/tmp/task15-100-sample.jsonl").write_text(
        "".join(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in child_first.records
        ),
        encoding="utf-8",
    )
    summary = {
        "source_dataset": "math-ai/StackMathQA",
        "source_revision": REVISION_TOKEN,
        "seed": SEED,
        "five_k_strata": FIVE_K_QUOTAS,
        "five_k_sample_sha256": first.manifest.sample_sha256,
        "hundred_strata": CERTIFICATION_QUOTAS,
        "hundred_sample_sha256": child_first.manifest.sample_sha256,
        "source_file_sha256": first.manifest.source_file_sha256,
        "deterministic_reproduction": True,
        "model_calls": 0,
    }
    Path("/tmp/task15-zero-model-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("TASK15_ZERO_MODEL_5K_CERTIFICATION=PASS")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
