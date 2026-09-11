from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from college_builder.pipeline.pilot import (
    CertificationSamplePlan,
    FiveKPilotPlan,
    build_certification_sample,
    sample_stackmathqa_source_root,
)
from college_builder.source.stackmathqa import OFFICIAL_SOURCE_FILES

REVISION = "13239ec8c8078bc8dfb43e1383069eb9be000ff7"
REVISION_TOKEN = f"git:{REVISION}"
SEED = 20260910
EXPECTED_5K_SHA = "e9d7b957a092114da35dbb839319a08ab555692a8a06863faf2b1980c2a681c5"
BASE_URL = (
    "https://huggingface.co/datasets/math-ai/StackMathQA/resolve/"
    + REVISION
    + "/preprocessed/stackexchange-math--1q1a"
)
SOURCE_ROOT = Path("/tmp/task15-real100-source")


def main() -> None:
    expected_100_sha = os.environ["TASK15_EXPECTED_100_SHA"]
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    source_hashes: dict[str, str] = {}
    for filename in OFFICIAL_SOURCE_FILES:
        destination = SOURCE_ROOT / filename
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", destination)
        source_hashes[filename] = hashlib.sha256(destination.read_bytes()).hexdigest()

    parent = sample_stackmathqa_source_root(
        SOURCE_ROOT,
        source_revision=REVISION_TOKEN,
        plan=FiveKPilotPlan(seed=SEED),
    )
    assert parent.manifest.sample_sha256 == EXPECTED_5K_SHA
    assert parent.manifest.source_file_sha256 == source_hashes

    child = build_certification_sample(
        parent,
        plan=CertificationSamplePlan(seed=SEED),
    )
    assert child.manifest.parent_sample_sha256 == EXPECTED_5K_SHA
    assert child.manifest.sample_sha256 == expected_100_sha
    assert child.manifest.total == 100

    Path("/tmp/task15-real100-sample.jsonl").write_text(
        "".join(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
            for record in child.records
        ),
        encoding="utf-8",
    )
    Path("/tmp/task15-real100-5k-manifest.json").write_text(
        json.dumps(parent.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path("/tmp/task15-real100-100-manifest.json").write_text(
        json.dumps(child.manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("TASK15_REAL100_SAMPLE_IDENTITY=PASS")
    print("TASK15_REAL100_5K_SHA=" + parent.manifest.sample_sha256)
    print("TASK15_REAL100_100_SHA=" + child.manifest.sample_sha256)


if __name__ == "__main__":
    main()
