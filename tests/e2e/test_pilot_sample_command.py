from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from typer.testing import CliRunner

from college_builder.cli import app
from college_builder.pipeline.pilot import FIVE_K_QUOTAS

SOURCES = {
    "math": ("math.stackexchange.com.jsonl", "math.stackexchange.com", 2000),
    "physics": ("physics.stackexchange.com.jsonl", "physics.stackexchange.com", 1000),
    "statistics": ("stats.stackexchange.com.jsonl", "stats.stackexchange.com", 1000),
    "mathoverflow": ("mathoverflow.net.jsonl", "mathoverflow.net", 1000),
}


def _write_source_root(root: Path) -> None:
    root.mkdir(parents=True)
    question_id = 1
    for site, (filename, host, count) in SOURCES.items():
        path = root / filename
        with path.open("w", encoding="utf-8") as handle:
            for local_index in range(count):
                row = {
                    "Q": f"{site} sample question {local_index}",
                    "A": f"Source solution {local_index}. Therefore x = {local_index}.",
                    "meta": {
                        "url": f"https://{host}/questions/{question_id}/sample",
                        "answer_id": question_id,
                        "language": "en",
                    },
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                question_id += 1


def test_pilot_sample_command_materializes_exact_5k_and_safe_manifest(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _write_source_root(source_root)
    output = tmp_path / "runtime" / "sample.jsonl"
    manifest = tmp_path / "runtime" / "manifest.json"
    revision = "git:" + "c" * 40

    result = CliRunner().invoke(
        app,
        [
            "pilot",
            "sample",
            "--source-root",
            str(source_root),
            "--source-revision",
            revision,
            "--seed",
            "20260910",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
    )

    assert result.exit_code == 0, result.stdout
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(rows) == 5000
    assert payload["total"] == 5000
    assert payload["quotas"] == FIVE_K_QUOTAS
    assert payload["seed"] == 20260910
    assert payload["source_revisions"] == {"stackmathqa": revision}
    assert len(payload["sampled_record_ids"]) == 5000
    assert len(payload["sampled_raw_sha256"]) == 5000
    assert set(payload["source_file_sha256"]) == {item[0] for item in SOURCES.values()}
    counts = Counter(str(row["metadata"]["source_site"]) for row in rows)
    assert counts == FIVE_K_QUOTAS


def test_pilot_sample_command_rejects_symbolic_revision_before_writing(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    source_root.mkdir()
    output = tmp_path / "sample.jsonl"
    manifest = tmp_path / "manifest.json"
    result = CliRunner().invoke(
        app,
        [
            "pilot",
            "sample",
            "--source-root",
            str(source_root),
            "--source-revision",
            "main",
            "--output",
            str(output),
            "--manifest",
            str(manifest),
        ],
    )
    assert result.exit_code != 0
    assert not output.exists()
    assert not manifest.exists()
