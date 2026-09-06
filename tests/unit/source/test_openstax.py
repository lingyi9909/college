from __future__ import annotations

from pathlib import Path

import pytest

from college_builder.source.base import SourceAdapter
from college_builder.source.openstax import OpenTextbookAdapter

FIXTURE = Path("tests/fixtures/sources/openstax_sample.xml")


def _config(data_file: Path = FIXTURE) -> dict[str, object]:
    return {
        "dataset_name": "openstax",
        "data_file": str(data_file),
        "source_url": "https://openstax.org/books/university-physics-volume-1",
        "revision": "fixture-v1",
        "gold_role": "HIGH_CONFIDENCE",
        "license_metadata": {
            "declared": "CC-BY-4.0",
            "status": "UNREVIEWED",
        },
    }


def test_openstax_pairs_problem_and_solution_deterministically_without_rewriting() -> None:
    adapter = OpenTextbookAdapter()
    assert isinstance(adapter, SourceAdapter)
    descriptor = tuple(adapter.discover(_config()))[0]

    records = list(adapter.acquire(descriptor))

    assert [record.source_id for record in records] == [
        "osbooks-university-physics-volume-1:ex-1",
        "osbooks-university-physics-volume-1:ex-2",
    ]
    first, second = records
    assert "2x = 8" in first.raw_question
    assert "The source solution is x = 4; no wording is added." in first.raw_analysis
    assert first.raw_answer == ""
    assert "3 m/s for 4 s" in second.raw_question
    assert "The source solution is 12 m" in second.raw_analysis
    assert "x = 4" not in second.raw_analysis

    first_dumped = first.model_dump(mode="json")
    assert first_dumped["raw_payload"]["document_id"] == (
        "osbooks-university-physics-volume-1"
    )
    assert first_dumped["raw_payload"]["exercise_id"] == "ex-1"
    assert first_dumped["raw_payload"]["problem_xml"] == first.raw_question
    assert first_dumped["raw_payload"]["solution_xml"] == first.raw_analysis
    assert "exercise_xml" in first_dumped["raw_payload"]
    assert first_dumped["metadata"]["textbook_title"] == "University Physics Volume 1"
    assert first_dumped["metadata"]["gold_role"] == "HIGH_CONFIDENCE"
    assert first_dumped["metadata"]["calibration_holdout"] is False
    assert first_dumped["metadata"]["quality_tier_candidate"] == "GOLD"
    assert "accepted" not in first_dumped["metadata"]
    assert first_dumped["license_metadata"] == _config()["license_metadata"]


def test_openstax_identity_and_checkpoint_resume_are_stable() -> None:
    first_adapter = OpenTextbookAdapter()
    first_descriptor = tuple(first_adapter.discover(_config()))[0]
    iterator = iter(first_adapter.acquire(first_descriptor))
    first_record = next(iterator)
    checkpoint = first_adapter.checkpoint()
    assert checkpoint.next_row_offset == 1

    resumed_adapter = OpenTextbookAdapter(resume_from=checkpoint)
    resumed_descriptor = tuple(resumed_adapter.discover(_config()))[0]
    resumed = list(resumed_adapter.acquire(resumed_descriptor))

    clean_adapter = OpenTextbookAdapter()
    clean_descriptor = tuple(clean_adapter.discover(_config()))[0]
    clean = list(clean_adapter.acquire(clean_descriptor))

    assert [record.source_id for record in resumed] == [
        "osbooks-university-physics-volume-1:ex-2"
    ]
    assert first_record.record_id == clean[0].record_id
    assert resumed[0].record_id == clean[1].record_id
    assert clean[0].raw_sha256 == first_record.raw_sha256


def test_openstax_missing_solution_fails_closed(tmp_path: Path) -> None:
    bad_xml = tmp_path / "missing-solution.xml"
    bad_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="http://cnx.rice.edu/cnxml" id="book-1">
  <title>Book 1</title>
  <content>
    <exercise id="ex-1"><problem><para>Original problem.</para></problem></exercise>
  </content>
</document>
""",
        encoding="utf-8",
    )
    adapter = OpenTextbookAdapter()
    descriptor = tuple(adapter.discover(_config(bad_xml)))[0]

    with pytest.raises(ValueError, match=r"exercise ex-1.*exactly one.*solution"):
        list(adapter.acquire(descriptor))


def test_openstax_multiple_solutions_fail_closed(tmp_path: Path) -> None:
    bad_xml = tmp_path / "multiple-solutions.xml"
    bad_xml.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="http://cnx.rice.edu/cnxml" id="book-1">
  <title>Book 1</title>
  <content>
    <exercise id="ex-1">
      <problem><para>Original problem.</para></problem>
      <solution><para>Solution one.</para></solution>
      <solution><para>Solution two.</para></solution>
    </exercise>
  </content>
</document>
""",
        encoding="utf-8",
    )
    adapter = OpenTextbookAdapter()
    descriptor = tuple(adapter.discover(_config(bad_xml)))[0]

    with pytest.raises(ValueError, match=r"exercise ex-1.*exactly one.*solution"):
        list(adapter.acquire(descriptor))
