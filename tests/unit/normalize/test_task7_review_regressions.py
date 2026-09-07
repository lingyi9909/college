from __future__ import annotations

from pathlib import Path

from college_builder.normalize.html import normalize_html
from college_builder.normalize.normalizer import normalize
from college_builder.source.openstax import OpenTextbookAdapter


def test_reversed_ordered_list_uses_descending_default_start() -> None:
    result = normalize_html(
        "<ol reversed><li>three</li><li>two</li><li>one</li></ol>"
    )

    assert result.text == "3. three\n2. two\n1. one"
    assert result.unresolved_structure == ()


def test_reversed_ordered_list_honors_start_and_type_marker() -> None:
    result = normalize_html(
        '<ol reversed start="5" type="A"><li>five</li><li>four</li></ol>'
    )

    assert result.text == "E. five\nD. four"
    assert result.unresolved_structure == ()


def test_reversed_ordered_list_li_value_resets_descending_sequence() -> None:
    result = normalize_html(
        '<ol reversed start="5"><li>five</li><li value="2">two</li><li>one</li></ol>'
    )

    assert result.text == "5. five\n2. two\n1. one"
    assert result.unresolved_structure == ()


def test_unrepresentable_reversed_marker_combination_fails_closed() -> None:
    source = '<ol reversed type="A"><li value="0">zero</li></ol>'

    result = normalize_html(source)

    assert "<ol" in result.text
    assert "reversed" in result.text
    assert result.unresolved_structure
    assert result.unresolved_structure[0]["code"] == "STRUCTURE_UNRESOLVED"


def test_openstax_cnxml_target_id_link_is_preserved_with_unresolved_evidence(
    tmp_path: Path,
) -> None:
    fixture = tmp_path / "cnxml-target-id.xml"
    fixture.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="http://cnx.rice.edu/cnxml" id="book-target-id">
  <title>Target ID Link</title>
  <content>
    <exercise id="ex-1">
      <problem>
        <para>Inspect figure <link target-id="fig-1"/> before answering.</para>
      </problem>
      <solution>
        <para>The source points to figure fig-1.</para>
      </solution>
    </exercise>
  </content>
</document>
""",
        encoding="utf-8",
    )
    adapter = OpenTextbookAdapter()
    descriptor = tuple(
        adapter.discover(
            {
                "dataset_name": "openstax",
                "data_file": str(fixture),
                "source_url": "https://example.test/book-target-id",
                "revision": "fixture-target-id-v1",
                "gold_role": "HIGH_CONFIDENCE",
                "license_metadata": {"declared": "CC-BY-4.0"},
            }
        )
    )[0]
    raw = list(adapter.acquire(descriptor))[0]

    normalized = normalize(raw)
    evidence = normalized.model_dump(mode="json")["normalization_evidence"]["unresolved"]

    assert "target-id=\"fig-1\"" in normalized.question
    assert any(
        item["code"] == "STRUCTURE_UNRESOLVED"
        and "target-id=\"fig-1\"" in item["source_markup"]
        for item in evidence
    )
