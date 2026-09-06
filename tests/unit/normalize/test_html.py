from __future__ import annotations

import hashlib
import json
from pathlib import Path

from college_builder.domain.source import RawSourceRecord
from college_builder.normalize.html import normalize_html
from college_builder.normalize.normalizer import normalize
from college_builder.source.openstax import OpenTextbookAdapter

GOLDEN_CASES = Path("tests/golden/normalization_cases.json")


def _raw_record(*, question: str, answer: str, analysis: str) -> RawSourceRecord:
    return RawSourceRecord.model_validate(
        {
            "record_id": "raw_gold_task7_fixture",
            "source_type": "dataset",
            "source_dataset": "stemq",
            "source_id": "fixture-7",
            "source_url": "https://example.test/stemq/fixture-7",
            "raw_question": question,
            "raw_answer": answer,
            "raw_analysis": analysis,
            "raw_payload": {
                "question": question,
                "answer": answer,
                "analysis": analysis,
            },
            "metadata": {
                "gold_role": "CALIBRATION_GOLD",
                "calibration_holdout": True,
                "quality_tier_candidate": "GOLD",
                "institution": "Example University",
            },
            "license_metadata": {"declared": "CC-BY-4.0"},
            "raw_sha256": "a" * 64,
        }
    )


def test_html_normalization_matches_golden_cases() -> None:
    cases = json.loads(GOLDEN_CASES.read_text(encoding="utf-8"))

    for case in cases:
        result = normalize_html(case["input"])
        assert result.text == case["expected"], case["name"]
        assert list(result.images) == case["images"], case["name"]


def test_normalize_preserves_raw_duality_gold_role_and_hash_evidence() -> None:
    raw = _raw_record(
        question="<p>Solve 2x+3=7.</p>",
        answer="B",
        analysis="<p>Keep $x^2$ and the source expression 2x+3=7 unchanged.</p>",
    )
    before = raw.model_dump(mode="json")

    normalized = normalize(raw)
    dumped = normalized.model_dump(mode="json")

    assert normalized.source_record_id == raw.record_id
    assert normalized.question == "Solve 2x+3=7."
    assert normalized.answer == "B"
    assert normalized.analysis == "Keep $x^2$ and the source expression 2x+3=7 unchanged."
    assert "x=2" not in normalized.question
    assert "x=2" not in normalized.analysis
    assert "$x^2$" in normalized.analysis
    assert normalized.metadata["gold_role"] == "CALIBRATION_GOLD"
    assert normalized.metadata["calibration_holdout"] is True
    assert normalized.metadata["quality_tier_candidate"] == "GOLD"
    assert raw.model_dump(mode="json") == before

    evidence = dumped["normalization_evidence"]
    assert evidence["version"] == "normalization_v1"
    assert evidence["source_record_id"] == raw.record_id
    assert evidence["input_hashes"]["question"] == hashlib.sha256(
        raw.raw_question.encode()
    ).hexdigest()
    assert evidence["output_hashes"]["question"] == hashlib.sha256(
        normalized.question.encode()
    ).hexdigest()
    assert "content" in evidence["input_hashes"]
    assert "content" in evidence["output_hashes"]
    assert evidence["unresolved"] == []


def test_answer_option_is_not_expanded_or_inferred() -> None:
    raw = _raw_record(
        question="<p>Choose the correct option.</p>",
        answer="B",
        analysis="<p>The source solution says option B.</p>",
    )

    normalized = normalize(raw)

    assert normalized.answer == "B"
    assert normalized.answer not in {"Option B", "B. inferred answer"}
    assert normalized.analysis == "The source solution says option B."


def test_images_are_structurally_preserved_without_classification() -> None:
    raw = _raw_record(
        question=(
            '<p>Inspect <img src="https://example.test/q.png" alt="source diagram"></p>'
        ),
        answer="B",
        analysis=(
            '<p>See <img src="https://example.test/q.png" alt="same diagram"> and '
            '<img src="https://example.test/a.png" alt="analysis diagram"></p>'
        ),
    )

    normalized = normalize(raw)

    assert normalized.images == (
        "https://example.test/q.png",
        "https://example.test/a.png",
    )
    assert normalized.subject_candidates == ()


def test_mixed_html_native_latex_and_markdown_code_are_not_reparsed() -> None:
    raw = _raw_record(
        question=(
            "<p>Given $a<b<c$ and $x^2$, solve.</p>\n"
            "```python\nif a < b:\n    print(a)\n```"
        ),
        answer="<p>Keep `a < b` literally.</p>",
        analysis=(
            "<p>Source-native display math:</p>\n"
            "$$a<b<c$$\n"
            "```text\nleft < right\n  indented\n```"
        ),
    )

    normalized = normalize(raw)

    assert "Given $a<b<c$ and $x^2$, solve." in normalized.question
    assert "```python\nif a < b:\n    print(a)\n```" in normalized.question
    assert normalized.answer == "Keep `a < b` literally."
    assert "$$a<b<c$$" in normalized.analysis
    assert "```text\nleft < right\n  indented\n```" in normalized.analysis
    assert normalized.model_dump(mode="json")["normalization_evidence"]["unresolved"] == []


def test_plain_entities_and_markdown_images_are_preserved_structurally() -> None:
    result = normalize_html(
        "A &amp; B &lt; C. See ![diagram 7](https://example.test/md.png)."
    )

    assert result.text == "A & B < C. See ![diagram 7](https://example.test/md.png)."
    assert result.images == ("https://example.test/md.png",)


def test_html_sup_sub_table_and_source_ordered_labels_keep_semantics() -> None:
    superscript = normalize_html("<p>x<sup>2</sup> + y<sub>1</sub></p>")
    table = normalize_html(
        "<table><tr><td>1</td><td>2</td></tr>"
        "<tr><td>3</td><td>4</td></tr></table>"
    )
    ordered = normalize_html(
        '<ol type="A" start="3"><li>third</li><li value="5">fifth</li>'
        "<li>sixth</li></ol>"
    )
    nested = normalize_html("<ul><li>parent<ul><li>child</li></ul></li></ul>")

    assert superscript.text == "x^{2} + y_{1}"
    assert table.text == "| 1 | 2 |\n| 3 | 4 |"
    assert ordered.text == "C. third\nE. fifth\nF. sixth"
    assert nested.text == "- parent\n  - child"


def test_list_and_blockquote_preserve_code_blocks_and_indentation() -> None:
    list_result = normalize_html(
        "<ul><li>code<pre>if x < 2:\n    print(x)</pre></li></ul>"
    )
    quote_result = normalize_html(
        "<blockquote><pre>if y < 3:\n    print(y)</pre></blockquote>"
    )

    assert "- code" in list_result.text
    assert "```\n  if x < 2:\n      print(x)\n  ```" in list_result.text
    assert "> ```\n> if y < 3:\n>     print(y)\n> ```" in quote_result.text


def test_complex_table_fails_closed_without_silent_flattening() -> None:
    source = (
        '<table><tr><td rowspan="2">1</td><td>2</td></tr>'
        "<tr><td>3</td></tr></table>"
    )

    result = normalize_html(source)

    assert "<table" in result.text
    assert "rowspan" in result.text
    assert len(result.unresolved_structure) == 1
    assert result.unresolved_structure[0]["code"] == "STRUCTURE_UNRESOLVED"
    assert result.unresolved_structure[0]["reason"] == "UNSUPPORTED_TABLE_SPAN"


def test_open_textbook_adapter_to_normalize_handles_namespaced_cnxml(tmp_path: Path) -> None:
    fixture = tmp_path / "cnxml-mathml.xml"
    fixture.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<document xmlns="http://cnx.rice.edu/cnxml"
          xmlns:m="http://www.w3.org/1998/Math/MathML" id="book-ns">
  <title>Namespaced STEM</title>
  <content>
    <exercise id="ex-1">
      <problem>
        <para>Find <m:math><m:msup><m:mi>x</m:mi><m:mn>2</m:mn></m:msup></m:math>.</para>
        <para>Use <image src="diagram.png" alt="diagram"/> and
        <link url="https://example.test/reference">reference</link>.</para>
      </problem>
      <solution>
        <para>Keep <m:math><m:mrow><m:mn>7</m:mn><m:mo>-</m:mo><m:mn>3</m:mn>
        <m:mo>=</m:mo><m:mn>4</m:mn></m:mrow></m:math>.</para>
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
                "source_url": "https://example.test/book-ns",
                "revision": "fixture-ns-v1",
                "gold_role": "HIGH_CONFIDENCE",
                "license_metadata": {"declared": "CC-BY-4.0"},
            }
        )
    )[0]
    raw = list(adapter.acquire(descriptor))[0]
    assert "ns0:" in raw.raw_question
    assert "ns1:" in raw.raw_question

    normalized = normalize(raw)

    assert "$x^{2}$" in normalized.question
    assert "Find $x^{2}$.\n\nUse" in normalized.question
    assert "![diagram](diagram.png)" in normalized.question
    assert "[reference](https://example.test/reference)" in normalized.question
    assert normalized.images == ("diagram.png",)
    assert "$7-3=4$" in normalized.analysis
    assert normalized.model_dump(mode="json")["normalization_evidence"]["unresolved"] == []
