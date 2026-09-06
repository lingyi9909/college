from __future__ import annotations

from college_builder.domain.source import RawSourceRecord
from college_builder.normalize.html import normalize_html
from college_builder.normalize.math import MathStatus, convert_mathml
from college_builder.normalize.normalizer import normalize


def _raw_record(*, question: str, answer: str, analysis: str) -> RawSourceRecord:
    return RawSourceRecord.model_validate(
        {
            "record_id": "raw_math_task7_fixture",
            "source_type": "dataset",
            "source_dataset": "math-fixture",
            "source_id": "math-7",
            "source_url": "https://example.test/math-7",
            "raw_question": question,
            "raw_answer": answer,
            "raw_analysis": analysis,
            "raw_payload": {"question": question, "answer": answer, "analysis": analysis},
            "metadata": {"gold_role": "TRUSTED_TRAINING", "calibration_holdout": False},
            "license_metadata": {"declared": "CC-BY-4.0"},
            "raw_sha256": "b" * 64,
        }
    )


def test_mathml_converts_supported_structure_deterministically() -> None:
    source = (
        "<math><mrow><mn>7</mn><mo>-</mo><mn>3</mn><mo>=</mo>"
        "<mn>4</mn></mrow></math>"
    )

    result = convert_mathml(source)

    assert result.status is MathStatus.CONVERTED
    assert result.latex == "7-3=4"
    assert result.source_mathml == source


def test_mathml_preserves_exponent_exactly() -> None:
    source = "<math><msup><mi>x</mi><mn>2</mn></msup></math>"

    result = convert_mathml(source)

    assert result.status is MathStatus.CONVERTED
    assert result.latex == "x^{2}"
    assert "^{2}" in result.latex
    assert "^{3}" not in result.latex


def test_unsupported_mathml_is_unresolved_and_retains_original_evidence() -> None:
    source = (
        "<math><munderover><mo>∑</mo><mn>1</mn><mi>n</mi></munderover></math>"
    )

    result = convert_mathml(source)

    assert result.status is MathStatus.MATH_UNRESOLVED
    assert result.latex is None
    assert result.source_mathml == source
    assert result.reason == "UNSUPPORTED_MATHML_TAG:munderover"


def test_malformed_mathml_is_unresolved_without_guessing() -> None:
    source = "<math><mfrac><mn>1</mn></math>"

    result = convert_mathml(source)

    assert result.status is MathStatus.MATH_UNRESOLVED
    assert result.latex is None
    assert result.source_mathml == source
    assert result.reason == "MATHML_PARSE_ERROR"


def test_html_keeps_unresolved_mathml_and_marks_evidence() -> None:
    source = (
        "<p>Do not guess "
        "<math><munderover><mo>∑</mo><mn>1</mn><mi>n</mi></munderover></math>."
        "</p>"
    )

    result = normalize_html(source)

    assert "<math>" in result.text
    assert "munderover" in result.text
    assert len(result.unresolved_math) == 1
    unresolved = result.unresolved_math[0]
    assert unresolved["code"] == "MATH_UNRESOLVED"
    assert unresolved["reason"] == "UNSUPPORTED_MATHML_TAG:munderover"
    assert "munderover" in unresolved["source_mathml"]


def test_source_solution_numbers_operators_and_formula_meaning_survive() -> None:
    source = (
        "<p>From <math><mrow><mn>7</mn><mo>-</mo><mn>3</mn><mo>=</mo>"
        "<mn>4</mn></mrow></math>, keep $x^2$ unchanged.</p>"
    )

    result = normalize_html(source)

    assert "$7-3=4$" in result.text
    assert "$x^2$" in result.text
    assert "7-3=4" in result.text


def test_mathml_semantic_attributes_fail_closed_in_formal_normalize() -> None:
    malformed = "<p>Broken <math><msup><mi>x</mi><mn>2</mn></p>"
    bold = '<math><mi mathvariant="bold">v</mi></math>'
    zero_line_fraction = '<math><mfrac linethickness="0"><mn>1</mn><mn>2</mn></mfrac></math>'
    raw = _raw_record(question=malformed, answer=bold, analysis=zero_line_fraction)

    normalized = normalize(raw)
    evidence = normalized.model_dump(mode="json")["normalization_evidence"]["unresolved"]

    assert "<math><msup><mi>x</mi><mn>2</mn>" in normalized.question
    assert normalized.answer == bold
    assert normalized.analysis == zero_line_fraction
    assert {item["field"] for item in evidence} == {"question", "answer", "analysis"}
    reasons = {item["field"]: item["reason"] for item in evidence}
    assert reasons["question"] == "MATHML_PARSE_ERROR"
    assert reasons["answer"] == "UNSUPPORTED_MATHML_ATTRIBUTE:mi:mathvariant"
    assert reasons["analysis"] == "UNSUPPORTED_MATHML_ATTRIBUTE:mfrac:linethickness"


def test_mfenced_empty_separator_and_mtext_special_chars_preserve_semantics() -> None:
    fenced = convert_mathml(
        '<math><mfenced separators=""><mi>x</mi><mi>y</mi></mfenced></math>'
    )
    text = convert_mathml("<math><mtext>50% chance</mtext></math>")

    assert fenced.status is MathStatus.CONVERTED
    assert fenced.latex == "(xy)"
    assert "," not in fenced.latex
    assert text.status is MathStatus.CONVERTED
    assert text.latex == r"\text{50\% chance}"


def test_formal_normalize_preserves_mfenced_and_mtext_across_fields() -> None:
    raw = _raw_record(
        question='<p><math><mfenced separators=""><mi>x</mi><mi>y</mi></mfenced></math></p>',
        answer="<math><mtext>50% chance</mtext></math>",
        analysis=(
            "<p><math><mrow><mn>7</mn><mo>-</mo><mn>3</mn><mo>=</mo>"
            "<mn>4</mn></mrow></math></p>"
        ),
    )

    normalized = normalize(raw)

    assert normalized.question == "$(xy)$"
    assert normalized.answer == r"$\text{50\% chance}$"
    assert normalized.analysis == "$7-3=4$"
    assert normalized.model_dump(mode="json")["normalization_evidence"]["unresolved"] == []
