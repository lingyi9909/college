from __future__ import annotations

from college_builder.normalize.html import normalize_html
from college_builder.normalize.math import MathStatus, convert_mathml


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
