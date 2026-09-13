from __future__ import annotations

import pytest

from college_builder.domain.source import NormalizedQA
from college_builder.quality.answer import extract_source_answer


def _candidate(
    analysis: str,
    *,
    answer: str = "",
    question: str = "Solve the source problem.",
) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16-answer",
        source_record_id="raw-task16-answer",
        question=question,
        answer=answer,
        analysis=analysis,
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        (
            "Using the beta-function identity gives the closed form.\n"
            "$(1010!)^2/2021!$",
            "$(1010!)^2/2021!$",
        ),
        (
            "After evaluating the integral over the stated region, the value is\n"
            "$45\\pi/4$",
            "$45\\pi/4$",
        ),
        (
            "Combining the two constraints gives the equivalent radius\n"
            "$r_{eq}=2l$",
            "$r_{eq}=2l$",
        ),
        (
            "Applying the WKB boundary conditions yields the quantization condition\n"
            "$\\int_{x_1}^{x_2} p(x)\\,dx=(n+1/2)\\pi\\hbar$",
            "$\\int_{x_1}^{x_2} p(x)\\,dx=(n+1/2)\\pi\\hbar$",
        ),
    ],
)
def test_gate3_extracts_explicit_terminal_source_conclusion(
    analysis: str,
    expected: str,
) -> None:
    extraction = extract_source_answer(_candidate(analysis))

    assert extraction.final_answer == expected
    assert extraction.source_field == "analysis"
    assert extraction.start_offset is not None
    assert extraction.end_offset is not None
    assert analysis[extraction.start_offset : extraction.end_offset] == expected
    assert extraction.source_span == (
        f"analysis:{extraction.start_offset}:{extraction.end_offset}"
    )


@pytest.mark.parametrize("expected", ["42", "3.5", "-7", "1/2"])
def test_gate3_extracts_standalone_terminal_numeric_source_answer(expected: str) -> None:
    analysis = f"The source computation ends with the following result.\n{expected}"

    extraction = extract_source_answer(_candidate(analysis))

    assert extraction.final_answer == expected
    assert extraction.source_field == "analysis"
    assert extraction.start_offset is not None
    assert extraction.end_offset is not None
    assert analysis[extraction.start_offset : extraction.end_offset] == expected
    assert extraction.source_span == (
        f"analysis:{extraction.start_offset}:{extraction.end_offset}"
    )


def test_gate3_preserves_complete_causal_answer_for_why_question() -> None:
    question = "Why can geometric optics be used to analyze a microscope image?"
    analysis = "Microscopes create images of macroscopic size, so geometric optics applies."

    extraction = extract_source_answer(_candidate(analysis, question=question))

    assert extraction.final_answer == analysis
    assert extraction.source_field == "analysis"
    assert extraction.start_offset == 0
    assert extraction.end_offset == len(analysis)
    assert extraction.source_span == f"analysis:0:{len(analysis)}"


def test_gate3_preserves_complete_causal_answer_for_explain_why_question() -> None:
    question = "Explain why the gas temperature rises during compression."
    analysis = "Compression increases molecular collision frequency, therefore temperature rises."

    extraction = extract_source_answer(_candidate(analysis, question=question))

    assert extraction.final_answer == analysis
    assert extraction.source_field == "analysis"
    assert extraction.start_offset == 0
    assert extraction.end_offset == len(analysis)


def test_gate3_generic_explain_preserves_mechanically_explicit_causal_premise() -> None:
    question = "Explain the observed temperature increase."
    analysis = (
        "The gas is compressed because external work raises its internal energy, "
        "therefore the temperature increases."
    )

    extraction = extract_source_answer(_candidate(analysis, question=question))

    assert extraction.final_answer == analysis
    assert extraction.source_field == "analysis"
    assert extraction.start_offset == 0
    assert extraction.end_offset == len(analysis)


def test_gate3_generic_explain_keeps_self_contained_formal_result() -> None:
    question = "Explain the result."
    analysis = "Substituting the boundary values cancels the remaining terms; therefore 4"

    extraction = extract_source_answer(_candidate(analysis, question=question))

    assert extraction.final_answer == "4"
    assert extraction.source_field == "analysis"
    assert extraction.start_offset == analysis.rindex("4")
    assert extraction.end_offset == len(analysis)


def test_gate3_calculation_keeps_terminal_formal_result() -> None:
    question = "Calculate x."
    analysis = "Subtracting 3 from both sides and dividing by 2. Therefore x = 4"

    extraction = extract_source_answer(_candidate(analysis, question=question))

    assert extraction.final_answer == "x = 4"
    assert extraction.source_field == "analysis"
    assert extraction.start_offset == analysis.rindex("x = 4")
    assert extraction.end_offset == len(analysis)


def test_gate3_does_not_derive_answer_when_terminal_source_text_is_absent() -> None:
    analysis = "Solving 2x + 3 = 11 requires isolating x by ordinary algebra."

    extraction = extract_source_answer(_candidate(analysis))

    assert extraction.final_answer is None
    assert extraction.source_field is None


def test_gate3_does_not_treat_trailing_explanatory_sentence_as_answer() -> None:
    analysis = (
        "The computation gives $6$. This demonstrates how substitution simplifies "
        "the original expression."
    )

    extraction = extract_source_answer(_candidate(analysis))

    assert extraction.final_answer is None
    assert extraction.source_field is None
