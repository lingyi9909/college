from __future__ import annotations

import pytest

from college_builder.domain.source import NormalizedQA
from college_builder.quality.answer import extract_source_answer


def _candidate(analysis: str, *, answer: str = "") -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16-answer",
        source_record_id="raw-task16-answer",
        question="Solve the source problem.",
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
