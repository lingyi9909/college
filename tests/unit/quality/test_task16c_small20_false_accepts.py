from __future__ import annotations

from pathlib import Path

from college_builder.domain.source import NormalizedQA
from college_builder.quality.answer import extract_source_answer


def _candidate(*, question: str, analysis: str) -> NormalizedQA:
    return NormalizedQA(
        record_id="norm-task16c-small20-fa",
        source_record_id="raw-task16c-small20-fa",
        question=question,
        answer="",
        analysis=analysis,
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def test_gate3_rejects_context_dependent_generic_conclusion_fragment() -> None:
    question = "Prove that (J:I)=J when I+J=R. Is this correct?"
    analysis = (
        "Since I+J=R, write 1=i+j. For r in (J:I), r=ri+rj is in J, "
        "so (J:I) is contained in J. J is always contained in (J:I), so we have equality."
    )

    extraction = extract_source_answer(_candidate(question=question, analysis=analysis))

    assert extraction.final_answer is None
    assert extraction.source_field is None


def test_gate5_prompts_require_formal_answer_completeness_not_analysis_substitution() -> None:
    for path in (
        Path("prompts/qa_alignment/v2.txt"),
        Path("prompts/correctness_verify/v2.txt"),
    ):
        prompt = path.read_text(encoding="utf-8").lower()
        assert "formal answer must" in prompt
        assert "all requested" in prompt
        assert "analysis must not substitute" in prompt


def test_correctness_prompt_explicitly_rejects_unfinished_required_operations() -> None:
    prompt = Path("prompts/correctness_verify/v2.txt").read_text(encoding="utf-8").lower()

    assert "orthonormal" in prompt
    assert "orthogonal" in prompt
    assert "normalization" in prompt
    assert "fail" in prompt
