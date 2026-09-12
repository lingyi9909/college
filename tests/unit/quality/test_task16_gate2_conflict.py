from __future__ import annotations

import pytest

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import NormalizedQA
from college_builder.providers.base import ModelDecision
from college_builder.providers.fake import FakeStructuredModelProvider
from college_builder.quality.classify import ProblemGate
from college_builder.quality.engine import GateContext, GateEngine

_DEFAULT_QUESTION = "Prove the stated identity and derive the requested expression."

# Frozen from Task 15 real-model artifact 10183643305. Decision fields are exact;
# question excerpts are exact source excerpts sufficient to validate the recorded references.
_TASK15_REAL_POSITIVE_CONFLICTS = ({'source_id': 'math:2142021:0',
  'raw_sha256': '58d66855aa3323bcf34290eaa1afcc7bc0c60a865b1e3796115ae4cbbdbc8ca7',
  'question': 'A pole in a multiple integral I want to know whether the following integral '
              'converge in the region $k\\to \\infty$ or not. How should we treat this pole in the '
              'multiple integral?',
  'primary': {'label': 'CONCEPTUAL',
              'score': 0.92,
              'evidence_references': ('question: "A pole in a multiple integral I want to know '
                                      'whether the following integral converge in the region '
                                      '$k\\to \\infty$ or not."',
                                      'question: "How should we treat this pole in the multiple '
                                      'integral?"'),
              'reason_code': 'MATHEMATICAL_CONVERGENCE_AND_REGULARIZATION_QUESTION'},
  'verifier': {'label': 'CALCULATION',
               'score': 0.95,
               'evidence_references': ('question: A pole in a multiple integral I want to know '
                                       'whether the following integral converge in the region k→∞ '
                                       'or not. ∫ d^4k/(2π)^4 1/k^4 e^{-ik·ε} ... How should we '
                                       'treat this pole in the multiple integral?',),
               'reason_code': 'PROBLEM'},
  'expected_verdict': 'REJECT',
  'expected_reason': 'PROBLEM_TYPE_UNCERTAIN'},
 {'source_id': 'math:3463165:0',
  'raw_sha256': '269d28fe5b84625f596398c3e3cf8d9bac8f37bc0edb259b1c1d47ac2f456c70',
  'question': 'Determine the upper limit of the step size $h$ in this Euler method simulation One '
              'of the theoretical questions, i.e. that should be done by calculation and not by '
              'running the simulation, is to find at what exact value of $h$ the model breaks down '
              'that $x_i^n > x_{i+1}^n$ for some time step $n$',
  'primary': {'label': 'CALCULATION',
              'score': 0.95,
              'evidence_references': ('question: Determine the upper limit of the step size $h$ in '
                                      'this Euler method simulation',
                                      'question: One of the theoretical questions, i.e. that '
                                      'should be done by calculation and not by running the '
                                      'simulation, is to find at what exact value of $h$ the model '
                                      'breaks down',
                                      'question: that $x_i^n > x_{i+1}^n$ for some time step $n$'),
              'reason_code': 'POSITIVE_CALCULATION_PROBLEM'},
  'verifier': {'label': 'DERIVATION',
               'score': 0.9,
               'evidence_references': ('question: Determine the upper limit of the step size h in '
                                       'this Euler method simulation',
                                       'question: One of the theoretical questions, i.e. that '
                                       'should be done by calculation and not by running the '
                                       'simulation, is to find at what exact value of h the model '
                                       'breaks down in that the cars can drive past each other, in '
                                       'other words, that x_i^n > x_{i+1}^n for some time step n.'),
               'reason_code': 'ASSESSABLE_PROBLEM'},
  'expected_verdict': 'REJECT',
  'expected_reason': 'PROBLEM_TYPE_UNCERTAIN'},
 {'source_id': 'physics:23682:1',
  'raw_sha256': 'ad3f6b84198bd09d27b2e2fb2de6a1d892638444966796e68e5f5e228b6cd822',
  'question': 'Free particle propagation amplitude calculation I have a quick calculational '
              'question. In Peskin and Schroeder, Chapter 2, they want to look at the amplitude '
              'for a particle to propagate between two arbitrary points, $x$ and $x_0$, in an '
              "arbitrary amount of time t. I'm not quite sure where to proceed from here... Peskin "
              "and Schroeder somehow reach: Aren't we summing over all possible momenta?",
  'primary': {'label': 'DERIVATION',
              'score': 0.92,
              'evidence_references': ('question:In Peskin and Schroeder, Chapter 2, they want to '
                                      'look at the amplitude for a particle to propagate between '
                                      'two arbitrary points...',
                                      "question:I'm not quite sure where to proceed from here... "
                                      'Peskin and Schroeder somehow reach...',
                                      'question:To get here, however, it seems that you would have '
                                      'to assume that p points in the same direction as x-x0... '
                                      "Why can we do this? Aren't we summing over all possible "
                                      'momenta?'),
              'reason_code': 'CONCRETE_DERIVATION_QUESTION'},
  'verifier': {'label': 'CALCULATION',
               'score': 1.0,
               'evidence_references': ('question:Free particle propagation amplitude calculation I '
                                       "have a quick calculational question... Aren't we summing "
                                       'over all possible momenta?',),
               'reason_code': 'CONCRETE_PROBLEM'},
  'expected_verdict': 'VERIFY',
  'expected_reason': 'PROBLEM_REVIEW_REQUIRED'},
 {'source_id': 'mathoverflow:88899:0',
  'raw_sha256': '038be9cab1e0d3404548bcdf2ae8ccf4023bdd03c7b2d109b733b40c38dd2bbf',
  'question': 'If we consider an induced representation $Ind_{H}^{G}1_{H}$ of finite groups, where '
              '$1_{H}$ is the trivial caracter of $H$, how we decompose this representation into '
              "irreducible ? I don't know if $ W $ is irreducible and if it is not irreducible how "
              'decomposed there into irreducible ? The question arises also for profinite groups : '
              'If $G$ is totaly disconnected compact group and $H$ is an open subgroup, how to '
              'decompose the representation $Ind_{H}^{G}1_{H}$ into irreducibles ?',
  'primary': {'label': 'CONCEPTUAL',
              'score': 0.9,
              'evidence_references': ('question:how we decompose this representation into '
                                      'irreducible ?',
                                      "question:I don't know if $ W $ is irreducible and if it is "
                                      'not irreducible how decomposed there into irreducible ?'),
              'reason_code': 'POSITIVE_CONCEPTUAL'},
  'verifier': {'label': 'DERIVATION',
               'score': 0.95,
               'evidence_references': ('question:If we consider an induced representation '
                                       '$Ind_{H}^{G}1_{H}$ of finite groups, where $1_{H}$ is the '
                                       'trivial caracter of $H$, how we decompose this '
                                       'representation into irreducible? ... The question arises '
                                       'also for profinite groups : If $G$ is totaly disconnected '
                                       'compact group and $H$ is an open subgroup, how to '
                                       'decompose the representation $Ind_{H}^{G}1_{H}$ into '
                                       'irreducibles ?',),
               'reason_code': 'PROBLEM'},
  'expected_verdict': 'REJECT',
  'expected_reason': 'PROBLEM_TYPE_UNCERTAIN'})


def _decision(
    label: str,
    score: float = 0.95,
    *,
    evidence_references: tuple[str, ...] = ("question:0-20",),
    reason_code: str = "TASK16_REGRESSION",
) -> ModelDecision:
    return ModelDecision(
        label=label,
        score=score,
        evidence_references=evidence_references,
        reason_code=reason_code,
    )


def _candidate(
    question: str = _DEFAULT_QUESTION,
    *,
    source_record_id: str = "raw-task16-gate2",
) -> NormalizedQA:
    return NormalizedQA(
        record_id=f"norm-{source_record_id}",
        source_record_id=source_record_id,
        question=question,
        answer="source answer",
        analysis="source analysis",
        subject_candidates=("mathematics",),
        images=(),
        metadata={},
        normalization_evidence={},
    )


def _run(
    primary: ModelDecision,
    verifier: ModelDecision,
    *,
    candidate: NormalizedQA | None = None,
):
    primary_provider = FakeStructuredModelProvider(
        provider="fake",
        model="primary-v1",
        decisions=(primary,),
    )
    verifier_provider = FakeStructuredModelProvider(
        provider="fake",
        model="verifier-v2",
        decisions=(verifier,),
    )
    gate = ProblemGate(
        primary=primary_provider,
        verifier=verifier_provider,
        prompt="problem prompt",
        prompt_version="v1",
    )
    result = GateEngine(
        gates=(gate,),
        context=GateContext(config_version="task16a1-regression"),
    ).run(candidate or _candidate())
    return result.evidence[0]


def test_gate2_compatible_positive_labels_with_valid_evidence_verify() -> None:
    evidence = _run(_decision("PROOF"), _decision("DERIVATION", 0.99))

    assert evidence.verdict is GateVerdict.VERIFY
    assert evidence.reason_code == "PROBLEM_REVIEW_REQUIRED"


@pytest.mark.parametrize(
    ("primary_refs", "verifier_refs"),
    [
        (("question:not-a-span",), ("question:0-20",)),
        (("question:0-20",), ("question:garbage",)),
        (("question:not-a-span",), ("question:garbage",)),
    ],
    ids=("malformed-primary", "malformed-verifier", "malformed-both"),
)
def test_gate2_compatible_positive_malformed_evidence_fails_closed(
    primary_refs: tuple[str, ...],
    verifier_refs: tuple[str, ...],
) -> None:
    evidence = _run(
        _decision("PROOF", evidence_references=primary_refs),
        _decision("DERIVATION", 0.99, evidence_references=verifier_refs),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "PROBLEM_TYPE_UNCERTAIN"


def test_gate2_compatible_positive_unsupported_evidence_fails_closed() -> None:
    evidence = _run(
        _decision("PROOF", evidence_references=("analysis:0-20",)),
        _decision("DERIVATION", 0.99),
    )

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "PROBLEM_TYPE_UNCERTAIN"


def test_gate2_positive_negative_disagreement_still_fails_closed() -> None:
    evidence = _run(_decision("PROOF"), _decision("DISCUSSION", 0.99))

    assert evidence.verdict is GateVerdict.REJECT
    assert evidence.reason_code == "MODEL_DECISION_CONFLICT"


@pytest.mark.parametrize(
    "fixture",
    _TASK15_REAL_POSITIVE_CONFLICTS,
    ids=lambda fixture: fixture["source_id"],
)
def test_task15_real_positive_conflicts_use_grounded_evidence_semantics(
    fixture: dict[str, object],
) -> None:
    primary_data = fixture["primary"]
    verifier_data = fixture["verifier"]
    assert isinstance(primary_data, dict)
    assert isinstance(verifier_data, dict)

    primary = _decision(
        str(primary_data["label"]),
        float(primary_data["score"]),
        evidence_references=tuple(str(item) for item in primary_data["evidence_references"]),
        reason_code=str(primary_data["reason_code"]),
    )
    verifier = _decision(
        str(verifier_data["label"]),
        float(verifier_data["score"]),
        evidence_references=tuple(str(item) for item in verifier_data["evidence_references"]),
        reason_code=str(verifier_data["reason_code"]),
    )
    evidence = _run(
        primary,
        verifier,
        candidate=_candidate(
            str(fixture["question"]),
            source_record_id=str(fixture["source_id"]),
        ),
    )

    assert evidence.reason_code != "MODEL_DECISION_CONFLICT"
    assert evidence.verdict is GateVerdict(str(fixture["expected_verdict"]))
    assert evidence.reason_code == fixture["expected_reason"]
