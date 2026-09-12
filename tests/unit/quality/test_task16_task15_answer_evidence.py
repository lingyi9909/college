from __future__ import annotations

import hashlib

import pytest

from college_builder.domain.source import NormalizedQA
from college_builder.quality.answer import extract_source_answer

_TASK15_PRODUCTION_SHA = "d3b35964ac92d799405fd88101fa11fd05ff1460"
_TASK15_RUN_ID = 34558426996
_TASK15_ARTIFACT_ID = 10183643305
_TASK15_SOURCE_REVISION = "git:13239ec8c8078bc8dfb43e1383069eb9be000ff7"

# Frozen from Task 15 real-model artifact 10183643305 and the 100/100 manual audit.
# Each tuple binds source_id -> immutable raw record identity -> exact source evidence.
_TASK15_GATE3_FALSE_REJECTS = (
    ("math:4453111:4", "raw_stackmathqa_f7762046904511bb6ebfaf2e4f125ea85bd73ee35d16f895fd5a0d73fc24bb52", "73cfeeaa55454bef8db46ad0c1d777484f7ebb03752acfd33299159bfe29206e", "terminal_math_before_notes", r"\frac{(1010!)^2}{2021!}"),
    ("math:2544901:0", "raw_stackmathqa_7945c0c47a1ecf60ba26888f9eaccc2cd983deb004087e8862c28810ee20dab9", "02d2103c5cf4cea57a97a6816f43a2bf01fe42a2353fd242f4f8cb15d810decd", "natural_language_named_condition", r"\Ass N =\Ass M -\Phi, \quad \Ass M/N=\Phi"),
    ("math:3701703:0", "raw_stackmathqa_a089c58c53d2ffb300be3b889bbef0444c58f765d8a6e0da2a1cf50d204e0636", "17abcd415fb67deb4740b4624a231de65ac1b4daf7b94c22f5860a8ffdfc3bd2", "leading_yes_no", "Yes."),
    ("math:4338045:0", "raw_stackmathqa_d1a80c7820635b8175e300332bb3a721e218f8dd2563dae40e3b8fe160d19f26", "e0337766bebe0ebb3719ad7866ac8b209cfb88936e9528fa58bc78dfef0ffd8c", "natural_language_named_conclusion", r"$\Phi$ is injective."),
    ("math:2975169:0", "raw_stackmathqa_a5ccb47933f6b5a4d5ebd1ee2a5ac0aad7b943279822b126d9212a681c94ccf9", "ab5324a083529f27b48c89b9fe5c2cec91241c3c57d6fca1dc77f771fcfe3a56", "natural_language_result", r"$\sigma_1+\sigma_1=2\sigma_1$ is $0$"),
    ("math:2176500:0", "raw_stackmathqa_6d2ce95741d4b0bffdefe8bbaa54703a22c726fa6b4884f39ea21bfc9182de3d", "d147bc27f390e68674d1e7d786537a8d5199bab10deab32125edcb3246ad3378", "conclusion_marker_display_math", r"$$\int_S\nabla\times \vec V\cdot \hat n\,dS=\int_0^{2\pi}\int_1^2 3\rho^3\cos^2(\phi)\,d\rho\,d\phi=45\pi/4$$"),
    ("math:3969220:0", "raw_stackmathqa_36383322c62efa98ac513621c69ff25f05bdb72a2c74dadc9870c33f8b9c030d", "e32e233508cb23c63e5eee7024e7ea15eaa6a667cf636cd2826108205868c1ea", "conclusion_marker_so", r"So $B$ is an open set."),
    ("math:277083:2", "raw_stackmathqa_e1ba557e2fd7257ce379c838903caab6bcd89caeddb0df28aa58d8f8b919ab87", "0f1f82fe1f0af223b7f10b79b848cdeeeb519dd2003f258ed9d38e403e2eb6f7", "leading_yes_no", "Yes, as Julien's answer says, you can apply L'Hopital."),
    ("math:193977:0", "raw_stackmathqa_872fb608cf18c280288cb5d4ca3b838a702025ed647261da49d4ce0100eb010e", "69e546864ddce98658ed49374ab5f62c3c04b082d88646e00ae163f9800bd182", "symbolic_conclusion", "∴¬q"),
    ("math:4094740:0", "raw_stackmathqa_d38475520fa9d7152ab07d1ee2a92951cbf29ba1692e14aae27368550e2ccf9c", "6dc89bbf6f14313e45711d10fb3c303a7674e620191fe950dab9dc5385d7ee29", "named_construction", r"$$\pi(x,y) \triangleq \frac{(x+y)(x+y+1)}{2} + y$$"),
    ("physics:658710:0", "raw_stackmathqa_2ecaaaa01aa99aaef794eee1d29611745acc32c5722a00d467de00cecb97c0a7", "4aa2823186b2b414c0d2847c620697949fa468973ecf6796c09b05a33f80ac88", "terminal_math_before_closure", r"\boxed{r_\text{eq} = 2l}"),
    ("physics:390716:0", "raw_stackmathqa_831d2384d89550f980da78ffeaa2c08073407fd51a00e25d68a06b10b8ba080b", "c74bcffa9d388a74b54cf402e02eae0beb4e0957d7e2f690445464f27a63aab2", "named_state", r"This is how you obtain the $^1\Sigma_\mathrm u^-$ state"),
    ("physics:540721:0", "raw_stackmathqa_32b398558474c0a6b5673614df83c3dc25fcc7905d46acdc83b01b5a632bd1c4", "20ed8b4946ee934f343db9af35bd02268f7025e69aaee15e6fed3011c403833a", "named_condition", r"$$n h = \int_0^L \sqrt{2 m (E_n - V(x))} \, dx$$"),
    ("statistics:206494:1", "raw_stackmathqa_1c4381e616889f6b78d899c8520c891cc6416683e764225410193b1962f9b0c8", "1c8d1511d6e8b04e3fbe0e9dcc043f6e1cafefb680e63073d7a4f2b2c00ccd19", "terminal_numeric_display_math", r"$$\int_{-\infty}^\infty F(x)f(x) \text{dx}=\frac{1}{2}$$"),
    ("mathoverflow:325965:0", "raw_stackmathqa_5973c7eabbaa8130f6ccdf9b9737975bb5c183a7e2fc946f9bee1a92414ea48b", "7ddd38775a95f1e48cf8217af43425da6b5efdaacbdae37fc265fa447dccaeaf", "natural_language_isomorphism_conclusion", r"$\mathcal{O}_{X,x}/\mathfrak{a}$ is isomorphic, as a local $\mathbb{C}$-algebra, to $\mathcal{O}_{X^G,x}$"),
    ("mathoverflow:377204:0", "raw_stackmathqa_a90f524eb66bd0b1f923e9b2bf418e4a505ecc26c5076a4c30a34056eabbfd10", "c7a236ccf808703e6856d5311f49bb65dfd01ff51e8b25eb87069091847fa039", "explicit_answer_sentence", r"the maximum $k$ is given by $\lfloor \frac{p-c}{2}\rfloor$"),
)


def _candidate(source_record_id: str, analysis: str) -> NormalizedQA:
    return NormalizedQA(
        record_id=f"norm-{source_record_id[-12:]}",
        source_record_id=source_record_id,
        question="Task 15 frozen source question",
        answer="",
        analysis=analysis,
        subject_candidates=("mathematics",),
        images=(),
        metadata={
            "task15_production_sha": _TASK15_PRODUCTION_SHA,
            "task15_run_id": _TASK15_RUN_ID,
            "task15_artifact_id": _TASK15_ARTIFACT_ID,
            "source_revision": _TASK15_SOURCE_REVISION,
        },
        normalization_evidence={},
    )


def test_task15_gate3_false_reject_inventory_is_frozen_to_real_source_identity() -> None:
    assert len(_TASK15_GATE3_FALSE_REJECTS) == 16
    assert len({case[0] for case in _TASK15_GATE3_FALSE_REJECTS}) == 16
    assert len({case[1] for case in _TASK15_GATE3_FALSE_REJECTS}) == 16
    for source_id, record_id, raw_sha256, _category, excerpt in _TASK15_GATE3_FALSE_REJECTS:
        assert source_id
        assert record_id.startswith("raw_stackmathqa_")
        assert len(raw_sha256) == 64
        int(raw_sha256, 16)
        assert excerpt


_REAL_EXTRACTION_CASES = (
    pytest.param(
        "math:2176500:0",
        "raw_stackmathqa_6d2ce95741d4b0bffdefe8bbaa54703a22c726fa6b4884f39ea21bfc9182de3d",
        "d147bc27f390e68674d1e7d786537a8d5199bab10deab32125edcb3246ad3378",
        "Let $\\vec V=\\hat y x^3+\\hat z z^3$.  Then, $\\nabla \\times \\vec V=3\\hat z x^2$.  \n"
        "A vector point on the surface can be written as $\\vec r=\\hat \\rho\\rho +\\hat z\\rho^2$.\n"
        "Therefore, \n"
        "$$\\int_S\\nabla\\times \\vec V\\cdot \\hat n\\,dS=\\int_0^{2\\pi}\\int_1^2 3\\rho^3\\cos^2(\\phi)\\,d\\rho\\,d\\phi=45\\pi/4$$",
        r"$$\int_S\nabla\times \vec V\cdot \hat n\,dS=\int_0^{2\pi}\int_1^2 3\rho^3\cos^2(\phi)\,d\rho\,d\phi=45\pi/4$$",
        id="task15-marker-display-math-45pi-over-4",
    ),
    pytest.param(
        "math:3969220:0",
        "raw_stackmathqa_36383322c62efa98ac513621c69ff25f05bdb72a2c74dadc9870c33f8b9c030d",
        "e32e233508cb23c63e5eee7024e7ea15eaa6a667cf636cd2826108205868c1ea",
        "You can show $B^C$ is closed.\nSo $B^C$ is closed.\nSo $B$ is an open set.",
        r"$B$ is an open set.",
        id="task15-so-conclusion-open-set",
    ),
    pytest.param(
        "math:4094740:0",
        "raw_stackmathqa_d38475520fa9d7152ab07d1ee2a92951cbf29ba1692e14aae27368550e2ccf9c",
        "6dc89bbf6f14313e45711d10fb3c303a7674e620191fe950dab9dc5385d7ee29",
        "The bijection provided by Cantor's Pairing Function is polynomially bounded.\n"
        "Indeed it is a polynomial itself -- $\\pi : \\mathbb{N}^2 \\to \\mathbb{N}$ is given by\n"
        "$$\\pi(x,y) \\triangleq \\frac{(x+y)(x+y+1)}{2} + y$$\n\nI hope this helps ^_^",
        r"$$\pi(x,y) \triangleq \frac{(x+y)(x+y+1)}{2} + y$$",
        id="task15-named-cantor-pairing-construction",
    ),
    pytest.param(
        "physics:540721:0",
        "raw_stackmathqa_32b398558474c0a6b5673614df83c3dc25fcc7905d46acdc83b01b5a632bd1c4",
        "20ed8b4946ee934f343db9af35bd02268f7025e69aaee15e6fed3011c403833a",
        "You impose the condition $\\psi(s) = \\psi(s + L)$, which implies that\n"
        "$$\\phi(L) - \\phi(0) = 2 \\pi n$$\n"
        "which $n$ is the energy level. Then you get\n"
        "$$n h = \\int_0^L \\sqrt{2 m (E_n - V(x))} \\, dx$$\n"
        "which is a typical WKB quantization integral, from which you compute the $E_n$ in the usual way.",
        r"$$n h = \int_0^L \sqrt{2 m (E_n - V(x))} \, dx$$",
        id="task15-named-wkb-condition",
    ),
    pytest.param(
        "statistics:206494:1",
        "raw_stackmathqa_1c4381e616889f6b78d899c8520c891cc6416683e764225410193b1962f9b0c8",
        "1c8d1511d6e8b04e3fbe0e9dcc043f6e1cafefb680e63073d7a4f2b2c00ccd19",
        "Integration by parts:\n"
        "$$2\\int_{-\\infty}^\\infty F(x)f(x) \\text{dx}=1$$\n"
        "Dividing by 2\n"
        "$$\\int_{-\\infty}^\\infty F(x)f(x) \\text{dx}=\\frac{1}{2}$$",
        r"$$\int_{-\infty}^\infty F(x)f(x) \text{dx}=\frac{1}{2}$$",
        id="task15-terminal-numeric-display-one-half",
    ),
)


@pytest.mark.parametrize(
    ("source_id", "record_id", "raw_sha256", "analysis", "expected"),
    _REAL_EXTRACTION_CASES,
)
def test_task15_real_gate3_false_rejects_extract_exact_source_evidence(
    source_id: str,
    record_id: str,
    raw_sha256: str,
    analysis: str,
    expected: str,
) -> None:
    frozen = next(case for case in _TASK15_GATE3_FALSE_REJECTS if case[0] == source_id)
    assert frozen[1] == record_id
    assert frozen[2] == raw_sha256
    assert expected in analysis

    extraction = extract_source_answer(_candidate(record_id, analysis))

    assert extraction.final_answer == expected
    assert extraction.source_field == "analysis"
    assert extraction.start_offset is not None
    assert extraction.end_offset is not None
    assert analysis[extraction.start_offset : extraction.end_offset] == expected
    assert extraction.source_span == f"analysis:{extraction.start_offset}:{extraction.end_offset}"


def test_task15_source_evidence_fingerprint_is_stable() -> None:
    digest = hashlib.sha256()
    for source_id, record_id, raw_sha256, category, excerpt in _TASK15_GATE3_FALSE_REJECTS:
        digest.update("\x1f".join((source_id, record_id, raw_sha256, category, excerpt)).encode())
        digest.update(b"\n")
    assert digest.hexdigest() == "4ea988ee8232b25d6f38e37907852ba2c5ff2c65d545271198a59c7a8bbff44e"
