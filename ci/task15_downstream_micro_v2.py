from __future__ import annotations

import task15_downstream_micro as micro
from college_builder.domain.question import AnalysisType, Discipline, ProblemType

micro.SELECTED = {
    "raw_stackmathqa_36383322c62efa98ac513621c69ff25f05bdb72a2c74dadc9870c33f8b9c030d": {
        "raw_sha256": "e32e233508cb23c63e5eee7024e7ea15eaa6a667cf636cd2826108205868c1ea",
        "answer": "$B$ is an open set",
        "start": 305,
        "end": 323,
        "discipline": Discipline.MATHEMATICS,
        "problem_type": ProblemType.PROOF,
        "analysis_type": AnalysisType.PROOF,
    },
    "raw_stackmathqa_ed6e1c277a74b99fce6262bf96f70461badc63ad18bb22a3371a27d30f99ecc6": {
        "raw_sha256": "fbddef7171e8654c2e5e1b5c5787057bfa2aa83ad725c807e5de4f9e987f1023",
        "answer": (
            "the total differential of $f$ at a point $a \\in \\Omega$ is a linear map "
            "$Df(a): \\Bbb R^n \\to \\Bbb R^m$"
        ),
        "start": 52,
        "end": 154,
        "discipline": Discipline.MATHEMATICS,
        "problem_type": ProblemType.CONCEPTUAL,
        "analysis_type": AnalysisType.DERIVATION,
    },
    "raw_stackmathqa_acad4b62acbb4a6e4284ac9777d02fe98b6180808a3e923e2ef17a9709c0f5e6": {
        "raw_sha256": "4deb3f4f77b724cc0da2e6722eba372272ba86e2ff6be089c89bf450375e9406",
        "answer": "logically equivalent",
        "start": 80,
        "end": 100,
        "discipline": Discipline.MATHEMATICS,
        "problem_type": ProblemType.CONCEPTUAL,
        "analysis_type": AnalysisType.PROOF,
    },
}

micro.main()
