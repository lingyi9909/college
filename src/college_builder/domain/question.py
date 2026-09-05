"""Typed university-question intermediate representation contracts."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from college_builder.domain.evidence import GateEvidence

NonEmptyStr = Annotated[str, Field(min_length=1)]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-fA-F]{64}$")]


class Discipline(StrEnum):
    MATHEMATICS = "MATHEMATICS"
    STATISTICS = "STATISTICS"
    PHYSICS = "PHYSICS"
    CHEMISTRY = "CHEMISTRY"
    COMPUTER_SCIENCE = "COMPUTER_SCIENCE"
    ELECTRONICS = "ELECTRONICS"
    ELECTRICAL_ENGINEERING = "ELECTRICAL_ENGINEERING"
    AUTOMATION = "AUTOMATION"
    MECHANICAL_ENGINEERING = "MECHANICAL_ENGINEERING"
    MATERIALS = "MATERIALS"
    CIVIL_ENGINEERING = "CIVIL_ENGINEERING"
    BIOLOGICAL_SCIENCE = "BIOLOGICAL_SCIENCE"
    OTHER_STEM = "OTHER_STEM"
    NON_STEM = "NON_STEM"


class UniversityLevel(StrEnum):
    UNDERGRADUATE = "UNDERGRADUATE"
    MASTERS = "MASTERS"
    DOCTORAL = "DOCTORAL"
    GRADUATE = "GRADUATE"
    UNIVERSITY_UNKNOWN = "UNIVERSITY_UNKNOWN"
    NON_UNIVERSITY = "NON_UNIVERSITY"


class ProblemType(StrEnum):
    CALCULATION = "CALCULATION"
    PROOF = "PROOF"
    DERIVATION = "DERIVATION"
    CONCEPTUAL = "CONCEPTUAL"
    MULTIPLE_CHOICE = "MULTIPLE_CHOICE"
    PROGRAMMING = "PROGRAMMING"
    ALGORITHM = "ALGORITHM"
    ENGINEERING = "ENGINEERING"
    SOFTWARE_USAGE = "SOFTWARE_USAGE"
    DEBUG_HELP = "DEBUG_HELP"
    CAREER_ADVICE = "CAREER_ADVICE"
    OPINION = "OPINION"
    RESOURCE_REQUEST = "RESOURCE_REQUEST"
    DISCUSSION = "DISCUSSION"
    NEWS = "NEWS"
    META = "META"


class AnalysisType(StrEnum):
    DERIVATION = "DERIVATION"
    STEP_BY_STEP = "STEP_BY_STEP"
    PROOF = "PROOF"
    CONCEPTUAL_REASONING = "CONCEPTUAL_REASONING"
    ALGORITHM_EXPLANATION = "ALGORITHM_EXPLANATION"
    CODE_EXPLANATION = "CODE_EXPLANATION"
    ENGINEERING_REASONING = "ENGINEERING_REASONING"


class DomainModel(BaseModel):
    """Base for strict internal contracts that reject ad-hoc mapping substitution."""

    model_config = ConfigDict(extra="forbid", strict=True)


class QuestionContent(DomainModel):
    raw: str
    normalized: str
    assets: tuple[str, ...] = ()


class AnswerContent(DomainModel):
    raw: str
    final_answer: str | None = None
    source_span: str | None = None


class AnalysisContent(DomainModel):
    raw: str
    type: AnalysisType | None = None


class Classification(DomainModel):
    discipline: Discipline
    course: str | None = None
    level: UniversityLevel
    problem_type: ProblemType


class QuestionMetadata(DomainModel):
    knowledge_points: tuple[str, ...] = ()
    exam_points: tuple[str, ...] = ()
    language: NonEmptyStr


class QualityState(DomainModel):
    gates: tuple[GateEvidence, ...] = ()


class QuestionProvenance(DomainModel):
    source_dataset: NonEmptyStr
    source_id: NonEmptyStr
    source_url: str | None = None
    raw_sha256: Sha256Hex


class DedupState(DomainModel):
    exact_hash: str | None = None
    duplicate_of: str | None = None
    candidate_ids: tuple[str, ...] = ()


class UniversityQuestionIR(DomainModel):
    candidate_id: NonEmptyStr
    source_record_id: NonEmptyStr
    question: QuestionContent
    answer: AnswerContent
    analysis: AnalysisContent
    classification: Classification
    metadata: QuestionMetadata
    quality: QualityState
    provenance: QuestionProvenance
    dedup: DedupState
