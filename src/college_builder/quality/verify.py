"""Gate 5 source alignment and independent correctness verification."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import ValidationError
from sympy import simplify, sympify  # type: ignore[import-untyped]

from college_builder.domain.evidence import GateVerdict
from college_builder.domain.source import JsonValue, NormalizedQA
from college_builder.providers.base import (
    ModelClassificationRequest,
    ModelDecision,
    StructuredModelProvider,
)
from college_builder.quality.answer import AnswerExtraction, extract_source_answer
from college_builder.quality.engine import GateContext, GateResultEvidence

VERIFICATION_LABELS = ("PASS", "FAIL", "UNCERTAIN")
_ANSWER_AUTHORITY_THRESHOLD = 0.995
_SOURCE_SPAN_RE = re.compile(
    r"^(?P<field>question|answer|analysis):(?P<start>\d+)-(?P<end>\d+)$",
    re.IGNORECASE,
)
_ARITHMETIC_PREFIX_RE = re.compile(
    r"^(?:compute|evaluate|calculate|find\s+the\s+value\s+of)\s+(?P<expr>.+)$",
    re.IGNORECASE,
)
_SOLVE_PREFIX_RE = re.compile(r"^(?:solve|find\s+\w+\s+in)\s+(?P<equation>.+)$", re.IGNORECASE)
_SAFE_ARITHMETIC_RE = re.compile(r"^[0-9eE+\-*/^().\s]+$")
_SAFE_ALGEBRA_RE = re.compile(r"^[A-Za-z0-9_+\-*/^().\s]+$")
_SIMPLE_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

ScoreCalibrator = Callable[[float], float]


class DeterministicVerificationStatus(StrEnum):
    """Conservative deterministic verification outcome."""

    CONFIRMED = "CONFIRMED"
    CONTRADICTED = "CONTRADICTED"
    NOT_VERIFIED = "NOT_VERIFIED"


@dataclass(frozen=True, slots=True)
class DeterministicVerificationResult:
    """Result of a deterministic check that never rewrites source content."""

    status: DeterministicVerificationStatus
    reason_code: str


class DeterministicMathVerifier:
    """Conservatively verify safely parseable arithmetic and one-variable equations."""

    def verify(self, candidate: NormalizedQA) -> DeterministicVerificationResult:
        extraction = extract_source_answer(candidate)
        source_answer = _formal_final_answer(extraction)
        if source_answer is None:
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_ANSWER_UNAVAILABLE",
            )

        question = _strip_terminal_punctuation(candidate.question)
        arithmetic_match = _ARITHMETIC_PREFIX_RE.fullmatch(question)
        if arithmetic_match is not None:
            return self._verify_arithmetic(arithmetic_match.group("expr"), source_answer)

        solve_match = _SOLVE_PREFIX_RE.fullmatch(question)
        if solve_match is not None:
            return self._verify_equation(solve_match.group("equation"), source_answer)

        return DeterministicVerificationResult(
            DeterministicVerificationStatus.NOT_VERIFIED,
            "DETERMINISTIC_NOT_APPLICABLE",
        )

    def _verify_arithmetic(
        self,
        expression: str,
        source_answer: str,
    ) -> DeterministicVerificationResult:
        expected = _safe_sympify(expression, allow_symbols=False)
        answer = _safe_answer_expression(source_answer, allow_symbols=False)
        if expected is None or answer is None:
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_PARSE_UNAVAILABLE",
            )
        try:
            matches = simplify(expected - answer) == 0  # type: ignore[operator]
        except Exception:  # noqa: BLE001 - unsupported SymPy object is not a contradiction.
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_PARSE_UNAVAILABLE",
            )
        return DeterministicVerificationResult(
            DeterministicVerificationStatus.CONFIRMED
            if matches
            else DeterministicVerificationStatus.CONTRADICTED,
            "DETERMINISTIC_CONFIRMED" if matches else "DETERMINISTIC_MISMATCH",
        )

    def _verify_equation(
        self,
        equation: str,
        source_answer: str,
    ) -> DeterministicVerificationResult:
        if equation.count("=") != 1:
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_PARSE_UNAVAILABLE",
            )
        left_text, right_text = (part.strip() for part in equation.split("=", 1))
        left = _safe_sympify(left_text, allow_symbols=True)
        right = _safe_sympify(right_text, allow_symbols=True)
        if left is None or right is None:
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_PARSE_UNAVAILABLE",
            )
        try:
            expression = left - right  # type: ignore[operator]
            symbols = tuple(expression.free_symbols)
        except Exception:  # noqa: BLE001 - fail open to independent verifier, never correct.
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_PARSE_UNAVAILABLE",
            )
        if len(symbols) != 1:
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_PARSE_UNAVAILABLE",
            )
        symbol = symbols[0]
        answer = _safe_equation_answer(source_answer, str(symbol))
        if answer is None:
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_PARSE_UNAVAILABLE",
            )
        try:
            matches = simplify(expression.subs(symbol, answer)) == 0
        except Exception:  # noqa: BLE001 - unsupported substitution is not a contradiction.
            return DeterministicVerificationResult(
                DeterministicVerificationStatus.NOT_VERIFIED,
                "DETERMINISTIC_PARSE_UNAVAILABLE",
            )
        return DeterministicVerificationResult(
            DeterministicVerificationStatus.CONFIRMED
            if matches
            else DeterministicVerificationStatus.CONTRADICTED,
            "DETERMINISTIC_CONFIRMED" if matches else "DETERMINISTIC_MISMATCH",
        )


class AlignmentGate:
    """Gate 5 alignment: source analysis must answer this question and support its answer."""

    name = "gate_5_qa_alignment"

    def __init__(
        self,
        *,
        provider: StructuredModelProvider,
        prompt: str,
        prompt_version: str,
        pass_threshold: float = 0.995,
    ) -> None:
        self.provider_impl = provider
        self.provider, self.model = _provider_identity(provider)
        self.prompt = _nonempty(prompt, "alignment prompt")
        self.prompt_version = _nonempty(prompt_version, "prompt_version")
        if not 0.0 <= pass_threshold <= 1.0:
            raise ValueError("alignment pass_threshold must be within [0,1]")
        self.pass_threshold = pass_threshold

    def evaluate(self, candidate: NormalizedQA, context: GateContext) -> GateResultEvidence:
        extraction = extract_source_answer(candidate)
        authority_payload = _answer_authority_payload(extraction)
        final_answer = _formal_final_answer(extraction)
        if final_answer is None:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=extraction.answer_extract_score,
                reason_code=_answer_authority_reject_reason(extraction),
                evidence_payload={"answer_authority": authority_payload},
            )

        request = _verification_request(self.name, self.prompt, candidate, final_answer)
        decision, error = _call_and_validate(self.provider_impl, request)
        if error is not None:
            error_payload = dict(error.payload)
            error_payload["answer_authority"] = authority_payload
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code=error.reason_code,
                evidence_payload=error_payload,
            )
        assert decision is not None
        if decision.label not in VERIFICATION_LABELS:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="MALFORMED_MODEL_DECISION",
                evidence_payload={
                    "provider": self.provider,
                    "model": self.model,
                    "answer_authority": authority_payload,
                },
            )

        validated_refs, evidence_complete, invalid_ref_count = _validated_source_references(
            decision,
            candidate,
            extraction,
        )
        alignment_payload: dict[str, JsonValue] = {
            "provider": self.provider,
            "model": self.model,
            "verdict": decision.label,
            "alignment_score": decision.score,
            "reason_code": decision.reason_code,
            "evidence_references": list(validated_refs),
        }
        if invalid_ref_count:
            alignment_payload["invalid_evidence_reference_count"] = invalid_ref_count
        payload: dict[str, JsonValue] = {
            "answer_authority": authority_payload,
            "alignment": alignment_payload,
        }
        if not evidence_complete:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=decision.score,
                reason_code="QA_ALIGNMENT_EVIDENCE_INVALID",
                evidence_payload=payload,
            )
        if decision.label == "FAIL":
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=decision.score,
                reason_code="QA_ALIGNMENT_MISMATCH",
                evidence_payload=payload,
            )
        if decision.label == "UNCERTAIN" or decision.score < self.pass_threshold:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=decision.score,
                reason_code="QA_ALIGNMENT_UNCERTAIN",
                evidence_payload=payload,
            )
        return self._result(
            context,
            verdict=GateVerdict.PASS,
            score=decision.score,
            reason_code="QA_ALIGNMENT_CONFIRMED",
            evidence_payload=payload,
        )

    def _result(
        self,
        context: GateContext,
        *,
        verdict: GateVerdict,
        score: float,
        reason_code: str,
        evidence_payload: dict[str, JsonValue],
    ) -> GateResultEvidence:
        return GateResultEvidence(
            gate_name=self.name,
            verdict=verdict,
            score=score,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code=reason_code,
            evidence_payload=evidence_payload,
            timestamp=datetime.now(UTC),
        )


class CorrectnessVerifier:
    """Independent correctness gate; verifier reasoning never becomes formal source truth."""

    name = "independent_correctness_verification"

    def __init__(
        self,
        *,
        provider: StructuredModelProvider,
        prompt: str,
        prompt_version: str,
        calibration_id: str = "identity-v1",
        calibrator: ScoreCalibrator | None = None,
        deterministic_verifier: DeterministicMathVerifier | None = None,
    ) -> None:
        self.provider_impl = provider
        self.provider, self.model = _provider_identity(provider)
        self.prompt = _nonempty(prompt, "correctness prompt")
        self.prompt_version = _nonempty(prompt_version, "prompt_version")
        self.calibration_id = _nonempty(calibration_id, "calibration_id")
        self.calibrator = calibrator or _identity_calibrator
        self.deterministic_verifier = deterministic_verifier or DeterministicMathVerifier()

    def evaluate(self, candidate: NormalizedQA, context: GateContext) -> GateResultEvidence:
        extraction = extract_source_answer(candidate)
        authority_payload = _answer_authority_payload(extraction)
        final_answer = _formal_final_answer(extraction)
        if final_answer is None:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=extraction.answer_extract_score,
                reason_code=_answer_authority_reject_reason(extraction),
                evidence_payload={"answer_authority": authority_payload},
            )

        deterministic = self.deterministic_verifier.verify(candidate)
        deterministic_payload: dict[str, JsonValue] = {
            "status": deterministic.status.value,
            "reason_code": deterministic.reason_code,
        }
        if deterministic.status is DeterministicVerificationStatus.CONTRADICTED:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="DETERMINISTIC_CORRECTNESS_MISMATCH",
                evidence_payload={
                    "answer_authority": authority_payload,
                    "deterministic": deterministic_payload,
                },
            )

        request = _verification_request(self.name, self.prompt, candidate, final_answer)
        decision, error = _call_and_validate(self.provider_impl, request)
        if error is not None:
            error_payload = dict(error.payload)
            error_payload["answer_authority"] = authority_payload
            error_payload["deterministic"] = deterministic_payload
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code=error.reason_code,
                evidence_payload=error_payload,
            )
        assert decision is not None
        if decision.label not in VERIFICATION_LABELS:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="MALFORMED_MODEL_DECISION",
                evidence_payload={
                    "answer_authority": authority_payload,
                    "deterministic": deterministic_payload,
                    "provider": self.provider,
                    "model": self.model,
                },
            )

        calibrated_score = self._calibrate(decision.score)
        if calibrated_score is None:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=0.0,
                reason_code="CALIBRATION_ERROR",
                evidence_payload={
                    "answer_authority": authority_payload,
                    "deterministic": deterministic_payload,
                    "provider": self.provider,
                    "model": self.model,
                    "raw_score": decision.score,
                    "calibration_id": self.calibration_id,
                },
            )

        validated_refs, evidence_complete, invalid_ref_count = _validated_source_references(
            decision,
            candidate,
            extraction,
        )
        verifier_payload: dict[str, JsonValue] = {
            "provider": self.provider,
            "model": self.model,
            "verdict": decision.label,
            "raw_score": decision.score,
            "calibrated_score": calibrated_score,
            "calibration_id": self.calibration_id,
            "reason_code": decision.reason_code,
            "evidence_references": list(validated_refs),
        }
        if invalid_ref_count:
            verifier_payload["invalid_evidence_reference_count"] = invalid_ref_count
        payload: dict[str, JsonValue] = {
            "answer_authority": authority_payload,
            "deterministic": deterministic_payload,
            "verifier": verifier_payload,
        }
        if not evidence_complete:
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=calibrated_score,
                reason_code="CORRECTNESS_EVIDENCE_INVALID",
                evidence_payload=payload,
            )
        if decision.label == "FAIL":
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=calibrated_score,
                reason_code="CORRECTNESS_MISMATCH",
                evidence_payload=payload,
            )
        if decision.label == "UNCERTAIN":
            return self._result(
                context,
                verdict=GateVerdict.REJECT,
                score=calibrated_score,
                reason_code="CORRECTNESS_UNCERTAIN",
                evidence_payload=payload,
            )
        return self._result(
            context,
            verdict=GateVerdict.PASS,
            score=calibrated_score,
            reason_code="CORRECTNESS_CONFIRMED",
            evidence_payload=payload,
        )

    def _calibrate(self, raw_score: float) -> float | None:
        try:
            calibrated = self.calibrator(raw_score)
        except Exception:  # noqa: BLE001 - calibration boundary fails closed.
            return None
        if isinstance(calibrated, bool) or not isinstance(calibrated, (int, float)):
            return None
        value = float(calibrated)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            return None
        return value

    def _result(
        self,
        context: GateContext,
        *,
        verdict: GateVerdict,
        score: float,
        reason_code: str,
        evidence_payload: dict[str, JsonValue],
    ) -> GateResultEvidence:
        return GateResultEvidence(
            gate_name=self.name,
            verdict=verdict,
            score=score,
            provider=self.provider,
            model=self.model,
            prompt_version=self.prompt_version,
            config_version=context.config_version,
            reason_code=reason_code,
            evidence_payload=evidence_payload,
            timestamp=datetime.now(UTC),
        )


@dataclass(frozen=True, slots=True)
class _ProviderError:
    reason_code: str
    payload: dict[str, JsonValue]


def _call_and_validate(
    provider: StructuredModelProvider,
    request: ModelClassificationRequest,
) -> tuple[ModelDecision | None, _ProviderError | None]:
    try:
        output = provider.classify(request)
    except Exception as exc:  # noqa: BLE001 - provider boundary fails closed.
        return None, _ProviderError(
            "MODEL_PROVIDER_ERROR",
            {
                "provider": getattr(provider, "provider", "unknown"),
                "model": getattr(provider, "model", "unknown"),
                "exception_type": type(exc).__name__,
            },
        )
    if not isinstance(output, ModelDecision):
        return None, _ProviderError(
            "MALFORMED_MODEL_DECISION",
            {
                "provider": getattr(provider, "provider", "unknown"),
                "model": getattr(provider, "model", "unknown"),
            },
        )
    try:
        validation_input = output.model_dump(mode="python", warnings=False)
        return ModelDecision.model_validate(validation_input), None
    except (ValidationError, TypeError, ValueError):
        return None, _ProviderError(
            "MALFORMED_MODEL_DECISION",
            {
                "provider": getattr(provider, "provider", "unknown"),
                "model": getattr(provider, "model", "unknown"),
            },
        )


def _verification_request(
    task: str,
    prompt: str,
    candidate: NormalizedQA,
    final_answer: str,
) -> ModelClassificationRequest:
    return ModelClassificationRequest(
        task=task,
        prompt=prompt,
        inputs={
            "question": candidate.question,
            "answer": final_answer,
            "analysis": candidate.analysis,
        },
        allowed_labels=VERIFICATION_LABELS,
    )


def _provider_identity(provider: StructuredModelProvider) -> tuple[str, str]:
    provider_name = getattr(provider, "provider", None)
    model = getattr(provider, "model", None)
    if not isinstance(provider_name, str) or not provider_name.strip():
        raise ValueError("provider identity must be non-empty")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model identity must be non-empty")
    return provider_name, model


def _nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _identity_calibrator(score: float) -> float:
    return score


def _formal_final_answer(extraction: AnswerExtraction) -> str | None:
    if extraction.answer_extract_score < _ANSWER_AUTHORITY_THRESHOLD:
        return None
    if extraction.final_answer is None:
        return None
    if (
        extraction.source_field is None
        or extraction.start_offset is None
        or extraction.end_offset is None
        or extraction.source_span is None
    ):
        return None
    return extraction.final_answer


def _answer_authority_reject_reason(extraction: AnswerExtraction) -> str:
    if not extraction.answer_source_exists:
        return "ANSWER_MISSING"
    return "ANSWER_NOT_EXTRACTABLE"


def _answer_authority_payload(extraction: AnswerExtraction) -> dict[str, JsonValue]:
    return {
        "final_answer": extraction.final_answer,
        "source_field": extraction.source_field,
        "start_offset": extraction.start_offset,
        "end_offset": extraction.end_offset,
        "source_span": extraction.source_span,
        "answer_source_exists": extraction.answer_source_exists,
        "answer_extract_score": extraction.answer_extract_score,
    }


def _validated_source_references(
    decision: ModelDecision,
    candidate: NormalizedQA,
    extraction: AnswerExtraction,
) -> tuple[tuple[str, ...], bool, int]:
    source = {
        "question": candidate.question,
        "answer": candidate.answer,
        "analysis": candidate.analysis,
    }
    source_field = extraction.source_field
    authority_start = extraction.start_offset
    authority_end = extraction.end_offset
    if source_field is None or authority_start is None or authority_end is None:
        return (), False, len(decision.evidence_references)

    required_fields = {"question", "analysis", source_field}
    covered: set[str] = set()
    authority_covered = False
    validated: list[str] = []
    invalid_count = 0
    for reference in decision.evidence_references:
        match = _SOURCE_SPAN_RE.fullmatch(reference.strip())
        if match is None:
            invalid_count += 1
            continue
        field = match.group("field").lower()
        text = source[field]
        start = int(match.group("start"))
        end = int(match.group("end"))
        if not (0 <= start < end <= len(text) and text[start:end].strip()):
            invalid_count += 1
            continue
        validated.append(reference)
        covered.add(field)
        if field == source_field and start <= authority_start and authority_end <= end:
            authority_covered = True
    complete = (
        required_fields.issubset(covered)
        and authority_covered
        and invalid_count == 0
    )
    return tuple(validated), complete, invalid_count


def _strip_terminal_punctuation(text: str) -> str:
    stripped = text.strip()
    while stripped.endswith((".", "?", "!")):
        stripped = stripped[:-1].rstrip()
    return stripped


def _is_safe_expression_syntax(text: str, *, allow_symbols: bool) -> bool:
    normalized = text.replace("^", "**")
    try:
        tree = ast.parse(normalized, mode="eval")
    except (SyntaxError, ValueError):
        return False

    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.UAdd,
        ast.USub,
        ast.Constant,
        ast.Name,
        ast.Load,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            return False
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                return False
        if isinstance(node, ast.Name):
            if not allow_symbols or _SIMPLE_SYMBOL_RE.fullmatch(node.id) is None:
                return False
    return True


def _safe_sympify(text: str, *, allow_symbols: bool) -> object | None:
    stripped = text.strip()
    allowed = _SAFE_ALGEBRA_RE if allow_symbols else _SAFE_ARITHMETIC_RE
    if not stripped or allowed.fullmatch(stripped) is None:
        return None
    if not _is_safe_expression_syntax(stripped, allow_symbols=allow_symbols):
        return None
    try:
        parsed: object = sympify(stripped.replace("^", "**"), evaluate=True)
        return parsed
    except Exception:  # noqa: BLE001 - unsupported syntax is simply not verified.
        return None


def _safe_answer_expression(source_answer: str, *, allow_symbols: bool) -> object | None:
    text = _strip_terminal_punctuation(source_answer)
    if "=" in text:
        if text.count("=") != 1:
            return None
        _, text = text.split("=", 1)
    return _safe_sympify(text.strip(), allow_symbols=allow_symbols)


def _safe_equation_answer(source_answer: str, symbol_name: str) -> object | None:
    text = _strip_terminal_punctuation(source_answer)
    if "=" in text:
        if text.count("=") != 1:
            return None
        left, right = (part.strip() for part in text.split("=", 1))
        if not _SIMPLE_SYMBOL_RE.fullmatch(left) or left != symbol_name:
            return None
        text = right
    return _safe_sympify(text, allow_symbols=False)
