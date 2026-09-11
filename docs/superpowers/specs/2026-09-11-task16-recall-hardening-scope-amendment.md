# Task 16 Recall Hardening Scope Amendment

## Authority

This amendment changes Task 16 of `docs/superpowers/plans/2026-09-05-university-stem-dataset-builder-pilot-implementation-plan.md`.

Task 15 is formally ACCEPTED at exact HEAD `5572fe691a866d048309806a8db051244a2d46d3` for Engineering Certification only. Production-scale quality/yield certification has not passed.

Task 15 measured 43/100 false rejects and identified four systemic recall/evidence limitations:

1. Gate2 classification/conflict handling.
2. Gate3 source-grounded answer extraction recall.
3. Gate4 prompt/validator evidence contract mismatch.
4. Gate5/Independent Correctness character-offset evidence protocol ambiguity.

Where this amendment conflicts with the original Task 16 plan, this amendment is authoritative.

## Goal

Recover recall without weakening Precision First, source authority, provenance, correctness, or fail-closed behavior; prove the fixes on a small real-model certification set before spending model budget on the 50K validation pilot.

## Non-negotiable constraints

- Do not start the 50K pilot until Task 16A and Task 16B are independently accepted.
- False Accept must remain zero in the Task 16B audited certification set.
- LLMs remain forbidden from generating, completing, rewriting, repairing, or correcting formal question, answer, or analysis content.
- Gate3 answers must remain exact source-grounded content or deterministic extraction from source content.
- Unknown or ambiguous critical evidence remains fail-closed.
- Do not solve recall by lowering thresholds without evidence, accepting malformed references, or treating conflicting negative labels as positive.
- Every behavior change requires RED evidence before production implementation and permanent regression coverage.
- Provider/model/config/prompt identities must be versioned in certification evidence.

## Task 16A — False-Reject and Evidence Protocol Hardening

### 16A.1 Gate2 classification/conflict

Distinguish semantic conflicts from multiple compatible positive labels. A positive classification must not be rejected merely because two compatible positive labels are returned. Evidence validation remains strict. Negative/positive semantic disagreement remains fail-closed.

Acceptance evidence must include Task 15 false-reject examples and synthetic boundary cases proving:
- compatible positives do not become `MODEL_DECISION_CONFLICT`;
- positive vs negative disagreement still rejects;
- malformed/unsupported evidence still fails closed.

### 16A.2 Gate3 answer extraction recall

Improve deterministic extraction of explicit final answers already present in source analysis. The extractor may recognize additional conclusion forms and mathematical terminal expressions, but must return a source substring/span and must never synthesize an answer.

Acceptance evidence must include Task 15 examples where explicit answers such as terminal formulas/numeric conclusions were previously `ANSWER_NOT_EXTRACTABLE`, plus anti-generation tests.

### 16A.3 Gate4 evidence contract

Make the analysis prompt and validator share one explicit reference grammar. The protocol must define field names, 0-based indexing, and end-exclusive half-open spans `[start,end)`. Gate4 must reject malformed, out-of-range, or semantically irrelevant references.

### 16A.4 Gate5 / Independent Correctness evidence protocol

Version the alignment and correctness prompts so evidence references explicitly use 0-based, end-exclusive `[start,end)` spans. Validators remain strict and do not silently reinterpret inclusive-end or malformed offsets. PASS requires the same source-authority coverage rules as before.

The change is a protocol clarification, not a relaxation of evidence requirements.

## Task 16B — Small Real-Model Re-certification Gate

Use 20–50 records, prioritizing the Task 15 manually identified false rejects/known-good records and fixed Gold cases. Do not use a large random run merely to increase sample size.

The certification must exercise the real configured model providers and the complete applicable path:

`Gate2 → Gate3 → Gate4 → Gate5 Alignment → Independent Correctness → Final Dedup → exact 19-field export`

The report must bind:
- exact code SHA;
- source record IDs and raw SHA256 values;
- parent Task 15 evidence identity where applicable;
- config and prompt versions/SHA256 values;
- provider/model identities;
- per-gate funnel and reject reasons;
- cache/call/error/latency/token/cost evidence, using unavailable rather than fake zero where unavailable;
- every Accepted export to its immutable raw source and exact source-grounded answer authority.

### Task 16B acceptance gate

All of the following are required before 50K may start:

1. The four Task 15 known limitation classes have permanent regression coverage.
2. The small real-model set demonstrates materially improved recall on the known false-reject cases; the report must show before/after counts rather than claiming improvement qualitatively.
3. False Accept = 0 under manual audit of the entire small certification set.
4. At least one genuine source-grounded record traverses Gate5, Independent Correctness, Final Dedup and exact 19-field export successfully.
5. No formal question/answer/analysis content is model-generated or repaired.
6. Fresh focused tests, full pytest, Ruff and MyPy pass on the exact certification HEAD.

No production-scale precision/yield claim may be made from this small set.

## Task 16C — 50K Validation Pilot and Scale Decision

Only after explicit acceptance of 16A and 16B may the original Task 16 50K pilot begin.

The 50K run must use the newly certified versioned config/prompt identities rather than treating `pilot-5k-frozen-v1` as production-scale certified. The original Task 16 requirements for stratification, reporting, manual audit, acceptance-rate analysis, cost/runtime measurement, duplicate measurement, source mix, and scale decision remain in force.

## Stop rule

If 16B shows that recall has not materially recovered, False Accept is non-zero, source authority is weakened, or downstream real-model export cannot be demonstrated, stop Task 16 before 50K and return to 16A. Do not compensate by increasing sample size or lowering quality gates.