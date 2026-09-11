# Task 15 — Downstream Real-Model Micro Evidence

## Purpose

This note records the additional real-model downstream checks performed after the formal 100-record certification. It is evidence only; it does not change production code, frozen configuration, prompt versions, source data, or the formal 100-record result.

## Certified production identity

- Production code SHA exercised by the formal 100-record run: `d3b35964ac92d799405fd88101fa11fd05ff1460`
- Formal 100-record run: `34558426996`
- Formal 100-record artifact: `10183643305`
- 100-sample SHA256: `ae513933791c2ac27d3916a4149193c2477e00ce993c59b597b27ad02101ae8e`
- Verifier: `openai_compatible / deepseek-v4-pro`
- Frozen correctness prompt: `prompts/correctness_verify/v1.txt`
- Frozen alignment prompt: `prompts/qa_alignment/v1.txt`

The feature branch commits after the production code SHA contain certification evidence/docs only.

## Additional downstream runs

### Run `34589616128`

Three immutable records from the certified 100-child sample were exercised through the real `deepseek-v4-pro` Gate 5 alignment and independent correctness verifier.

Observed behavior:

- One record completed both real verifier gates successfully.
- Two records received model-level `PASS` decisions with score `1.0`, but were rejected by the production evidence validator because one or more character-offset source references were invalid.
- A representative record had independent correctness `PASS` while alignment was rejected only as `QA_ALIGNMENT_EVIDENCE_INVALID`.
- No source question, answer, or analysis was rewritten or repaired.

The CI harness intentionally stopped before final export when any selected record failed the production gate contract; therefore this failed run is not represented as a successful downstream certification artifact.

### Run `34590949726`

A previously successful immutable known-good record was isolated to remove sample-selection variance and exercise the same production verifier code.

Observed result:

- Alignment gate: `PASS`, score `1.0`, reason `QA_ALIGNMENT_CONFIRMED`.
- Independent correctness model decision: `PASS`, raw/calibrated score `1.0`, reason `valid_quantifier_shift`.
- Production correctness gate: `REJECT`, reason `CORRECTNESS_EVIDENCE_INVALID`.
- The verifier returned three evidence references, but all three were invalid under the strict production character-offset validator; the validated reference list was empty and `invalid_evidence_reference_count=3`.
- The run therefore correctly stopped before final dedup/export instead of bypassing the production gate.

## Root cause

Both frozen verifier prompts require references in the form:

`question:<start>-<end>`, `answer:<start>-<end>`, `analysis:<start>-<end>`

but do not define the coordinate convention (for example, zero-based indexing and an end-exclusive boundary). Production validation applies exact Python-style character slicing semantics and, for a `PASS`, requires all required source fields to be covered, the answer authority to be fully covered, and zero invalid references.

Real-model runs repeatedly showed semantically correct `PASS` decisions failing only because the model-generated exact character offsets were unreliable. This is a verifier-evidence protocol reliability limitation, not evidence of a wrong answer being accepted.

## Safety conclusion

This behavior is fail-closed and is therefore consistent with the project's Precision First policy:

- false accepts observed: **0**;
- invalid evidence never enters Accepted training output;
- source authority remains immutable;
- no generated repair/correction is used as training data;
- the validator was **not** weakened to manufacture an Accepted record;
- further repeated model calls were stopped rather than spending tokens until offsets happened to pass.

The limitation affects recall/yield and should not be interpreted as proof of production-final throughput. A future evidence-protocol redesign should prefer machine-derived or otherwise deterministic source-span binding rather than trusting an LLM to count exact character offsets. That redesign is outside this Task 15 certification evidence and is not implemented here.

## Task 15 interpretation

The formal 100-record run remains the authoritative E2E certification result: 100 records executed the production pipeline, 0 were accepted, and all 100 were manually audited. These micro runs add evidence that downstream real-model semantic judgments execute correctly while also exposing a strict, safe false-reject mode in the source-reference contract. They do not override the formal run report and do not support a >=99% precision claim or any 500K yield extrapolation.
