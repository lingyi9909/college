# Task 16B Small Real-Model Re-certification — 50/50 Manual Audit

## Certification identity

- Exact production/evidence feature SHA under certification: `456174b0ed0497239733ddd2909eea39eda83ecb`
- GitHub Actions Run: `34703681938`
- Artifact: `10301257383`
- Frozen 50-record payload SHA256: `50026b04d0d5a55a27c8a796f54f35eeea756410835a82c29bbd155013d08304`
- Pipeline run ID: `run-4595aa5faafe6dcfe431d0f52ef70245aea3d3eeb92b1f9fe1ec994034d5dfc7`
- Config: `task16-recertification-v2`
- Classifier: `openai_compatible / deepseek-flash`
- Verifier: `openai_compatible / deepseek-v4-pro`

The sample was frozen before Task16B model calls. Task15 source IDs and raw SHA256 identities were revalidated against the immutable Task15 real-100 parent artifact before the real-model run.

## Audit method

All 50 Task16B outcomes were reconciled against the immutable source identity and the Task15 manual ground-truth label for the same exact raw record. The 7 precision controls were rechecked as reject-eligible controls; the single Task16 Accepted record was re-read against source content and its final answer authority was verified against the accepted IR/source span. No raw copyrighted source text is copied into this audit.

Classification rule used for the new result:

- Task15 `FALSE_REJECT` + Task16 ACCEPT → `CORRECT_ACCEPT`
- Task15 `FALSE_REJECT` + Task16 REJECT → `FALSE_REJECT`
- Task15 `REJECT_CONFIRMED` + Task16 ACCEPT → `FALSE_ACCEPT`
- Task15 `REJECT_CONFIRMED` + Task16 REJECT → `CORRECT_REJECT`

## Final audit result

- Correct Accept: **1**
- False Reject: **42**
- Correct Reject: **7**
- False Accept: **0**
- **False Accept = 0: PASS**
- Known Task15 false rejects in this frozen set: **43**
- Task16B recovered known false rejects: **1**
- Remaining known false rejects: **42**
- Recall recovery is measurable but small: **43 → 42 false rejects**.

The single Correct Accept is `physics:540721:0`. It completed Gate5 → independent correctness → final dedup → exact 19-field export. Its exported final answer is an exact source-derived substring represented by the accepted IR authority span; no generated/repaired formal answer was introduced.

## Per-record audit

| # | Source ID | Coverage | Task15 ground truth | Task16 result | Task16 reason | Audit |
|---:|---|---|---|---|---|---|
| 1 | `math:800029:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 2 | `math:3384278:0` | control | REJECT_CONFIRMED | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **CORRECT_REJECT** |
| 4 | `math:4453111:4` | G3-answer | FALSE_REJECT | REJECT | `UNIVERSITY_LEVEL_UNCERTAIN` | **FALSE_REJECT** |
| 5 | `math:2311173:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 6 | `math:4478385:0` | G2-threshold | FALSE_REJECT | REJECT | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** |
| 7 | `math:2544901:0` | G3-answer | FALSE_REJECT | REJECT | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** |
| 9 | `math:2142021:0` | G2-positive-conflict | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 10 | `math:3701703:0` | G3-answer | FALSE_REJECT | REJECT | `QA_ALIGNMENT_EVIDENCE_INVALID` | **FALSE_REJECT** |
| 12 | `math:1484602:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 13 | `math:4338045:0` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 14 | `math:2521299:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 15 | `math:868000:1` | G5/correctness-span, control | REJECT_CONFIRMED | REJECT | `UNIVERSITY_LEVEL_UNCERTAIN` | **CORRECT_REJECT** |
| 16 | `math:3331406:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 17 | `math:1432059:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 18 | `math:3463165:0` | G2-positive-conflict | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 19 | `math:2975169:0` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 21 | `math:4640072:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 24 | `math:2176500:0` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 25 | `math:3108521:0` | control | REJECT_CONFIRMED | REJECT | `NOT_UNIVERSITY_LEVEL` | **CORRECT_REJECT** |
| 27 | `math:1856187:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 28 | `math:3969220:0` | G3-answer, G5/correctness-span | FALSE_REJECT | REJECT | `QA_ALIGNMENT_EVIDENCE_INVALID` | **FALSE_REJECT** |
| 29 | `math:277083:2` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 31 | `math:193977:0` | G3-answer | FALSE_REJECT | REJECT | `UNIVERSITY_LEVEL_UNCERTAIN` | **FALSE_REJECT** |
| 34 | `math:4094740:0` | G3-answer | FALSE_REJECT | REJECT | `ANALYSIS_UNCERTAIN` | **FALSE_REJECT** |
| 35 | `math:2621527:0` | G4-span, G5/correctness-span | FALSE_REJECT | REJECT | `CORRECTNESS_EVIDENCE_INVALID` | **FALSE_REJECT** |
| 38 | `math:939151:1` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 41 | `physics:160816:1` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 42 | `physics:348145:0` | control | REJECT_CONFIRMED | REJECT | `UNIVERSITY_LEVEL_UNCERTAIN` | **CORRECT_REJECT** |
| 44 | `physics:191712:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 46 | `physics:658710:0` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 47 | `physics:390716:0` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 56 | `physics:23682:1` | G2-positive-conflict | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 58 | `physics:76308:3` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 60 | `physics:540721:0` | G3-answer | FALSE_REJECT | ACCEPT | `-` | **CORRECT_ACCEPT** |
| 63 | `statistics:206494:1` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 64 | `statistics:4980:3` | control | REJECT_CONFIRMED | REJECT | `NON_STEM` | **CORRECT_REJECT** |
| 70 | `statistics:271780:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 71 | `statistics:494429:1` | G2-threshold | FALSE_REJECT | REJECT | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** |
| 73 | `statistics:541038:0` | control | REJECT_CONFIRMED | REJECT | `ANSWER_NOT_EXTRACTABLE` | **CORRECT_REJECT** |
| 81 | `mathoverflow:88899:0` | G2-positive-conflict | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 82 | `mathoverflow:415885:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 83 | `mathoverflow:7247:2` | control | REJECT_CONFIRMED | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **CORRECT_REJECT** |
| 84 | `mathoverflow:423011:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 87 | `mathoverflow:120975:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 88 | `mathoverflow:354215:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 91 | `mathoverflow:206332:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 92 | `mathoverflow:434799:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 95 | `mathoverflow:179618:0` | G2-threshold | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 98 | `mathoverflow:325965:0` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |
| 99 | `mathoverflow:377204:0` | G3-answer | FALSE_REJECT | REJECT | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** |

## Limitation-class findings

- **Gate2 compatible-positive conflict:** all 4 former `MODEL_DECISION_CONFLICT` false rejects no longer fail for that reason, but all 4 still reject as `PROBLEM_TYPE_UNCERTAIN`. The semantic-conflict fix is active; Gate2 confidence/evidence remains the dominant end-to-end blocker.
- **Gate3 source-grounded answer extraction:** 16 covered records produced 1 final Correct Accept. Several records now pass Gate3 but fail later at Gate4/Gate5/correctness; many are still blocked upstream by Gate2.
- **Gate4 span protocol:** the known Gate4 record `math:2621527:0` passes analysis and alignment under the new strict span contract, then rejects at independent correctness evidence validation. The original Gate4 prompt/validator mismatch is no longer its rejection point.
- **Gate5/correctness span protocol:** real-model execution reached these gates. Strict invalid-evidence rejection remains fail-closed; no malformed reference was silently repaired.

## Remaining false-reject blockers

- `PROBLEM_TYPE_UNCERTAIN`: **33**
- `ANSWER_NOT_EXTRACTABLE`: **3**
- `UNIVERSITY_LEVEL_UNCERTAIN`: **2**
- `QA_ALIGNMENT_EVIDENCE_INVALID`: **2**
- `ANALYSIS_UNCERTAIN`: **1**
- `CORRECTNESS_EVIDENCE_INVALID`: **1**

The dominant remaining blocker is Gate2: `PROBLEM_TYPE_UNCERTAIN` accounts for 33 of 42 false rejects. Task16B therefore certifies safety and a measurable recall recovery, but it does **not** demonstrate healthy production yield.

## Verification evidence

- Task16 focused regressions: **59 passed**
- Full pytest: **547 passed**
- Ruff: **PASS**
- MyPy: **PASS**, 33 source files
- Provider errors: **0**
- Exact 19-field export validation: **PASS**
- Artifact SHA256: `353928556b46bd3cd755e70c3d6f89f9e845d1c5595a33ca10be521afbc24218`

## Task16B acceptance-gate assessment

- Frozen 20–50 real-model sample: PASS (50)
- All four limitation classes covered: PASS
- Full 50/50 audit: PASS
- False Accept = 0: PASS
- Measurable known-false-reject reduction: PASS (43 → 42)
- At least one genuine source-grounded Gate5 → Correctness → Final Dedup → 19-field export: PASS
- No generated/repaired formal content observed: PASS

**Task16B status: CANDIDATE READY FOR EXPLICIT REVIEW. Do not start Task16C until explicit acceptance.**
