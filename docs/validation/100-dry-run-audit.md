# Task 15 — 100-Record Real-Model E2E Manual Audit

## Certification identity

- Production exact code SHA: `d3b35964ac92d799405fd88101fa11fd05ff1460`
- Real-model GitHub Actions run: `34558426996`
- Real-model artifact: `task15-real100-certification` (`10183643305`, `sha256:7fea5140b50ee96abc08872dd67312b77ab2b6b60811b5746269d824d0c7e8be`)
- Source: `math-ai/StackMathQA` @ `git:13239ec8c8078bc8dfb43e1383069eb9be000ff7`
- 5K sample SHA256: `e9d7b957a092114da35dbb839319a08ab555692a8a06863faf2b1980c2a681c5`
- 100-sample SHA256: `ae513933791c2ac27d3916a4149193c2477e00ce993c59b597b27ad02101ae8e`
- Strata: Math 40 / Physics 20 / Statistics 20 / MathOverflow 20

## Run result

- Total **100**; Accepted **0**; Rejected **100**.
- Funnel: `100 → 100 normalized → 69 STEM → 63 university → 27 problem → 1 answer_valid → 0 analysis_valid → 0 alignment → 0 final_accepted`.
- Provider errors: **0**.
- Tokens/cost: **unavailable** (`null`), not zero.
- This run is Engineering Certification only; it is not used to claim >=99% precision or forecast 500K yield.

## Reject reason counts

- `ANALYSIS_UNCERTAIN`: **1**
- `ANSWER_NOT_EXTRACTABLE`: **26**
- `MODEL_DECISION_CONFLICT`: **9**
- `NON_STEM`: **2**
- `NOT_PROBLEM`: **3**
- `PROBLEM_TYPE_UNCERTAIN`: **28**
- `UNIVERSITY_LEVEL_UNCERTAIN`: **31**

## 100/100 manual conclusion

- All 100 records were reviewed against source question/analysis, terminal rejection, and gate evidence.
- **57/100 REJECT_CONFIRMED**: current fail-closed rejection is reasonable.
- **43/100 FALSE_REJECT**:
  - Gate2 problem classification / confidence / evidence handling: **22**
  - Gate3 deterministic source-answer extraction: **16**
  - Gate2 conflict between two positive problem labels: **4**
  - Gate4 prompt/evidence-reference contract mismatch: **1**
- False accepts: **0** (there were no Accepted records).
- Parser bug: none identified.
- Normalization bug: none identified.
- Answer/analysis fabrication: none identified; source authority remained intact.
- The 0% acceptance rate therefore reflects both Precision-First fail-closed behavior **and** material over-rejection. The config must not be treated as production-final solely from this run.

### Confirmed systematic issues

1. **Gate3 recall** — 16 records contain an explicit source-grounded conclusion that the deterministic extractor failed to expose as `final_answer` (examples include `(1010!)^2/2021!`, `45π/4`, `r_eq=2l`, `1/2`, Cantor pairing, the WKB quantization condition).
2. **Gate2 positive-positive conflicts** — 4 records were rejected because classifier/verifier chose different positive labels such as `CALCULATION` vs `DERIVATION`; that disagreement should not by itself mean “not a problem.”
3. **Gate4 contract mismatch** — record #35 received `PROOF`, score `1.0`, but the prompt asks for `analysis:<span>` while the validator only accepts numeric `analysis:<start>-<end>` offsets, causing `ANALYSIS_UNCERTAIN`.
4. **Gate2 threshold/evidence sensitivity** — 22 concrete assessable STEM questions were manually judged false rejects at Gate2.

## Per-record audit

| # | Site | Source ID | Reject reason | Manual verdict | Issue |
|---:|---|---|---|---|---|
| 1 | math | `math:800029:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 2 | math | `math:3384278:0` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 3 | math | `math:4453773:0` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 4 | math | `math:4453111:4` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 5 | math | `math:2311173:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 6 | math | `math:4478385:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 7 | math | `math:2544901:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 8 | math | `math:3216659:0` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 9 | math | `math:2142021:0` | `MODEL_DECISION_CONFLICT` | **FALSE_REJECT** | Gate2 positive-label conflict |
| 10 | math | `math:3701703:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 11 | math | `math:969846:2` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 12 | math | `math:1484602:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 13 | math | `math:4338045:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 14 | math | `math:2521299:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 15 | math | `math:868000:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 16 | math | `math:3331406:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 17 | math | `math:1432059:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 18 | math | `math:3463165:0` | `MODEL_DECISION_CONFLICT` | **FALSE_REJECT** | Gate2 positive-label conflict |
| 19 | math | `math:2975169:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 20 | math | `math:1858833:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 21 | math | `math:4640072:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 22 | math | `math:4114624:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 23 | math | `math:182757:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 24 | math | `math:2176500:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 25 | math | `math:3108521:0` | `MODEL_DECISION_CONFLICT` | **REJECT_CONFIRMED** | none |
| 26 | math | `math:552229:2` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 27 | math | `math:1856187:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 28 | math | `math:3969220:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 29 | math | `math:277083:2` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 30 | math | `math:200617:6` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 31 | math | `math:193977:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 32 | math | `math:28975:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 33 | math | `math:575668:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 34 | math | `math:4094740:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 35 | math | `math:2621527:0` | `ANALYSIS_UNCERTAIN` | **FALSE_REJECT** | Gate4 prompt/evidence contract |
| 36 | math | `math:4127477:1` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 37 | math | `math:1470121:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 38 | math | `math:939151:1` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 39 | math | `math:2090296:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 40 | math | `math:3088692:0` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 41 | physics | `physics:160816:1` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 42 | physics | `physics:348145:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 43 | physics | `physics:222495:0` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 44 | physics | `physics:191712:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 45 | physics | `physics:372836:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 46 | physics | `physics:658710:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 47 | physics | `physics:390716:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 48 | physics | `physics:705459:3` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 49 | physics | `physics:8227:7` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 50 | physics | `physics:118697:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 51 | physics | `physics:557646:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 52 | physics | `physics:105433:3` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 53 | physics | `physics:215323:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 54 | physics | `physics:423861:6` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 55 | physics | `physics:564392:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 56 | physics | `physics:23682:1` | `MODEL_DECISION_CONFLICT` | **FALSE_REJECT** | Gate2 positive-label conflict |
| 57 | physics | `physics:668602:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 58 | physics | `physics:76308:3` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 59 | physics | `physics:531266:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 60 | physics | `physics:540721:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 61 | statistics | `statistics:255317:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 62 | statistics | `statistics:9334:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 63 | statistics | `statistics:206494:1` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 64 | statistics | `statistics:4980:3` | `NON_STEM` | **REJECT_CONFIRMED** | none |
| 65 | statistics | `statistics:596193:0` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 66 | statistics | `statistics:459354:0` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 67 | statistics | `statistics:129628:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 68 | statistics | `statistics:436878:0` | `MODEL_DECISION_CONFLICT` | **REJECT_CONFIRMED** | none |
| 69 | statistics | `statistics:153246:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 70 | statistics | `statistics:271780:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 71 | statistics | `statistics:494429:1` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 72 | statistics | `statistics:10066:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 73 | statistics | `statistics:541038:0` | `PROBLEM_TYPE_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 74 | statistics | `statistics:23445:1` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 75 | statistics | `statistics:230388:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 76 | statistics | `statistics:595516:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 77 | statistics | `statistics:598749:0` | `UNIVERSITY_LEVEL_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 78 | statistics | `statistics:336261:0` | `MODEL_DECISION_CONFLICT` | **REJECT_CONFIRMED** | none |
| 79 | statistics | `statistics:261479:0` | `PROBLEM_TYPE_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 80 | statistics | `statistics:346182:0` | `PROBLEM_TYPE_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 81 | mathoverflow | `mathoverflow:88899:0` | `MODEL_DECISION_CONFLICT` | **FALSE_REJECT** | Gate2 positive-label conflict |
| 82 | mathoverflow | `mathoverflow:415885:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 83 | mathoverflow | `mathoverflow:7247:2` | `NOT_PROBLEM` | **REJECT_CONFIRMED** | none |
| 84 | mathoverflow | `mathoverflow:423011:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 85 | mathoverflow | `mathoverflow:336928:0` | `PROBLEM_TYPE_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 86 | mathoverflow | `mathoverflow:359827:0` | `NON_STEM` | **REJECT_CONFIRMED** | none |
| 87 | mathoverflow | `mathoverflow:120975:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 88 | mathoverflow | `mathoverflow:354215:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 89 | mathoverflow | `mathoverflow:363720:16` | `NOT_PROBLEM` | **REJECT_CONFIRMED** | none |
| 90 | mathoverflow | `mathoverflow:422604:2` | `NOT_PROBLEM` | **REJECT_CONFIRMED** | none |
| 91 | mathoverflow | `mathoverflow:206332:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 92 | mathoverflow | `mathoverflow:434799:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 93 | mathoverflow | `mathoverflow:3398:0` | `PROBLEM_TYPE_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 94 | mathoverflow | `mathoverflow:10408:2` | `ANSWER_NOT_EXTRACTABLE` | **REJECT_CONFIRMED** | none |
| 95 | mathoverflow | `mathoverflow:179618:0` | `PROBLEM_TYPE_UNCERTAIN` | **FALSE_REJECT** | Gate2 problem classification/threshold |
| 96 | mathoverflow | `mathoverflow:48771:2` | `PROBLEM_TYPE_UNCERTAIN` | **REJECT_CONFIRMED** | none |
| 97 | mathoverflow | `mathoverflow:280760:2` | `MODEL_DECISION_CONFLICT` | **REJECT_CONFIRMED** | none |
| 98 | mathoverflow | `mathoverflow:325965:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 99 | mathoverflow | `mathoverflow:377204:0` | `ANSWER_NOT_EXTRACTABLE` | **FALSE_REJECT** | Gate3 answer extraction |
| 100 | mathoverflow | `mathoverflow:11084:19` | `MODEL_DECISION_CONFLICT` | **REJECT_CONFIRMED** | none |

## Required disposition

- Do not rerun another 100-record model job.
- Use a tiny **3–5 record real-model downstream certification** with manually confirmed source-grounded known-good records to exercise Gate5 Alignment → Independent Correctness → Final Dedup → exact 19-field export.
- Before a production-scale config freeze, repair the identified Gate2/Gate3/Gate4 false-reject mechanisms with regressions; preserve Precision-First and source-authority constraints.
- Task 15 remains **NOT YET ACCEPTED** until downstream real-model coverage is demonstrated and final fresh verification is complete.
