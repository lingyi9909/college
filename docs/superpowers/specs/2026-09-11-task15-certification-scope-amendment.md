# Task 15 Certification Scope Amendment

**Date:** 2026-09-11

**Authority:** This amendment supersedes the Task 15 requirement for a 5K real-model dry run in `docs/superpowers/plans/2026-09-05-university-stem-dataset-builder-pilot-implementation-plan.md`. All other approved precision-first, provenance, no-rewrite, exact-19-field, provider-independence, and Task-boundary requirements remain in force.

## Revised Task 15 Goal

Task 15 certification is now:

**5K Deterministic Sampling Certification + 100-record Real Model E2E + Gold Regression + Manual Audit**

Task 16 MUST NOT start until the reviewer explicitly accepts Task 15 under this amended gate.

## 1. 5K Deterministic Sampling Certification

The approved 5K StackMathQA sample remains:

- Math: 2000
- Physics: 1000
- Statistics: 1000
- MathOverflow: 1000
- Seed: `20260910`

The 5K stage MUST NOT invoke any model provider. It certifies only deterministic sampling, exact strata, immutable source revision, sampled `record_id`, `raw_sha256`, source-file SHA256, sample SHA256, manifest reproducibility, and exact identity reproduction from the same source + revision + seed.

Existing bounded-memory sampling, immutable revision enforcement, manifest identity, and raw-hash binding are approved and MUST NOT be rewritten without new blocker evidence.

## 2. 100-record Real Model Certification Sample

Select exactly 100 records deterministically from the certified 5K sample:

- Math: 40
- Physics: 20
- Statistics: 20
- MathOverflow: 20

The 100-record sample requires its own manifest and MUST bind to the parent 5K sample SHA256. It is an engineering-certification sample, not a statistical precision estimate.

The 100 records MUST execute the complete production path:

`Raw -> Normalize -> Gate 0 Integrity -> Gate 1 University/STEM -> Gate 2 Problem -> Gate 3 Original Answer -> Gate 4 Original Analysis -> Early Dedup -> Gate 5 Alignment -> Independent Correctness -> Final Dedup -> Final IR -> exact 19-field export`

Provider identities are frozen candidates:

- classifier: `openai_compatible / deepseek-flash`
- verifier: `openai_compatible / deepseek-v4-pro`

Fake providers are forbidden for the real-model certification run.

## 3. Run Evidence

The formal committed evidence is `artifacts/100/run_report.json` and MUST include:

- exact code SHA
- pipeline version
- source dataset and immutable revision
- parent 5K sample SHA256
- 100-sample SHA256
- seed and exact strata
- config SHA256 and config version
- all prompt versions
- classifier/verifier provider + model
- raw/accepted/rejected counts
- complete funnel and reject-reason counts
- acceptance rate
- provider call counts
- cache hits and misses
- provider errors and latency
- wall time
- token/cost evidence when the provider exposes it; otherwise an explicit unavailable marker, never fabricated zero usage

The historical `artifacts/5k/run_report.json` BLOCKED report may remain as investigation history but is not the new Task 15 certification report.

## 4. CALIBRATION_GOLD Holdout Contract

`CALIBRATION_GOLD` records MAY traverse all quality gates and produce verdict/evidence, but MUST NOT enter formal SFT training export and MUST NOT count as Final Training Accepted.

`TRUSTED_TRAINING` and `HIGH_CONFIDENCE` receive no gate bypass and still require the complete quality path.

A production E2E regression MUST mechanically enforce this contract.

## 5. Gold Regression Pack

The existing repeated `2*x+3=11` case remains only a wiring smoke test. The fixed regression pack MUST include STEMQ/CalibrationGold, SciBench, CFE, calculation, conceptual/derivation, answer extraction, analysis completeness, QA/analysis alignment, known-correct cases, and known mismatch/reject cases.

Known wrong matches MUST never enter Accepted. CalibrationGold MUST never enter formal SFT export.

## 6. Manual Audit

All 100 certification records MUST be manually audited. Accepted records are reviewed for university-STEM fit, problem/exercise status, question immutability, source-grounded final answer, original analysis, answer/analysis alignment, obvious answer error, provenance, and 19-field reasonableness.

Rejected records are reviewed by reject reason, with every record examined. The committed audit is `docs/validation/100-dry-run-audit.md` and records parser, normalization, prompt/gate, authority, false-accept, and false-reject findings.

The audit MUST NOT claim >=99% precision metrics or extrapolate 100-record acceptance rate to the final 500K yield.

## 7. Frozen Config

`config/pilot-5k-frozen.yaml` remains the Task 15 candidate frozen config. It becomes formally frozen only after the 100-record real-model E2E, Gold Regression, manual audit, blocker fixes, and fresh verification all pass.

After freeze, threshold/model/prompt/pipeline behavior changes require re-certification of affected evidence.

## 8. Final Verification

Fresh verification MUST include the Task 15 focused suite specified by the reviewer, full `pytest -q`, `ruff check src tests`, and `mypy src`. The 100-record workflow MUST explicitly checkout the final Task 15 exact HEAD and record its GitHub Actions run ID.

## 9. Revised Acceptance Gate

Task 15 passes only when all are true:

1. 5K deterministic sampling certification PASS.
2. 100-record stratified real-model E2E PASS.
3. All 100 records manually audited.
4. Gold Regression PASS.
5. CALIBRATION_GOLD excluded from formal training export and Final Training Accepted count.
6. No known Question/Answer wrong match enters Accepted.
7. Exact 19-field export validates.
8. Provenance and answer source remain traceable.
9. Config/code/prompt/provider identity exactly matches run evidence.
10. Fresh pytest/Ruff/MyPy pass.

No 5K real-model task may be launched. `verification/task15-*` and `ci/task15-*` branches are evidence-only and MUST NOT be merged into formal development branches.