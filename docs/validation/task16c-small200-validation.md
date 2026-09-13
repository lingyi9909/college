# Task 16C — Small-200 Production Validation

## Result

**NOT ACCEPTED FOR SCALE-UP.**

This validation intentionally replaced the planned 50K pilot with a bounded real-model production check before any large-scale data production.

## Exact execution identity

- Production code SHA: `456174b0ed0497239733ddd2909eea39eda83ecb`
- GitHub Actions Run: `34734958843`
- Job: `103664606841`
- Artifact: `10311081780`
- Artifact SHA256: `a73531087bd043ffcf547a17684eb3903c5b01d666bee11efe912c5cdf7fc9d4`
- Sample count: `200`
- Sample payload SHA256: `d47c87205417fcc1607c3a8736c4859a1c0ae7ea4818c7d2396d9a28b002912a`
- Record identity SHA256: `3c7876808773b45452ad917f781990faef46ba899d22af947dd3ba102dc489e8`
- Sample was frozen before model calls.

Strata: Math 80, MathOverflow 20, Statistics 40, Physics 40, OpenStax 20.

The exact accepted Task16 configuration was used unchanged: `task16-recertification-v2`, classifier `deepseek-flash`, verifier `deepseek-v4-pro`.

## Engineering result

The production pipeline completed successfully with no provider errors. All 200 records reached a terminal state and the exported record was structurally valid exact 19-field `QuestionRecordV1`.

Funnel:

- raw: 200
- normalized: 200
- STEM: 119
- university: 105
- problem: 30
- answer valid: 7
- analysis valid: 7
- after dedup: 7
- alignment pass: 2
- final accepted: 1

Rejects: 199. The two dominant terminal reasons were `UNIVERSITY_LEVEL_UNCERTAIN` (74) and `PROBLEM_TYPE_UNCERTAIN` (67).

Fresh verification also passed: Task16 focused pytest 59 PASS; full pytest 547 PASS; Ruff PASS; MyPy PASS for 33 source files.

## Manual quality audit

### Blocker 1 — False Accept in the only exported record

All accepted records were audited; there was exactly one.

Source question asks why geometric optics can be used to analyze a microscope image. The exact source solution is:

`Microscopes create images of macroscopic size, so geometric optics applies.`

Gate3 extracted only:

`geometric optics applies.`

The exported `text_answer` therefore drops the causal premise that actually answers the why-question. The downstream correctness gate still passed the record. This is classified as a **False Accept**.

Root-cause direction: the deterministic conclusion-fragment extraction treats generic `so`/conclusion anchors as sufficient even when the preceding clause contains essential answer semantics.

### Blocker 2 — Gate1 material False Rejects

A deterministic stratified sample of 10 `UNIVERSITY_LEVEL_UNCERTAIN` rejects was manually reviewed.

Conservative classification:

- 5 clear False Rejects
- 2 borderline
- 3 plausible correct/low-level rejects

Clear examples include questions from `chemistry-atoms-first-2e`, `college-physics-2e`, `university-physics-volume-1`, plus advanced time-series cross-validation and ordinal-regression questions.

The issue is not merely noisy source data: unmistakable university STEM content is being lost at Gate1.

### Blocker 3 — Gate2 material False Rejects

A deterministic stratified sample of 10 `PROBLEM_TYPE_UNCERTAIN` rejects was manually reviewed.

Conservative classification:

- 8 clear False Rejects
- 2 borderline

Examples include linear algebra determinant reasoning, complex analysis, set theory, universal algebra, definite integration, plasma physics and MLE standard-error questions.

The approved `problem_classify/v1` prompt explicitly lists `CONCEPTUAL`, `PROOF`, `CALCULATION`, etc. as positive labels. In observed evidence, valid conceptual items are correctly labeled positive by the model but rejected because confidence is below the global 0.98 threshold. Example: the determinant/non-zero-solution question was labeled `CONCEPTUAL` at 0.85 and then terminally rejected as `PROBLEM_TYPE_UNCERTAIN`.

### Blocker 4 — License provenance loss

The accepted OpenStax source carries `license_metadata.license = cc-by-4.0`, while the exported `static_info.source_license` is `UNKNOWN`.

This must be fixed before any scale production because provenance/licensing is part of the dataset contract.

## Scale decision

Do **not** run 5K or 50K.

Before any expansion, fix the four blockers above and rerun the same bounded small-production validation. The next run must demonstrate:

1. zero False Accepts among all accepted records;
2. materially improved Gate1/Gate2 recall on known-good controls without weakening fail-closed behavior;
3. source-grounded complete answer semantics, not merely substring traceability;
4. exact source license preservation in exported provenance;
5. provider errors = 0 and all existing Task16 regressions remain green.
