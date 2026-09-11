# Task 15 Certification Rebaseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-certify Task 15 with zero-model 5K deterministic sampling, one 100-record real-model production E2E, an enforceable CALIBRATION_GOLD holdout contract, expanded Gold regression, complete manual audit, and fresh exact-head verification.

**Architecture:** Preserve the existing bounded-memory 5K sampler and quality pipeline. Add a deterministic child-sample manifest bound to the certified 5K identity; enforce training-export eligibility only after all quality gates; use CI-only workflows for source materialization and real-model execution while committing only safe manifests/reports/docs to the feature branch. Real-model certification runs exactly once on the final candidate HEAD unless a blocker changes production behavior.

**Tech Stack:** Python 3.12+, Pydantic v2, Typer, GitHub Actions, OpenAI-compatible providers (`deepseek-flash`, `deepseek-v4-pro`), pytest, Ruff, MyPy.

**Spec:** `docs/superpowers/specs/2026-09-11-task15-certification-scope-amendment.md`

## Global Constraints

- Candidate starts from Task 15 branch `feature/task15-pilot-freeze`; Task 14 accepted base is `c7a57d9142156232444d2e7f8fc83031d09d039e`.
- Never launch a 5K real-model run.
- 5K sampling quotas remain Math 2000 / Physics 1000 / Statistics 1000 / MathOverflow 1000 with seed `20260910`.
- 100 real-model quotas are Math 40 / Physics 20 / Statistics 20 / MathOverflow 20 and must be a deterministic child of the certified 5K sample.
- LLMs classify/verify only; they never generate, fill, rewrite, correct, or improve formal question/answer/analysis content.
- `CALIBRATION_GOLD` may complete quality gates but is not training-export eligible and is not Final Training Accepted.
- `TRUSTED_TRAINING` and `HIGH_CONFIDENCE` have no quality-gate bypass.
- Raw 5K/100 source rows are runtime evidence only and must not be committed.
- `ci/task15-*` / `verification/task15-*` branches are evidence-only and never merged.
- Task 16 remains forbidden until reviewer acceptance.

---

### Task 15A: Reproduce CALIBRATION_GOLD Blocker and Lock the Contract

**Files:**
- Modify: `tests/e2e/test_gold_regression.py`
- Modify: `src/college_builder/pipeline/runner.py`
- Modify only if required by reporting semantics: `src/college_builder/reporting/pilot_report.py`

**Interfaces:**
- Consumes Gold metadata `gold_role` and `calibration_holdout` emitted by `GoldDatasetAdapter`.
- Produces a pipeline-level training-export eligibility decision after complete quality validation.

- [ ] **Step 1: Write RED E2E regression**

Create a `CALIBRATION_GOLD` record that passes all Gates with the existing regression provider. Assert all model tasks are invoked, final quality evidence exists, `questions.jsonl` contains zero records, and `PipelineRunResult.accepted_count == 0`.

Also assert equivalent `TRUSTED_TRAINING` and `HIGH_CONFIDENCE` records still traverse all Gates and remain export-eligible only when quality passes.

- [ ] **Step 2: Run RED**

Run: `pytest tests/e2e/test_gold_regression.py -q`

Expected before fix: CALIBRATION_GOLD appears in formal export and increments accepted count.

- [ ] **Step 3: Implement minimal holdout behavior**

Add one helper in the runner that reads the normalized/raw Gold metadata and returns `True` only for an explicit calibration holdout. Run the complete pipeline unchanged through verification/final schema validation, persist quality evidence, then exclude holdout IRs from `export_jsonl` and from `RecordAudit.accepted` / Final Training Accepted count.

Resume behavior must apply the same holdout rule so a completed quality-pass holdout cannot reappear in a resumed formal export.

- [ ] **Step 4: Run GREEN and related regressions**

Run:

```bash
pytest tests/e2e/test_gold_regression.py tests/e2e/test_pipeline_resume.py tests/e2e/test_final_export.py -q
ruff check src tests
mypy src
```

- [ ] **Step 5: Commit**

Commit only the holdout contract and its regression coverage.

**Acceptance:** CalibrationGold completes quality evaluation but never reaches formal training export/count; trusted/high-confidence Gold receives no bypass.

---

### Task 15B: Deterministic 100-of-5K Child Sample Manifest

**Files:**
- Modify: `src/college_builder/pipeline/pilot.py`
- Modify: `tests/unit/pipeline/test_pilot.py`
- Modify: `tests/e2e/test_pilot_sample_command.py` only if a CLI path is added.

**Interfaces:**
- Produces a 100-record child sample with a manifest containing parent `sample_sha256`, child seed/quotas, child `sample_sha256`, ordered `sampled_record_ids`, and ordered `sampled_raw_sha256`.

- [ ] **Step 1: Write RED deterministic-lineage tests**

Construct a synthetic parent 5K-shaped sample at reduced fixture scale and assert:

- exact child quotas;
- identical parent sample + seed gives byte-identical child manifest regardless of input iteration order;
- changing parent sample SHA or one raw hash changes child identity;
- every child `record_id` exists in the parent manifest and has the same raw hash;
- insufficient parent stratum fails closed.

- [ ] **Step 2: Run RED**

Run: `pytest tests/unit/pipeline/test_pilot.py -q` and confirm missing child-sample contract failure.

- [ ] **Step 3: Implement minimal child sampler**

Reuse deterministic priority selection, but bind priority/manifest identity to the parent 5K sample SHA. Do not reacquire from the full upstream dataset for the 100 selection; input is the already-certified 5K runtime sample.

- [ ] **Step 4: Run GREEN**

Run:

```bash
pytest tests/unit/pipeline/test_pilot.py tests/e2e/test_pilot_5k.py tests/e2e/test_pilot_sample_command.py -q
ruff check src tests
mypy src
```

- [ ] **Step 5: Commit**

Commit the child-sampling contract and tests.

**Acceptance:** The 100-record sample is mechanically proven to be an exact deterministic child of the 5K identity.

---

### Task 15C: Expand Fixed Gold Regression Pack

**Files:**
- Modify: `tests/e2e/test_gold_regression.py`
- Create if useful: `tests/fixtures/gold/task15_gold_regression.jsonl`

**Interfaces:**
- Uses existing production `GoldDatasetAdapter`, pipeline gates, export contract, and source-grounded answer extraction.

- [ ] **Step 1: Add fixed known cases**

Cover at minimum:

- STEMQ / `CALIBRATION_GOLD` calculation, quality-pass but no export;
- SciBench / `TRUSTED_TRAINING` conceptual or derivation case;
- CFE / `HIGH_CONFIDENCE` proof/derivation case;
- source answer extraction from an explicit answer;
- source answer extraction from a conclusion in analysis;
- complete analysis pass;
- shallow/missing analysis reject;
- known QA/analysis mismatch reject;
- known correctness mismatch reject.

Each case contains fixed source text and fixed expected terminal result. Providers only return classification/verdict evidence; they do not repair source content.

- [ ] **Step 2: Run Gold regression**

Run: `pytest tests/e2e/test_gold_regression.py -q`.

- [ ] **Step 3: Verify false-accept protections**

Assert every known mismatch has zero formal export rows and a deterministic reject reason.

- [ ] **Step 4: Commit**

Commit only the fixed regression pack/test changes.

**Acceptance:** Gold pack is multi-dataset, multi-problem-type, covers correct and incorrect cases, and mechanically protects holdout/export semantics.

---

### Task 15D: Zero-Model 5K Sampling Certification

**Files:**
- Create: `artifacts/5k/sampling_certification.json`
- No raw sample file is committed.

**Interfaces:**
- Reuses `sample_stackmathqa_source_root()` and current approved immutable StackMathQA revision.

- [ ] **Step 1: Run CI-only source materialization**

Checkout the current Task 15 feature exact HEAD. Download/stage the immutable StackMathQA source revision, compute source-file SHA256s, and run only `college-builder pilot sample`. Do not set or read model API secrets in this job.

- [ ] **Step 2: Verify canonical identity**

Assert exact quotas, seed `20260910`, immutable revision, 5000 record IDs/raw hashes, source-file SHA256s, and canonical sample SHA256. Re-run sampling from the same staged source and assert both manifests are byte-identical.

- [ ] **Step 3: Produce safe certification JSON**

Commit only identity/evidence metadata in `artifacts/5k/sampling_certification.json`; raw source rows remain CI-only.

**Acceptance:** 5K sampling is reproducible and model-free.

---

### Task 15E: Final-Head 100-record Real Model E2E

**Files:**
- Create after execution: `artifacts/100/sample_manifest.json`
- Create after execution: `artifacts/100/run_report.json`
- Runtime-only review bundle remains in GitHub Actions artifact storage.

**Interfaces:**
- Uses final Task 15 candidate code, certified 5K runtime sample, deterministic 100 child sampler, `config/pilot-5k-frozen.yaml`, and production CLI/pipeline.

- [ ] **Step 1: Freeze exact candidate identity for execution**

The workflow must checkout a literal Task 15 commit SHA, verify clean status, compute config SHA256 and all prompt SHA256s, and assert classifier/verifier are exactly `openai_compatible/deepseek-flash` and `openai_compatible/deepseek-v4-pro`.

- [ ] **Step 2: Recreate certified 5K without models**

Materialize the 5K sample and assert its manifest identity equals `artifacts/5k/sampling_certification.json`.

- [ ] **Step 3: Materialize deterministic 100 child sample**

Produce exactly 40/20/20/20, assert parent 5K SHA binding, and save the safe manifest.

- [ ] **Step 4: Execute production pipeline once**

Expose only the two OpenAI-compatible secrets to this step. Execute `college-builder run` over the 100 runtime records with the frozen config. Fake providers are forbidden.

- [ ] **Step 5: Validate terminal artifacts**

Mechanically validate all exported rows have exactly 19 fields, every final answer has source authority, every accepted record has provenance, no `CALIBRATION_GOLD` record is exported, and no provider identity mismatch/bypass exists.

- [ ] **Step 6: Build run evidence**

Create `run_report.json` with exact code/config/prompt/source/sample/provider identities, counts, funnel, reject reasons, provider calls/cache/errors/latency/wall time. Record provider token usage if exposed; otherwise set an explicit unavailable marker. Do not convert unavailable token/cost evidence into numeric zero.

- [ ] **Step 7: Upload review bundle**

Upload the 100 runtime records, normalized/audit/evidence artifacts, rejection facts, and final export as a private workflow artifact for manual audit. Do not commit raw source content.

**Acceptance:** One final-head 100-record real-model run completes the production chain with auditable identities and safe committed evidence.

---

### Task 15F: Manual Audit of All 100 Records

**Files:**
- Create: `docs/validation/100-dry-run-audit.md`

**Interfaces:**
- Consumes the Task 15E workflow review bundle and run evidence.

- [ ] **Step 1: Review all accepted records**

For every Accepted row record university-STEM fit, problem status, question immutability, source-grounded final answer, original analysis, answer/analysis alignment, obvious answer correctness, provenance, and 19-field reasonableness.

- [ ] **Step 2: Review all rejected records**

Group by reject reason but inspect every record; record whether the rejection is justified or a suspected false reject.

- [ ] **Step 3: Record systematic findings**

Explicitly state whether any parser, normalization, prompt/gate, answer/analysis authority, false-accept, or false-reject issue was found.

- [ ] **Step 4: Commit audit**

Commit aggregated audit findings only; do not reproduce copyrighted raw 100-row contents.

**Acceptance:** 100/100 certification records have a human audit disposition and no known false accept remains unresolved.

---

### Task 15G: Final Fresh Certification

**Files:**
- No production changes are allowed after the real-model run unless a blocker is found. If behavior changes, Task 15E must be rerun.

- [ ] **Step 1: Run reviewer-focused suite**

```bash
pytest tests/e2e/test_pilot_5k.py \
  tests/e2e/test_pilot_sample_command.py \
  tests/e2e/test_gold_regression.py \
  tests/e2e/test_task15_pipeline_concurrency.py \
  tests/e2e/test_task15_provider_prompt_protocol.py \
  tests/unit/pipeline/test_pilot.py \
  tests/unit/providers/test_openai_compatible_resilience.py \
  tests/unit/test_provider_config_resilience.py -q
```

- [ ] **Step 2: Run full quality gates**

```bash
pytest -q
ruff check src tests
mypy src
```

- [ ] **Step 3: Verify identity/scope guards**

Assert no raw 5K/100 data is tracked, no `ci/task15-*` or `verification/task15-*` workflow leaks into feature, run evidence code/config/prompt/provider identity equals the exact final HEAD, and Task 16 files/commits are absent.

- [ ] **Step 4: Submit Task 15 for reviewer acceptance**

Report exact feature HEAD, 5K sampling-certification evidence, 100 real-model Actions run ID, accepted/rejected funnel, manual-audit disposition, Gold result, and fresh pytest/Ruff/MyPy evidence. Stop; do not enter Task 16.

**Task 15 Acceptance Gate:** All ten criteria in the scope amendment must PASS on one consistent final Task 15 exact HEAD.