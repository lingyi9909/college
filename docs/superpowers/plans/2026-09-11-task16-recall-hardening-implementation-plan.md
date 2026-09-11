# Task 16 Recall Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the four Task 15 false-reject/evidence protocol limitations, prove recall recovery with a small real-model certification, and only then permit the 50K validation pilot.

**Architecture:** Preserve the existing precision-first pipeline and source-authority contracts. Task 16A makes four narrowly scoped, regression-driven fixes at existing gates; Task 16B certifies those changes on a bounded real-model set with full manual audit; Task 16C re-enters the original 50K plan only after explicit acceptance.

**Tech Stack:** Python 3.12+, Pydantic v2, pytest, Ruff, MyPy, existing provider protocols and OpenAI-compatible real-model configuration.

**Spec:** `docs/superpowers/specs/2026-09-11-task16-recall-hardening-scope-amendment.md`

## Global Constraints

- Accepted Task 15 base: `5572fe691a866d048309806a8db051244a2d46d3`.
- Do not start 50K before Task 16A and 16B are explicitly accepted.
- False Accept must remain zero in the fully audited Task 16B set.
- No LLM generation/repair/rewrite of formal question, answer, or analysis.
- Gate3 output remains exact source-grounded content.
- Evidence remains fail-closed; protocol fixes must not silently reinterpret malformed references.
- Every production behavior change follows RED → GREEN → fresh regression verification.
- Evidence-only CI branches/runs must not become production authority.

---

### Task 16A.1: Gate2 Compatible-Positive Conflict Semantics

**Files:**
- Modify: `src/college_builder/quality/classify.py`
- Test: `tests/unit/quality/test_classify.py`
- Test: `tests/e2e/test_gold_regression.py`

**Interfaces:**
- Consumes existing Gate2 model decision/evidence objects.
- Produces the same Gate2 public result type; no downstream interface change.

- [ ] **Step 1: Add RED regressions from Task 15 false rejects**

Add cases where two supported positive problem labels/evidence are compatible and currently collapse to `MODEL_DECISION_CONFLICT`. Assert the desired result is a positive problem decision without inventing a new label.

Add control cases proving positive-vs-negative disagreement remains conflict/reject and malformed evidence remains fail-closed.

- [ ] **Step 2: Run focused tests and capture RED**

Run the exact new tests plus existing classification tests. The compatible-positive case must fail for the current production reason, not fixture/setup errors.

- [ ] **Step 3: Implement minimal conflict-semantic fix**

Change only semantic conflict resolution. Do not lower Gate2 thresholds and do not accept unsupported evidence. Compatible positive labels may co-exist; contradictory polarity may not.

- [ ] **Step 4: Run focused GREEN and regression suite**

Run classification unit tests, Gold regression, full pytest, Ruff, MyPy.

- [ ] **Step 5: Commit Task 16A.1 and stop for review**

Commit message: `fix: preserve compatible positive problem classifications`

**Acceptance Gate:** Known compatible-positive false rejects are recovered while contradictory/malformed decisions remain fail-closed.

---

### Task 16A.2: Gate3 Source-Grounded Answer Extraction Recall

**Files:**
- Modify: `src/college_builder/quality/answer.py`
- Test: `tests/unit/quality/test_answer.py`
- Test: `tests/e2e/test_gold_regression.py`

**Interfaces:**
- Existing answer validation/extraction result type remains unchanged.
- Every extracted answer must carry exact source authority/span.

- [ ] **Step 1: Add RED cases from Task 15 `ANSWER_NOT_EXTRACTABLE` false rejects**

Cover explicit terminal numeric answers, terminal mathematical expressions, named constructions/conditions, and existing supported conclusion-marker forms. Each expected answer must be an exact substring of source answer/analysis.

- [ ] **Step 2: Add anti-generation RED/guard cases**

Cases requiring algebraic derivation, paraphrase, correction, or completion must remain rejected when the exact answer text is absent from source content.

- [ ] **Step 3: Verify RED**

Run only the new answer-extraction tests and confirm failures are caused by current extraction recall.

- [ ] **Step 4: Implement minimal deterministic extraction expansion**

Recognize only bounded source forms proven by tests. Return source substring/span; never calculate, normalize into a different semantic string, or call a model to create the answer.

- [ ] **Step 5: Verify GREEN plus full regression**

Run answer tests, Gold regression, full pytest, Ruff, MyPy.

- [ ] **Step 6: Commit Task 16A.2 and stop for review**

Commit message: `fix: recover explicit source-grounded final answers`

**Acceptance Gate:** Known explicit-answer false rejects recover without any generated/derived answer entering formal output.

---

### Task 16A.3: Gate4 Analysis Evidence Reference Contract

**Files:**
- Create: `prompts/analysis_classify/v3.txt`
- Modify: `config/pilot-5k-frozen.yaml` only through a new Task16 certification config/version; do not mutate historical Task15 evidence identity in place.
- Modify: `src/college_builder/quality/analysis.py` only if required to enforce the documented grammar consistently.
- Test: `tests/e2e/test_task15_provider_prompt_protocol.py`
- Test: `tests/unit/quality/test_analysis.py`

**Interfaces:**
- Evidence grammar: `<field>:<start>-<end>`, indexes are Unicode/Python string offsets, 0-based, end-exclusive `[start,end)`.

- [ ] **Step 1: Add RED protocol tests**

Assert the current prompt does not sufficiently define the indexing contract and reproduce the Task15 analysis evidence false reject.

- [ ] **Step 2: Add validator boundary tests**

Valid half-open spans pass; inclusive-end, negative, reversed, out-of-range, wrong-field and semantically irrelevant references reject.

- [ ] **Step 3: Create versioned v3 prompt and Task16 config identity**

State the exact grammar, 0-based indexing, end-exclusive semantics, examples, and requirement to quote only source fields. Preserve the no-generation rule.

- [ ] **Step 4: Implement only validator changes required for prompt/validator consistency**

Do not auto-correct malformed offsets.

- [ ] **Step 5: Verify focused/full tests, Ruff, MyPy**

- [ ] **Step 6: Commit Task 16A.3 and stop for review**

Commit message: `fix: version analysis evidence span protocol`

**Acceptance Gate:** Prompt and validator mechanically agree on one strict span grammar; Task15 contract mismatch is covered permanently.

---

### Task 16A.4: Gate5 and Correctness Evidence Span Protocol

**Files:**
- Create: `prompts/qa_alignment/v2.txt`
- Create: `prompts/correctness_verify/v2.txt`
- Modify: Task16 certification config to select the new versions.
- Modify: `src/college_builder/quality/verify.py` only if needed for explicit shared validation helpers.
- Test: `tests/e2e/test_task15_provider_prompt_protocol.py`
- Test: `tests/unit/quality/test_verify.py`

**Interfaces:**
- Same strict 0-based end-exclusive `[start,end)` grammar as Gate4.
- PASS still requires valid question/answer/analysis evidence and complete answer authority coverage.

- [ ] **Step 1: Add RED regressions from downstream micro evidence**

Reproduce PASS/1.0 semantic verdicts rejected solely because prompt-generated offsets are ambiguous/invalid.

- [ ] **Step 2: Add strict validator tests**

Prove valid half-open references pass and malformed/inclusive/out-of-range references reject; no silent offset repair.

- [ ] **Step 3: Add versioned v2 prompts**

Explicitly define indexing and include short examples for question, answer and analysis fields. Preserve independent-verification and no-generation language.

- [ ] **Step 4: Refactor shared span validation only if duplication risks contract drift**

Any helper must preserve existing fail-closed semantics.

- [ ] **Step 5: Verify Gate5/correctness focused tests, Gold regression, full pytest, Ruff, MyPy**

- [ ] **Step 6: Commit Task 16A.4 and stop for review**

Commit message: `fix: version downstream evidence span protocol`

**Acceptance Gate:** Alignment/correctness prompt and validator share the exact same strict reference protocol, with no relaxed source authority.

---

### Task 16B: 20–50 Record Real-Model Re-certification

**Files:**
- Create: `artifacts/task16/re-certification/sample_manifest.json`
- Create: `artifacts/task16/re-certification/run_report.json`
- Create: `docs/validation/task16-small-recert-audit.md`
- Add CI workflow/evidence scripts only on evidence branches unless production execution support is genuinely reusable.

**Interfaces:**
- Input records are deterministically selected from Task15 known false rejects/known-good records plus fixed Gold controls.
- Output Accepted rows use the existing exact 19-field contract.

- [ ] **Step 1: Freeze deterministic small sample before model calls**

Choose 20–50 records with explicit coverage of all four limitation classes. Store record IDs/raw SHA256 and parent Task15 sample identity. Do not select outcomes after seeing new model results.

- [ ] **Step 2: Record Task15 baseline labels for the same records**

The report must include before-state rejection class so recall recovery is measurable.

- [ ] **Step 3: Run one real-model certification on exact code/config/prompt SHA**

Use configured real classifier/verifier models. Run complete applicable production path through final dedup/export. Do not retry by changing the sample after observing outcomes; infrastructure retry may reuse the exact frozen sample/identity.

- [ ] **Step 4: Audit every record manually**

Classify every new Accepted/Rejected result against immutable source content. Record false accept, false reject, correct accept, correct reject and gate reason.

- [ ] **Step 5: Enforce acceptance gate**

Require: False Accept=0; measurable reduction of known false rejects with before/after counts; all four limitation classes covered; at least one genuine source-grounded record completes Gate5→Correctness→Final Dedup→19-field export; no generated/repaired formal content.

- [ ] **Step 6: Run fresh exact-head certification**

Run focused Task16 regressions, full `pytest -q`, `ruff check src tests`, `mypy src`, plus identity/raw-data guards.

- [ ] **Step 7: Commit safe certification evidence and stop for explicit review**

Do not start 50K automatically.

**Acceptance Gate:** Task16B is independently reviewable and explicitly accepted. Failure returns to the relevant Task16A subtask.

---

### Task 16C: 50K Validation Pilot, Manual Audit, and Scale Decision

**Files:**
- Create: `docs/validation/50k-pilot-report.md`
- Create: `docs/validation/50k-manual-audit.md`
- Create: `docs/validation/scale-decision.md`

- [ ] **Step 1: Verify explicit Task16A + Task16B acceptance evidence**

No acceptance evidence means stop; do not run 50K.

- [ ] **Step 2: Execute the original Task16 deterministic 50K stratified pilot using the newly certified config/prompt versions**

Preserve original stratification and provenance requirements.

- [ ] **Step 3: Produce full funnel, reject reasons, duplicate rate, provider/cache/error/latency/token/cost evidence**

Unavailable usage values remain unavailable, never fake zero.

- [ ] **Step 4: Produce the original manual-audit pack and calculate measured precision/yield with explicit denominators**

- [ ] **Step 5: Write scale decision**

Decide scale-out, source-mix changes, further hardening, or stop based on measured 50K evidence.

- [ ] **Step 6: Fresh verification and explicit Task16 review**

No task after Task16 may begin before reviewer acceptance.

**Acceptance Gate:** The 50K evidence supports a defensible production-scale quality/yield decision; otherwise stop and iterate without claiming 500K readiness.
