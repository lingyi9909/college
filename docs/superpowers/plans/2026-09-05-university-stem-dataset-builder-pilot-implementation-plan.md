# University STEM Dataset Builder V1 Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-shaped University STEM Dataset Builder from an empty repository through a fully auditable 5K dry run and 50K validation pilot, without yet scaling to the final 500K release.

**Architecture:** The system ingests heterogeneous public university/STEM sources through strict SourceAdapter contracts, preserves immutable raw provenance, normalizes content without semantic rewriting, applies staged precision-first quality gates, performs multi-level deduplication, exports the existing exact 19-field `QuestionRecordV1` under `UniversitySTEMProfileV1`, and produces auditable pilot reports. The plan deliberately stops after the 50K Pilot Decision Gate because the approved design requires the scale-out and vendor-purchase strategy to be chosen from measured acceptance rates rather than assumptions.

**Tech Stack:** Python 3.12+, Pydantic v2, Typer, DuckDB, PyArrow/Parquet, SQLite (`sqlite3`), httpx, BeautifulSoup4/lxml, markdownify, SymPy, datasketch, pytest, pytest-asyncio, Ruff, MyPy. Model access must be behind provider protocols; no business/domain code may depend on a vendor SDK.

**Spec:** `docs/superpowers/specs/2026-09-05-university-stem-dataset-builder-design.md`

## Global Constraints

- Final formal release target remains `Final Accepted >= 500,000`, but this implementation plan stops at the 50K Validation Pilot and Decision Gate.
- Formal records must use the existing exact 19-field `QuestionRecordV1`; this project adds `UniversitySTEMProfileV1` but must not break the K12 profile contract.
- Accepted records require original source question, original source answer, and original source analysis; `answer_analysis == ""` or `answer_analysis == "略"` is rejected for the university profile.
- LLMs may classify, extract metadata, judge completeness/alignment, and independently verify correctness; they must never generate,补写,改写,纠正 formal questions, answers, or analyses.
- Every Accepted record must be traceable to immutable Raw Source data and quality evidence.
- License/copyright does not block Pilot acquisition, but provenance and license metadata must be captured from the first acquisition step.
- Unknown or ambiguous critical evidence fails closed; do not choose the “best looking” candidate when evidence conflicts.
- V1 must not introduce Kafka, Flink, Spark, Kubernetes, a Web UI, or synthetic question generation.
- All thresholds, provider/model identities, prompt versions, and pipeline versions must be configuration/version controlled and stored in run evidence.
- Development follows TDD. Each Task gets its own commit and must be independently reviewable before the next Task starts.
- No Task after Task 16 may begin until Task 16 is formally accepted by the reviewer.

---

## File Structure Locked By This Plan

```text
college/
├── pyproject.toml
├── README.md
├── config/
│   ├── pilot.yaml
│   └── profiles/
│       └── university_stem_v1.yaml
├── prompts/
│   ├── university_classify/v1.txt
│   ├── problem_classify/v1.txt
│   ├── analysis_classify/v1.txt
│   ├── qa_alignment/v1.txt
│   └── correctness_verify/v1.txt
├── src/college_builder/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── domain/
│   │   ├── source.py
│   │   ├── question.py
│   │   ├── evidence.py
│   │   └── final_record.py
│   ├── providers/
│   │   ├── base.py
│   │   ├── fake.py
│   │   └── openai_compatible.py
│   ├── source/
│   │   ├── base.py
│   │   ├── huggingface.py
│   │   ├── stackmathqa.py
│   │   ├── gold.py
│   │   └── openstax.py
│   ├── normalize/
│   │   ├── html.py
│   │   ├── math.py
│   │   └── normalizer.py
│   ├── storage/
│   │   ├── state.py
│   │   ├── parquet.py
│   │   └── cache.py
│   ├── quality/
│   │   ├── engine.py
│   │   ├── integrity.py
│   │   ├── classify.py
│   │   ├── answer.py
│   │   ├── analysis.py
│   │   ├── dedup.py
│   │   └── verify.py
│   ├── export/
│   │   ├── profile.py
│   │   └── jsonl.py
│   ├── reporting/
│   │   └── pilot_report.py
│   └── pipeline/
│       ├── runner.py
│       └── pilot.py
└── tests/
    ├── unit/
    ├── contract/
    ├── golden/
    └── e2e/
```

---

### Task 1: Repository Bootstrap, Configuration, and Core Domain Contracts

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `src/college_builder/__init__.py`
- Create: `src/college_builder/config.py`
- Create: `src/college_builder/domain/source.py`
- Create: `src/college_builder/domain/question.py`
- Create: `src/college_builder/domain/evidence.py`
- Create: `tests/unit/test_config.py`
- Create: `tests/unit/domain/test_source.py`
- Create: `tests/unit/domain/test_question.py`

**Interfaces:**
- Produces `RawSourceRecord`, `NormalizedQA`, `UniversityQuestionIR`, `GateEvidence`, `PipelineConfig`.
- Later Tasks must import these types rather than declaring parallel dict schemas.

- [ ] **Step 1: Add project dependencies and quality commands**

Use Python `>=3.12`. Add runtime dependencies: `pydantic>=2,<3`, `typer>=0.12,<1`, `duckdb>=1,<2`, `pyarrow>=17,<20`, `httpx>=0.27,<1`, `beautifulsoup4>=4.12,<5`, `lxml>=5,<6`, `markdownify>=0.13,<1`, `sympy>=1.13,<2`, `datasketch>=1.6,<2`, `pyyaml>=6,<7`. Add dev dependencies: `pytest>=8,<9`, `pytest-asyncio>=0.24,<1`, `ruff>=0.8,<1`, `mypy>=1.13,<2`.

- [ ] **Step 2: Write failing domain-contract tests**

```python
from college_builder.domain.source import RawSourceRecord


def test_raw_source_requires_stable_identity_and_raw_hash() -> None:
    record = RawSourceRecord(
        record_id="raw_stackmathqa_42",
        source_type="dataset",
        source_dataset="stackmathqa",
        source_id="42",
        source_url="https://math.stackexchange.com/questions/42",
        raw_question="Q",
        raw_answer="A",
        raw_analysis="Because A",
        raw_payload={"id": 42},
        metadata={"tags": ["linear-algebra"]},
        license_metadata={"declared": "UNREVIEWED"},
        raw_sha256="a" * 64,
    )
    assert record.source_id == "42"
    assert len(record.raw_sha256) == 64
```

Also test `UniversityQuestionIR` keeps raw and normalized question separately and that `GateEvidence` requires `gate_name`, `verdict`, `score`, `provider`, `model`, `prompt_version`, and `config_version`.

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest tests/unit/domain tests/unit/test_config.py -q`
Expected: import/module failures because contracts do not exist.

- [ ] **Step 4: Implement exact Pydantic contracts**

Use enums for `GateVerdict={PASS,REJECT,VERIFY}`, `Discipline`, `UniversityLevel`, `ProblemType`, `AnalysisType`. `RawSourceRecord` raw fields are immutable after construction (`frozen=True`). `NormalizedQA` holds normalized content plus `source_record_id`; it never overwrites raw content. `UniversityQuestionIR` contains `question`, `answer`, `analysis`, `classification`, `metadata`, `quality`, `provenance`, and `dedup` objects.

- [ ] **Step 5: Implement typed configuration loading**

`PipelineConfig.load(path: Path) -> PipelineConfig` must validate pipeline version, profile, threshold ranges `[0,1]`, provider names, prompt versions, and concurrency values `>=1`. Invalid configuration must fail before source acquisition.

- [ ] **Step 6: Verify Task 1**

Run:

```bash
pytest tests/unit/domain tests/unit/test_config.py -q
ruff check src tests
mypy src
```

Expected: all PASS, Ruff clean, MyPy clean.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml README.md src tests
git commit -m "feat: establish dataset builder domain contracts"
```

**Task 1 Acceptance Gate:** No untyped dict may act as a core domain contract; raw source fields are immutable; configuration fails closed.

---

### Task 2: Exact 19-Field Contract and UniversitySTEMProfileV1

**Files:**
- Create: `src/college_builder/domain/final_record.py`
- Create: `src/college_builder/export/profile.py`
- Create: `config/profiles/university_stem_v1.yaml`
- Create: `tests/contract/test_final_record_contract.py`
- Create: `tests/contract/test_university_profile.py`

**Interfaces:**
- Produces `FinalQuestionRecord`, `UniversitySTEMProfile`, `slim_question_md5_v1(text: str) -> str`.
- Task 13 exporter must use these definitions without redefining fields.

- [ ] **Step 1: Write exact-field failure test**

```python
EXPECTED_FIELDS = {
    "text_question", "is_pic_included", "text_answer", "answer_analysis",
    "text_course", "text_grade_level", "text_grade", "knowledge_points",
    "exam_points", "publisher", "text_paper", "textbook_version",
    "static_info", "language", "text_year", "entrance_exam_type",
    "text_city", "question_type", "competition_event",
}


def test_final_record_has_exact_19_fields() -> None:
    assert set(FinalQuestionRecord.model_fields) == EXPECTED_FIELDS
    assert len(FinalQuestionRecord.model_fields) == 19
```

- [ ] **Step 2: Write university-profile rejection tests**

Test that `text_grade="大学"` is valid; K12-only values are not silently accepted by the university profile; `answer_analysis=""` and `answer_analysis="略"` reject; `language` requires lowercase ISO-639-1-like two-letter codes; `static_info` must parse as a JSON object; `is_pic_included` only accepts `0|1`.

- [ ] **Step 3: Run tests and verify RED**

Run: `pytest tests/contract -q`
Expected: failures because contract/profile do not exist.

- [ ] **Step 4: Implement `slim_md5_v1` exactly**

Canonicalization order: Unicode NFC; CRLF/CR to LF; trim outer whitespace; collapse runs of blank lines to one blank line; preserve digits, formulas, image references, and semantic punctuation; return lowercase MD5 hex.

- [ ] **Step 5: Implement UniversitySTEMProfileV1 enum rules**

`text_grade` accepted value for formal university records is `大学`; `text_grade_level` accepts `本科一年级/本科二年级/本科三年级/本科四年级/本科/硕士/博士/研究生/大学未知`; `text_course` accepts the approved stable first-level disciplines; `entrance_exam_type` remains `未知` for ordinary university course data.

- [ ] **Step 6: Verify Task 2**

Run: `pytest tests/contract -q && ruff check src tests && mypy src`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add src/college_builder/domain/final_record.py src/college_builder/export/profile.py config/profiles tests/contract
git commit -m "feat: add university 19-field profile contract"
```

**Task 2 Acceptance Gate:** Exact 19-field shape must be mechanically enforced; University profile must not relax analysis completeness.

---

### Task 3: Run State, Parquet Storage, and Deterministic Cache

**Files:**
- Create: `src/college_builder/storage/state.py`
- Create: `src/college_builder/storage/parquet.py`
- Create: `src/college_builder/storage/cache.py`
- Create: `tests/unit/storage/test_state.py`
- Create: `tests/unit/storage/test_parquet.py`
- Create: `tests/unit/storage/test_cache.py`

**Interfaces:**
- Produces `RunStateStore`, `ParquetStageStore`, `CallCache`.
- `CallCache.key(content_hash, gate_name, provider, model, prompt_version, config_version) -> str` is the only cache-key construction path.

- [ ] **Step 1: Write failing state-transition tests**

Allow only: `ACQUIRED -> NORMALIZED -> CLASSIFIED -> ANSWER_VALIDATED -> ANALYSIS_VALIDATED -> DEDUPED -> VERIFIED -> ACCEPTED|REJECTED`. Re-running a completed stage with the same fingerprint must be idempotent.

- [ ] **Step 2: Write failing Parquet round-trip test**

Persist three `RawSourceRecord` rows, reopen with DuckDB, and assert all `record_id`, raw text, provenance JSON, and hashes survive exactly.

- [ ] **Step 3: Write failing cache-key test**

Changing only `prompt_version` or `model` must change the cache key; identical inputs must produce identical keys.

- [ ] **Step 4: Implement storage**

Use SQLite for run/stage/call state and cache index; use Parquet partitions by `run_id/stage/source_dataset`; write via temp file then atomic rename. Store JSON objects as deterministic JSON strings with sorted keys.

- [ ] **Step 5: Verify crash recovery**

Test a run interrupted after `NORMALIZED`: reopening must resume at `CLASSIFIED` and must not reacquire or renormalize completed records.

- [ ] **Step 6: Verify and commit**

Run: `pytest tests/unit/storage -q && ruff check src tests && mypy src`

Commit: `git commit -am "feat: add resumable storage and cache"` after staging new files.

**Task 3 Acceptance Gate:** A killed process can resume without losing raw records or repeating a cached expensive call.

---

### Task 4: SourceAdapter Contract, Generic HF Loader, and Immutable Provenance

**Files:**
- Create: `src/college_builder/source/base.py`
- Create: `src/college_builder/source/huggingface.py`
- Create: `tests/unit/source/test_base.py`
- Create: `tests/unit/source/test_huggingface.py`
- Create: `tests/fixtures/sources/hf_sample.jsonl`

**Interfaces:**
- Produces `SourceAdapter.discover(config)`, `SourceAdapter.acquire(descriptor)`, `SourceAdapter.checkpoint()` and `HFDatasetAdapter`.

- [ ] **Step 1: Write adapter contract test**

A fake adapter must yield stable records across two runs and preserve `source_dataset`, `source_id`, original URL, original raw payload, license metadata, acquisition metadata, and `raw_sha256`.

- [ ] **Step 2: Write HF fixture mapping test**

Use a local JSONL fixture rather than internet in unit tests. Assert field mapping is explicit configuration; unknown source fields stay inside `raw_payload` rather than being dropped.

- [ ] **Step 3: Implement base adapter and HF loader**

Network fetch is isolated behind one loader method. Adapter code may map fields but must not classify university level or rewrite content.

- [ ] **Step 4: Add checkpoint behavior**

Checkpoint must record source descriptor, next row offset/shard, and source revision when available. Resume must not duplicate already persisted `record_id` values.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/source/test_base.py tests/unit/source/test_huggingface.py -q`.

Commit: `git commit -am "feat: add source adapter and huggingface ingestion"` after staging files.

**Task 4 Acceptance Gate:** Raw provenance is lossless and stable; no quality/business decision exists in source adapters.

---

### Task 5: StackMathQA Adapter and Deterministic Stratified Sampling

**Files:**
- Create: `src/college_builder/source/stackmathqa.py`
- Create: `tests/unit/source/test_stackmathqa.py`
- Create: `tests/golden/stackmathqa_mapping.json`

**Interfaces:**
- Produces `StackMathQAAdapter` and `stratified_sample(records, quotas, seed) -> list[RawSourceRecord]`.

- [ ] **Step 1: Add fixture covering Math, MathOverflow, Statistics, Physics**

Each fixture row must include question identity, question body, accepted/high-quality answer body, tags, score, source URL/site, timestamps when present, and original payload.

- [ ] **Step 2: Write mapping tests**

Assert answer and explanation are not collapsed into generated content. If the source only provides one long answer body, keep it as source analysis and leave final answer extraction for Task 10.

- [ ] **Step 3: Write sampling tests**

For quotas `{math: 20, mathoverflow: 5, statistics: 10, physics: 10}`, the sampler must return exact counts when enough rows exist, must be deterministic for a fixed seed, and must fail explicitly when a requested stratum lacks rows.

- [ ] **Step 4: Implement adapter and sampler**

Pilot strata must be source-site based, not inferred course labels. No `StackMathQA100K` convenience subset may replace the approved stratified Pilot composition.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/source/test_stackmathqa.py -q`.

Commit: `git commit -am "feat: add stackmathqa stratified source"` after staging files.

**Task 5 Acceptance Gate:** Sampling is reproducible and cannot silently become math-heavy.

---

### Task 6: Gold Sources and OpenStax/Open-Textbook Acquisition

**Files:**
- Create: `src/college_builder/source/gold.py`
- Create: `src/college_builder/source/openstax.py`
- Create: `tests/unit/source/test_gold.py`
- Create: `tests/unit/source/test_openstax.py`
- Create: `tests/fixtures/sources/openstax_sample.xml`

**Interfaces:**
- Produces `GoldDatasetAdapter`, `OpenTextbookAdapter`.
- Gold rows must set internal `quality_tier_candidate="GOLD"` but still pass structural/contract gates.

- [ ] **Step 1: Write Gold identity tests**

STEMQ/SciBench/CFE/OpenStax fixture records retain dataset name, source record id, institution/textbook metadata, source question, source solution, and license metadata.

- [ ] **Step 2: Write OpenStax pair extraction test**

From a local fixture containing problem and solution containers, assert one deterministic problem-solution pair is emitted with stable source identity.

- [ ] **Step 3: Implement adapters**

Do not mark Gold records Accepted directly. Gold means trusted calibration provenance, not bypass of final schema/integrity checks.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/unit/source/test_gold.py tests/unit/source/test_openstax.py -q`.

Commit: `git commit -am "feat: add gold and open textbook sources"` after staging files.

**Task 6 Acceptance Gate:** Gold provenance is explicit; no Gold adapter can bypass later content or schema validation.

---

### Task 7: Source Normalization Without Semantic Rewrite

**Files:**
- Create: `src/college_builder/normalize/html.py`
- Create: `src/college_builder/normalize/math.py`
- Create: `src/college_builder/normalize/normalizer.py`
- Create: `tests/unit/normalize/test_html.py`
- Create: `tests/unit/normalize/test_math.py`
- Create: `tests/golden/normalization_cases.json`

**Interfaces:**
- Produces `normalize(record: RawSourceRecord) -> NormalizedQA`.

- [ ] **Step 1: Write golden normalization cases**

Cover HTML paragraphs, lists, blockquotes, code blocks, `<pre>`, inline math, display math, MathML, escaped entities, links, images, and CRLF. Expected output must preserve numbers, operators, option labels, code, and source meaning.

- [ ] **Step 2: Write anti-rewrite tests**

For `2x+3=7`, normalization must not produce `x=2`; for `$x^2$`, it must not change exponent; for answer option `B`, it must not infer option text.

- [ ] **Step 3: Implement deterministic normalization**

Use parser-based HTML conversion. Convert recognized MathML deterministically to LaTeX; if conversion fails, mark evidence `MATH_UNRESOLVED` instead of asking an LLM to rewrite formula content.

- [ ] **Step 4: Preserve raw/normalized duality**

`NormalizedQA.source_record_id` must resolve back to immutable RawSourceRecord. Store normalization evidence and content hashes.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/normalize tests/golden -q`.

Commit: `git commit -am "feat: add lossless source normalization"` after staging files.

**Task 7 Acceptance Gate:** Deterministic formatting changes are allowed; semantic answer/question rewriting is impossible by API design.

---

### Task 8: Gate Engine and Gate 0 Integrity/Provenance

**Files:**
- Create: `src/college_builder/quality/engine.py`
- Create: `src/college_builder/quality/integrity.py`
- Create: `tests/unit/quality/test_engine.py`
- Create: `tests/unit/quality/test_integrity.py`

**Interfaces:**
- Produces `QualityGate.evaluate(candidate, context) -> GateEvidence` and `GateEngine.run(candidate) -> GateRunResult`.

- [ ] **Step 1: Write fail-closed engine tests**

A gate exception, malformed gate output, missing score, or unknown verdict must result in `REJECT` with explicit reason; it must never become PASS.

- [ ] **Step 2: Write Gate 0 tests**

Reject when raw record cannot be found, raw hash mismatches, source id is absent, source question is empty, source answer/analysis evidence is missing, or provenance JSON is malformed. License status may be `UNREVIEWED`; missing license metadata key itself is rejected from Pilot Accepted output.

- [ ] **Step 3: Implement engine and integrity gate**

Every gate result must store gate name, verdict, normalized score, reason code, evidence payload, provider/model identity when applicable, prompt/config versions, and timestamp.

- [ ] **Step 4: Verify and commit**

Run: `pytest tests/unit/quality/test_engine.py tests/unit/quality/test_integrity.py -q`.

Commit: `git commit -am "feat: add fail-closed quality gate engine"` after staging files.

**Task 8 Acceptance Gate:** No exception or missing evidence path can accidentally PASS.

---

### Task 9: Provider Contract, Gate 1 University/STEM, and Gate 2 Problem Classification

**Files:**
- Create: `src/college_builder/providers/base.py`
- Create: `src/college_builder/providers/fake.py`
- Create: `src/college_builder/providers/openai_compatible.py`
- Create: `src/college_builder/quality/classify.py`
- Create: `prompts/university_classify/v1.txt`
- Create: `prompts/problem_classify/v1.txt`
- Create: `tests/unit/providers/test_contract.py`
- Create: `tests/unit/quality/test_classify.py`

**Interfaces:**
- Produces `StructuredModelProvider.classify(request) -> ModelDecision`, `UniversityStemGate`, `ProblemGate`.

- [ ] **Step 1: Define strict model-decision schema**

Model outputs contain only labels, scores, evidence references, and reason codes. They must not contain rewritten question/answer/analysis fields.

- [ ] **Step 2: Write threshold tests**

Default Pilot v1: score `>=0.98 => PASS`; `0.90 <= score < 0.98 => VERIFY`; `<0.90 => REJECT`. A Primary/Verifier conflict rejects. Verifier must be independently configured and its provider+model identity stored.

- [ ] **Step 3: Write taxonomy tests**

University STEM positives: linear algebra eigenvalue calculation, Maxwell-equation derivation, algorithm complexity proof. Negatives: IDE recommendation, software installation, career advice, ordinary K12 arithmetic. Problem positives: calculation/proof/derivation/conceptual/algorithm/programming/engineering problem. Reject software-use/debug-help/opinion/resource-request/meta discussions.

- [ ] **Step 4: Implement provider protocol and fake provider**

All unit tests use `FakeStructuredModelProvider`; network model calls belong only in integration tests. The OpenAI-compatible adapter receives endpoint/base URL and model through config/environment and must not leak provider DTOs into domain code.

- [ ] **Step 5: Implement Gate 1 and Gate 2**

Course/site tags may contribute evidence but cannot force PASS. `math.stackexchange.com` alone never proves university level.

- [ ] **Step 6: Verify and commit**

Run: `pytest tests/unit/providers tests/unit/quality/test_classify.py -q && ruff check src tests && mypy src`.

Commit: `git commit -am "feat: add university and problem classification gates"` after staging files.

**Task 9 Acceptance Gate:** University and Problem decisions are evidence-based, independently verifiable, and never inferred solely from source site.

---

### Task 10: Gate 3 Original Answer Extraction and Gate 4 Original Analysis Completeness

**Files:**
- Create: `src/college_builder/quality/answer.py`
- Create: `src/college_builder/quality/analysis.py`
- Create: `prompts/analysis_classify/v1.txt`
- Create: `tests/unit/quality/test_answer.py`
- Create: `tests/unit/quality/test_analysis.py`
- Create: `tests/golden/answer_analysis_cases.json`

**Interfaces:**
- Produces `extract_source_answer(candidate) -> AnswerExtraction`, `AnswerGate`, `AnalysisGate`.

- [ ] **Step 1: Write answer source-span tests**

A final answer is valid only when it is a literal or deterministically normalized span of source answer/solution content. Store start/end offsets or equivalent source-span evidence. If extraction requires solving the problem, reject with `ANSWER_NOT_EXTRACTABLE`.

- [ ] **Step 2: Write explicit anti-generation tests**

Given source analysis `"Solving gives the result shown above"` with no explicit result, a fake model suggesting `x=4` must not be accepted. Given source text ending `"therefore x = 4"`, deterministic extraction may return `x = 4` with that exact span.

- [ ] **Step 3: Write analysis completeness tests**

Reject empty analysis, `略`, answer-only text, link-only response, page-reference-only response, and commentary unrelated to solving. Accept derivation, step-by-step, proof, conceptual reasoning, algorithm explanation, code explanation, and engineering reasoning when they materially support the answer.

- [ ] **Step 4: Implement AnswerGate thresholds**

`answer_source_exists=True`, `answer_extract_score>=0.995`, and source span evidence are required. No model-generated value may populate `final_answer`.

- [ ] **Step 5: Implement AnalysisGate thresholds**

Default Pilot v1: `analysis_score>=0.98 PASS`, `0.90-0.98 VERIFY`, `<0.90 REJECT`; verifier conflict rejects. Persist `AnalysisType`.

- [ ] **Step 6: Verify and commit**

Run: `pytest tests/unit/quality/test_answer.py tests/unit/quality/test_analysis.py tests/golden -q`.

Commit: `git commit -am "feat: validate original answers and analyses"` after staging files.

**Task 10 Acceptance Gate:** Every surviving answer and analysis is source-grounded; no generated answer/analysis path exists.

---

### Task 11: Exact, Near, and Semantic Deduplication

**Files:**
- Create: `src/college_builder/quality/dedup.py`
- Create: `tests/unit/quality/test_dedup.py`
- Create: `tests/golden/dedup_cases.json`

**Interfaces:**
- Produces `ExactDeduper`, `NearDuplicateIndex`, `DuplicateDecision={SAME_PROBLEM,VARIANT,DIFFERENT}`.

- [ ] **Step 1: Write exact duplicate tests**

Questions differing only by Unicode normalization, line endings, or meaningless blank lines must collide under `slim_md5_v1`; questions differing in a digit, exponent, sign, option, or image reference must not.

- [ ] **Step 2: Write near-duplicate candidate tests**

Use MinHash/LSH only to retrieve candidates. Similarity alone must never delete a record.

- [ ] **Step 3: Write variant-preservation tests**

`∫x dx` and `∫2x dx` are `VARIANT`, not duplicate. Same question mirrored across two datasets is `SAME_PROBLEM`; keep the higher provenance/quality record and preserve duplicate lineage.

- [ ] **Step 4: Implement dedup evidence**

Store exact hash, near-duplicate signatures, candidate ids, decision source, and kept/dropped record ids. Semantic verifier is invoked only for candidate pairs above configured retrieval threshold.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/quality/test_dedup.py tests/golden -q`.

Commit: `git commit -am "feat: add evidence-based multi-level dedup"` after staging files.

**Task 11 Acceptance Gate:** Similar-but-different math/STEM variants cannot be deleted solely by embedding/MinHash similarity.

---

### Task 12: Gate 5 QA/Analysis Alignment and Independent Correctness Verification

**Files:**
- Create: `src/college_builder/quality/verify.py`
- Create: `prompts/qa_alignment/v1.txt`
- Create: `prompts/correctness_verify/v1.txt`
- Create: `tests/unit/quality/test_verify.py`
- Create: `tests/golden/verification_cases.json`

**Interfaces:**
- Produces `AlignmentGate`, `CorrectnessVerifier`, `DeterministicMathVerifier`.

- [ ] **Step 1: Write alignment tests**

PASS only if analysis addresses the same question and materially supports the extracted source answer. Question A + Answer B, or explanation supporting a different option/result, must reject.

- [ ] **Step 2: Write independent correctness tests**

Verifier verdicts are exactly `PASS|FAIL|UNCERTAIN`. `UNCERTAIN` rejects from formal Accepted data. Store verifier provider/model and calibrated score. The verifier may solve internally but its generated solution must never enter `text_answer` or `answer_analysis`.

- [ ] **Step 3: Add deterministic math verification**

For safely parseable arithmetic/algebra identities, SymPy may confirm a source answer. Failure to parse means “not deterministically verified”, not correction. A deterministic contradiction rejects.

- [ ] **Step 4: Enforce final quality thresholds**

Default Pilot config: QA alignment `>=0.995`; independent correctness verdict `PASS`; any high-confidence verifier disagreement is `CORRECTNESS_UNCERTAIN` or explicit mismatch and rejects.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/quality/test_verify.py tests/golden -q`.

Commit: `git commit -am "feat: add alignment and correctness verification"` after staging files.

**Task 12 Acceptance Gate:** A model may judge correctness but cannot replace source truth; disagreement fails closed.

---

### Task 13: Metadata Mapping, UniversityQuestionIR Finalization, and 19-Field Export

**Files:**
- Create: `src/college_builder/export/jsonl.py`
- Modify: `src/college_builder/export/profile.py`
- Create: `tests/unit/export/test_jsonl.py`
- Create: `tests/e2e/test_final_export.py`

**Interfaces:**
- Produces `to_final_record(ir: UniversityQuestionIR) -> FinalQuestionRecord`, `export_jsonl(records, output_dir)`.

- [ ] **Step 1: Write metadata mapping tests**

Map first-level discipline to `text_course`; use reliable source level or calibrated inference for `text_grade_level`; set `text_grade="大学"`; use `knowledge_points` for topic concepts and `exam_points` for assessment capability; do not fabricate publisher, city, textbook version, year, or competition.

- [ ] **Step 2: Write `static_info` contract test**

It must be a JSON string containing at least `slim_question_md5`, `copyright`, `contract_version`, `profile`, `source_dataset`, `source_id`, `source_url`, `source_license`, `source_question_hash`, `source_answer_hash`, `pipeline_version`, and `quality_tier`.

- [ ] **Step 3: Write image/reference test**

`is_pic_included` is computed from final question content/assets; every `<img src="image/...">` target must exist in export output or the record rejects.

- [ ] **Step 4: Implement exporter**

Only records in terminal `ACCEPTED` state may be exported. JSONL is UTF-8, one JSON object per line, deterministic field order, no internal provider secrets.

- [ ] **Step 5: Verify and commit**

Run: `pytest tests/unit/export tests/e2e/test_final_export.py -q`.

Commit: `git commit -am "feat: export university records to exact 19-field jsonl"` after staging files.

**Task 13 Acceptance Gate:** Exported formal output is 100% schema-valid and source-traceable.

---

### Task 14: Pipeline Runner, CLI, Reporting, and Resumability E2E

**Files:**
- Create: `src/college_builder/pipeline/runner.py`
- Create: `src/college_builder/reporting/pilot_report.py`
- Create: `src/college_builder/cli.py`
- Create: `config/pilot.yaml`
- Create: `tests/e2e/test_pipeline_resume.py`
- Create: `tests/e2e/test_pipeline_report.py`

**Interfaces:**
- Produces CLI commands `college-builder run`, `resume`, `report`, `config validate`.

- [ ] **Step 1: Write E2E happy-path fixture**

Use 20 local records: valid, non-university, non-problem, missing-answer, missing-analysis, mismatch, duplicate, and ambiguous cases. Assert only valid rows reach `questions.jsonl`.

- [ ] **Step 2: Write interruption/resume E2E**

Force process interruption after normalization. Resume must reuse acquisition, normalization, and cached model decisions and continue at the next incomplete stage.

- [ ] **Step 3: Implement run fingerprint**

Compute from input/source revision + config hash + pipeline version. Reusing the same fingerprint must maximize cache reuse; changed prompt/model/config must invalidate only affected decisions.

- [ ] **Step 4: Implement pilot report metrics**

Report Raw, STEM, University, Problem, Answer Valid, Analysis Valid, Alignment PASS, After Dedup, Final Accepted; split by source/subject; include reject reasons, provider call counts, cache hit rate, cost fields, latency, and acceptance rate.

- [ ] **Step 5: Verify full quality suite**

Run:

```bash
pytest -q
ruff check src tests
mypy src
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src config tests
git commit -m "feat: add resumable pipeline and pilot reporting"
```

**Task 14 Acceptance Gate:** Local E2E proves acquisition-to-export behavior, fail-closed rejection, report counts, and resume/cache semantics.

---

### Task 15: 5K Dry Run Execution and Pilot Configuration Freeze

**Files:**
- Create: `src/college_builder/pipeline/pilot.py`
- Create: `docs/validation/5k-dry-run-report.md`
- Create: `config/pilot-5k-frozen.yaml`
- Add generated summary only: `artifacts/5k/run_report.json` (do not commit raw copyrighted dataset rows unless explicitly approved)

**Interfaces:**
- Produces a frozen, reviewed configuration for the 50K Pilot.

- [ ] **Step 1: Implement deterministic 5K sampling command**

Exact strata: Math 2,000; Physics 1,000; Statistics 1,000; MathOverflow 1,000. Seed must be recorded in run metadata.

- [ ] **Step 2: Execute 5K through the complete pipeline**

Run example:

```bash
college-builder run --config config/pilot.yaml --pilot-size 5000 --output artifacts/5k
```

The engineer must attach `run_report.json`, source revision, config hash, prompt versions, provider/model identities, and reject distribution to the review handoff.

- [ ] **Step 3: Audit failure clusters before changing thresholds**

Manually inspect representative rejects/accepts from each major reason. Fix parser/normalization/contract bugs first; do not lower precision thresholds merely to increase acceptance.

- [ ] **Step 4: Run Gold Regression**

All Gold fixtures and manually confirmed Gold records must pass intended gates. Any known answer mismatch in Accepted blocks Task 15.

- [ ] **Step 5: Freeze 50K Pilot config**

Commit the exact config/prompt versions used after reviewer approval as `config/pilot-5k-frozen.yaml`; no threshold/model/prompt change is allowed during the 50K run without invalidating the run and restarting from the affected cached stage.

- [ ] **Step 6: Verification commands**

Run: `pytest -q && ruff check src tests && mypy src` after all 5K-derived code fixes.

- [ ] **Step 7: Commit**

```bash
git add src config docs/validation tests
git commit -m "test: certify 5k dry run configuration"
```

**Task 15 Acceptance Gate:** 5K engineering dry run is stable, reproducible, Gold regression passes, and the reviewer explicitly accepts the frozen config before Task 16.

---

### Task 16: 50K Validation Pilot, Manual Audit Pack, and Scale Decision Gate

**Files:**
- Create: `docs/validation/50k-pilot-report.md`
- Create: `docs/validation/50k-manual-audit.md`
- Create: `docs/validation/scale-decision.md`
- Add generated summary only: `artifacts/50k/run_report.json`

**Interfaces:**
- Produces the only approved evidence for deciding public-data scale-out vs vendor supplementation.

- [ ] **Step 1: Execute approved 50K strata**

Required target composition: Math 20K; MathOverflow 5K; Statistics 10K; Physics 10K; other approved pilot source(s) 5K. If the fifth stratum is unavailable, stop and document the blocker; do not silently reallocate the 5K to Math.

- [ ] **Step 2: Produce full funnel report**

The report must include `University Rate`, `Problem Rate`, `Analysis Complete Rate`, `Answer Extractable Rate`, `Dedup Rate`, and `Final Acceptance Rate`, both overall and by source/subject.

- [ ] **Step 3: Build manual audit sample**

From Accepted: Math 500; Physics 500; Statistics 300; CS/other 300 when present; Electronics/other 300 when present. If a category has fewer Accepted than the requested sample, audit all of it and state the limitation. Gold regression is audited separately in full.

- [ ] **Step 4: Enforce Pilot quality thresholds**

Formal acceptance requires: known Question/Answer wrong match = 0 in audited Accepted; University Precision >=99%; Problem Precision >=99%; Analysis Valid Precision >=99%; exact 19-field validity =100%; provenance available =100%; answer source traceable =100%; exact duplicates=0.

- [ ] **Step 5: Compute public-source capacity forecast**

Use observed per-source acceptance and dedup rates against actually available raw-source counts; do not extrapolate from a single math-heavy subset. Document assumptions and confidence bounds/sensitivity cases.

- [ ] **Step 6: Apply the approved procurement Decision Gate**

- Forecast public Accepted `>=600K`: no main vendor purchase; vendor sample only for quality comparison.
- Forecast `350K-600K`: public sources remain primary; plan vendor supplementation of approximately `100K-200K` or measured gap plus safety margin.
- Forecast `<350K`: vendor sources become a primary input and require separate vendor validation before scale production.

- [ ] **Step 7: Run fresh final verification**

Run:

```bash
pytest -q
ruff check src tests
mypy src
college-builder report --run-id <50k-run-id>
```

Record the exact output/commit SHA/config hash in `50k-pilot-report.md`.

- [ ] **Step 8: Commit pilot evidence**

```bash
git add docs/validation config src tests
git commit -m "test: complete 50k university stem validation pilot"
```

**Task 16 Acceptance Gate:** STOP. Do not implement P1 mass acquisition, vendor scale adapters, or 500K production batches until the reviewer formally accepts Task 16 and issues the next implementation plan.

---

## Reviewer Handoff Requirements For Every Task

Every development handoff must include:

```text
Task number and name
Branch
Base SHA
HEAD SHA
Changed files
Exact test commands executed
Fresh test outputs / counts
Ruff result
MyPy result
Any network/provider integration evidence required by the Task
Known limitations
Explicit statement that no later Task work is included
```

The reviewer should reject a Task if its commit contains unrelated future-task implementation, if verification evidence is stale, or if fail-closed requirements were weakened to increase recall.

## Branch / Commit Discipline

Recommended execution model:

```text
main (approved design + plan)
  ↓
feature/task-01-domain-contracts
  ↓ review/accept
main or accepted integration baseline
  ↓
feature/task-02-university-profile
  ↓ review/accept
...
```

One Task = one reviewable logical commit (or a small TDD commit series squashed before acceptance). The next Task must branch from the latest Accepted Baseline, not from unaccepted work.

## Explicitly Deferred Until After Task 16 Acceptance

The following are intentionally not implementation Tasks in this plan because their correct shape depends on measured Pilot evidence:

- P1 mass StackExchange Electronics/CS/Chemistry/DSP/Engineering acquisition.
- Vendor production adapter specifics for Nexdata/Dataocean or another supplier.
- Exact raw-data target required to reach 500K Accepted.
- Final subject quotas if public-source availability materially differs from the design target.
- Production batch sizing adjustments beyond the approved default 50K/100K staged model.
- Final 500K release certification.

After Task 16 is accepted, the reviewer will issue a second `Production Scale Implementation Plan` grounded in measured acceptance, dedup, cost, source-capacity, and procurement results. This prevents building expensive scale-out machinery against assumptions that the Pilot is specifically intended to test.

## Plan Self-Review Result

- Spec coverage through the mandatory 50K Pilot: covered by Tasks 1-16.
- No AI-generated formal question/answer/analysis path is permitted.
- Exact 19-field output and University profile are covered before ingestion scale work.
- Provenance, license metadata, cache, resume, provider identity, evidence, dedup, reporting, Gold regression, manual audit, and Decision Gate are all assigned to concrete Tasks.
- Scale production is deliberately deferred rather than left ambiguous; the approved design requires a data-driven procurement/scale decision after Pilot.
- No Task may begin scale-out before Task 16 formal acceptance.
