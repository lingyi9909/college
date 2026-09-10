# Task 15 — 5K Engineering Dry Run Report

Status: **BLOCKED_EXTERNAL_PROVIDER**

The approved deterministic sampling plan is frozen at seed **20260910** with exact strata: Math 2,000; Physics 1,000; Statistics 1,000; MathOverflow 1,000.

Source dataset: `math-ai/StackMathQA`  
Source revision authority: `git:13239ec8c8078bc8dfb43e1383069eb9be000ff7`  
Frozen config version: `pilot-5k-frozen-v1`  
Frozen config SHA-256: `0c13691fce9a8300cf159db55fc2ae3ba0135d8abb123605686697aab54341b4`

## Engineering preflight

The Task 15 implementation now provides an executable, bounded-memory sampling path for the four official StackMathQA source files. Sampling is deterministic by seed and stable record identity, while the sample identity also binds each selected record's `raw_sha256`. The safe manifest records the immutable source revision, the 5,000 selected record IDs, their raw hashes, and a SHA-256 for each staged source file.

The fresh Task 15 E2E suite materializes an exact 5,000-row sample using the real StackMathQA adapter contract and checks the required 2,000 / 1,000 / 1,000 / 1,000 source-site distribution. Raw sampled rows are runtime-only and are not committed to the repository.

The Gold regression runs representative STEMQ, SciBench, and CFE records through `GoldDatasetAdapter`, the complete `PipelineRunner`, all intended gate stages, and exact 19-field export. Gold source language is explicit source configuration provenance rather than an inferred default.

## Reproducible commands

```text
college-builder pilot sample --source-root <staged-stackmathqa> --source-revision git:13239ec8c8078bc8dfb43e1383069eb9be000ff7 --seed 20260910 --output artifacts/5k/sample.jsonl --manifest artifacts/5k/sample_manifest.json
college-builder run --input artifacts/5k/sample.jsonl --config config/pilot-5k-frozen.yaml --workspace artifacts/5k/workspace --output artifacts/5k/output
```

The frozen configuration preserves the approved Pilot thresholds and prompt versions; no threshold has been lowered to improve acceptance.

## Remaining execution blocker

The complete **real model-backed 5K run has not been executed** because the repository CI environment does not provide a usable `OPENAI_COMPATIBLE_BASE_URL` and matching provider credentials. Therefore this report deliberately does **not** claim or invent final acceptance count, rejection distribution, cost, latency, model-quality, or answer-correctness metrics for the real 5K sample.

Task 15 must remain blocked until the exact frozen configuration is run against the staged immutable StackMathQA snapshot with the configured classifier/verifier providers and that resulting run report is independently reviewed. The 50K Pilot must not begin before that acceptance.
