# Task 15 — 5K Engineering Dry Run Report

Status: **BLOCKED_STABLE_PROVIDER_IDENTITY**

The approved deterministic sampling plan is frozen at seed **20260910** with exact strata: Math 2,000; Physics 1,000; Statistics 1,000; MathOverflow 1,000.

Source dataset: `math-ai/StackMathQA`  
Source revision authority: `git:13239ec8c8078bc8dfb43e1383069eb9be000ff7`  
Real 5K sample SHA-256: `e9d7b957a092114da35dbb839319a08ab555692a8a06863faf2b1980c2a681c5`  
Frozen config version: `pilot-5k-frozen-v1`  
Frozen config SHA-256: `0c13691fce9a8300cf159db55fc2ae3ba0135d8abb123605686697aab54341b4`

## Engineering preflight

The Task 15 implementation provides an executable, bounded-memory sampling path for the four official StackMathQA source files. Sampling is deterministic by seed and stable record identity, while the sample identity also binds each selected record's `raw_sha256`. The safe manifest records the immutable source revision, the 5,000 selected record IDs, their raw hashes, and a SHA-256 for each staged source file.

The real upstream sampling certification scanned 1,957,006 StackMathQA rows and produced the exact 5,000-row strata required by the approved plan. Raw sampled rows remain runtime-only and are not committed to the repository.

The Gold regression runs representative STEMQ, SciBench, and CFE records through `GoldDatasetAdapter`, the complete `PipelineRunner`, all intended gate stages, and exact 19-field export. Gold source language is explicit source configuration provenance rather than an inferred default.

## Reproducible commands

```text
college-builder pilot sample --source-root <staged-stackmathqa> --source-revision git:13239ec8c8078bc8dfb43e1383069eb9be000ff7 --seed 20260910 --output artifacts/5k/sample.jsonl --manifest artifacts/5k/sample_manifest.json
college-builder run --input artifacts/5k/sample.jsonl --config config/pilot-5k-frozen.yaml --workspace artifacts/5k/workspace --output artifacts/5k/output
```

The frozen configuration preserves the approved Pilot thresholds and prompt versions; no threshold has been lowered to improve acceptance.

## Provider execution investigation

The repository CI environment does not contain the traditional `OPENAI_COMPATIBLE_BASE_URL`, `OPENAI_COMPATIBLE_API_KEY`, and fixed model configuration required for a direct external provider run. A CI-only Copilot proxy was therefore investigated without changing the production provider contract.

A real four-strata micro run proved that the existing production path can execute end to end through:

```text
StackMathQA real source record
  -> production PipelineRunner
  -> OpenAICompatibleStructuredModelProvider
  -> CI-only localhost proxy
  -> GitHub Copilot CLI auto routing
  -> ModelDecision
  -> production gates
```

Evidence run: `34445107273` against code SHA `ba7056219064c529ea0d7eabd898a901fffdc6e4`.

Observed non-content metrics:

```text
raw_count              4
accepted_count         0
provider_errors        0
proxy_nonzero_calls    0
proxy_model_calls      6

rejects:
ANALYSIS_TOO_SHALLOW         1
UNIVERSITY_LEVEL_UNCERTAIN   2
VERIFIER_NOT_INDEPENDENT     1
```

The zero Accepted result is not treated as a defect by itself because the pipeline is precision-first and fail-closed. The important result is that the production provider boundary and real model call path work without provider errors.

However, Copilot `auto` is not an auditable frozen model identity and both temporary roles used the same auto alias, so this run cannot satisfy the Task 15 provider freeze or independent-verifier acceptance requirements.

### Explicit fixed-model diagnostics

A second CI-only diagnostic used GitHub Copilot CLI `1.0.83` and tested the exact model strings documented by the current Copilot CLI reference:

```text
claude-sonnet-4.6
gpt-5.4
claude-haiku-4.5
gpt-5.3-codex
gemini-3.1-pro-preview
gemini-3.5-flash
gemini-3.6-flash
gemini-3.7-flash
mai-code-1-flash
```

Evidence run: `34448910429`.

All nine explicit selectors failed consistently before model inference with:

```text
Error: Model "<model>" from --model flag is not available.
```

Explicit model success count: `0`.

This evidence is sufficient to reject the hypothesis that the current repository Actions execution context can be turned into a valid frozen dual-model provider merely by choosing a documented Copilot CLI model string. It does not establish which account plan, repository policy, or GitHub-side entitlement causes explicit selection to be unavailable, so this report does not speculate about that cause.

## Remaining execution blocker

The complete **real model-backed 5K run has not been executed** because Task 15 still lacks an execution environment that exposes at least two stable, explicit model identities for the existing `openai_compatible` provider contract.

The production implementation already supports distinct classifier and verifier model names on the same OpenAI-compatible base URL. The classification gates enforce independence using `(provider, model)` execution identity. Therefore no production provider-enum change and no weakening of `VERIFIER_NOT_INDEPENDENT` is justified by this blocker.

Task 15 can continue only when one of the following is true:

1. a real OpenAI-compatible endpoint reachable by the execution environment is configured and exposes at least two stable explicit model identities for classifier and verifier; or
2. the GitHub Copilot execution entitlement/policy is changed so at least two explicit CLI models are actually selectable and can be pinned reproducibly through the existing CI-only proxy.

Until then:

- Copilot auto routing must not be represented as a frozen model identity;
- the classifier/verifier independence gate must not be bypassed;
- thresholds must not be reduced to increase acceptance;
- no real 5K acceptance, reject-distribution, cost, latency, or quality metrics may be invented;
- Task 16 must not begin.

Task 15 therefore remains blocked at provider identity execution, not at sampling, pipeline wiring, Gold regression, or schema/export implementation.
