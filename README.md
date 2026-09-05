# University STEM Dataset Builder

Precision-first pipeline for producing source-grounded university STEM problem/solution records.

This repository follows the approved Pilot implementation plan. The current bootstrap establishes only:

- immutable `RawSourceRecord` provenance snapshots;
- typed normalization/question/evidence domain contracts;
- fail-closed, versioned pipeline configuration;
- Python 3.12+ project and quality-tool configuration.

Formal question, answer, and analysis content must always originate from source data. Model providers may later classify or verify records, but provider DTOs and generated content are not domain contracts and may not replace source truth.

## Task 1 quality gate

```bash
pytest tests/unit/domain tests/unit/test_config.py -q
ruff check src tests
mypy src
```
