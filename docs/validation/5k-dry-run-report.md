# Task 15 — 5K Engineering Dry Run Report

Status: **BLOCKED_EXTERNAL_PROVIDER**

The approved deterministic sampling plan is frozen at seed **20260910** with exact strata: Math 2,000; Physics 1,000; Statistics 1,000; MathOverflow 1,000.

Source dataset: `math-ai/StackMathQA`  
Source revision authority: `git:13239ec8c8078bc8dfb43e1383069eb9be000ff7`  
Frozen config version: `pilot-5k-frozen-v1`  
Frozen config SHA-256: `0c13691fce9a8300cf159db55fc2ae3ba0135d8abb123605686697aab54341b4`

The complete model-backed 5K pipeline cannot truthfully be executed in repository CI because `OPENAI_COMPATIBLE_BASE_URL` and matching provider credentials are not available there. No acceptance, reject-distribution, cost, latency, or quality numbers have been fabricated.

Gold/source regression remains part of the fresh repository test suite. The frozen config preserves the approved Pilot precision thresholds and prompt versions. The 50K Pilot must not begin until a real 5K run using this frozen configuration is attached and independently accepted.
