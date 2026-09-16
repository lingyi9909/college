from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path('/tmp/task16c-small200')
SAMPLE = ROOT / 'sample.jsonl'
MANIFEST = ROOT / 'sample_manifest.json'

# Fixed high-value cohort selected from the frozen Small-200 sample.
# Includes prior Gate4/Gate5 failures, known positive controls, and representative rejects.
RECORD_IDS = (
    'raw_stackmathqa_dca4a5b486d798c00e8919583156cc2e173bdd33b5a3c6a2840170730c33fbcb',
    'raw_stackmathqa_a9930e918a91825a4e50f4993efa22bb1fe5211d93470e7391b4eacbd5c19bf0',
    'raw_stackmathqa_2bb32b74998c7b36c6ee9f4c3f081070de941ba44062249109fcb7b6323768f7',
    'raw_stackmathqa_03cbf3eadc36867f228df219b264d2a58bd7fb0159f18ec1a813b93f662efefc',
    'raw_stackmathqa_133cd9d04fc39fc34b38416d83df49650ddc6ccc777513c2d06c23d7f092c634',
    'raw_stackmathqa_89ee59e3914c4d46da47b79f765969e55bb65df3c8d88d32ef2fd01eea95202a',
    'raw_stackmathqa_68cbc45120c0ceeff13ef8ac36d838c401569d228d8233590d5c1505586da333',
    'raw_hf_ce3a79a1a9e6952791099fbd98c03e3fcbd1cd7c07bb3bcefc95fcfa3d538be2',
    'raw_stackmathqa_905f488da8dc4ced22497e5fad1f6926c6ac1f4a382883f74dcbb90f9ace879d',
    'raw_stackmathqa_eef4f43da9e4dbfabaeb3525fac7b953ae33f5b3b63367226e179246c3d4285f',
    'raw_stackmathqa_7e8ec169d54e980ffd309e22e2d01df93c8acafcfc8f6d75501bdcca8468247e',
    'raw_hf_78c1cf2722e9b4b6f5d6958b6e9cc092e21bb9969c428755c0f3081cd915e43b',
    'raw_hf_81f8a33fccbf0f4a63e5b4c6ddde5259320fa6e1f2e13707bb19bb05b2d559d9',
    'raw_hf_296f16c32377d59d31abbe5819cdf49d78f4c595abc03b09b1368b4c8ba4a277',
    'raw_hf_d401f6a0c83d3f44e884ad55b22be8e89a96dca8a90ff738c9dc5c66d3a464de',
    'raw_hf_796e6695a1e9b7701c3d04dfd29a7ea6e728c36a56ba1681ec1ab100cd775389',
    'raw_stackmathqa_050c44657009a3aed677c60669ad925f7cf1795923f4ee2306a9b96896841a52',
    'raw_stackmathqa_056382e545690e64d8de19f39713308e23ec4e43adfcb01652a6c5f768a64700',
    'raw_hf_000ccc4c07072d1a9297b72f32c3dbf1b7f8fc91e83575ada4751047839dcbe6',
    'raw_hf_1cfc249f8e169cae620a9f1b801fe19d5003c74f68f1b2f65c023bfa6016a176',
)


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def main() -> None:
    rows = [json.loads(line) for line in SAMPLE.read_text(encoding='utf-8').splitlines() if line.strip()]
    by_id = {row['record_id']: row for row in rows}
    missing = [record_id for record_id in RECORD_IDS if record_id not in by_id]
    if missing:
        raise AssertionError(f'missing fixed Small-20 records: {missing}')
    selected = [by_id[record_id] for record_id in RECORD_IDS]
    if len(selected) != 20 or len({row['record_id'] for row in selected}) != 20:
        raise AssertionError('Small-20 cohort must contain exactly 20 unique records')

    payload = b''.join(canonical_json(row) + b'\n' for row in selected)
    payload_sha = hashlib.sha256(payload).hexdigest()
    identity_sha = hashlib.sha256(canonical_json([(row['record_id'], row['raw_sha256']) for row in selected])).hexdigest()
    SAMPLE.write_bytes(payload)

    old = json.loads(MANIFEST.read_text(encoding='utf-8'))
    source_counts = Counter(str(row['source_dataset']) for row in selected)
    manifest = {
        'schema_version': 'task16c-small20-regression-v1',
        'selection_frozen_before_model_calls': True,
        'seed': old['seed'],
        'total': 20,
        'cohort': 'fixed-high-value-regression',
        'source_dataset_counts': dict(sorted(source_counts.items())),
        'sample_payload_sha256': payload_sha,
        'record_identity_sha256': identity_sha,
        'parent_small200_payload_sha256': old['sample_payload_sha256'],
        'records': [
            {
                'ordinal': index,
                'record_id': row['record_id'],
                'source_dataset': row['source_dataset'],
                'source_id': row['source_id'],
                'raw_sha256': row['raw_sha256'],
            }
            for index, row in enumerate(selected, start=1)
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('TASK16C_SMALL20_SAMPLE=PASS')
    print('TASK16C_SMALL20_SAMPLE_SHA256=' + payload_sha)
    print('TASK16C_SMALL20_IDENTITY_SHA256=' + identity_sha)


if __name__ == '__main__':
    main()
