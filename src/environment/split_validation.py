"""Check exact runtime-input overlap before claiming held-out evaluation."""
from __future__ import annotations

import hashlib
import json
from itertools import combinations


def audit_splits(splits: dict[str, list[dict]]) -> dict:
    """Return counts and cross-split duplicate IDs/inputs with concrete examples.

    Supports incident JSONL and controlled-task records. Input comparison omits
    IDs, grading targets and source labels. This checks exact duplicates, not
    shared log windows, template similarity or semantic leakage.
    """
    fingerprints = {}
    ids = {}
    for name, rows in splits.items():
        ids[name] = [row.get('task_id', row.get('incident_id')) for row in rows]
        inputs = []
        for row in rows:
            if 'raw_text' in row:
                value = [row['raw_text'], row['question']]
            else:
                value = [row['question'], row['observations']]
            inputs.append(hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest())
        fingerprints[name] = inputs
    pairs = {}
    for left, right in combinations(splits, 2):
        indexed = dict(zip(fingerprints[left], ids[left]))
        examples = [{'source_id': indexed[fp], 'destination_id': identifier}
                    for fp, identifier in zip(fingerprints[right], ids[right]) if fp in indexed]
        pairs[f'{left}->{right}'] = {
            'duplicate_ids': sorted(set(ids[left]) & set(ids[right])),
            'duplicate_destination_inputs': len(examples),
            'examples': examples[:5],
        }
    return {
        'counts': {name: len(rows) for name, rows in splits.items()},
        'cross_split_pairs': pairs,
        'within_split_duplicate_ids': {
            name: len(values) - len(set(values)) for name, values in ids.items()},
        'valid': not any(p['duplicate_ids'] or p['duplicate_destination_inputs'] for p in pairs.values())
                 and all(len(v) == len(set(v)) for v in ids.values()),
        'scope': 'Exact inputs and IDs; not a semantic/template/source-window independence proof.',
    }
