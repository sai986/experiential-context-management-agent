"""Recompute study metrics and check evidence from artifacts, without a model call."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import re

from src.constants import SUSPECT_TAGS
from src.environment.split_validation import audit_splits

ROOT = Path(__file__).resolve().parents[2]
TEST_CONDITIONS = {
    'full_context', 'empty_store', 'learned_a', 'legacy_transfer_b',
    'trusted_transfer_b', 'without_archive_lessons', 'without_compression_lessons',
}


def active_context(items: list[dict]) -> str:
    return '\n'.join(f"[{item['item_id']}] {item['content']}" for item in
                     sorted((item for item in items if item['active']), key=lambda item: item['step']))


def visible_answer(question: str, context: str) -> dict:
    facts = dict(re.findall(r'FACT key=(k_[a-z0-9_]+) value=(v_[a-z0-9_]+)', context))
    return {key: facts.get(key) for key in json.loads(question)['keys']}


def read_jsonl(path):
    content = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    return [json.loads(line) for line in content.decode().splitlines() if line.strip()]


def verify_artifacts(folder: Path) -> dict:
    manifest = json.loads((folder / 'manifest.json').read_text())
    expected = json.loads((folder / 'summary.json').read_text())
    if manifest['protocol_version'] != 1 or set(expected['aggregate_test']) != TEST_CONDITIONS:
        raise ValueError('Unexpected protocol version or test conditions.')
    for name, digest in manifest['implementation_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Implementation differs from recorded run: {name}')
    for name, digest in manifest['generated_dataset_sha256'].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Dataset differs from recorded run: {name}')
    for name, digest in manifest['teammate_dataset_sha256'].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Teammate dataset differs from recorded run: {name}')
    teammate_audits = json.loads((folder / 'teammate-dataset-audit.json').read_text())
    for domain, saved_audit in teammate_audits.items():
        splits = {name: read_jsonl(ROOT / 'data' / domain / f'{name}.jsonl')
                  for name in ('train', 'validation', 'test')}
        if audit_splits(splits) != saved_audit:
            raise ValueError('Teammate dataset audit differs from source inputs.')
    aggregate = defaultdict(lambda: {'n': 0, 'successes': 0, 'words': 0})
    episodes_checked = replays_checked = null_checks = 0
    for seed in manifest['seeds']:
        seed_folder = folder / f'seed-{seed}'
        splits = {name: read_jsonl(seed_folder / 'data' / f'{name}.jsonl')
                  for name in ('train', 'validation', 'test')}
        if not audit_splits(splits)['valid']:
            raise ValueError('Cross-split input/ID overlap.')
        cases = {row['task_id']: row for rows in splits.values() for row in rows}
        test_ids = {row['task_id'] for row in splits['test']}
        records = read_jsonl(seed_folder / 'raw' / 'episodes.jsonl.gz')
        training = {r['task_id']: r for r in records if r['condition'] == 'train'}
        conditions = TEST_CONDITIONS
        required_pairs = {(row['task_id'], condition) for row in splits['test'] for condition in conditions}
        required_pairs.update((row['task_id'], 'train') for row in splits['train'])
        required_pairs.update((row['task_id'], 'validation_a') for row in splits['validation'])
        seen = set()
        for record in records:
            identity = (record['task_id'], record['condition'])
            if identity in seen:
                raise ValueError(f'Duplicate episode: {identity}')
            seen.add(identity)
            targets = cases[record['task_id']]['targets']
            success = int(json.loads(record['answer']) == targets)
            if (record['targets'] != targets or record['evaluation']['success'] != success
                    or record['family'] != cases[record['task_id']]['family']):
                raise ValueError('Saved episode targets or grading disagree with the dataset.')
            context = active_context(record['ledger']['items'])
            words = max(1, len(context.split()))
            if context != record['ledger']['active_context'] or words != record['active_context_words']:
                raise ValueError('Ledger and reported context words disagree.')
            question = cases[record['task_id']]['question']
            if record['question'] != question or json.loads(record['answer']) != visible_answer(question, context):
                raise ValueError('Saved answer differs from the facts visible in the ledger.')
            episodes_checked += 1
            if record['condition'] == 'train' and not success and 'null_intervention_answer' not in record:
                raise ValueError('Missing null intervention for a failed training episode.')
            if 'null_intervention_answer' in record:
                if record['null_intervention_answer'] != record['answer']:
                    raise ValueError('Null intervention changed the answer.')
                null_checks += 1
            if record['task_id'] in test_ids:
                if record['condition'] not in conditions:
                    raise ValueError('Unknown test condition.')
                metrics = aggregate[record['condition']]
                metrics['n'] += 1
                metrics['successes'] += success
                metrics['words'] += words
        if seen != required_pairs:
            raise ValueError('Missing episodes or episodes assigned to the wrong split/condition.')
        replay_rows = read_jsonl(seed_folder / 'raw' / 'replays.jsonl')
        evidence, independent_episodes = Counter(), defaultdict(set)
        for row in replay_rows:
            targets = cases[row['task_id']]['targets']
            episode = training[row['task_id']]
            decision = row['decision']
            if (decision not in episode['ledger']['decisions']
                    or row['target_decision_id'] != decision['decision_id']
                    or row['original_answer'] != episode['answer']
                    or row['episode_id'] != episode['ledger']['episode_id']):
                raise ValueError('Replay decision or original answer differs from the training ledger.')
            replay_items = deepcopy(episode['ledger']['items'])
            indexed = {item['item_id']: item for item in replay_items}
            if decision['operation'] == 'COMPRESS' and row['intervention_type'] == 'RESTORE_ORIGINAL':
                view = indexed[decision['output_item_id']]
                view['active'] = False
                indexed[view['parent_item_ids'][0]]['active'] = True
            elif decision['operation'] == 'ARCHIVE' and row['intervention_type'] == 'UNARCHIVE_KEEP':
                for identifier in decision['input_item_ids']:
                    indexed[identifier]['active'] = True
            else:
                raise ValueError('Replay intervention does not match its target operation.')
            if json.loads(row['counterfactual_answer']) != visible_answer(
                    episode['question'], active_context(replay_items)):
                raise ValueError('Counterfactual answer differs from the reconstructed intervention.')
            original = int(json.loads(row['original_answer']) == targets)
            alternative = int(json.loads(row['counterfactual_answer']) == targets)
            if row['original_score'] != original or row['counterfactual_score'] != alternative:
                raise ValueError('Replay scores disagree with exact grading.')
            if row['label'] != ('HARMFUL' if alternative > original else 'NEUTRAL'):
                raise ValueError('Replay label disagrees with the two recorded answers.')
            if row['label'] == 'HARMFUL':
                items = {item['item_id']: item for item in training[row['task_id']]['ledger']['items']}
                tags = set(tag for identifier in row['decision']['input_item_ids'] for tag in items[identifier]['tags'])
                operation = row['decision']['operation']
                pattern = (operation, tuple(sorted(tags if operation == 'ARCHIVE' else tags & SUSPECT_TAGS)))
                evidence[pattern] += 1
                independent_episodes[pattern].add(row['task_id'])
            replays_checked += 1
        seed_summary = json.loads((seed_folder / 'summary.json').read_text())
        if seed_summary['replay_labels'] != dict(Counter(r['label'] for r in replay_rows)):
            raise ValueError('Seed replay counts disagree with raw evidence.')
        if (seed_summary['training_episodes'] != len(training)
                or seed_summary['training_replays'] != len(replay_rows)
                or seed_summary['null_intervention_checks'] != sum('null_intervention_answer' in r for r in training.values())):
            raise ValueError('Seed episode, replay or null-intervention counts disagree with raw evidence.')
        source_rules = json.loads((seed_folder / 'lessons/agent-a.json').read_text())
        if {(l['operation'], tuple(sorted(l['trigger_tags']))) for l in source_rules} != set(evidence):
            raise ValueError('Saved lesson patterns differ from harmful training evidence.')
        for lesson in source_rules:
            pattern = (lesson['operation'], tuple(sorted(lesson['trigger_tags'])))
            if (lesson['support_count'] != evidence[pattern]
                    or lesson['support_count'] != len(independent_episodes[pattern])):
                raise ValueError('Lesson support does not match independent harmful training episodes.')
        transferred = json.loads((seed_folder / 'lessons/trusted-b.json').read_text())
        qualified = {(l['operation'], tuple(sorted(l['trigger_tags'])), l['instruction']): l
                     for l in source_rules if l['status'] == 'transferable'}
        if len(transferred) != len(qualified):
            raise ValueError('Transferred rule count differs from eligible source rules.')
        for lesson in transferred:
            source = qualified[(lesson['operation'], tuple(sorted(lesson['trigger_tags'])), lesson['instruction'])]
            if lesson['support_count'] != source['support_count'] or lesson['status'] != 'transferable':
                raise ValueError('Transferred evidence/status differs from the qualifying source.')
        if source_rules != seed_summary['learned_lessons'] or transferred != seed_summary['transferred_b']:
            raise ValueError('Seed summary and saved lesson stores disagree.')
    computed = {}
    for condition, metrics in aggregate.items():
        saved = expected['aggregate_test'][condition]
        n, wins, words = metrics['n'], metrics['successes'], metrics['words']
        computed[condition] = {'n': n, 'successes': wins, 'success_rate': wins / n,
                               'avg_active_context_words': words / n}
        if saved['n'] != n or saved['successes'] != wins or abs(saved['avg_active_context_words'] - words / n) > 1e-9:
            raise ValueError(f'Summary differs from raw episodes: {condition}')
        if abs(saved['success_rate'] - wins / n) > 1e-9:
            raise ValueError(f'Success rate differs from raw episodes: {condition}')
    return {'valid': True, 'api_calls': 0, 'episodes_checked': episodes_checked,
            'replays_checked': replays_checked, 'null_interventions_checked': null_checks,
            'recomputed_test': computed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path, nargs='?', default=ROOT / 'results/transfer-validation')
    args = parser.parse_args()
    print(json.dumps(verify_artifacts(args.folder), indent=2))


if __name__ == '__main__':
    main()
