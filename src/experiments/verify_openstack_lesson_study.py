"""Recompute saved study outcomes, context accounting and hosted provenance."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from src.agents.task_agent import llm_task_agent
from src.environment.evaluator import evaluate_answer
from src.experiments.run_openstack_lesson_study import (
    ROOT, CONDITIONS, LEVELS, aggregate, completed_pairs, digest, manager_for, manage,
    read_jsonl, task_from, write_json,
)


def check_ledger(row: dict) -> None:
    ledger = row['ledger']
    active = sorted((v for v in ledger['items'] if v['active']), key=lambda v: v['step'])
    context = '\n'.join(f"[{v['item_id']}] {v['content']}" for v in active)
    assert context == ledger['active_context'], 'Saved active view does not match ledger items.'
    assert row['active_context_words'] == max(1, len(context.split()))
    trace = row['trajectory']
    assert row['step_sum_active_words'] == sum(r['after_management_words'] for r in trace)
    assert row['peak_active_context_words'] == max(max(r['before_management_words'], r['after_management_words']) for r in trace)
    assert trace[-1]['after_management_words'] == row['active_context_words']


def verify(output: Path) -> dict:
    manifest = json.loads((output / 'manifest.json').read_text())
    for relative, expected in manifest['source_sha256'].items():
        assert digest(ROOT / relative) == expected, f'Study source changed: {relative}'
    assert digest(output / 'data/selected_cases.json') == manifest['selected_cases_sha256']
    selected = json.loads((output / 'data/selected_cases.json').read_text())
    assert len(selected) == 24
    selection = {(r['dataset'], r['task_id']): r for r in selected}
    assert len(selection) == 24
    for source in ('openstack', 'synthetic'):
        original = json.loads((output / f'lessons/source-{source}.json').read_text())
        eligible = [r for r in original if r['status'] == 'transferable']
        imported = json.loads((output / f'lessons/imported-{source}.json').read_text())
        for field in ('operation', 'trigger_tags', 'support_count', 'status', 'instruction'):
            assert [r[field] for r in imported] == [r[field] for r in eligible], 'Import lost or invented evidence.'
    offline = read_jsonl(output / 'offline/episodes.jsonl.gz')
    raw_cases = {}
    for seed in (7, 19, 41):
        raw_cases.update({(seed, r['task_id']): r for r in read_jsonl(ROOT / f'results/transfer-validation/seed-{seed}/data/test.jsonl')})
    expected_conditions = {'full_context', 'empty_store', 'openstack_any', 'openstack_all', 'synthetic_any', 'synthetic_all'}
    identities = set()
    for row in offline:
        identity = (row['seed'], row['task_id'], row['condition'], row['noise_level'])
        assert identity not in identities, 'Duplicate offline episode.'
        identities.add(identity)
        source = raw_cases[(row['seed'], row['task_id'])]
        assert row['targets'] == source['targets'] and row['question'] == source['question']
        assert row['condition'] in expected_conditions and row['noise_level'] in LEVELS
        originals = sorted((r for r in row['ledger']['items'] if r['immutable']), key=lambda r: r['step'])
        unchanged = [r for r in originals if not r['content'].startswith('Irrelevant study observation')]
        assert [(r['content'], r['tags']) for r in unchanged] == [(r['content'], r['tags']) for r in source['observations']]
        assert len(originals) - len(unchanged) == row['noise_level']
        check_ledger(row)
        import re
        visible = dict(re.findall(r'FACT key=(k_[a-z0-9_]+) value=(v_[a-z0-9_]+)', row['ledger']['active_context']))
        requested = json.loads(row['question'])['keys']
        expected_answer = {key: visible.get(key) for key in requested}
        assert json.loads(row['answer']) == expected_answer, 'Offline answer cannot be reproduced from visible facts.'
        assert row['evaluation']['task_success'] == int(expected_answer == row['targets'])
        irrelevant = sum(r['active'] and r['immutable'] and r['content'].startswith('Irrelevant study observation') for r in originals)
        assert row['irrelevant_verbatim_items'] == irrelevant
    expected_grid = {(seed, task_id, condition, level) for seed, task_id in raw_cases
                     for condition in expected_conditions for level in LEVELS}
    assert identities == expected_grid, 'Missing offline cases or conditions.'
    offline_summary = json.loads((output / 'offline/summary.json').read_text())
    assert offline_summary['aggregate'] == aggregate(offline, ('condition', 'noise_level'))
    assert offline_summary['by_family'] == aggregate(offline, ('condition', 'noise_level', 'family'))
    assert offline_summary['api_calls'] == 0

    hosted = read_jsonl(output / 'hosted/episodes.jsonl')
    for row in hosted:
        original = selection[(row['dataset'], row['task_id'])]
        if row['dataset'] == 'openstack':
            observations = json.loads((output / f'hosted/observations/{row["task_id"]}.json').read_text())
            lines = [r for r in original['raw_text'].splitlines() if r.strip()]
            assert [r['content'] for r in observations[:-1]] == lines
            assert observations[-1]['content'] == original['question'] and observations[-1]['tags'] == ['trigger']
        else:
            observations = original['observations']
        assert row['task']['observations'] == observations
        for field in ('question', 'required_outputs', 'forbidden_outputs', 'synonym_groups'):
            assert row['task'][field] == original[field]
        task = task_from(row['task'])
        simulated = manage(task, manager_for(output, row['condition']))
        for field in ('ledger', 'trajectory', 'active_context_words', 'step_sum_active_words',
                      'peak_active_context_words', 'matched_rule_ids', 'lesson_override_unique_items'):
            assert simulated[field] == row[field], f'Hosted policy trace differs: {field}'
        check_ledger(row)
        assert evaluate_answer(task, row['answer']).__dict__ == row['evaluation']
        cached = json.loads((output / f'hosted/cache/{row["model_call"]["prompt_sha256"]}.json').read_text())
        assert cached['text'] == row['answer'], 'Hosted answer differs from provider response.'

        class Capture:
            def complete(self, prompt):
                self.prompt = prompt
                return row['answer']

        capture = Capture()
        llm_task_agent(task, row['ledger']['active_context'], capture)
        body = {'model': cached['model'], 'temperature': 0.0, 'max_tokens': 512,
                'messages': [{'role': 'user', 'content': capture.prompt}]}
        assert hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == row['model_call']['prompt_sha256']
    complete, pairs = completed_pairs(hosted)
    hosted_complete = False
    if (output / 'hosted/summary.json').exists():
        hs = json.loads((output / 'hosted/summary.json').read_text())
        assert hs['aggregate'] == aggregate(complete)
        assert hs['paired_outcomes'] == pairs
        assert hs['paired_cases'] == len(complete) // 4
        calls = read_jsonl(output / 'hosted/api_calls.jsonl')
        assert hs['api_attempts'] == len(calls) <= 110
        hosted_complete = len(complete) == 96 and hs['status'] == 'complete'
    record = {'valid': True, 'offline_episodes_checked': len(offline), 'offline_grid_complete': True,
              'hosted_episodes_checked': len(hosted), 'hosted_paired_cases': len(complete) // 4,
              'hosted_complete': hosted_complete, 'source_hashes_verified': True,
              'imports_preserve_evidence': True, 'hosted_prompts_and_provider_answers_verified': True,
              'context_metrics_recomputed': True, 'verification_implementation_sha256': digest(Path(__file__))}
    write_json(output / 'verification.json', record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/openstack-lesson-study')
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))


if __name__ == '__main__':
    main()
