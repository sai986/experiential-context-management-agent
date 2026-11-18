"""Reproducible offline experiment for lesson ablation and evidence-preserving transfer.

Run: python -m src.experiments.run_transfer_validation
No model client, environment-file loading or network calls are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import gzip
import hashlib
import html
import json
from pathlib import Path
import platform
import random
import re
import subprocess

from src.attribution.replay import (create_lessons_from_replays, find_suspect_archives,
                                    find_suspect_compressions, replay_archive_keep,
                                    replay_restore_original)
from src.context.ledger import ContextLedger
from src.context.manager import FullContextManager, LessonAwareManager, NonLearningManager
from src.environment.split_validation import audit_splits
from src.environment.task_generator import ControlledTask, Observation
from src.lessons.lesson_store import LessonStore

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ('critical_compression', 'delayed_compression', 'false_archive',
            'benign_recent', 'info_mistag', 'joint_loss')
FACT = re.compile(r'FACT key=(k_[a-z0-9_]+) value=(v_[a-z0-9_]+)')


def frozen_executor(task: ControlledTask, context: str) -> str:
    """Read only the question and visible facts, never targets, tags or family.

    This is a generic deterministic fact reader, not an LLM or an NLP model.
    Missing values produce null; the evaluator must not supply the answer.
    """
    requested = json.loads(task.question)['keys']
    visible = dict(FACT.findall(context))
    return json.dumps({key: visible.get(key) for key in requested}, sort_keys=True)


def truncate_compressor(content: str) -> str:
    """One explicitly lossy compressor shared by every managed condition."""
    return ' '.join(content.split()[:7])


def generate_cases(seed: int, split: str, per_family: int) -> list[dict]:
    """New IDs, keys and values per split; six fixed, explicitly engineered families."""
    rng = random.Random(f'{seed}:{split}')
    cases = []
    for family in FAMILIES:
        for index in range(per_family):
            identity = f'{split}_{seed}_{family}_{index:03d}'
            suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
            count = 2 if family == 'joint_loss' else 1
            targets = {f'k_{suffix}_{n}': f'v_{rng.getrandbits(64):016x}' for n in range(count)}
            tags = {
                'critical_compression': ['critical', 'hard_constraint'],
                'delayed_compression': ['critical', 'delayed_dependency'],
                'false_archive': ['critical', 'distractor'],
                'benign_recent': ['preference'],
                'info_mistag': ['info'],
                'joint_loss': ['negation'],
            }[family]
            target_obs = [Observation(0, f'Recorded configuration for workload {suffix} at checkpoint '
                                        f'{rng.randrange(1000)} FACT key={key} value={value}', list(tags))
                          for key, value in targets.items()]
            noise = [Observation(0, f'Background probe {suffix} sample {n} reports healthy '
                                   f'queue depth {rng.randrange(100)} and heartbeat latency '
                                   f'{rng.randrange(100)} milliseconds.', ['info'])
                     for n in range(rng.randrange(8, 13))]
            if family == 'benign_recent':
                old = Observation(0, f'Recorded configuration for old workload {suffix} '
                                      f'at checkpoint 0 FACT key=k_old_{suffix} value=v_old_{suffix}', ['preference'])
                observations = [old, *noise, *target_obs]
            else:
                observations = [*target_obs, *noise]
            question = json.dumps({'keys': list(targets)}, sort_keys=True)
            observations.append(Observation(0, question, ['trigger']))
            task = ControlledTask(identity, family,
                                  [Observation(i + 1, o.content, o.tags) for i, o in enumerate(observations)],
                                  question, [], [], list(targets.values()))
            cases.append({**asdict(task), 'targets': targets})
    rng.shuffle(cases)
    return cases


def to_task(case: dict) -> ControlledTask:
    return ControlledTask(case['task_id'], case['family'],
                          [Observation(**obs) for obs in case['observations']],
                          case['question'], case['expected_constraints'],
                          case['forbidden_outputs'], case['required_outputs'])


def exact_grade(answer: str, targets: dict) -> dict:
    try:
        parsed = json.loads(answer)
    except json.JSONDecodeError:
        return {'success': 0, 'format_valid': False, 'missing_or_wrong_keys': sorted(targets)}
    valid = isinstance(parsed, dict)
    wrong = [key for key, value in targets.items() if not valid or parsed.get(key) != value]
    return {'success': int(valid and parsed == targets), 'format_valid': valid,
            'missing_or_wrong_keys': sorted(wrong)}


def run_episode(case: dict, manager, condition: str) -> dict:
    task = to_task(case)
    ledger = ContextLedger(f'{condition}:{task.task_id}')
    for obs in task.observations:
        ledger.add_raw(obs.step, obs.content, obs.tags)
        manager.manage_after_observation(ledger, obs.step)
    answer = frozen_executor(task, ledger.active_context())
    return {'task_id': task.task_id, 'family': task.family, 'condition': condition,
            'question': task.question, 'targets': case['targets'], 'answer': answer,
            'evaluation': exact_grade(answer, case['targets']),
            'active_context_words': ledger.word_estimate(), 'ledger': ledger}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows).encode()
    if path.suffix == '.gz':
        # Fixed timestamp and no embedded filename make the archive reproducible.
        path.write_bytes(gzip.compress(content, mtime=0))
    else:
        path.write_bytes(content)


def serialized_episode(episode: dict) -> dict:
    return {**episode, 'ledger': episode['ledger'].to_dict()}


def summarize(episodes: list[dict]) -> dict:
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode['condition']].append(episode)
    summary = {}
    for condition, rows in groups.items():
        families = {}
        for family in FAMILIES:
            chosen = [row for row in rows if row['family'] == family]
            wins = sum(row['evaluation']['success'] for row in chosen)
            families[family] = {'n': len(chosen), 'successes': wins,
                                'success_rate': wins / len(chosen)}
        wins = sum(row['evaluation']['success'] for row in rows)
        summary[condition] = {'n': len(rows), 'successes': wins, 'success_rate': wins / len(rows),
                              'avg_active_context_words': sum(r['active_context_words'] for r in rows) / len(rows),
                              'by_family': families}
    return summary


def run_seed(seed: int, folder: Path, per_family: int) -> dict:
    splits = {name: generate_cases(seed, name, count) for name, count in
              [('train', per_family), ('validation', max(2, per_family // 2)), ('test', per_family)]}
    audit = audit_splits(splits)
    if not audit['valid']:
        raise ValueError('The generated dataset has cross-split duplicate inputs or IDs.')
    for name, cases in splits.items():
        write_jsonl(folder / 'data' / f'{name}.jsonl', cases)
    write_json(folder / 'split-audit.json', audit)
    store = LessonStore()
    training, replays = [], []
    null_checks = 0
    for case in splits['train']:
        episode = run_episode(case, NonLearningManager(compressor=truncate_compressor), 'train')
        training.append(serialized_episode(episode))
        if episode['evaluation']['success']:
            continue
        ledger, task = episode['ledger'], to_task(case)
        # A null intervention on the identical context must not repair a failure.
        null_answer = frozen_executor(task, ledger.active_context())
        if null_answer != episode['answer']:
            raise AssertionError('Executor changed an answer without a context intervention.')
        training[-1]['null_intervention_answer'] = null_answer
        null_checks += 1
        suspects = [(d, replay_restore_original) for d in find_suspect_compressions(ledger)[:2]]
        suspects += [(d, replay_archive_keep) for d in find_suspect_archives(ledger)[:2]]
        for decision, replay_fn in suspects:
            replay = replay_fn(task, ledger, decision, episode['answer'], frozen_executor)
            exact = exact_grade(replay.counterfactual_answer, case['targets'])
            if (replay.label == 'HARMFUL') != bool(exact['success']):
                raise AssertionError('Replay grader disagreed with exact structured grading.')
            create_lessons_from_replays([replay], ledger, store)
            replays.append({'task_id': task.task_id, 'family': task.family,
                            'episode_id': ledger.episode_id, 'decision': asdict(decision),
                            **asdict(replay), 'exact_counterfactual_evaluation': exact})
    store.save(folder / 'lessons' / 'agent-a.json')
    legacy, transferred = LessonStore(), LessonStore()
    for lesson in store.get_transferable():
        legacy.add(lesson)
    transferred.import_transferable(store)
    legacy.save(folder / 'lessons' / 'legacy-b.json')
    transferred.save(folder / 'lessons' / 'trusted-b.json')
    ablations = {}
    for label, excluded in [('without_archive_lessons', 'ARCHIVE'),
                            ('without_compression_lessons', 'COMPRESS')]:
        path = folder / 'lessons' / f'{label}.json'
        write_json(path, [row for row in store.to_dict() if row['operation'] != excluded])
        ablations[label] = LessonStore.load(path)
    before = store.to_dict()
    validation = [run_episode(case, LessonAwareManager(store, compressor=truncate_compressor), 'validation_a')
                  for case in splits['validation']]
    managers = {
        'full_context': FullContextManager(),
        'empty_store': LessonAwareManager(LessonStore(), compressor=truncate_compressor),
        'learned_a': LessonAwareManager(store, compressor=truncate_compressor),
        'legacy_transfer_b': LessonAwareManager(legacy, compressor=truncate_compressor),
        'trusted_transfer_b': LessonAwareManager(transferred, compressor=truncate_compressor),
        **{label: LessonAwareManager(s, compressor=truncate_compressor) for label, s in ablations.items()},
    }
    test = [run_episode(case, manager, condition)
            for condition, manager in managers.items() for case in splits['test']]
    if before != store.to_dict():
        raise AssertionError('Evaluation changed the learned store.')
    write_jsonl(folder / 'raw' / 'episodes.jsonl.gz',
                [*training, *(serialized_episode(e) for e in [*validation, *test])])
    write_jsonl(folder / 'raw' / 'replays.jsonl', replays)
    result = {'seed': seed, 'training_episodes': len(training), 'null_intervention_checks': null_checks,
              'training_replays': len(replays), 'replay_labels': dict(Counter(r['label'] for r in replays)),
              'replays_by_family': {family: dict(Counter(r['label'] for r in replays if r['family'] == family))
                                    for family in FAMILIES},
              'learned_lessons': store.to_dict(), 'legacy_b': legacy.to_dict(),
              'transferred_b': transferred.to_dict(), 'validation': summarize(validation),
              'test': summarize(test), 'evaluation_store_unchanged': True}
    write_json(folder / 'summary.json', result)
    return result


def run_study(output: Path, seeds=(7, 19, 41), per_family=8) -> dict:
    if per_family < 5 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Use at least five training cases per family and distinct seeds.')
    output.mkdir(parents=True, exist_ok=True)
    results = [run_seed(seed, output / f'seed-{seed}', per_family) for seed in seeds]
    dataset_audit = {}
    for domain in ('android', 'bgl', 'openstack'):
        data = ROOT / 'data' / domain
        if all((data / f'{name}.jsonl').exists() for name in ('train', 'validation', 'test')):
            records = {name: [json.loads(line) for line in (data / f'{name}.jsonl').read_text().splitlines()]
                       for name in ('train', 'validation', 'test')}
            dataset_audit[domain] = audit_splits(records)
    write_json(output / 'teammate-dataset-audit.json', dataset_audit)
    implementation = ['src/context/manager.py', 'src/context/ledger.py', 'src/lessons/lesson_store.py',
                      'src/attribution/replay.py', 'src/environment/evaluator.py',
                      'src/constants.py', 'src/environment/task_generator.py', 'src/environment/split_validation.py',
                      'src/experiments/run_transfer_validation.py', 'src/experiments/verify_transfer_validation.py']
    manifest = {'mode': 'deterministic_fact_executor', 'api_calls': 0, 'protocol_version': 1,
                'python_version': platform.python_version(), 'seeds': list(seeds),
                'train_and_test_per_family': per_family, 'families': list(FAMILIES),
                'compressor': 'first seven whitespace words; identical in all managed conditions',
                'source_revision': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
                'implementation_sha256': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                          for name in implementation},
                'generated_dataset_sha256': {
                    f'seed-{seed}/data/{split}.jsonl': hashlib.sha256(
                        (output / f'seed-{seed}' / 'data' / f'{split}.jsonl').read_bytes()).hexdigest()
                    for seed in seeds for split in ('train', 'validation', 'test')},
                'teammate_dataset_sha256': {
                    f'data/{domain}/{split}.jsonl': hashlib.sha256(
                        (ROOT / 'data' / domain / f'{split}.jsonl').read_bytes()).hexdigest()
                    for domain in dataset_audit for split in ('train', 'validation', 'test')},
                'limitations': ['Controlled fact-retention test, not an LLM or production incident result.',
                                'Held-out values and inputs; task families and authored tag rules are shared.',
                                'Seed variation is not evidence of model or semantic generalization.']}
    write_json(output / 'manifest.json', manifest)
    aggregate = {}
    for condition in results[0]['test']:
        rows = [r['test'][condition] for r in results]
        n, wins = sum(r['n'] for r in rows), sum(r['successes'] for r in rows)
        aggregate[condition] = {'n': n, 'successes': wins, 'success_rate': wins / n,
                                'avg_active_context_words': sum(r['avg_active_context_words'] * r['n'] for r in rows) / n,
                                'per_seed_success_rates': [r['success_rate'] for r in rows]}
    result = {'mode': manifest['mode'], 'api_calls': 0, 'seeds': list(seeds), 'aggregate_test': aggregate,
              'all_generated_splits_disjoint': True, 'evaluation_stores_frozen': True,
              'teammate_dataset_exact_overlap_checks': {d: a['valid'] for d, a in dataset_audit.items()}}
    write_json(output / 'summary.json', result)
    lines = ['# Controlled replay and transfer validation results', '',
             'Offline deterministic fact-retention study. These are not LLM or LogHub task success rates.', '',
             '| Condition | Successes | Success rate | Mean context words |',
             '|---|---:|---:|---:|']
    for condition, row in aggregate.items():
        lines.append(f"| {condition} | {row['successes']}/{row['n']} | {row['success_rate']:.1%} | {row['avg_active_context_words']:.1f} |")
    lines += ['', 'The first three loss families test learned protection; benign_recent is a safe control.',
              'info_mistag and joint_loss are negative controls exposing unchanged policy/replay limitations.',
              'All managed conditions share the executor, input cases, grader and compressor. Evaluation stores are frozen.',
              'Each seed uses new keys, values and nuisance observations. Shared templates do not establish unseen-rule generalization.',
              '', 'Inspect each seed summary for family results, null checks, lesson counts and import states.',
              'The raw gzip JSONL includes complete ledgers; replays include target decisions and both answers.']
    (output / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    table = ''.join(f"<tr><td>{html.escape(condition)}</td><td>{row['successes']}/{row['n']}</td>"
                    f"<td>{row['success_rate']:.1%}</td><td>{row['avg_active_context_words']:.1f}</td></tr>"
                    for condition, row in aggregate.items())
    (output / 'RESULTS.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Controlled replay and transfer validation</title><style>'
        'body{font:16px/1.6 system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 20px;color:#213a35;background:#f5f8f5}'
        'table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d5e2dc;padding:10px;text-align:left}'
        '.scroll{overflow:auto}.note{background:#fff0dc;border-left:4px solid #8a5c18;padding:15px}'
        '.flow{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}.flow span{background:#e3f1e7;padding:10px;border-radius:8px}'
        'code{font-size:14px}a{color:#176c57}</style><main><h1>Controlled replay and transfer validation</h1>'
        '<p class="note">Offline deterministic fact-retention study. These are not LLM or production incident success rates.</p>'
        '<div class="flow"><span>Training failures</span><span>→ local undo replays</span><span>→ supported rules</span>'
        '<span>→ preserve evidence on import</span><span>→ frozen held-out evaluation</span></div>'
        '<div class="scroll"><table><thead><tr><th>Condition</th><th>Successes</th><th>Success rate</th><th>Mean context words</th></tr></thead>'
        '<tbody>' + table + '</tbody></table></div><h2>What this establishes</h2>'
        '<p>The empty, learned, transferred and ablated managers share the executor, grading, input cases and compressor. '
        'Correct imports retain usable support and status. Legacy imports leave rules inactive. '
        'Removing each operation-specific lesson loses its corresponding protection.</p>'
        '<h2>Where the current policy fails</h2><p>An important item tagged as info is always archived. '
        'Restoring one of two lost facts leaves the joint task unsuccessful, so binary replay creates no lesson. '
        'These failures remain visible in the family summaries.</p>'
        '<p>Inputs and values differ across train, validation and test. Families and authored tag rules are shared. '
        'This is a mechanism validation, not proof of semantic or cross-model generalization. Context words are not API costs.</p>'
        '<p><a href="summary.json">Aggregate JSON</a> · <a href="manifest.json">Source and dataset hashes</a> · '
        '<a href="teammate-dataset-audit.json">Exact duplicate audit of existing datasets</a> · '
        '<a href="../../docs/ANIL_TRANSFER_VALIDATION.md">Protocol and reproduction instructions</a></p></main></html>')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results' / 'transfer-validation')
    parser.add_argument('--seeds', type=int, nargs='+', default=[7, 19, 41])
    parser.add_argument('--per-family', type=int, default=8)
    args = parser.parse_args()
    result = run_study(args.output, tuple(args.seeds), args.per_family)
    print(json.dumps(result, indent=2))
    print('Results:', args.output / 'RESULTS.md')


if __name__ == '__main__':
    main()
