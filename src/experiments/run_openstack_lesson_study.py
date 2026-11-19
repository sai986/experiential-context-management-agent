"""Matched OpenStack/synthetic lesson study with a bounded hosted evaluation.

Prepare/offline: python -m src.experiments.run_openstack_lesson_study
Hosted: add --hosted --key-file /tmp/ecm-study-qgenie.key
Plots: add --plots (requires matplotlib).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import gzip
import hashlib
import html
import json
from pathlib import Path
import subprocess

from src.agents.investigation_agent import _TAG_GUIDE, _strip_fences
from src.agents.task_agent import llm_task_agent
from src.context.ledger import ContextLedger
from src.context.manager import FullContextManager, LessonAwareManager, NonLearningManager
from src.environment.evaluator import evaluate_answer
from src.environment.task_generator import ControlledTask, Observation
from src.environment.task_generator_incident import generate_incident_tasks
from src.experiments.lesson_study_client import StudyClient, StudyStopped, credential
from src.experiments.run_transfer_validation import exact_grade, frozen_executor, truncate_compressor
from src.lessons.lesson_store import LessonStore

ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ('full_context', 'empty_store', 'openstack_transfer', 'synthetic_transfer')
LEVELS = (0, 8, 32)
ALLOWED_TAGS = {'critical', 'hard_constraint', 'negation', 'delayed_dependency',
                'distractor', 'preference', 'info'}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz' else path.read_text()
    return [json.loads(row) for row in data.splitlines() if row.strip()]


def task_from(row: dict) -> ControlledTask:
    return ControlledTask(row['task_id'], row['family'], [Observation(**v) for v in row['observations']],
                          row['question'], row.get('expected_constraints', []), row['forbidden_outputs'],
                          row['required_outputs'], row.get('synonym_groups', []))


def prepare(output: Path) -> dict:
    sources = [ROOT / 'data/openstack/test.jsonl', ROOT / 'results/openstack/real/processed/summary.json',
               ROOT / 'results/transfer-validation/seed-7/lessons/agent-a.json',
               ROOT / 'src/environment/task_generator_incident.py', ROOT / 'src/context/manager.py',
               ROOT / 'src/lessons/lesson_store.py', ROOT / 'src/environment/evaluator.py',
               Path(__file__), Path(__file__).with_name('lesson_study_client.py')]
    for seed in (7, 19, 41):
        sources.append(ROOT / f'results/transfer-validation/seed-{seed}/data/test.jsonl')
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    existing = output / 'manifest.json'
    if existing.exists() and read_jsonl(output / 'hosted/episodes.jsonl'):
        if json.loads(existing.read_text())['source_sha256'] != hashes:
            raise ValueError('Study inputs/implementation changed; use a new output folder for hosted results.')
    grouped = defaultdict(list)
    for row in read_jsonl(sources[0]):
        grouped[row['family']].append(row)
    if len(grouped) != 6 or any(len(v) < 2 for v in grouped.values()):
        raise ValueError('Expected at least two OpenStack cases in each of six families.')
    cases = []
    for family, rows in sorted(grouped.items()):
        for rank, row in enumerate(sorted(rows, key=lambda v: v['incident_id'])[:2]):
            cases.append({'dataset': 'openstack', 'selection_rank': rank, 'task_id': row['incident_id'],
                          'family': family, 'question': row['question'], 'raw_text': row['raw_text'],
                          'expected_constraints': row.get('expected_constraints', []),
                          'required_outputs': row['required_outputs'], 'forbidden_outputs': row['forbidden_outputs'],
                          'synonym_groups': row.get('synonym_groups', [])})
    ranks = defaultdict(int)
    for incident in generate_incident_tasks(n_per_family=2):
        row = asdict(incident.task)
        cases.append({'dataset': 'synthetic_incident', 'selection_rank': ranks[row['family']], **row})
        ranks[row['family']] += 1
    cases.sort(key=lambda c: (c['selection_rank'], c['family'], c['dataset']))
    write_json(output / 'data/selected_cases.json', cases)
    openstack = json.loads(sources[1].read_text())['learned_lessons_a']
    write_json(output / 'lessons/source-openstack.json', openstack)
    write_json(output / 'lessons/source-synthetic.json', json.loads(sources[2].read_text()))
    for source in ('openstack', 'synthetic'):
        store = LessonStore()
        store.import_transferable(LessonStore.load(output / f'lessons/source-{source}.json'))
        store.save(output / f'lessons/imported-{source}.json')
    manifest = {'protocol_version': 1, 'source_revision': subprocess.check_output(
                    ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                'source_sha256': hashes, 'selected_cases_sha256': digest(output / 'data/selected_cases.json'),
                'conditions': CONDITIONS, 'shared_compressor': 'first seven whitespace words',
                'lesson_stores_frozen': True, 'hosted_cases_per_dataset': 12,
                'selection': 'First two incident IDs per family; round-robin family/rank/dataset order.',
                'synthetic_store_source': 'Previously verified deterministic fact-retention seed-7 study.',
                'noise_levels': LEVELS,
                'limitations': ['Small paired pilot; no significance or universal-transfer claim.',
                                'Source stores differ in rule counts/tag coverage.',
                                'OpenStack is LogHub-derived with authored constraints and log additions.',
                                'Synthetic incidents have predefined observations; OpenStack tags are model-assigned.',
                                'Shared synthetic templates; these are additional probes, not a new production dataset.']}
    write_json(existing, manifest)
    return manifest


class AllTagsStore(LessonStore):
    """Experimental retrieval adapter; production retrieval is unchanged."""
    def retrieve(self, tags, operation, min_status='validated'):
        return [lesson for lesson in super().retrieve(tags, operation, min_status)
                if set(lesson.trigger_tags).issubset(tags)]


def manager_for(output: Path, condition: str):
    if condition == 'full_context':
        return FullContextManager()
    if condition == 'empty_store':
        return NonLearningManager(compressor=truncate_compressor)
    source = 'openstack' if condition.startswith('openstack') else 'synthetic'
    store_type = AllTagsStore if condition.endswith('_all') else LessonStore
    store = store_type.load(output / f'lessons/imported-{source}.json')
    return LessonAwareManager(store, compressor=truncate_compressor)


def manage(task: ControlledTask, manager) -> dict:
    before_store = manager.lesson_store.to_dict() if isinstance(manager, LessonAwareManager) else None
    ledger = ContextLedger(task.task_id)
    trajectory = []
    for obs in task.observations:
        ledger.add_raw(obs.step, obs.content, list(obs.tags))
        before = ledger.word_estimate()
        manager.manage_after_observation(ledger, obs.step)
        trajectory.append({'step': obs.step, 'before_management_words': before,
                           'after_management_words': ledger.word_estimate()})
    if before_store is not None:
        assert manager.lesson_store.to_dict() == before_store, 'Evaluation mutated the lesson store.'
    overrides = [d for d in ledger.decisions if d.rationale.startswith('Lesson:')]
    rule_ids = set()
    for decision in overrides:
        op = 'ARCHIVE' if 'archiving' in decision.rationale else 'COMPRESS'
        for item_id in decision.input_item_ids:
            rule_ids.update(l.lesson_id for l in manager.lesson_store.retrieve(ledger.items[item_id].tags, op))
    return {'ledger': ledger.to_dict(), 'active_context_words': ledger.word_estimate(),
            'peak_active_context_words': max(max(r['before_management_words'], r['after_management_words']) for r in trajectory),
            'step_sum_active_words': sum(r['after_management_words'] for r in trajectory),
            'trajectory': trajectory, 'lesson_override_decisions': len(overrides),
            'lesson_override_unique_items': len({i for d in overrides for i in d.input_item_ids}),
            'matched_rule_ids': sorted(rule_ids)}


def add_noise(case: dict, n: int) -> dict:
    extra = [{'step': 0, 'content': f'Irrelevant study observation {i} describes a separate workload '
             'with a routine configuration note and an unrelated operating constraint requiring no action.',
              'tags': ['critical'] if i % 2 == 0 else ['hard_constraint']} for i in range(n)]
    old = case['observations']
    # Keep the benign recent fact recent; other families retain their early fact placement.
    observations = extra + old if case['family'] == 'benign_recent' else old[:-1] + extra + old[-1:]
    return {**case, 'noise_level': n, 'observations': [{**o, 'step': i + 1} for i, o in enumerate(observations)]}


def aggregate(rows: list[dict], group_keys=('dataset', 'condition')) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in group_keys)].append(row)
    result = []
    for keys, selected in sorted(groups.items()):
        wins = sum(r['evaluation']['task_success'] for r in selected)
        record = {**dict(zip(group_keys, keys)), 'n': len(selected), 'successes': wins,
                  'success_rate': wins / len(selected)}
        for metric in ('active_context_words', 'peak_active_context_words', 'step_sum_active_words',
                       'lesson_override_unique_items', 'irrelevant_verbatim_items'):
            if metric in selected[0]:
                record['mean_' + metric] = sum(r[metric] for r in selected) / len(selected)
        result.append(record)
    return result


def run_offline(output: Path) -> dict:
    conditions = ('full_context', 'empty_store', 'openstack_any', 'openstack_all', 'synthetic_any', 'synthetic_all')
    episodes = []
    for seed in (7, 19, 41):
        cases = read_jsonl(ROOT / f'results/transfer-validation/seed-{seed}/data/test.jsonl')
        for case in cases:
            for level in LEVELS:
                altered = add_noise(case, level)
                task = task_from(altered)
                for condition in conditions:
                    managed = manage(task, manager_for(output, condition))
                    answer = frozen_executor(task, managed['ledger']['active_context'])
                    grade = exact_grade(answer, case['targets'])
                    irrelevant = sum(v['active'] and v['immutable'] and v['content'].startswith('Irrelevant study observation')
                                     for v in managed['ledger']['items'])
                    episodes.append({'dataset': 'controlled_fact', 'seed': seed, 'task_id': task.task_id,
                                     'family': task.family, 'condition': condition, 'noise_level': level,
                                     'targets': case['targets'], 'question': task.question, 'answer': answer,
                                     'evaluation': {'task_success': grade['success'], **grade},
                                     'irrelevant_verbatim_items': irrelevant, **managed})
    path = output / 'offline/episodes.jsonl.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(''.join(json.dumps(r, sort_keys=True) + '\n' for r in episodes).encode(), mtime=0))
    summary = {'mode': 'deterministic_fact_executor', 'api_calls': 0, 'episodes': len(episodes),
               'aggregate': aggregate(episodes, ('condition', 'noise_level')),
               'by_family': aggregate(episodes, ('condition', 'noise_level', 'family'))}
    write_json(output / 'offline/summary.json', summary)
    return summary


def evaluator_audit(output: Path) -> list[dict]:
    fixtures = [
        ('affirmed_constraint', ['offline', 'local'], ['cloud API'], [], 'Use a local offline model.', 1),
        ('negated_forbidden', ['offline', 'local'], ['cloud API'], [], 'Use a local offline model, not a cloud API.', 1),
        ('actual_violation', ['offline', 'local'], ['cloud API'], [], 'Use a cloud API instead of local offline inference.', 0),
        ('post_nominal_negation', ['PowerShell'], ['bash'], [], 'Use PowerShell; bash is unsupported.', 1),
        ('pre_nominal_negation', ['PowerShell'], ['bash'], [], 'Use PowerShell; do not use bash.', 1),
        ('unrelated_negation', ['local', 'offline'], ['cloud API'], [], 'No network. Use local offline inference.', 1),
        ('negated_synonym', ['PowerShell'], [], [['PowerShell', 'ps1']], 'Do not use ps1.', 0),
        ('dismissed_alternative', ['Glance', 'storage backend'], ['scheduler'], [],
         'Glance storage backend is the cause; scheduler errors are unrelated.', 1),
    ]
    rows = []
    for name, required, forbidden, synonyms, answer, intended in fixtures:
        task = ControlledTask(name, 'grader_audit', [], '', [], forbidden, required, synonyms)
        evaluated = asdict(evaluate_answer(task, answer))
        rows.append({'fixture': name, 'required': required, 'forbidden': forbidden, 'synonyms': synonyms,
                     'answer': answer, 'intended_success': intended, 'evaluation': evaluated,
                     'agrees': evaluated['task_success'] == intended})
    write_json(output / 'evaluator-audit.json', rows)
    return rows


def investigate(case: dict, client) -> list[dict]:
    lines = [line for line in case['raw_text'].splitlines() if line.strip()]
    prompt = (_TAG_GUIDE + '\ninfo: background information irrelevant to answering this question.\n'
              'Assign tags to every numbered input line. Preserve all line numbers. Return ONLY a JSON array '
              'of {"line": integer, "tags": [strings]}. No trigger tags: the question is appended separately.\n'
              f'Question: {case["question"]}\n' + '\n'.join(f'{i + 1}: {line}' for i, line in enumerate(lines)))
    raw = client.complete(prompt)
    try:
        rows = json.loads(_strip_fences(raw))
        if not isinstance(rows, list) or len(rows) != len(lines):
            raise ValueError()
        by_line = {r['line']: r['tags'] for r in rows}
        if set(by_line) != set(range(1, len(lines) + 1)):
            raise ValueError()
        if any(not isinstance(tags, list) or not tags or any(t not in ALLOWED_TAGS for t in tags)
               for tags in by_line.values()):
            raise ValueError()
    except (ValueError, TypeError, KeyError):
        raise StudyStopped('Investigation returned invalid or incomplete line tags; no fallback was used.') from None
    observations = [asdict(Observation(i, line, by_line[i])) for i, line in enumerate(lines, 1)]
    observations.append(asdict(Observation(len(lines) + 1, case['question'], ['trigger'])))
    return observations


def completed_pairs(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    grouped = defaultdict(dict)
    for row in rows:
        key = (row['dataset'], row['task_id'])
        if row['condition'] in grouped[key]:
            raise ValueError('Duplicate hosted episode.')
        grouped[key][row['condition']] = row
    complete, pairs = [], []
    for (dataset, task_id), conditions in sorted(grouped.items()):
        if set(conditions) != set(CONDITIONS):
            continue
        complete.extend(conditions.values())
        base = conditions['empty_store']['evaluation']['task_success']
        for name in CONDITIONS[2:]:
            score = conditions[name]['evaluation']['task_success']
            pairs.append({'dataset': dataset, 'task_id': task_id, 'family': conditions[name]['family'],
                          'condition': name, 'baseline': base, 'success': score,
                          'outcome': 'win' if score > base else 'loss' if score < base else 'tie'})
    return complete, pairs


def hosted_summary(output: Path, status: str, stop_reason: str | None = None) -> dict:
    rows = read_jsonl(output / 'hosted/episodes.jsonl')
    complete, pairs = completed_pairs(rows)
    calls = read_jsonl(output / 'hosted/api_calls.jsonl')
    usage = []
    for purpose in ('preflight', 'investigation', 'task'):
        selected = [c for c in calls if c['purpose'] == purpose]
        item = {'purpose': purpose, 'api_attempts': len(selected), 'successful_requests': sum(c['status'] == 'ok' for c in selected)}
        for field in ('input_tokens', 'output_tokens', 'cache_read_input_tokens', 'cache_creation_input_tokens'):
            reported = [c[field] for c in selected if c.get(field) is not None]
            item[field] = sum(reported) if reported else None
        usage.append(item)
    result = {'status': status, 'stop_reason': stop_reason, 'completed_episode_records': len(rows),
              'paired_cases': len(complete) // len(CONDITIONS), 'selected_cases': 24,
              'unpaired_episode_records': len(rows) - len(complete), 'aggregate': aggregate(complete),
              'by_family': aggregate(complete, ('dataset', 'condition', 'family')),
              'paired_outcomes': pairs, 'api_attempts': len(calls), 'provider_usage': usage}
    write_json(output / 'hosted/summary.json', result)
    return result


def run_hosted(output: Path, client: StudyClient) -> dict:
    cases = json.loads((output / 'data/selected_cases.json').read_text())
    previous = read_jsonl(output / 'hosted/episodes.jsonl')
    done = {(r['dataset'], r['task_id'], r['condition']) for r in previous}
    reason = None
    try:
        client.preflight()
        for case in cases:
            if all((case['dataset'], case['task_id'], c) in done for c in CONDITIONS):
                continue
            if case['dataset'] == 'openstack':
                frozen = output / f'hosted/observations/{case["task_id"]}.json'
                if frozen.exists():
                    observations = json.loads(frozen.read_text())
                else:
                    client.set_context(purpose='investigation', task_id=case['task_id'], dataset=case['dataset'], max_tokens=2048)
                    observations = investigate(case, client)
                    write_json(frozen, observations)
                task = task_from({**case, 'observations': observations})
            else:
                task = task_from(case)
            for condition in CONDITIONS:
                identity = (case['dataset'], case['task_id'], condition)
                if identity in done:
                    continue
                managed = manage(task, manager_for(output, condition))
                client.set_context(purpose='task', dataset=case['dataset'], task_id=case['task_id'], condition=condition)
                answer = llm_task_agent(task, managed['ledger']['active_context'], client)
                row = {'dataset': case['dataset'], 'task_id': task.task_id, 'family': task.family,
                       'condition': condition, 'task': asdict(task), 'answer': answer,
                       'evaluation': asdict(evaluate_answer(task, answer)), **managed,
                       'model_call': {k: client.last.get(k) for k in ['prompt_sha256', 'response_model', 'cache_hit',
                                                                   'input_tokens', 'output_tokens', 'response_id']}}
                with (output / 'hosted/episodes.jsonl').open('a') as stream:
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
                done.add(identity)
                print(f"{len(done):02d}/96 {case['dataset']} {task.family} {condition}: {row['evaluation']['task_success']}", flush=True)
    except StudyStopped as exc:
        reason = str(exc)
    finally:
        client.save_cache_events()
    return hosted_summary(output, 'complete' if len(done) == 96 else 'incomplete', reason)


def render_report(output: Path, plots=False) -> None:
    offline = json.loads((output / 'offline/summary.json').read_text())
    audit = json.loads((output / 'evaluator-audit.json').read_text())
    hosted = json.loads((output / 'hosted/summary.json').read_text()) if (output / 'hosted/summary.json').exists() else {'status': 'awaiting_credentials', 'aggregate': []}
    lines = ['# Matched OpenStack and synthetic lesson study', '',
             'This contribution isolates lesson-store changes under a shared seven-word compressor, then measures rule selectivity under irrelevant tagged noise.', '',
             '## Hosted paired pilot', '', f"Status: **{hosted['status']}**.", '',
             '| Dataset | Condition | Successes / cases | Final words | Peak words | Step-sum words |',
             '|---|---|---:|---:|---:|---:|']
    for r in hosted['aggregate']:
        lines.append(f"| {r['dataset']} | {r['condition']} | {r['successes']}/{r['n']} ({r['success_rate']:.1%}) | {r['mean_active_context_words']:.1f} | {r['mean_peak_active_context_words']:.1f} | {r['mean_step_sum_active_words']:.1f} |")
    if hosted.get('stop_reason'):
        lines.extend(['', f"Stop reason: {hosted['stop_reason']}"])
    lines.extend(['', 'Analysis uses only cases with all four conditions completed. Partial records remain saved and counted.', '',
                  '## Offline rule-selectivity experiment', '',
                  f"{offline['episodes']} deterministic episode records; zero model calls. Three source seeds, 144 base cases and three noise levels.", '',
                  '| Policy | Added noise | Success | Final words | Irrelevant items kept verbatim |', '|---|---:|---:|---:|---:|'])
    for r in offline['aggregate']:
        lines.append(f"| {r['condition']} | {r['noise_level']} | {r['success_rate']:.1%} | {r['mean_active_context_words']:.1f} | {r['mean_irrelevant_verbatim_items']:.1f} |")
    lines.extend(['', '## Evaluator audit', '',
                  'Hand-authored semantic fixtures diagnose grading behavior. They are not estimates of real-world grader accuracy.', '',
                  '| Fixture | Intended success | Grader success | Agreement |', '|---|---:|---:|---|'])
    for r in audit:
        lines.append(f"| {r['fixture']} | {r['intended_success']} | {r['evaluation']['task_success']} | {r['agrees']} |")
    lines.extend(['', '## Interpretation and reproducibility', '',
                  '- OpenStack source: one transferable compression rule with support 13; two candidates are excluded. Controlled synthetic source: three transferable rules, each with support 8.',
                  '- Source policies differ in tag coverage and rule count; this is a comparison of complete policies.',
                  '- Current any-tag matching and experimental all-tags matching share every other setting. Production retrieval is unchanged.',
                  '- Final/peak words include observation IDs. Step-sum words is the sum of active words after observation-management steps, not billed tokens.',
                  '- Provider token usage is reported separately for investigation and task requests. Shared preprocessing and prompt-cache reuse are explicit; cache hits are not independent samples.',
                  '- OpenStack raw lines are preserved exactly and tagged once; synthetic incidents use predefined observations. Grading metadata is excluded from task prompts.',
                  '- No tuning after outcomes, no new training, and no statistical or production-generalization claim. OpenStack includes authored constraints/additions; synthetic probes share existing templates.',
                  '', 'Run `python -m src.experiments.run_openstack_lesson_study --hosted --key-file /tmp/ecm-study-qgenie.key --plots`.',
                  'Source hashes, frozen inputs/stores, full ledgers, provider calls and verifier output accompany this report.'])
    markdown = '\n'.join(lines) + '\n'
    (output / 'RESULTS.md').write_text(markdown)
    (output / 'RESULTS.html').write_text('<!doctype html><meta charset="utf-8"><title>Matched lesson study</title>'
        '<style>body{max-width:1100px;margin:40px auto;font:17px system-ui;background:#f5f7fa;color:#20313f}pre{white-space:pre-wrap}img{max-width:100%}</style>'
        '<h1>Matched OpenStack and synthetic lesson study</h1><img src="rule-selectivity.svg"><pre>' + html.escape(markdown) + '</pre>')
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
        for source, color in [('openstack', '#237f78'), ('synthetic', '#496bb0')]:
            for mode, style in [('any', '-'), ('all', '--')]:
                rows = sorted((r for r in offline['aggregate'] if r['condition'] == source + '_' + mode), key=lambda r: r['noise_level'])
                for ax, field in zip(axes, ['mean_active_context_words', 'success_rate']):
                    ax.plot([r['noise_level'] for r in rows], [r[field] * (100 if field == 'success_rate' else 1) for r in rows],
                            style, marker='o', color=color, label=source + ': ' + mode + '-tag matching')
        for ax, ylabel in zip(axes, ['Mean final active words', 'Exact fact success (%)']):
            ax.set(xlabel='Added irrelevant tagged observations', ylabel=ylabel, xticks=LEVELS)
            ax.grid(alpha=.25)
        axes[1].set_ylim(0, 105)
        axes[0].legend(fontsize=8)
        fig.suptitle('Frozen lesson policies: selectivity and context cost (offline controlled probes)')
        for suffix in ('svg', 'pdf', 'png'):
            fig.savefig(output / f'rule-selectivity.{suffix}', dpi=170)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'results/openstack-lesson-study')
    parser.add_argument('--hosted', action='store_true')
    parser.add_argument('--plots', action='store_true')
    parser.add_argument('--key-file', type=Path, default=Path('/tmp/ecm-study-qgenie.key'))
    parser.add_argument('--max-calls', type=int, default=110)
    parser.add_argument('--max-inference-minutes', type=float, default=45)
    args = parser.parse_args()
    prepare(args.output)
    offline = run_offline(args.output)
    audit = evaluator_audit(args.output)
    print(f"Offline episodes: {offline['episodes']}; evaluator disagreements: {sum(not r['agrees'] for r in audit)}/{len(audit)}", flush=True)
    if args.hosted:
        try:
            client = StudyClient(args.output, credential(args.key_file), max_calls=args.max_calls, max_minutes=args.max_inference_minutes)
            summary = run_hosted(args.output, client)
        except StudyStopped as exc:
            summary = hosted_summary(args.output, 'incomplete', str(exc))
        print(f"Hosted: {summary['status']}; paired cases: {summary['paired_cases']}; API attempts: {summary['api_attempts']}")
    render_report(args.output, args.plots)


if __name__ == '__main__':
    main()
