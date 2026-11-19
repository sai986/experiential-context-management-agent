from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from src.context.manager import LessonAwareManager, NonLearningManager
from src.experiments.lesson_study_client import StudyClient, StudyStopped
from src.experiments.run_openstack_lesson_study import (
    AllTagsStore, CONDITIONS, add_noise, completed_pairs, evaluator_audit,
    investigate, manage, manager_for, prepare, task_from,
)
from src.experiments.run_transfer_validation import frozen_executor, generate_cases, truncate_compressor
from src.lessons.lesson_store import LessonStore


class ClientTests(unittest.TestCase):
    def test_cache_reuses_identical_prompts_and_preserves_usage_under_budget(self):
        requests = []

        def transport(url, headers, body, timeout):
            requests.append(body)
            return {'id': 'fixture', 'model': body['model'], 'content': [{'type': 'text', 'text': 'Answer'}],
                    'usage': {'input_tokens': 9, 'output_tokens': 2}, 'stop_reason': 'end_turn'}

        with tempfile.TemporaryDirectory() as folder:
            client = StudyClient(Path(folder), 'fixture-key-never-persist', transport=transport, max_calls=1)
            self.assertEqual(client.complete('Same prompt'), 'Answer')
            client.set_context(condition='another_condition')
            self.assertEqual(client.complete('Same prompt'), 'Answer')
            self.assertTrue(client.last['cache_hit'])
            self.assertEqual(len(requests), 1)
            self.assertEqual(requests[0]['temperature'], 0)
            with self.assertRaises(StudyStopped):
                client.complete('Different prompt')
            resumed = StudyClient(Path(folder), 'fixture-key-never-persist', transport=transport, max_calls=1)
            self.assertEqual(resumed.complete('Same prompt'), 'Answer')
            with self.assertRaises(StudyStopped):
                resumed.complete('Uncached prompt')
            for path in Path(folder).rglob('*.json*'):
                self.assertNotIn('fixture-key-never-persist', path.read_text())

    def test_failed_attempt_and_timeout_consume_budget(self):
        def failure(*args):
            raise StudyStopped('Provider returned HTTP 429.')

        with tempfile.TemporaryDirectory() as folder:
            client = StudyClient(Path(folder), 'fixture', transport=failure, max_calls=1)
            with self.assertRaisesRegex(StudyStopped, '429'):
                client.complete('Prompt')
            self.assertEqual(len(client.records), 1)
            with self.assertRaisesRegex(StudyStopped, 'limit'):
                client.complete('Prompt')
        clock = iter([0, 61])
        with tempfile.TemporaryDirectory() as folder:
            client = StudyClient(Path(folder), 'fixture', now=lambda: next(clock), max_minutes=1)
            with self.assertRaisesRegex(StudyStopped, 'time limit'):
                client.complete('Prompt')


class StudyTests(unittest.TestCase):
    def test_empty_store_equivalence_and_metric_accounting(self):
        case = generate_cases(2, 'probe', 1)[0]
        task = task_from(case)
        baseline = manage(task, NonLearningManager(compressor=truncate_compressor))
        learned = manage(task, LessonAwareManager(LessonStore(), compressor=truncate_compressor))
        self.assertEqual(baseline['ledger']['active_context'], learned['ledger']['active_context'])
        self.assertEqual(baseline['active_context_words'], len(baseline['ledger']['active_context'].split()))
        self.assertEqual(baseline['step_sum_active_words'], sum(r['after_management_words'] for r in baseline['trajectory']))
        hidden = replace(task, family='different', required_outputs=['hidden'], forbidden_outputs=['hidden'])
        self.assertEqual(frozen_executor(task, baseline['ledger']['active_context']),
                         frozen_executor(hidden, baseline['ledger']['active_context']))

    def test_preparation_balances_cases_and_preserves_imported_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            prepare(output)
            cases = json.loads((output / 'data/selected_cases.json').read_text())
            self.assertEqual(len(cases), 24)
            for dataset in ('openstack', 'synthetic_incident'):
                rows = [r for r in cases if r['dataset'] == dataset]
                self.assertEqual(len(rows), 12)
                self.assertEqual(len({r['family'] for r in rows}), 6)
                self.assertEqual(len({r['task_id'] for r in rows}), 12)
            store = manager_for(output, 'openstack_transfer').lesson_store
            self.assertEqual(len(store.to_dict()), 1)
            self.assertEqual(store.to_dict()[0]['support_count'], 13)
            self.assertEqual(store.to_dict()[0]['status'], 'transferable')
            restricted = manager_for(output, 'openstack_all').lesson_store
            self.assertIsInstance(restricted, AllTagsStore)
            self.assertTrue(store.retrieve(['critical'], 'COMPRESS'))
            self.assertFalse(restricted.retrieve(['critical'], 'COMPRESS'))
            self.assertTrue(restricted.retrieve(['critical', 'hard_constraint'], 'COMPRESS'))
            before = store.to_dict()
            case = next(c for c in generate_cases(2, 'probe', 1) if c['family'] == 'critical_compression')
            manage(task_from(case), LessonAwareManager(store, compressor=truncate_compressor))
            self.assertEqual(store.to_dict(), before)

    def test_noise_does_not_change_targets_or_make_recent_control_old(self):
        case = next(c for c in generate_cases(3, 'probe', 1) if c['family'] == 'benign_recent')
        altered = add_noise(case, 32)
        self.assertEqual(altered['targets'], case['targets'])
        self.assertEqual(altered['question'], case['question'])
        self.assertEqual(altered['observations'][-2]['content'], case['observations'][-2]['content'])
        self.assertEqual(altered['observations'][-1]['tags'], ['trigger'])
        self.assertEqual([o['step'] for o in altered['observations']], list(range(1, len(altered['observations']) + 1)))

    def test_investigation_preserves_lines_and_rejects_partial_tags(self):
        case = {'raw_text': '  Constraint text.\nLog text.', 'question': 'What happened?'}

        class Client:
            def complete(self, prompt):
                return '[{"line": 2, "tags": ["info"]}, {"line": 1, "tags": ["critical"]}]'

        observations = investigate(case, Client())
        self.assertEqual(observations[0]['content'], '  Constraint text.')
        self.assertEqual(observations[-1]['content'], case['question'])

        class Incomplete:
            def complete(self, prompt):
                return '[{"line": 1, "tags": ["critical"]}]'

        with self.assertRaises(StudyStopped):
            investigate(case, Incomplete())

    def test_partial_cases_are_excluded_and_duplicates_rejected(self):
        rows = [{'dataset': 'openstack', 'task_id': 'complete', 'family': 'constraint',
                 'condition': c, 'evaluation': {'task_success': int(c != 'empty_store')}} for c in CONDITIONS]
        rows.append({**rows[0], 'task_id': 'partial'})
        complete, pairs = completed_pairs(rows)
        self.assertEqual(len(complete), 4)
        self.assertEqual([p['outcome'] for p in pairs], ['win', 'win'])
        with self.assertRaises(ValueError):
            completed_pairs(rows + [rows[0]])

    def test_semantic_audit_keeps_disagreements_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            rows = evaluator_audit(Path(folder))
        by_name = {r['fixture']: r for r in rows}
        self.assertEqual(by_name['negated_synonym']['intended_success'], 0)
        self.assertFalse(by_name['negated_synonym']['agrees'])
        self.assertFalse(by_name['post_nominal_negation']['agrees'])
        self.assertTrue(by_name['affirmed_constraint']['agrees'])


if __name__ == '__main__':
    unittest.main()
