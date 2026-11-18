from __future__ import annotations

from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from src.context.manager import LessonAwareManager, NonLearningManager
from src.environment.split_validation import audit_splits
from src.experiments.run_transfer_validation import (frozen_executor, generate_cases,
                                                     run_episode, run_study, to_task,
                                                     truncate_compressor)
from src.lessons.lesson_store import Lesson, LessonStore
from src.experiments.verify_transfer_validation import verify_artifacts


def supported_store(count=5, instruction='Preserve this pattern.'):
    store = LessonStore()
    lesson = Lesson('', 'compression_safety', 'COMPRESS', ('critical', 'hard_constraint'),
                    instruction, 'dec_0001', 0.0, 1.0)
    for _ in range(count):
        store.add(lesson)
    return store


class TransferImportTests(unittest.TestCase):
    def test_import_enables_a_real_management_decision(self):
        case = next(c for c in generate_cases(1, 'test', 1) if c['family'] == 'critical_compression')
        source, destination = supported_store(), LessonStore()
        before = source.to_dict()
        empty = run_episode(case, LessonAwareManager(destination, compressor=truncate_compressor), 'before')
        self.assertEqual(empty['evaluation']['success'], 0)
        self.assertEqual(destination.import_transferable(source), 1)
        protected = run_episode(case, LessonAwareManager(destination, compressor=truncate_compressor), 'after')
        self.assertEqual(protected['evaluation']['success'], 1)
        self.assertTrue(any(d.operation == 'KEEP' and d.rationale.startswith('Lesson:')
                            for d in protected['ledger'].decisions))
        self.assertEqual(destination.get_support_count(destination.lessons[0].lesson_id), 5)
        self.assertEqual(source.to_dict(), before)

    def test_repeat_import_does_not_invent_confirmations(self):
        source, destination = supported_store(7), supported_store(3)
        destination.import_transferable(source)
        before = destination.to_dict()
        self.assertEqual(destination.import_transferable(source), 0)
        self.assertEqual(destination.to_dict(), before)
        self.assertEqual(before[0]['support_count'], 7)
        self.assertEqual(before[0]['status'], 'transferable')

    def test_ineligible_lessons_are_excluded_and_ids_do_not_collide(self):
        destination = supported_store(1, 'Unrelated local candidate.')
        self.assertEqual(destination.import_transferable(supported_store(2)), 0)
        destination.import_transferable(supported_store(5))
        self.assertEqual(len({lesson.lesson_id for lesson in destination.lessons}), 2)
        self.assertEqual(len(destination.retrieve(['critical'], 'COMPRESS')), 1)

    def test_import_survives_save_and_reload(self):
        destination = LessonStore()
        destination.import_transferable(supported_store(8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'store.json'
            destination.save(path)
            restored = LessonStore.load(path)
        self.assertEqual(restored.to_dict(), destination.to_dict())
        self.assertEqual(len(restored.retrieve(['critical'], 'COMPRESS')), 1)

    def test_invalid_source_promotion_fails_before_partial_import(self):
        source = supported_store()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad-store.json'
            rows = source.to_dict()
            rows[0]['support_count'] = 1
            path.write_text(json.dumps(rows))
            invalid = LessonStore.load(path)
        destination = LessonStore()
        with self.assertRaisesRegex(ValueError, 'promotion threshold'):
            destination.import_transferable(invalid)
        self.assertEqual(destination.to_dict(), [])


class ExperimentTests(unittest.TestCase):
    def test_same_compressor_makes_empty_managers_equivalent(self):
        case = next(c for c in generate_cases(2, 'test', 1) if c['family'] == 'critical_compression')
        baseline = run_episode(case, NonLearningManager(compressor=truncate_compressor), 'baseline')
        class NoModelCalls:
            def complete(self, *args, **kwargs):
                raise AssertionError('An explicit compressor must bypass the model client.')
        learning = run_episode(case, LessonAwareManager(LessonStore(), NoModelCalls(),
                                                        compressor=truncate_compressor), 'learning')
        self.assertEqual(baseline['ledger'].active_context(), learning['ledger'].active_context())
        self.assertEqual(baseline['answer'], learning['answer'])

    def test_executor_cannot_get_answer_from_hidden_metadata(self):
        case = generate_cases(2, 'test', 1)[0]
        task = to_task(case)
        wrong_metadata = replace(task, family='invented', required_outputs=['fake_answer'],
                                 expected_constraints=['fake_answer'], forbidden_outputs=['fake_answer'])
        context = '\n'.join(o.content for o in task.observations)
        self.assertEqual(json.loads(frozen_executor(wrong_metadata, context)), case['targets'])
        self.assertTrue(all(value is None for value in json.loads(frozen_executor(wrong_metadata, '')).values()))

    def test_split_check_catches_duplicate_inputs_with_different_ids(self):
        left = {'incident_id': 'a', 'raw_text': 'same log', 'question': 'same question'}
        right = {**left, 'incident_id': 'b', 'required_outputs': ['different target']}
        audit = audit_splits({'train': [left], 'test': [right]})
        self.assertFalse(audit['valid'])
        self.assertEqual(audit['cross_split_pairs']['train->test']['duplicate_ids'], [])
        self.assertEqual(audit['cross_split_pairs']['train->test']['duplicate_destination_inputs'], 1)

    def test_generated_splits_have_distinct_inputs(self):
        splits = {name: generate_cases(2, name, 5) for name in ['train', 'validation', 'test']}
        self.assertTrue(audit_splits(splits)['valid'])
        self.assertEqual(generate_cases(2, 'test', 5), splits['test'])

    def test_verifier_rejects_forged_answers_replays_and_missing_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run_study(root, seeds=(7,), per_family=5)
            episode_path = root / 'seed-7/raw/episodes.jsonl.gz'
            original_episodes = episode_path.read_bytes()
            for mutation in ['invent_answer', 'remove_null']:
                with self.subTest(mutation=mutation):
                    records = [json.loads(line) for line in gzip.decompress(original_episodes).decode().splitlines()]
                    failed = next(r for r in records if r['condition'] == 'train' and not r['evaluation']['success'])
                    if mutation == 'invent_answer':
                        failed['answer'] = json.dumps(failed['targets'], sort_keys=True)
                        failed['evaluation']['success'] = 1
                    else:
                        del failed['null_intervention_answer']
                    episode_path.write_bytes(gzip.compress(('\n'.join(json.dumps(r) for r in records) + '\n').encode()))
                    with self.assertRaisesRegex(ValueError, 'visible in the ledger|Missing null intervention'):
                        verify_artifacts(root)
                    episode_path.write_bytes(original_episodes)
            replay_path = root / 'seed-7/raw/replays.jsonl'
            replay_rows = [json.loads(line) for line in replay_path.read_text().splitlines()]
            harmful = next(r for r in replay_rows if r['label'] == 'HARMFUL')
            harmful['counterfactual_answer'] = '{}'
            harmful['counterfactual_score'] = 0.0
            harmful['label'] = 'NEUTRAL'
            replay_path.write_text('\n'.join(json.dumps(r) for r in replay_rows) + '\n')
            with self.assertRaisesRegex(ValueError, 'reconstructed intervention'):
                verify_artifacts(root)

    def test_end_to_end_ablation_transfer_negative_controls_and_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = run_study(root, seeds=(7,), per_family=5)
            aggregate = result['aggregate_test']
            self.assertEqual(aggregate['full_context']['successes'], 30)
            self.assertEqual(aggregate['empty_store']['successes'], 5)
            self.assertEqual(aggregate['legacy_transfer_b'], aggregate['empty_store'])
            self.assertEqual(aggregate['trusted_transfer_b'], aggregate['learned_a'])
            self.assertEqual(aggregate['learned_a']['successes'], 20)
            summary = json.loads((root / 'seed-7/summary.json').read_text())
            self.assertEqual(summary['test']['without_archive_lessons']['by_family']['false_archive']['successes'], 0)
            self.assertEqual(summary['test']['without_compression_lessons']['by_family']['critical_compression']['successes'], 0)
            for family in ['info_mistag', 'joint_loss']:
                self.assertEqual(summary['test']['learned_a']['by_family'][family]['successes'], 0)
            self.assertEqual(summary['replays_by_family']['joint_loss'], {'NEUTRAL': 10})
            self.assertEqual(summary['null_intervention_checks'], 25)
            records = [json.loads(line) for line in gzip.decompress(
                (root / 'seed-7/raw/episodes.jsonl.gz').read_bytes()).decode().splitlines()]
            test_ids = {c['task_id'] for c in [json.loads(line) for line in
                        (root / 'seed-7/data/test.jsonl').read_text().splitlines()]}
            test_records = [r for r in records if r['task_id'] in test_ids]
            self.assertEqual(len(test_records), 7 * 30)
            wins = sum(json.loads(r['answer']) == r['targets'] for r in test_records if r['condition'] == 'learned_a')
            self.assertEqual(wins, 20)
            self.assertTrue(all('items' in r['ledger'] and 'decisions' in r['ledger'] for r in records))
            verified = verify_artifacts(root)
            self.assertTrue(verified['valid'])
            self.assertEqual(verified['replays_checked'], 25)
            result['aggregate_test']['learned_a']['successes'] = 29
            (root / 'summary.json').write_text(json.dumps(result))
            with self.assertRaisesRegex(ValueError, 'Summary differs'):
                verify_artifacts(root)


if __name__ == '__main__':
    unittest.main()
