import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from carc_profile import Profiler, check_reference, method_costs, subset_size
from carc_pilot import write_parsed, verify_sources
from parse import group_id, split_of


class CarcPilotTests(unittest.TestCase):
    def test_source_snapshot(self):
        verify_sources(json.loads((ROOT / 'configs/carc_pilot.json').read_text()))

    def test_budget_boundaries(self):
        self.assertEqual(subset_size(2000, .1), 200)
        self.assertEqual(subset_size(20000, .2), 4000)
        self.assertEqual(subset_size(13, .1), 1)
        for n, f in [(0, .1), (5, .1), (10, 0), (10, 1.1)]:
            with self.assertRaises(ValueError):
                subset_size(n, f)

    def test_reference_guard(self):
        check_reference({'count': 2}, {'count': 2}, ['a', 'b'], ['a', 'b'])
        with self.assertRaises(ValueError):
            check_reference({'count': 3}, {'count': 2}, ['a', 'b'], ['a', 'b'])
        with self.assertRaises(ValueError):
            check_reference({'count': 2}, {'count': 2}, ['b', 'a'], ['a', 'b'])

    def test_stage_failure_is_saved_and_raised(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = Profiler(Path(directory) / 'stages.jsonl')
            with self.assertRaisesRegex(RuntimeError, 'test error'):
                with profile.stage('failure'):
                    raise RuntimeError('test error')
            record = json.loads(profile.path.read_text())
            self.assertEqual(record['status'], 'failed')
            self.assertGreaterEqual(record['seconds'], 0)

    def test_method_accounting_excludes_unneeded_embeddings(self):
        records = [dict(stage=s, seconds=t, status='ok') for s, t in [
            ('read_pool_and_validation', 2), ('encoder_load', 3), ('embed_context', 5),
            ('random/selection', 1), ('kcenter/selection', 4), ('reward_import_tokenizer', 2),
            ('tokenize_validation', 3), ('random/training_epoch_1', 10),
            ('kcenter/training_epoch_1', 11), ('full_pool/training_epoch_1', 20),
            ('feature_artifact_write', 100), ('verify_prepared_inputs', 200),
        ]]
        totals = method_costs(records, ['random', 'kcenter', 'full_pool'], 7)
        self.assertEqual(totals['random']['sum_measured_required_stage_seconds'], 18)
        self.assertEqual(totals['kcenter']['sum_measured_required_stage_seconds'], 30)
        self.assertEqual(totals['full_pool']['sum_measured_required_stage_seconds'], 27)
        self.assertEqual(totals['random']['plus_one_preprocessing_pass_seconds'], 25)

    def test_streamed_parser_keeps_source_ids_and_group_policy(self):
        contexts = {}
        for i in range(200):
            context = f'\n\nHuman: question number {i}'
            contexts.setdefault(split_of(group_id(context)), context)
        self.assertEqual(set(contexts), {'train', 'val', 'test'})
        def row(context, a='answer A', b='answer B'):
            return {'chosen': context + '\n\nAssistant: ' + a,
                    'rejected': context + '\n\nAssistant: ' + b}
        # Official test rows may become training in Katherine's pooled split.
        dataset = {'train': [row(contexts['val']), row(contexts['test']), row(contexts['train'], 'same', 'same')],
                   'test': [row(contexts['train'])]}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            report, ids = write_parsed(dataset, destination)
            self.assertEqual(report['counts'], {'train': 1, 'val': 1, 'test': 1})
            self.assertEqual(ids, ['test-0'])
            self.assertEqual(report['dropped'], {'identical_responses': 1})
            self.assertTrue(all(v == 0 for v in report['cross_split_group_overlap'].values()))
            record = json.loads((destination / 'train.jsonl').read_text())
            self.assertEqual(record['source'], 'test')
            self.assertEqual(record['context'], contexts['train'].strip())


if __name__ == '__main__':
    unittest.main()
