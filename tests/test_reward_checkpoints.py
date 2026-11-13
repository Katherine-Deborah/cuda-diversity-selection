import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

import test_reward_inputs as fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from carc_profile import dump_json, sha_file
from reward_from_ids import source_hashes
from reward_inputs import content_sha
from verify_reward_checkpoints import compare_metrics, load_bundle, token_length_stats


class RewardCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ExternalRewardInputsTests('test_valid_pooled_source_ids_and_preserved_order')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.work = self.fixture.work
        self.run = self.work / 'runs' / 'fixture'
        self.run.mkdir(parents=True)
        config = self.fixture.config()
        inputs, _, _, selections = self.fixture.validate()
        summary = dict(inputs, runner='reward_from_ids', status='passed', resolved_config=config,
                       resolved_config_sha256=content_sha(config), source_sha256=source_hashes(), results={})
        dump_json(self.run / 'resolved_config.json', config)
        dump_json(self.run / 'pool_ids.json', self.fixture.ids['pool'])
        dump_json(self.run / 'validation_ids.json', self.fixture.ids['val'])
        for method, ids in selections.items():
            checkpoint = self.run / f'{method}_best.pt'
            # Bundle inspection checks bytes/hashes only, never deserializes these fixtures.
            checkpoint.write_bytes(f'fixture:{method}'.encode())
            expected = dict(epoch=1, acc=0.51, loss=0.69)
            metadata = dict(
                state_dict=checkpoint.name, state_dict_sha256=sha_file(checkpoint),
                reward=config['reward'], training_seed=config['training_seed'],
                checkpoint_policy='lowest validation loss among trained epochs',
                best_validation_epoch=expected, selected_ids_sha256=content_sha(ids),
                ordered_pool_ids_sha256=inputs['ordered_pool_ids_sha256'],
                ordered_validation_ids_sha256=inputs['ordered_validation_ids_sha256'],
                resolved_config_sha256=summary['resolved_config_sha256'])
            summary['results'][method] = dict(checkpoint_saved=True, checkpoint=metadata,
                                             best_validation_epoch=expected)
            dump_json(self.run / f'{method}_selected_ids.json', ids)
            dump_json(self.run / f'{method}_best_metadata.json', metadata)
        self.summary = summary
        dump_json(self.run / 'summary.json', summary)

    def load(self):
        return load_bundle(self.run, self.work)

    def test_valid_bundle_uses_saved_ids_without_changing_training_summary(self):
        before = (self.run / 'summary.json').read_bytes()
        summary, pool, val, methods, checkpoints = self.load()
        self.assertEqual(list(checkpoints), ['random', 'kcenter'])
        self.assertEqual(methods['random'], self.fixture.ids['random'])
        self.assertEqual([r['id'] for r in pool], self.fixture.ids['pool'])
        self.assertEqual((self.run / 'summary.json').read_bytes(), before)

    def test_changed_checkpoint_bytes_rejected(self):
        (self.run / 'random_best.pt').write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError, 'checkpoint bytes'):
            self.load()

    def test_changed_sidecar_rejected(self):
        path = self.run / 'random_best_metadata.json'
        metadata = json.loads(path.read_text())
        metadata['training_seed'] = 2
        dump_json(path, metadata)
        with self.assertRaisesRegex(ValueError, 'sidecar differs'):
            self.load()

    def test_changed_selected_id_order_rejected(self):
        dump_json(self.run / 'random_selected_ids.json', list(reversed(self.fixture.ids['random'])))
        with self.assertRaisesRegex(ValueError, 'selected_ids_sha256'):
            self.load()

    def test_changed_saved_configuration_rejected(self):
        config = copy.deepcopy(self.summary['resolved_config'])
        config['epochs'] = 2
        dump_json(self.run / 'resolved_config.json', config)
        with self.assertRaisesRegex(ValueError, 'resolved_config.json differs'):
            self.load()

    def test_changed_source_hash_rejected(self):
        self.summary['source_sha256']['scripts/reward_inputs.py'] = 'wrong hash'
        dump_json(self.run / 'summary.json', self.summary)
        with self.assertRaisesRegex(ValueError, 'changed since training'):
            self.load()

    def test_inconsistent_method_lists_rejected(self):
        self.summary['methods'] = ['random']
        dump_json(self.run / 'summary.json', self.summary)
        with self.assertRaisesRegex(ValueError, 'Method lists differ'):
            self.load()

    def test_changed_prepared_content_rejected(self):
        self.fixture.train[0]['chosen'] = 'changed text'
        self.fixture.write_prepared()
        with self.assertRaisesRegex(ValueError, 'differs from the training run'):
            self.load()

    def test_matching_metrics_pass(self):
        metrics = dict(acc=0.507, loss=0.6925)
        self.assertTrue(compare_metrics(metrics, metrics)['passed'])

    def test_loss_tolerance(self):
        expected = dict(acc=0.5, loss=0.69)
        self.assertTrue(compare_metrics(expected, dict(acc=0.5, loss=0.6900005))['passed'])
        self.assertFalse(compare_metrics(expected, dict(acc=0.5, loss=0.69001))['passed'])

    def test_one_pair_accuracy_difference_is_not_hidden(self):
        self.assertFalse(compare_metrics(dict(acc=0.5, loss=0.69),
                                         dict(acc=0.501, loss=0.69))['passed'])

    def test_nonfinite_metrics_rejected(self):
        for value in (float('nan'), float('inf'), -0.1, 1.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                compare_metrics(dict(acc=0.5, loss=0.69), dict(acc=value, loss=0.69))

    def test_token_length_boundary_passes(self):
        stats = token_length_stats([([1, 2, 3, 4], [1, 2, 3])], 4)
        self.assertEqual(stats['max_tokens'], 4)

    def test_over_limit_tokens_rejected(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            token_length_stats([([1, 2, 3, 4, 5], [1, 2, 3])], 4)

    def test_empty_or_short_encoded_pairs_rejected(self):
        for pairs in ([], [([1, 2], [1, 2, 3])], [([1, 2, 3],)]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                token_length_stats(pairs, 4)

    def test_inspect_cli_needs_no_model_import(self):
        p = subprocess.run([sys.executable, str(ROOT / 'scripts/verify_reward_checkpoints.py'),
                            'inspect', '--run-dir', str(self.run), '--work-root', str(self.work)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('CHECKPOINT BUNDLE INSPECTION PASSED', p.stdout)


if __name__ == '__main__':
    unittest.main()
