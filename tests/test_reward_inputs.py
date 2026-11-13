import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from carc_profile import sha_file
from reward_inputs import configure_reward, resolve_config, validate_inputs


class ExternalRewardInputsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.prepared = self.work / 'prepared'
        self.prepared.mkdir()
        self.reference = json.loads((ROOT / 'configs/carc_pilot.json').read_text())
        def row(identity, group, split='train'):
            return dict(id=identity, group=group, split=split, context='Human: question',
                        chosen='preferred answer', rejected='other answer')
        # A source-prefixed test ID can legitimately be in the pooled training split.
        self.train = [row('test-0', 'g0'), row('train-1', 'g1'), row('train-2', 'g2')]
        self.val = [row('train-9', 'v0', 'val')]
        self.ids = {'pool': ['train-1', 'test-0', 'train-2'], 'val': ['train-9'],
                    'random': ['train-2', 'test-0'], 'kcenter': ['test-0', 'train-1']}
        for name, ids in self.ids.items():
            self.write_ids(name, ids)
        self.raw = dict(schema_version=1, epochs=1, training_seed=0, budget=2,
                        save_checkpoints=True, reward={}, inputs={
                            'pool_ids': 'pool.json', 'validation_ids': 'val.json',
                            'selected_ids': {'random': 'random.json', 'kcenter': 'kcenter.json'}})
        self.write_prepared()

    def write_ids(self, name, ids):
        (self.work / f'{name}.json').write_text(json.dumps(ids))

    def write_prepared(self):
        for split, rows in [('train', self.train), ('val', self.val)]:
            (self.prepared / f'{split}.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows))
        manifest = {key: self.reference[key] for key in
                    ('dataset', 'encoder', 'reward', 'source_sha256')}
        manifest['file_sha256'] = {
            f'{split}.jsonl': sha_file(self.prepared / f'{split}.jsonl')
            for split in ('train', 'val')}
        (self.prepared / 'manifest.json').write_text(json.dumps(manifest))

    def config(self, raw=None):
        return resolve_config(raw if raw is not None else self.raw, self.reference['reward'])

    def validate(self, raw=None):
        return validate_inputs(self.work, self.config(raw), self.reference)

    def test_valid_pooled_source_ids_and_preserved_order(self):
        report, pool, val, methods = self.validate()
        self.assertEqual([r['id'] for r in pool], self.ids['pool'])
        self.assertEqual([r['id'] for r in val], self.ids['val'])
        self.assertEqual(methods['random'], self.ids['random'])
        self.assertEqual((report['N'], report['k'], report['validation_pairs']), (3, 2, 1))
        self.assertFalse(report['test_set_evaluated'])
        self.assertFalse((self.prepared / 'test.jsonl').exists())

    def test_pool_order_is_fingerprinted(self):
        first, *_ = self.validate()
        self.write_ids('pool', list(reversed(self.ids['pool'])))
        second, *_ = self.validate()
        self.assertNotEqual(first['ordered_pool_ids_sha256'], second['ordered_pool_ids_sha256'])
        self.assertNotEqual(first['pool_content_sha256'], second['pool_content_sha256'])

    def test_duplicate_pool_ids_rejected(self):
        self.write_ids('pool', ['test-0', 'test-0', 'train-1'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.validate()

    def test_duplicate_selected_ids_rejected(self):
        self.write_ids('random', ['test-0', 'test-0'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.validate()

    def test_row_indices_cannot_masquerade_as_pair_ids(self):
        self.write_ids('random', [0, 1])
        with self.assertRaisesRegex(ValueError, 'not row indices'):
            self.validate()

    def test_selected_id_outside_pool_rejected(self):
        self.write_ids('random', ['train-9', 'test-0'])
        with self.assertRaisesRegex(ValueError, 'outside'):
            self.validate()

    def test_budget_mismatch_rejected(self):
        self.write_ids('random', ['test-0'])
        with self.assertRaisesRegex(ValueError, 'exactly budget'):
            self.validate()

    def test_empty_validation_rejected(self):
        self.write_ids('val', [])
        with self.assertRaisesRegex(ValueError, 'nonempty'):
            self.validate()

    def test_pair_overlap_rejected(self):
        self.write_ids('val', ['test-0'])
        with self.assertRaisesRegex(ValueError, 'pair IDs overlap'):
            self.validate()

    def test_prompt_group_overlap_rejected(self):
        self.val[0]['group'] = 'g0'
        self.write_prepared()
        with self.assertRaisesRegex(ValueError, 'prompt groups overlap'):
            self.validate()

    def test_missing_training_id_rejected(self):
        self.train = self.train[:-1]
        self.write_prepared()
        with self.assertRaisesRegex(ValueError, 'requested IDs are absent'):
            self.validate()

    def test_wrong_split_metadata_rejected(self):
        self.train[0]['split'] = 'val'
        self.write_prepared()
        with self.assertRaisesRegex(ValueError, 'wrong split metadata'):
            self.validate()

    def test_duplicate_prepared_rows_rejected(self):
        self.train.append(dict(self.train[0]))
        self.write_prepared()
        with self.assertRaisesRegex(ValueError, 'duplicate row ID'):
            self.validate()

    def test_changed_prepared_text_rejected_by_hash(self):
        with (self.prepared / 'train.jsonl').open('a') as stream:
            stream.write('\n')
        with self.assertRaisesRegex(ValueError, 'changed after preparation'):
            self.validate()

    def test_training_overrides_do_not_require_new_preparation(self):
        raw = copy.deepcopy(self.raw)
        raw['reward'] = dict(lr=0.0001, max_seq_length=64, reply_token_cap=16,
                             microbatch_pairs=3, effective_batch_pairs=7)
        report, *_ = self.validate(raw)
        self.assertEqual(report['k'], 2)
        self.assertEqual(self.config(raw)['reward']['lr'], 0.0001)

    def test_invalid_configuration_rejected_before_model_import(self):
        cases = [('epochs', 0), ('epochs', True), ('training_seed', -1),
                 ('training_seed', 2**32), ('budget', False), ('save_checkpoints', 'yes')]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                raw = copy.deepcopy(self.raw)
                raw[key] = value
                with self.assertRaises(ValueError):
                    self.config(raw)
        for changes in [dict(lr=float('nan')), dict(lr=0), dict(lr=True),
                        dict(max_seq_length=3), dict(max_seq_length=513),
                        dict(max_seq_length=8, reply_token_cap=6),
                        dict(microbatch_pairs=33), dict(precision='float16'),
                        dict(learning_rate=0.1)]:
            with self.subTest(reward=changes):
                raw = copy.deepcopy(self.raw)
                raw['reward'].update(changes)
                with self.assertRaises(ValueError):
                    self.config(raw)

    def test_unsafe_method_names_rejected(self):
        raw = copy.deepcopy(self.raw)
        raw['inputs']['selected_ids'] = {'../escape': 'random.json'}
        with self.assertRaisesRegex(ValueError, 'Invalid method name'):
            self.config(raw)

    def test_legacy_tokenizer_function_uses_configured_limits(self):
        spec = self.config()['reward']
        spec.update(max_seq_length=8, reply_token_cap=3,
                    microbatch_pairs=3, effective_batch_pairs=7, lr=0.0001)
        legacy = types.SimpleNamespace(NAME=spec['name'], REV=spec['revision'])
        class Tokenizer:
            cls_token_id, sep_token_id = 1, 2
            def __call__(self, text, **kwargs):
                return {'input_ids': [int(t) for t in text.split()]}
        legacy.tok = Tokenizer()
        # Exercise the existing encode function without importing models/Torch.
        tree = ast.parse((ROOT / 'src/reward.py').read_text())
        encode = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'encode')
        exec(compile(ast.Module(body=[encode], type_ignores=[]), 'legacy_encode', 'exec'), legacy.__dict__)
        configure_reward(legacy, spec)
        result = legacy.encode('10 11 12 13 14 15', '20 21 22 23')
        self.assertEqual(result, [1, 14, 15, 2, 20, 21, 22, 2])
        self.assertEqual((legacy.MICRO, legacy.EFF_BS, legacy.LR), (3, 7, 0.0001))

    def test_mismatched_model_revision_rejected(self):
        spec = self.config()['reward']
        legacy = types.SimpleNamespace(NAME=spec['name'], REV='another revision')
        with self.assertRaisesRegex(ValueError, 'model/revision'):
            configure_reward(legacy, spec)

    def test_validate_cli_on_fixture(self):
        path = self.work / 'config.json'
        path.write_text(json.dumps(self.raw))
        p = subprocess.run([sys.executable, str(ROOT / 'scripts/reward_from_ids.py'),
                            'validate', '--work-root', str(self.work), '--config', str(path)],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('EXTERNAL-ID INPUT VALIDATION PASSED', p.stdout)


if __name__ == '__main__':
    unittest.main()
