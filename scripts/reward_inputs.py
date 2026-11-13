"""CPU-only configuration and input checks for external-ID reward training."""
import hashlib
import json
import math
from pathlib import Path
import re

from carc_profile import sha_file


def positive_int(value, label, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{label} must be an integer >= {minimum}.')
    return value


def require_keys(value, allowed, required, label):
    if not isinstance(value, dict):
        raise ValueError(f'{label} must be an object.')
    unknown = set(value) - set(allowed)
    missing = set(required) - set(value)
    if unknown or missing:
        raise ValueError(f'{label}: unknown keys {sorted(unknown)}, missing keys {sorted(missing)}.')


def resolve_config(raw, reference_reward):
    """Validate overrides; retain the pinned model, FP32, and AdamW protocol."""
    keys = {'schema_version', 'epochs', 'training_seed', 'budget', 'reward',
            'inputs', 'save_checkpoints'}
    require_keys(raw, keys, keys, 'configuration')
    if type(raw['schema_version']) is not int or raw['schema_version'] != 1:
        raise ValueError('Require schema_version=1.')
    positive_int(raw['epochs'], 'epochs')
    positive_int(raw['budget'], 'budget')
    positive_int(raw['training_seed'], 'training_seed', minimum=0)
    if raw['training_seed'] >= 2**32:
        raise ValueError('training_seed must be < 2**32 for NumPy seeding.')
    if type(raw['save_checkpoints']) is not bool:
        raise ValueError('save_checkpoints must be true or false.')

    allowed_reward = {'max_seq_length', 'reply_token_cap', 'microbatch_pairs',
                      'effective_batch_pairs', 'lr'}
    require_keys(raw['reward'], allowed_reward, set(), 'reward overrides')
    reward = dict(reference_reward)
    reward.update(raw['reward'])
    if reward['precision'] != 'float32':
        raise ValueError('This runner supports the reviewed float32 protocol only.')
    for key in ('max_seq_length', 'reply_token_cap', 'microbatch_pairs', 'effective_batch_pairs'):
        positive_int(reward[key], f'reward.{key}')
    if not 4 <= reward['max_seq_length'] <= 512:
        raise ValueError('DistilBERT total token limit must be between 4 and 512.')
    if reward['reply_token_cap'] > reward['max_seq_length'] - 3:
        raise ValueError('reply_token_cap must leave room for three special tokens.')
    if reward['microbatch_pairs'] > reward['effective_batch_pairs']:
        raise ValueError('microbatch_pairs must not exceed effective_batch_pairs.')
    lr = reward['lr']
    if type(lr) not in (int, float) or not math.isfinite(lr) or lr <= 0:
        raise ValueError('reward.lr must be finite and positive.')

    inputs = raw['inputs']
    require_keys(inputs, {'pool_ids', 'validation_ids', 'selected_ids'},
                 {'pool_ids', 'validation_ids', 'selected_ids'}, 'inputs')
    for label in ('pool_ids', 'validation_ids'):
        if not isinstance(inputs[label], str) or not inputs[label].strip():
            raise ValueError(f'inputs.{label} must be a nonempty file path.')
    methods = inputs['selected_ids']
    if not isinstance(methods, dict) or not methods:
        raise ValueError('inputs.selected_ids must contain at least one method.')
    for method, path in methods.items():
        if not re.fullmatch(r'[a-z][a-z0-9_-]{0,31}', method):
            raise ValueError(f'Invalid method name: {method!r}.')
        if not isinstance(path, str) or not path.strip():
            raise ValueError(f'{method} selected IDs need a nonempty file path.')
    resolved = dict(raw)
    resolved['reward'] = reward
    # Copy mutable input mappings so callers cannot alter the original config.
    resolved['inputs'] = dict(inputs, selected_ids=dict(methods))
    return resolved


def configure_reward(reward, spec):
    """Adapt the legacy module's globals without altering its reviewed source.

    Existing encode/prep/train_one functions read these values at call time.
    The tokenizer and RM model remain pinned to the original name/revision.
    """
    if reward.NAME != spec['name'] or reward.REV != spec['revision']:
        raise ValueError('Reward module model/revision differs from the pinned reference.')
    reward.L = spec['max_seq_length']
    reward.R = spec['reply_token_cap']
    reward.MICRO = spec['microbatch_pairs']
    reward.EFF_BS = spec['effective_batch_pairs']
    reward.LR = spec['lr']


def content_sha(value):
    # Same ordered-row content hash convention used by carc_pilot.py.
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def input_path(work_root, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (work_root / path).resolve()


def read_ids(path, label):
    ids = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(ids, list) or not ids:
        raise ValueError(f'{label} must be a nonempty JSON array of pair-ID strings.')
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError(f'{label} must contain pair-ID strings, not row indices.')
    if len(set(ids)) != len(ids):
        raise ValueError(f'{label} contains duplicate IDs.')
    return ids


def read_requested_rows(path, wanted, split):
    """Stream the prepared split, preserving only requested rows in memory."""
    requested = set(wanted)
    found, seen = {}, set()
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f'{path.name}:{line_number} is not a row object.')
            identity = row.get('id')
            if not isinstance(identity, str) or not identity.strip():
                raise ValueError(f'{path.name}:{line_number} lacks a valid pair ID.')
            if identity in seen:
                raise ValueError(f'{path.name} contains duplicate row ID {identity!r}.')
            seen.add(identity)
            if row.get('split') != split:
                raise ValueError(f'{identity!r} has wrong split metadata in {path.name}.')
            if identity in requested:
                for key in ('group', 'context', 'chosen', 'rejected'):
                    if not isinstance(row.get(key), str) or not row[key].strip():
                        raise ValueError(f'{identity!r} lacks valid {key}.')
                found[identity] = row
    missing = requested - found.keys()
    if missing:
        raise ValueError(f'{split}: {len(missing)} requested IDs are absent; first: {sorted(missing)[:3]}.')
    return [found[i] for i in wanted]


def validate_inputs(work_root, config, reference):
    """Validate identities and prepared hashes; never open/evaluate test data."""
    prepared = work_root / 'prepared'
    manifest_path = prepared / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    for key in ('dataset', 'encoder', 'source_sha256'):
        if manifest.get(key) != reference[key]:
            raise ValueError(f'Prepared {key} differs from the reviewed reference.')
    for key in ('name', 'revision'):
        if manifest.get('reward', {}).get(key) != config['reward'][key]:
            raise ValueError(f'Prepared reward {key} differs from the configured model.')
    prepared_hashes = {}
    for split in ('train', 'val'):
        name = f'{split}.jsonl'
        actual = sha_file(prepared / name)
        if manifest.get('file_sha256', {}).get(name) != actual:
            raise ValueError(f'{name} changed after preparation.')
        prepared_hashes[name] = actual

    paths = config['inputs']
    pool_path = input_path(work_root, paths['pool_ids'])
    val_path = input_path(work_root, paths['validation_ids'])
    pool_ids = read_ids(pool_path, 'pool IDs')
    validation_ids = read_ids(val_path, 'validation IDs')
    pool_set = set(pool_ids)
    if config['budget'] > len(pool_ids):
        raise ValueError('budget exceeds pool size.')
    if pool_set & set(validation_ids):
        raise ValueError('Pool and validation pair IDs overlap.')

    selected_ids, selected_paths, input_hashes = {}, {}, {
        'pool_ids': sha_file(pool_path), 'validation_ids': sha_file(val_path)}
    for method, name in paths['selected_ids'].items():
        path = input_path(work_root, name)
        ids = read_ids(path, f'{method} selected IDs')
        if len(ids) != config['budget']:
            raise ValueError(f'{method}: require exactly budget={config["budget"]} selected IDs.')
        if not set(ids) <= pool_set:
            raise ValueError(f'{method}: selected IDs are outside the ordered pool.')
        selected_ids[method] = ids
        selected_paths[method] = str(path)
        input_hashes[f'{method}_selected_ids'] = sha_file(path)

    pool = read_requested_rows(prepared / 'train.jsonl', pool_ids, 'train')
    validation = read_requested_rows(prepared / 'val.jsonl', validation_ids, 'val')
    if {r['group'] for r in pool} & {r['group'] for r in validation}:
        raise ValueError('Pool and validation prompt groups overlap.')
    report = {
        'N': len(pool_ids), 'k': config['budget'], 'validation_pairs': len(validation_ids),
        'methods': list(selected_ids), 'input_file_sha256': input_hashes,
        'ordered_pool_ids_sha256': content_sha(pool_ids),
        'ordered_validation_ids_sha256': content_sha(validation_ids),
        'pool_content_sha256': content_sha(pool),
        'validation_content_sha256': content_sha(validation),
        'prepared_manifest_sha256': sha_file(manifest_path),
        'prepared_file_sha256': prepared_hashes,
        'input_paths': {'pool_ids': str(pool_path), 'validation_ids': str(val_path),
                        'selected_ids': selected_paths},
        'test_set_evaluated': False,
        'order_policy': 'Preserve each supplied ordered ID list; no sorting or resampling.',
    }
    return report, pool, validation, selected_ids
