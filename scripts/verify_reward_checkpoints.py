#!/usr/bin/env python3
"""Inspect saved reward checkpoints or reload and verify validation metrics.

Requires the earlier reward_inputs/reward_from_ids increment. The original
training summary and checkpoint files are read-only. No test data is used.
"""
import argparse
import copy
import gc
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import uuid

from carc_pilot import ROOT, command_output, package_versions, verify_sources
from carc_profile import Profiler, dump_json, sha_file
from reward_inputs import configure_reward, content_sha, resolve_config, validate_inputs


def load_bundle(run_dir, work_root):
    """CPU-only identity checks before loading model weights."""
    summary = json.loads((run_dir / 'summary.json').read_text(encoding='utf-8'))
    if summary.get('runner') != 'reward_from_ids' or summary.get('status') != 'passed':
        raise ValueError('Require a passed reward_from_ids run.')
    config = summary['resolved_config']
    if content_sha(config) != summary['resolved_config_sha256']:
        raise ValueError('Saved resolved configuration hash differs from the summary.')
    if json.loads((run_dir / 'resolved_config.json').read_text()) != config:
        raise ValueError('resolved_config.json differs from the training summary.')
    reference = json.loads((ROOT / 'configs/carc_pilot.json').read_text())
    verify_sources(reference)
    raw = copy.deepcopy(config)
    raw['reward'] = {key: config['reward'][key] for key in
        ('max_seq_length', 'reply_token_cap', 'microbatch_pairs', 'effective_batch_pairs', 'lr')}
    if resolve_config(raw, reference['reward']) != config:
        raise ValueError('Saved configuration differs from the supported pinned protocol.')
    for name in ('scripts/reward_inputs.py', 'scripts/carc_pilot.py',
                 'scripts/carc_profile.py', 'src/reward.py'):
        if sha_file(ROOT / name) != summary['source_sha256'].get(name):
            raise ValueError(f'{name} changed since training; review before checking reload.')
    methods = list(config['inputs']['selected_ids'])
    if summary['methods'] != methods or set(summary['results']) != set(methods):
        raise ValueError('Method lists differ between saved configuration and results.')
    saved = copy.deepcopy(config)
    saved['inputs'] = {
        'pool_ids': str(run_dir / 'pool_ids.json'),
        'validation_ids': str(run_dir / 'validation_ids.json'),
        'selected_ids': {m: str(run_dir / f'{m}_selected_ids.json') for m in methods},
    }
    inputs, pool, validation, selected_ids = validate_inputs(work_root, saved, reference)
    for key in ('N', 'k', 'validation_pairs', 'ordered_pool_ids_sha256',
                'ordered_validation_ids_sha256', 'pool_content_sha256',
                'validation_content_sha256', 'prepared_manifest_sha256', 'prepared_file_sha256'):
        if inputs[key] != summary[key]:
            raise ValueError(f'{key} differs from the training run.')
    checkpoints = {}
    for method in methods:
        result = summary['results'][method]
        if not result.get('checkpoint_saved'):
            raise ValueError(f'{method}: no saved checkpoint was recorded.')
        metadata = json.loads((run_dir / f'{method}_best_metadata.json').read_text())
        if metadata != result['checkpoint']:
            raise ValueError(f'{method}: checkpoint sidecar differs from the training summary.')
        filename = metadata['state_dict']
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError(f'{method}: invalid checkpoint filename.')
        path = run_dir / filename
        if sha_file(path) != metadata['state_dict_sha256']:
            raise ValueError(f'{method}: checkpoint bytes differ from the saved hash.')
        expected = {
            'reward': config['reward'], 'training_seed': config['training_seed'],
            'checkpoint_policy': 'lowest validation loss among trained epochs',
            'best_validation_epoch': result['best_validation_epoch'],
            'selected_ids_sha256': content_sha(selected_ids[method]),
            'ordered_pool_ids_sha256': summary['ordered_pool_ids_sha256'],
            'ordered_validation_ids_sha256': summary['ordered_validation_ids_sha256'],
            'resolved_config_sha256': summary['resolved_config_sha256'],
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise ValueError(f'{method}: checkpoint metadata {key} differs from the run.')
        checkpoints[method] = path
    return summary, pool, validation, selected_ids, checkpoints


def token_length_stats(pairs, limit):
    if not pairs or any(len(pair) != 2 for pair in pairs):
        raise ValueError('Require nonempty encoded chosen/rejected pairs.')
    lengths = [len(sequence) for pair in pairs for sequence in pair]
    if min(lengths) < 3 or max(lengths) > limit:
        raise ValueError(f'Encoded token length outside [3,{limit}]: min={min(lengths)}, max={max(lengths)}.')
    return dict(pairs=len(pairs), sequences=len(lengths), min_tokens=min(lengths),
                max_tokens=max(lengths), configured_token_limit=limit)


def compare_metrics(expected, actual, accuracy_atol=1e-7, loss_atol=1e-6):
    values = [expected['acc'], expected['loss'], actual['acc'], actual['loss']]
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError('Metrics must be finite numbers.')
    if any(not 0 <= m['acc'] <= 1 or m['loss'] < 0 for m in (expected, actual)):
        raise ValueError('Accuracy must be in [0,1] and loss must be nonnegative.')
    da = abs(actual['acc'] - expected['acc'])
    dl = abs(actual['loss'] - expected['loss'])
    return dict(expected=expected, reloaded=actual, accuracy_absolute_difference=da,
                loss_absolute_difference=dl, accuracy_atol=accuracy_atol, loss_atol=loss_atol,
                passed=da <= accuracy_atol and dl <= loss_atol)


def verify_gpu(args):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Checkpoint evaluation needs an allocated GPU compute session.')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    suffix = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    output = args.run_dir / f'checkpoint_reload_{suffix}.json'
    profile = Profiler(output.with_suffix('.stages.jsonl'), torch)
    report = dict(status='running', run_dir=str(args.run_dir), test_set_evaluated=False,
                  hostname=socket.gethostname(), slurm_job=os.environ.get('SLURM_JOB_ID'),
                  gpu=torch.cuda.get_device_name(0), packages=package_versions(),
                  git=command_output(['git', 'rev-parse', 'HEAD']),
                  verifier_sha256=sha_file(Path(__file__)), results={})
    try:
        with profile.stage('verify_bundle_and_inputs'):
            summary, pool, validation, selected_ids, checkpoints = load_bundle(args.run_dir, args.work_root)
            report['training_summary_sha256'] = sha_file(args.run_dir / 'summary.json')
            report['resolved_config_sha256'] = summary['resolved_config_sha256']
            report['validation_content_sha256'] = summary['validation_content_sha256']
        with profile.stage('reward_import_tokenizer'):
            import reward
            configure_reward(reward, summary['resolved_config']['reward'])
        with profile.stage('tokenize_validation_and_check_lengths'):
            vd = reward.prep(validation)
            report['validation_token_lengths'] = token_length_stats(vd, reward.L)
        by_id = {r['id']: r for r in pool}
        for method, path in checkpoints.items():
            with profile.stage(f'{method}/check_training_token_lengths'):
                tokens = reward.prep([by_id[i] for i in selected_ids[method]])
                training_lengths = token_length_stats(tokens, reward.L)
                del tokens
            with profile.stage(f'{method}/reload_checkpoint', gpu=True):
                state = torch.load(path, map_location='cpu', weights_only=True)
                model = reward.RM().to('cuda')
                model.load_state_dict(state, strict=True)
                del state
            with profile.stage(f'{method}/evaluate_validation', gpu=True):
                actual = reward.evaluate(model, vd)
            expected = summary['results'][method]['best_validation_epoch']
            comparison = compare_metrics(expected, actual)
            comparison['training_token_lengths'] = training_lengths
            comparison['checkpoint_sha256'] = sha_file(path)
            report['results'][method] = comparison
            dump_json(output, report)
            print(json.dumps({'method': method, **comparison}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        report['status'] = 'passed' if all(r['passed'] for r in report['results'].values()) else 'failed'
        if report['status'] != 'passed':
            raise ValueError('Reloaded validation metrics differ beyond the recorded tolerances.')
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['stages'] = profile.records
        dump_json(output, report)
        print(f'RELOAD REPORT: {output}', flush=True)
    print('CHECKPOINT RELOAD VERIFICATION PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['inspect', 'verify'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--work-root', type=Path, required=True)
    args = parser.parse_args()
    args.run_dir = args.run_dir.expanduser().resolve()
    args.work_root = args.work_root.expanduser().resolve()
    if args.action == 'inspect':
        summary, _, _, _, checkpoints = load_bundle(args.run_dir, args.work_root)
        print(json.dumps(dict(status='inspected', methods=list(checkpoints),
                              validation_pairs=summary['validation_pairs'],
                              test_set_evaluated=False), indent=2))
        print('CHECKPOINT BUNDLE INSPECTION PASSED')
    else:
        verify_gpu(args)


if __name__ == '__main__':
    main()
