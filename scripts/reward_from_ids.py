#!/usr/bin/env python3
"""Validate external pair IDs or train with the existing profiled CARC loop.

    python scripts/reward_from_ids.py validate --work-root ROOT --config CONFIG
    python scripts/reward_from_ids.py train --work-root ROOT --config CONFIG

validate uses only the standard library. train requires an allocated CUDA GPU
and the existing ee451-pilot environment/cache. Neither action uses test data.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import sys
import time
import uuid

from carc_pilot import ROOT, command_output, package_versions, train_one, verify_sources
from carc_profile import Profiler, dump_json, sha_file
from reward_inputs import configure_reward, content_sha, resolve_config, validate_inputs


def load_configuration(config_path):
    reference = json.loads((ROOT / 'configs/carc_pilot.json').read_text(encoding='utf-8'))
    verify_sources(reference)
    raw = json.loads(config_path.read_text(encoding='utf-8'))
    resolved = resolve_config(raw, reference['reward'])
    return reference, resolved


def source_hashes():
    names = ['scripts/reward_from_ids.py', 'scripts/reward_inputs.py',
             'scripts/carc_pilot.py', 'scripts/carc_profile.py', 'src/reward.py']
    return {name: sha_file(ROOT / name) for name in names}


def run_training(args, config, reference):
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Training needs an allocated GPU compute session.')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    torch.set_default_dtype(torch.float32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    suffix = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    output = args.work_root / 'runs' / f'reward_ids_seed{config["training_seed"]}_{suffix}'
    output.mkdir(parents=True, exist_ok=False)
    profile = Profiler(output / 'stages.jsonl', torch)
    wall_start = time.perf_counter()
    report = {
        'schema_version': 1, 'runner': 'reward_from_ids', 'status': 'running',
        'output': str(output), 'resolved_config': config,
        'resolved_config_sha256': content_sha(config), 'config_file_sha256': sha_file(args.config),
        'source_sha256': source_hashes(), 'git': command_output(['git', 'rev-parse', 'HEAD']),
        'git_status': command_output(['git', 'status', '--short']),
        'packages': package_versions(), 'python': sys.version, 'hostname': socket.gethostname(),
        'slurm_job': os.environ.get('SLURM_JOB_ID'),
        'slurm_cpus': os.environ.get('SLURM_CPUS_PER_TASK'),
        'gpu': torch.cuda.get_device_name(0), 'pytorch_cuda': torch.version.cuda,
        'gpu_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
        'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'tf32': False, 'test_set_evaluated': False, 'results': {},
        'notes': [
            'External selections are consumed as supplied; selection/embedding cost is not measured here.',
            'Model initialization, training, validation and checkpoint writes reuse carc_pilot.train_one.',
            'AdamW defaults and evaluation batches of 16 pairs are fixed by the reviewed source.',
            'Wall time starts before input validation; excludes imports and initial GPU/setup checks.',
            'Checkpoint selection is lowest validation loss among trained epochs; epoch 0 is diagnostic.',
            'This runner does not provide interrupted-training resume or final test evaluation.',
        ],
    }
    dump_json(output / 'resolved_config.json', config)
    dump_json(output / 'summary.json', report)
    try:
        with profile.stage('read_pool_and_validation'):
            inputs, pool, validation, selected_ids = validate_inputs(args.work_root, config, reference)
            report.update(inputs)
        dump_json(output / 'pool_ids.json', [r['id'] for r in pool])
        dump_json(output / 'validation_ids.json', [r['id'] for r in validation])
        for method, ids in selected_ids.items():
            dump_json(output / f'{method}_selected_ids.json', ids)
        by_id = {r['id']: r for r in pool}
        with profile.stage('reward_import_tokenizer'):
            import reward
            configure_reward(reward, config['reward'])
        with profile.stage('tokenize_validation'):
            vd = reward.prep(validation)
        for method, ids in selected_ids.items():
            with profile.stage(f'{method}/tokenize_training'):
                td = reward.prep([by_id[i] for i in ids])
            result = train_one(reward, torch, np, profile, method, td, vd,
                               config['epochs'], config['training_seed'], output,
                               config['save_checkpoints'])
            if config['save_checkpoints']:
                checkpoint = output / f'{method}_best.pt'
                metadata = {
                    'state_dict': checkpoint.name, 'state_dict_sha256': sha_file(checkpoint),
                    'reward': config['reward'], 'training_seed': config['training_seed'],
                    'checkpoint_policy': 'lowest validation loss among trained epochs',
                    'best_validation_epoch': result['best_validation_epoch'],
                    'selected_ids_sha256': content_sha(ids),
                    'ordered_pool_ids_sha256': report['ordered_pool_ids_sha256'],
                    'ordered_validation_ids_sha256': report['ordered_validation_ids_sha256'],
                    'resolved_config_sha256': report['resolved_config_sha256'],
                }
                dump_json(output / f'{method}_best_metadata.json', metadata)
                result['checkpoint'] = metadata
            report['results'][method] = result
            dump_json(output / 'summary.json', report)
            del td
            gc.collect()
        if len({r['initial_head_sha256'] for r in report['results'].values()}) != 1:
            raise ValueError('Methods did not start with the same reward-head initialization.')
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['total_run_wall_seconds'] = time.perf_counter() - wall_start
        report['stages'] = profile.records
        report['sum_stage_seconds'] = sum(s['seconds'] for s in profile.records)
        for method, result in report['results'].items():
            result['training_seconds'] = sum(s['seconds'] for s in profile.records
                if s['stage'].startswith(f'{method}/training_epoch_'))
            result['validation_seconds'] = sum(s['seconds'] for s in profile.records
                if s['stage'].startswith(f'{method}/validation_'))
        dump_json(output / 'summary.json', report)
        print(f'SUMMARY: {output / "summary.json"}', flush=True)
    print('EXTERNAL-ID REWARD RUN PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['validate', 'train'])
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    args.work_root = args.work_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    if args.work_root == ROOT or ROOT in args.work_root.parents:
        parser.error('--work-root must be outside the source repository.')
    reference, config = load_configuration(args.config)
    if args.action == 'validate':
        report, _, _, _ = validate_inputs(args.work_root, config, reference)
        report.update(status='validated', resolved_config=config,
                      resolved_config_sha256=content_sha(config), source_sha256=source_hashes())
        print(json.dumps(report, indent=2, allow_nan=False))
        print('EXTERNAL-ID INPUT VALIDATION PASSED')
    else:
        run_training(args, config, reference)


if __name__ == '__main__':
    main()
