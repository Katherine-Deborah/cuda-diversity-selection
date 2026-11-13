#!/usr/bin/env python3
"""Reproduce Katherine's pilot with additive CARC profiling and pinned inputs.

prepare: CPU compute node; pinned download, same parsing/splits, cache models.
run: GPU compute node; cached/offline model loading, stage timings, no test use.
Original src/*.py and data/*.json are deliberately not modified.
"""
import argparse
import collections
import csv
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from carc_profile import Profiler, check_reference, dump_json, method_costs, sha_file, subset_size


def command_output(args):
    try:
        p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, timeout=30)
        return {'returncode': p.returncode, 'stdout': p.stdout.strip(), 'stderr': p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'error': str(exc)}


def verify_sources(config):
    for name, expected in config['source_sha256'].items():
        if sha_file(ROOT / name) != expected:
            raise ValueError(f'{name} changed from the reviewed snapshot. Review the change before reproducing.')


def write_parsed(ds, destination):
    """Use the original parsing/grouping functions and original traversal order."""
    from parse import parse_pair, group_id, split_of
    counts = collections.Counter()
    groups = {s: set() for s in ('train', 'val', 'test')}
    train_ids = []
    drops = collections.Counter()
    handles = {s: (destination / f'{s}.jsonl').open('w', encoding='utf-8') for s in groups}
    try:
        with (destination / 'excluded_rows.csv').open('w', newline='') as audit:
            writer = csv.writer(audit)
            writer.writerow(['id', 'reason'])
            for source in ('train', 'test'):
                for i, example in enumerate(ds[source]):
                    parsed, reason = parse_pair(example['chosen'], example['rejected'])
                    identity = f'{source}-{i}'
                    if parsed is None:
                        drops[reason] += 1
                        writer.writerow([identity, reason])
                        continue
                    context, a, b = parsed
                    group = group_id(context)
                    split = split_of(group)
                    row = dict(id=identity, group=group, split=split, source=source,
                               context=context, chosen=a, rejected=b)
                    handles[split].write(json.dumps(row) + '\n')
                    counts[split] += 1
                    groups[split].add(group)
                    if split == 'train':
                        train_ids.append(identity)
    finally:
        for handle in handles.values():
            handle.close()
    overlap = {f'{a}&{b}': len(groups[a] & groups[b])
               for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
    if any(overlap.values()):
        raise ValueError('Cross-split opening-prompt overlap detected.')
    return dict(counts=dict(counts), groups={s: len(g) for s, g in groups.items()},
                dropped=dict(drops), cross_split_group_overlap=overlap), train_ids


def prepare(args, config):
    from datasets import load_dataset
    from huggingface_hub import snapshot_download
    data = args.work_root / 'prepared'
    if data.exists():
        raise FileExistsError(f'{data} already exists. Keep it; choose another --work-root for a new preparation.')
    data.mkdir(parents=True)
    profile = Profiler(data / 'prepare_stages.jsonl')
    with profile.stage('dataset_load_download_and_arrow'):
        ds = load_dataset(config['dataset']['name'], revision=config['dataset']['revision'])
    if {s: len(ds[s]) for s in ('train', 'test')} != {'train': 160800, 'test': 8552}:
        raise ValueError('Unexpected raw dataset counts.')
    with profile.stage('parse_filter_split_write'):
        report, ids = write_parsed(ds, data)
    dump_json(data / 'parse_report.json', report)
    expected = json.loads((ROOT / 'data/parse_report.json').read_text())
    reference_ids = json.loads((ROOT / 'data/pilot_ids.json').read_text())
    generated_ids = random.Random(0).sample(ids, len(reference_ids))
    check_reference(report, expected, generated_ids, reference_ids)
    print('PARSING COUNTS AND ORIGINAL PILOT IDS MATCH', flush=True)
    del ds
    gc.collect()

    snapshots = {}
    patterns = ['config.json', 'modules.json', 'sentence_bert_config.json',
                'config_sentence_transformers.json', '1_Pooling/config.json',
                'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                'special_tokens_map.json', 'vocab.txt', 'added_tokens.json']
    for key in ('encoder', 'reward'):
        spec = config[key]
        with profile.stage(f'{key}_download_or_cache'):
            snapshot = Path(snapshot_download(spec['name'], revision=spec['revision'], allow_patterns=patterns))
            if not (snapshot / 'model.safetensors').is_file():
                raise FileNotFoundError(f'Pinned {key} snapshot lacks model.safetensors.')
        snapshots[key] = str(snapshot.resolve())

    manifest = {
        'schema_version': 1, 'source_commit_reviewed': config['source_commit'],
        'source_sha256': config['source_sha256'], 'dataset': config['dataset'],
        'encoder': config['encoder'], 'reward': config['reward'],
        'snapshots': snapshots, 'parse_report': report,
        'file_sha256': {f'{s}.jsonl': sha_file(data / f'{s}.jsonl') for s in ('train', 'val', 'test')},
        'original_pilot_ids_verified': True,
        'prepare_stages': profile.records,
        'packages': package_versions(),
        'note': 'Katherine opening-prompt pooled split. Not Mohan original official-test split.',
    }
    dump_json(data / 'manifest.json', manifest)
    print(json.dumps(report, indent=2), flush=True)
    print(f'PREPARATION PASSED: {data / "manifest.json"}', flush=True)


def package_versions():
    result = {}
    for name in ('torch', 'numpy', 'transformers', 'sentence-transformers', 'datasets',
                 'huggingface-hub', 'tokenizers', 'safetensors', 'pyarrow'):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def load_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def train_one(reward, torch, np, profile, method, train_data, val_data, epochs, seed, output, save_checkpoint):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    gc.collect()
    torch.cuda.empty_cache()
    with profile.stage(f'{method}/model_and_optimizer', gpu=True):
        model = reward.RM().to('cuda')
        opt = torch.optim.AdamW(model.parameters(), lr=reward.LR)
    initial_head = hashlib.sha256(b''.join(
        p.detach().cpu().numpy().tobytes() for p in model.head.parameters())).hexdigest()
    history = []
    with profile.stage(f'{method}/validation_initial', gpu=True):
        history.append(dict(epoch=0, **reward.evaluate(model, val_data)))
    generator = random.Random(seed)
    steps = 0
    best = None
    for epoch in range(1, epochs + 1):
        model.train()
        indices = list(range(len(train_data)))
        generator.shuffle(indices)
        micro_losses = []
        weighted_loss = 0.0
        examples = 0
        with profile.stage(f'{method}/training_epoch_{epoch}', gpu=True):
            for start in range(0, len(indices), reward.EFF_BS):
                batch = [train_data[i] for i in indices[start:start + reward.EFF_BS]]
                opt.zero_grad()
                for j in range(0, len(batch), reward.MICRO):
                    micro = batch[j:j + reward.MICRO]
                    a, b = reward.scores(model, micro)
                    # Same sample-weighted gradient accumulation as the source.
                    loss = torch.nn.functional.softplus(-(a - b)).sum() / len(batch)
                    loss.backward()
                    scalar = loss.item()
                    micro_losses.append(scalar * len(batch) / len(micro))
                    weighted_loss += scalar * len(batch)
                    examples += len(micro)
                opt.step()
                steps += 1
                if steps % 25 == 0:
                    print(f'{method}: epoch {epoch}, optimizer step {steps}', flush=True)
        with profile.stage(f'{method}/validation_epoch_{epoch}', gpu=True):
            metrics = reward.evaluate(model, val_data)
        row = dict(epoch=epoch, train_loss_original_micro_mean=float(np.mean(micro_losses)),
                   train_loss_pair_weighted=weighted_loss / examples, optimizer_steps=steps, **metrics)
        history.append(row)
        if best is None or metrics['loss'] < best['loss']:
            best = dict(epoch=epoch, **metrics)
            if save_checkpoint:
                with profile.stage(f'{method}/checkpoint_epoch_{epoch}', gpu=True):
                    torch.save(model.state_dict(), output / f'{method}_best.pt')
        dump_json(output / f'{method}_history.json', history)
        print(json.dumps({'method': method, **row}), flush=True)
    result = dict(train_pairs=len(train_data), validation_pairs=len(val_data), history=history,
                  best_validation_epoch=best, initial_head_sha256=initial_head,
                  optimizer_steps=steps, checkpoint_saved=save_checkpoint)
    del model, opt
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run(args, config):
    # Models were staged by prepare. Do not include network transfers in timings.
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Run this stage in an allocated GPU compute session.')
    torch.set_num_threads(int(os.environ.get('SLURM_CPUS_PER_TASK', '2')))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    data = args.work_root / 'prepared'
    manifest = json.loads((data / 'manifest.json').read_text())
    for key in ('dataset', 'encoder', 'reward', 'source_sha256'):
        if manifest[key] != config[key]:
            raise ValueError(f'Prepared {key} does not match this configuration.')
    preset = config['presets'][args.preset].copy()
    if args.validation_size is not None:
        preset['validation_size'] = args.validation_size
    if args.budget is not None:
        preset['budget'] = args.budget
    n = preset['pool_size']
    k = subset_size(n, preset['budget'])
    methods = list(preset['methods'])
    if args.include_full_pool and 'full_pool' not in methods:
        methods.append('full_pool')
    identity = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '_' + uuid.uuid4().hex[:8]
    output = args.work_root / 'runs' / f'{args.preset}_seed{args.seed}_{identity}'
    output.mkdir(parents=True, exist_ok=False)
    profile = Profiler(output / 'stages.jsonl', torch)
    wall_start = time.perf_counter()
    report = dict(status='running', preset=args.preset, settings=preset, methods=methods,
                  N=n, k=k, training_and_random_selection_seed=args.seed, pool_and_validation_seed=0,
                  kcenter_start=0, output=str(output), model_config=config, packages=package_versions(),
                  python=sys.version, hostname=socket.gethostname(), slurm_job=os.environ.get('SLURM_JOB_ID'),
                  slurm_cpus=os.environ.get('SLURM_CPUS_PER_TASK'),
                  gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  pytorch_cuda=torch.version.cuda, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                  tf32=False, gpu_info=command_output(['nvidia-smi']), git=command_output(['git', 'rev-parse', 'HEAD']),
                  git_status=command_output(['git', 'status', '--short']),
                  test_set_evaluated=False, results={})
    dump_json(output / 'summary.json', report)
    try:
        with profile.stage('verify_prepared_inputs'):
            for split in ('train', 'val'):
                if sha_file(data / f'{split}.jsonl') != manifest['file_sha256'][f'{split}.jsonl']:
                    raise ValueError(f'{split}.jsonl changed after preparation.')
        with profile.stage('read_pool_and_validation'):
            all_train = load_rows(data / 'train.jsonl')
            if n == 2000:
                wanted = json.loads((ROOT / 'data/pilot_ids.json').read_text())
                by_id = {row['id']: row for row in all_train}
                pool = [by_id[i] for i in wanted]
                del by_id
            else:
                pool = random.Random(0).sample(all_train, n)
            del all_train
            validation = random.Random(0).sample(load_rows(data / 'val.jsonl'), preset['validation_size'])
            if {r['group'] for r in pool} & {r['group'] for r in validation}:
                raise ValueError('Pilot/validation prompt overlap.')
        dump_json(output / 'pool_ids.json', [r['id'] for r in pool])
        dump_json(output / 'validation_ids.json', [r['id'] for r in validation])
        # A content hash helps future runs detect identical IDs with different text.
        report['pool_content_sha256'] = hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()
        report['validation_content_sha256'] = hashlib.sha256(json.dumps(validation, sort_keys=True).encode()).hexdigest()

        selections = {}
        if 'random' in methods:
            with profile.stage('random/selection'):
                selections['random'] = random.Random(args.seed).sample(range(n), k)
        if 'kcenter' in methods:
            from sentence_transformers import SentenceTransformer
            from embed import encode_context, encode_response
            from features import checks, sym_features
            from normalize import feat_final
            from kcenter import kcenter
            with profile.stage('encoder_load', gpu=True):
                encoder = SentenceTransformer(manifest['snapshots']['encoder'], device='cuda')
                encoder.max_seq_length = config['encoder']['max_seq_length']
                if encoder.get_sentence_embedding_dimension() != 384:
                    raise ValueError('Unexpected encoder feature dimension.')
            with profile.stage('embed_context', gpu=True):
                P = encode_context(encoder, [r['context'] for r in pool])
            with profile.stage('embed_chosen', gpu=True):
                A = encode_response(encoder, [r['chosen'] for r in pool])
            with profile.stage('embed_rejected', gpu=True):
                B = encode_response(encoder, [r['rejected'] for r in pool])
            with profile.stage('features'):
                X = feat_final(P, A, B)
                report['feature_checks'] = checks(P, A, B, sym_features(P, A, B))
                if not np.allclose(np.linalg.norm(X, axis=1), 1, atol=1e-5):
                    raise ValueError('Feature normalization failed.')
                report['feature_shape'] = list(X.shape)
                report['feature_sha256'] = hashlib.sha256(X.tobytes()).hexdigest()
            with profile.stage('feature_artifact_write'):
                np.save(output / 'features.npy', X)
            del encoder, P, A, B
            gc.collect()
            torch.cuda.empty_cache()
            with profile.stage('kcenter/selection'):
                selections['kcenter'] = kcenter(X, k, start=0)
            del X
        if 'full_pool' in methods:
            selections['full_pool'] = list(range(n))
        for method, indices in selections.items():
            expected_size = n if method == 'full_pool' else k
            if len(indices) != expected_size or len(set(indices)) != expected_size:
                raise ValueError(f'Invalid selection for {method}.')
            dump_json(output / f'{method}_selected_ids.json', [pool[i]['id'] for i in indices])

        with profile.stage('reward_import_tokenizer'):
            import reward
            if reward.REV != config['reward']['revision']:
                raise ValueError('Reward-model revision differs from the reviewed source.')
        with profile.stage('tokenize_validation'):
            vd = reward.prep(validation)
        for method in methods:
            with profile.stage(f'{method}/tokenize_training'):
                td = reward.prep([pool[i] for i in selections[method]])
            report['results'][method] = train_one(reward, torch, np, profile, method, td, vd,
                preset['epochs'], args.seed, output, args.save_checkpoints)
            del td
            dump_json(output / 'summary.json', report)
        heads = {r['initial_head_sha256'] for r in report['results'].values()}
        if len(heads) != 1:
            raise ValueError('Methods did not start with the same reward-head initialization.')
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['total_run_wall_seconds'] = time.perf_counter() - wall_start
        report['stages'] = profile.records
        report['sum_stage_seconds'] = sum(r['seconds'] for r in profile.records)
        prep_seconds = sum(r['seconds'] for r in manifest['prepare_stages'] if r['stage'] == 'parse_filter_split_write')
        report['per_method_stage_cost_estimates'] = method_costs(profile.records, methods, prep_seconds)
        report['cost_notes'] = [
            'Actual run wall time covers all requested methods and orchestration; stage sums are separate.',
            'Per-method totals sum measured required stages, not independently measured standalone wall times.',
            'Random/full_pool do not require selector embeddings. Common stages charged once per scenario.',
            'Downloads, integrity checks and feature-artifact writes are excluded from per-method stage totals.',
            'Preprocessing includes parsing/filtering/splitting/writing the entire dataset, once per scenario.',
            'GPU peaks are absolute PyTorch allocator peaks; reserved includes cached allocator memory.',
            'RSS is process lifetime high-water mark, not whole-job or stage-specific RAM.',
            'Timing includes first-use compute overhead; no separate warm-up speedup claim.',
            'Cross-device/software numerical equality with laptop results is not guaranteed.',
        ]
        dump_json(output / 'summary.json', report)
        dump_json(ROOT / 'results/carc' / f'{output.name}.json', report)
        with (output / 'stages.csv').open('w', newline='') as stream:
            fields = ['stage', 'status', 'seconds', 'gpu_peak_allocated_mib', 'gpu_peak_reserved_mib', 'process_peak_rss_mib_so_far', 'error']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(profile.records)
        print(f'SUMMARY: {output / "summary.json"}', flush=True)
    print('CARC PILOT PASSED', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run'])
    parser.add_argument('--work-root', type=Path, required=True)
    parser.add_argument('--preset', choices=['reproduce', 'signal', 'main'], default='reproduce')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--validation-size', type=int)
    parser.add_argument('--budget', type=float)
    parser.add_argument('--include-full-pool', action='store_true')
    parser.add_argument('--save-checkpoints', action='store_true')
    args = parser.parse_args()
    args.work_root = args.work_root.expanduser().resolve()
    if args.work_root == ROOT or ROOT in args.work_root.parents:
        parser.error('--work-root must be outside the repository (use scratch storage).')
    config = json.loads((ROOT / 'configs/carc_pilot.json').read_text())
    verify_sources(config)
    if args.action == 'prepare':
        prepare(args, config)
    else:
        run(args, config)


if __name__ == '__main__':
    main()
