#!/usr/bin/env python3
"""Export reward metrics, curves and timings to the project download folder.

No model imports, GPU, training, or dataset access. CSV needs only Python's
standard library; plots additionally use Matplotlib. Pre-generated plots for
the verified pilot are included in the accompanying reporting ZIP.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def check_summary(summary):
    if summary.get('runner') != 'reward_from_ids' or summary.get('status') != 'passed':
        raise ValueError('Require a passed reward_from_ids summary.')
    config = summary['resolved_config']
    if digest(config) != summary['resolved_config_sha256']:
        raise ValueError('Resolved configuration hash differs from the saved summary.')
    methods = list(config['inputs']['selected_ids'])
    if summary['methods'] != methods or set(summary['results']) != set(methods):
        raise ValueError('Method lists differ between configuration and results.')
    expected_steps = math.ceil(summary['k'] / config['reward']['effective_batch_pairs']) * config['epochs']
    for method in methods:
        result = summary['results'][method]
        if (result['train_pairs'], result['validation_pairs']) != (summary['k'], summary['validation_pairs']):
            raise ValueError(f'{method}: train/validation sizes differ from the matched protocol.')
        if result['optimizer_steps'] != expected_steps:
            raise ValueError(f'{method}: optimizer-step count differs from the configured schedule.')
        history = result['history']
        if [h['epoch'] for h in history] != list(range(config['epochs'] + 1)):
            raise ValueError(f'{method}: incomplete or reordered epoch history.')
        for row in history:
            if not math.isfinite(row['acc']) or not math.isfinite(row['loss']):
                raise ValueError(f'{method}: nonfinite validation metrics.')
            if not 0 <= row['acc'] <= 1 or row['loss'] < 0:
                raise ValueError(f'{method}: invalid validation metrics.')
        best = min(history[1:], key=lambda row: row['loss'])
        if result['best_validation_epoch'] != {k: best[k] for k in ('epoch', 'acc', 'loss')}:
            raise ValueError(f'{method}: best epoch differs from the minimum trained-epoch loss.')
    if len({r['initial_head_sha256'] for r in summary['results'].values()}) != 1:
        raise ValueError('Methods have different initial reward-head hashes.')
    if summary.get('test_set_evaluated') is not False:
        raise ValueError('This reporting command labels validation-only runs; test use differs.')
    return methods


def check_reload(summary, reload_report, summary_path):
    if reload_report is None:
        return {method: '' for method in summary['methods']}
    if reload_report.get('status') != 'passed' or reload_report.get('test_set_evaluated') is not False:
        raise ValueError('Require a passed, validation-only checkpoint reload report.')
    # Editor downloads can convert LF to CRLF. Accept that serialization change
    # while still checking the original training summary hash and identities.
    raw = summary_path.read_bytes()
    hashes = {hashlib.sha256(raw).hexdigest(),
              hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()}
    if reload_report['training_summary_sha256'] not in hashes:
        raise ValueError('Reload report refers to different training-summary bytes.')
    for key in ('resolved_config_sha256', 'validation_content_sha256'):
        if reload_report[key] != summary[key]:
            raise ValueError(f'Reload {key} differs from training.')
    if set(reload_report['results']) != set(summary['methods']):
        raise ValueError('Reload methods differ from training.')
    verified = {}
    limit = summary['resolved_config']['reward']['max_seq_length']
    validation_lengths = reload_report['validation_token_lengths']
    if validation_lengths['pairs'] != summary['validation_pairs'] or validation_lengths['max_tokens'] > limit:
        raise ValueError('Reload validation token-length evidence differs from the configured data.')
    for method in summary['methods']:
        result = summary['results'][method]
        check = reload_report['results'][method]
        expected = result['best_validation_epoch']
        if check['expected'] != expected or check['checkpoint_sha256'] != result['checkpoint']['state_dict_sha256']:
            raise ValueError(f'{method}: reload checkpoint identity/expected metrics differ.')
        actual = check['reloaded']
        if (not check['passed'] or not math.isfinite(actual['acc']) or not math.isfinite(actual['loss'])
                or abs(actual['acc'] - expected['acc']) > 1e-7
                or abs(actual['loss'] - expected['loss']) > 1e-6):
            raise ValueError(f'{method}: reload metrics did not reproduce training.')
        lengths = check['training_token_lengths']
        if lengths['pairs'] != result['train_pairs'] or lengths['max_tokens'] > limit:
            raise ValueError(f'{method}: reload training token-length evidence differs.')
        verified[method] = True
    return verified


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def make_rows(summary, verified):
    run_id = Path(summary['output']).name
    metrics, curves = [], []
    config = summary['resolved_config']
    for method in summary['methods']:
        result = summary['results'][method]
        best = result['best_validation_epoch']
        metrics.append(dict(
            run_id=run_id, method=method, pool_size=summary['N'], selected_pairs=summary['k'],
            validation_pairs=summary['validation_pairs'], training_seed=config['training_seed'],
            epochs=config['epochs'], optimizer_steps=result['optimizer_steps'],
            best_validation_epoch=best['epoch'], best_validation_accuracy=best['acc'],
            best_validation_pairwise_loss=best['loss'],
            last_epoch_validation_accuracy=result['history'][-1]['acc'],
            last_epoch_validation_pairwise_loss=result['history'][-1]['loss'],
            training_seconds=result['training_seconds'], validation_seconds=result['validation_seconds'],
            checkpoint_saved=result['checkpoint_saved'], checkpoint_reload_verified=verified[method],
            test_set_evaluated=False, gpu=summary['gpu'], slurm_job=summary['slurm_job'],
            training_git_commit=summary['git']['stdout'],
            resolved_config_sha256=summary['resolved_config_sha256'],
            pool_content_sha256=summary['pool_content_sha256'],
            validation_content_sha256=summary['validation_content_sha256']))
        for epoch in result['history']:
            curves.append(dict(run_id=run_id, method=method, epoch=epoch['epoch'],
                               validation_accuracy=epoch['acc'], validation_pairwise_loss=epoch['loss'],
                               training_pairwise_loss=epoch.get('train_loss_pair_weighted', ''),
                               optimizer_steps=epoch.get('optimizer_steps', 0)))
    return metrics, curves


def save_plot(fig, path):
    fig.savefig(path, dpi=220, facecolor='white')
    if path.suffix == '.svg':
        lines = path.read_bytes().splitlines()
        path.write_bytes(b'\n'.join(line.rstrip(b' \t\r') for line in lines) + b'\n')


def make_plots(summary, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, FormatStrFormatter
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    colors = ['#2563EB', '#C05B12', '#16857A', '#A43B76']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    all_accuracies = [50.0]
    all_losses = [math.log(2)]
    for index, method in enumerate(summary['methods']):
        history = summary['results'][method]['history']
        epochs = [r['epoch'] for r in history]
        acc = [100 * r['acc'] for r in history]
        loss = [r['loss'] for r in history]
        all_accuracies.extend(acc)
        all_losses.extend(loss)
        label = {'random': 'Random', 'kcenter': 'k-center'}.get(method, method)
        style = dict(label=label, color=colors[index % len(colors)], marker=['o', 's', '^', 'D'][index % 4],
                     linewidth=2.1, markersize=6)
        axes[0].plot(epochs, acc, **style)
        axes[1].plot(epochs, loss, **style)
    axes[0].axhline(50, color='#667085', linestyle='--', linewidth=1, label='Chance (50%)')
    axes[1].axhline(math.log(2), color='#667085', linestyle='--', linewidth=1, label='Equal-score loss')
    axes[0].set(ylabel='Validation accuracy (%)', title='Preference accuracy')
    axes[1].set(ylabel='Validation pairwise loss', title='Pairwise loss')
    axes[0].set_ylim(max(0, 5 * math.floor((min(all_accuracies) - 5) / 5)),
                     min(100, 5 * math.ceil((max(all_accuracies) + 5) / 5)))
    padding = max(0.001, 0.15 * (max(all_losses) - min(all_losses)))
    axes[1].set_ylim(max(0, min(all_losses) - padding), max(all_losses) + padding)
    axes[1].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    for ax in axes:
        ax.set_xlabel('Epoch (0 = before training)')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', alpha=0.18)
        ax.legend(frameon=False, fontsize=9, loc='best')
    fig.suptitle('Reward model integration pilot', x=0.07, ha='left', fontsize=17, fontweight='bold')
    fig.text(0.07, 0.875, f'N={summary["N"]:,}; selected={summary["k"]:,} per method; '
             f'validation={summary["validation_pairs"]:,}; training seed={summary["resolved_config"]["training_seed"]}',
             fontsize=10, color='#475467')
    fig.text(0.07, 0.035, 'Single run. Accuracy remains near chance. Test set unused.',
             fontsize=10, color='#475467')
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.19, wspace=0.30)
    for ext in ('png', 'svg'):
        save_plot(fig, output / f'validation_curves.{ext}')
    plt.close(fig)

    methods = summary['methods']
    labels = [{'random': 'Random', 'kcenter': 'k-center'}.get(m, m) for m in methods]
    train = [summary['results'][m]['training_seconds'] for m in methods]
    validation = [summary['results'][m]['validation_seconds'] for m in methods]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.bar(labels, train, width=0.48, color='#2563EB', label='Training')
    ax.bar(labels, validation, bottom=train, width=0.48, color='#72B6AC',
           label='Validation (initial + post-epoch)')
    for i, (a, b) in enumerate(zip(train, validation)):
        ax.text(i, a / 2, f'{a:.2f} s', ha='center', va='center', color='white', fontweight='bold')
        ax.text(i, a + b / 2, f'{b:.2f} s', ha='center', va='center', color='#123E3B', fontweight='bold')
    ax.set_ylabel('Measured stage time (seconds)')
    ax.set_ylim(0, max(a + b for a, b in zip(train, validation)) * 1.25)
    ax.grid(axis='y', alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    fig.suptitle('Reward training and validation stages', x=0.11, ha='left', fontsize=16, fontweight='bold')
    fig.text(0.11, 0.865, f'{summary["gpu"]}; {summary["resolved_config"]["epochs"]} epoch; '
             f'{summary["k"]} training pairs per method', fontsize=10, color='#475467')
    fig.text(0.11, 0.045, 'Single ordered run; first-use overhead differs. Descriptive timings only.\n'
             'Excludes model setup, tokenization, checkpoint I/O, embeddings and selection.', fontsize=9, color='#475467')
    fig.subplots_adjust(left=0.11, right=0.98, top=0.75, bottom=0.22)
    for ext in ('png', 'svg'):
        save_plot(fig, output / f'training_validation_time.{ext}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--reload-report', type=Path)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--no-plots', action='store_true', help='Export CSV and JSON using only the standard library.')
    parser.add_argument('--overwrite', action='store_true', help='Replace previously generated report files.')
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding='utf-8'))
    check_summary(summary)
    reload_report = json.loads(args.reload_report.read_text(encoding='utf-8')) if args.reload_report else None
    verified = check_reload(summary, reload_report, args.summary)
    metrics, curves = make_rows(summary, verified)
    run_id = Path(summary['output']).name
    if not run_id or run_id in ('.', '..'):
        raise ValueError('Invalid run directory name.')
    output = args.output_dir or ROOT / 'results' / 'downloads' / run_id
    output = output.expanduser().resolve()
    names = ['reward_metrics.csv', 'learning_curves.csv', 'stages.csv', 'report_provenance.json']
    if not args.no_plots:
        try:
            import matplotlib
        except ImportError:
            parser.error('Matplotlib is required for plots. Use --no-plots for CSV-only export.')
        names.extend(['validation_curves.png', 'validation_curves.svg',
                      'training_validation_time.png', 'training_validation_time.svg'])
    if not args.overwrite and any((output / name).exists() for name in names):
        parser.error('Generated reports already exist; inspect them or use --overwrite intentionally.')
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'reward_metrics.csv', metrics)
    write_csv(output / 'learning_curves.csv', curves)
    stage_rows = [{key: stage.get(key, '') for key in
                   ('stage', 'status', 'seconds', 'process_peak_rss_mib_so_far',
                    'gpu_peak_allocated_mib', 'gpu_peak_reserved_mib')} for stage in summary['stages']]
    write_csv(output / 'stages.csv', stage_rows)
    if not args.no_plots:
        make_plots(summary, output)
    provenance = dict(
        schema_version=1, run_id=run_id, source_run=summary['output'],
        summary_sha256=hashlib.sha256(args.summary.read_bytes()).hexdigest(),
        summary_lf_sha256=hashlib.sha256(args.summary.read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
        summary_content_sha256=digest(summary),
        reload_report_sha256=hashlib.sha256(args.reload_report.read_bytes()).hexdigest() if args.reload_report else None,
        reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        report_text_format='LF line endings; SVG trailing spaces removed',
        resolved_config_sha256=summary['resolved_config_sha256'], checkpoint_reload_verified=verified,
        test_set_evaluated=False, files=names,
        notes=['Integration pilot; one seed and a small selected subset.',
               'Best trained epoch is selected by validation loss; epoch 0 is diagnostic.',
               'Timings describe one ordered run and include first-use effects.',
               'No test result, uncertainty across seeds, or full-data quality comparison is available.'])
    (output / 'report_provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
    archive = shutil.make_archive(str(output.parent / f'{run_id}_reports'), 'zip', output)
    print(f'REPORT DIRECTORY: {output}')
    print(f'DOWNLOAD ZIP: {archive}')
    print('REWARD REPORT EXPORT PASSED')


if __name__ == '__main__':
    main()
