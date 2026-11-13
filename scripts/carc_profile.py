"""Small, CPU-testable helpers for the CARC feasibility runner."""
import contextlib
import hashlib
import json
import math
import resource
import time
from pathlib import Path


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def subset_size(n, fraction):
    if n < 1 or not 0 < fraction <= 1:
        raise ValueError('Require N >= 1 and 0 < budget <= 1.')
    k = math.floor(n * fraction)
    if k < 1:
        raise ValueError('The budget selects zero examples.')
    return k


def check_reference(actual_report, expected_report, sampled_ids, reference_ids):
    if actual_report != expected_report:
        raise ValueError('Parsing/split counts differ from Katherine\'s committed report. '
                         'Inspect parse_report.json; do not continue to training.')
    if sampled_ids != reference_ids:
        raise ValueError('Pilot IDs differ from the committed pilot_ids.json. '
                         'Dataset row ordering may differ; do not continue to training.')


class Profiler:
    """Synchronized wall time; absolute GPU peaks; process RSS high-water mark."""
    def __init__(self, path, torch_module=None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.torch = torch_module
        self.records = []

    @contextlib.contextmanager
    def stage(self, name, gpu=False):
        if gpu:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        record = {'stage': name, 'status': 'ok'}
        try:
            yield
            if gpu:
                self.torch.cuda.synchronize()
        except BaseException as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            raise
        finally:
            record['seconds'] = time.perf_counter() - start
            # Linux ru_maxrss is KiB. This is cumulative process peak, not a
            # resettable stage-local measurement and not total job memory.
            record['process_peak_rss_mib_so_far'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
            if gpu:
                record['gpu_peak_allocated_mib'] = self.torch.cuda.max_memory_allocated() / 2**20
                record['gpu_peak_reserved_mib'] = self.torch.cuda.max_memory_reserved() / 2**20
            self.records.append(record)
            with self.path.open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record), flush=True)


def method_costs(records, methods, preprocessing_seconds):
    """Accounting estimates from measured stages, not independent wall runs."""
    common_names = {'read_pool_and_validation', 'reward_import_tokenizer', 'tokenize_validation'}
    selection_names = {'encoder_load', 'embed_context', 'embed_chosen', 'embed_rejected', 'features'}
    result = {}
    for method in methods:
        chosen = [r for r in records if r['status'] == 'ok' and (
            r['stage'] in common_names
            or r['stage'].startswith(method + '/')
            or (method == 'kcenter' and r['stage'] in selection_names)
        )]
        measured = sum(r['seconds'] for r in chosen)
        result[method] = {
            'sum_measured_required_stage_seconds': measured,
            'plus_one_preprocessing_pass_seconds': measured + preprocessing_seconds,
            'stages_included': [r['stage'] for r in chosen],
        }
    return result
