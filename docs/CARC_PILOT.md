# CARC reproduction and feasibility profiling

Reviewed source: `a9c1e5eb3aaf29ca98a97504fb1bb808bb5c5b3d`.
This is an additive runner. Katherine's source, original reports, and committed
pilot IDs stay unchanged. It uses her pooled, first-human-message split and her
cleaning rules provisionally, not Mohan's earlier full-context/official-test split.
The team still needs to select one final data protocol.

## 1. Install in a separate environment on a CPU or GPU compute node

If using SSH, request a CPU node from the login shell first (skip if already on
an allocated compute node):

```bash
srun --account=prasanna_933 --partition=main --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=16G --time=01:00:00 --pty bash -l
```

Upload `carc_pilot_additions.zip` to the home directory, then:

```bash
cd ~/ee451-katherine-pilot
unzip ~/carc_pilot_additions.zip -d ~/ee451-katherine-pilot
module load gcc/13.3.0 cuda/12.6.3 conda
eval "$(conda shell.bash hook)"
conda create -n ee451-pilot --clone ee451 -y
conda activate ee451-pilot
source scripts/carc_env.sh
python -m pip install -r configs/requirements-carc-pilot.txt -c configs/carc-torch-constraint.txt
python -m pip check
python -m pytest -q tests
python -m pip freeze > configs/requirements-carc-resolved.lock.txt
```

Create the environment only once. On later sessions just activate it. If pip
reports a dependency conflict or tests fail, stop and retain the output. The
constraint protects the verified `torch==2.14.1+cu126` installation. The cloned
environment permits older Transformers/Sentence Transformers and a compatible
Dataset/Hub stack without changing `ee451`. These dependency versions need an
installation/import check on CARC; they have not been executed in the preparation
workspace, which has no Torch/CUDA installation.

## 2. Prepare pinned inputs (CPU sufficient)

```bash
python -u scripts/carc_pilot.py prepare --work-root "$EE451_PILOT_ROOT"
```

This loads the pinned HH-RLHF revision using Katherine's default file discovery
and source-row order, then applies her original functions. It verifies all
counts against `data/parse_report.json` and the ordered 2,000-pair sample against
`data/pilot_ids.json`. It stops if either differs. IDs alone do not prove
identical text; the new manifest also fingerprints the resulting JSONL files.
Expected counts: train 133,535; val 16,490; test 16,626; dropped 2,701.

It caches pinned model snapshots and writes
`$EE451_PILOT_ROOT/prepared/manifest.json`. The MiniLM revision was recorded after
the laptop runs, so an exact match to its originally loaded weights is not
established. DistilBERT's revision is pinned in the actual source. Existing
prepared directories are never overwritten. If preparation fails, preserve its
report and use a new work-root for a corrected retry.

Data and model download/cache time is separate from parsing/filtering/splitting/
serialization time. Model downloads include only the required safetensors and
tokenizer/config files. No reward-model test-set evaluation is performed.

## 3. First GPU reproduction

Request an OnDemand GPU session (Discovery, account prasanna_933, gpu partition,
one GPU, 2-4 CPUs, 16 GB RAM, one hour for the initial small pilot). The original
verified device was a V100; a different model is allowed but its times must be
labelled separately. The first full run will establish the required wall time.

In the new terminal:

```bash
module load gcc/13.3.0 cuda/12.6.3 conda
eval "$(conda shell.bash hook)"
conda activate ee451-pilot
cd ~/ee451-katherine-pilot
source scripts/carc_env.sh
nvidia-smi
python -u scripts/carc_pilot.py run --work-root "$EE451_PILOT_ROOT" --preset reproduce
```

This runs the original N=2,000, k=200 comparison for THREE epochs, with random
seed 0, 1,000 validation pairs, FP32, microbatch 8 pairs, and effective batch 32.
It reuses the original parsing, embedding/truncation, feature, normalization,
selection, reward scoring, tokenization, and evaluation functions. The training
loop preserves the gradient accumulation, optimizer, shuffle, and initialization
rules while measuring training and validation separately. The original head
initialization is checked to match across methods. Added reporting includes a
pair-weighted training loss alongside the original microbatch-mean display.

Model downloads are disabled for the run. Each stage synchronizes CUDA where
needed and logs its wall time and GPU allocator peaks. CPU memory is the process
lifetime RSS high-water mark, not a separate per-stage or whole-job measurement.
All timing is descriptive single-run feasibility data, not a speedup benchmark.

## 4. Follow-up experiments after inspecting the first run

Learning signal: all original 2,000 pilot training pairs, TWO epochs:

```bash
python -u scripts/carc_pilot.py run --work-root "$EE451_PILOT_ROOT" --preset signal
```

Proposed larger experiment: REAL N=20,000 pairs, k=2,000, TWO epochs:

```bash
python -u scripts/carc_pilot.py run --work-root "$EE451_PILOT_ROOT" --preset main
```

Optional full-pool baseline adds training on all 20,000 examples:

```bash
python -u scripts/carc_pilot.py run --work-root "$EE451_PILOT_ROOT" --preset main --include-full-pool
```

Do not launch all commands automatically. Inspect measured runtime/memory before
choosing the next allocation. `--validation-size 16490` uses the full validation
split; `--seed 1` changes training and random-selection seeds while keeping pool
and validation sampling seed 0 and k-center start 0. `--budget 0.2` selects 20%
from the same main pool; both selection and training work increase. Each run
currently regenerates embeddings, so repeated runs are not a cache-amortization
experiment. Feature matrices are saved for later CPU/CUDA work.

## Outputs and accounting

Large artifacts, selected IDs, per-epoch histories, stage CSV/JSONL, and the full
summary go in a unique `$EE451_PILOT_ROOT/runs/...` directory. A small summary
copy goes in the checkout's `results/carc/`. Partial completed-stage results are
saved if a Python exception occurs; a forced Slurm kill may leave only the
already-flushed stage log and last saved history/summary. There is no automatic
training resume after interruption; restart in a new run directory.

The actual measured run wall time covers all requested methods. Per-method
costs are explicitly accounting estimates obtained by summing relevant measured
stages. Random and full-pool baselines are not charged selection embeddings.
Downloads, integrity verification, and diagnostic feature writes are recorded
separately. A second cost total adds one complete preprocessing pass.

`best_validation_epoch` selects the lowest loss among TRAINED epochs. All epoch
metrics are retained, including epoch 0. Checkpoints are not saved by default;
use `--save-checkpoints` to retain the best model per method, with checkpoint I/O
timed separately. The test split is untouched for modeling/model selection.

## Review findings and limitations

- Original `train_sec` includes validation AFTER each epoch, but excludes initial
  validation, model loading, and tokenization. It is not training-only time.
- The `overfit_16` mode actually trains on `d[:8]` each step. Its saved result is
  an eight-pair overfit check despite its name. The new runner does not use it.
- The reward module docstring has a leading minus-sign typo; the executed
  softplus loss has the correct positive sign. No loss change was needed.
- No learning-size sweep script or 20k runner is included in the source ZIP;
  the added main preset supports the larger real-data run.
- Original k-center lacks validation for k=0/k>N; this runner validates budgets
  and the number/uniqueness of returned IDs before any reward training.
- Echo filtering and pooled test construction remain team decisions. This
  reproduction keeps them fixed so data changes do not confound hardware checks.
- The first pilot cannot demonstrate downstream quality superiority at k=200.
  Different GPU/PyTorch versions may change floating-point outputs and selection
  ties; exact agreement with laptop accuracies is not an acceptance condition.
- No custom CUDA selector or CPU-vs-GPU selection speedup is claimed here.
