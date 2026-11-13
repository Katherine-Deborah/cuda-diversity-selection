# Reward training from externally selected pair IDs — first increment

This additive patch targets the reviewed `a9c1e5e` source plus the existing CARC
additions, committed on CARC as `c1ac5b7`. It adds five files. It changes no
existing source, configuration, dataset or runner. CUDA branch files are not
required to develop or validate the reward input path.

## What this increment adds

- An explicit reward-training configuration with epochs, training seed, subset
  size, learning rate, total/reply token limits, microbatch and effective batch.
- External ordered pool, validation and per-method selected pair-ID files.
- Checks for duplicate/missing IDs, exact common budget, pool membership,
  current split metadata, pool/validation group overlap and prepared-file hashes.
- A standard-library-only `validate` action and CPU regression tests.
- A `train` action reusing `carc_pilot.train_one`, `Profiler`, the original
  pairwise loss/evaluation, initialization and optional best-checkpoint saving.
- Resolved configurations, ID/content/source hashes, copied input IDs and
  metadata beside existing best-model state dictionaries.

The adapter applies supported configuration to the legacy reward module before
tokenization and training. The pinned model/revision, FP32, AdamW defaults and
evaluation batch of 16 pairs remain part of the fixed reviewed protocol. Unknown
configuration options are rejected. This preserves the existing source-hash
checks and avoids re-preparing data just to change training hyperparameters.

Checkpoint reload verification, interrupted-training resume, a consolidated
metrics CSV, plots and final test evaluation are later increments. Saving weights
already exists; this patch reuses it and adds metadata. GPU training for this
patch has not been executed in the preparation workspace.

## Install and verify source changes

From the existing repository on `feature/reward`, extract the ZIP at its root:

```bash
unzip -n ~/EE451_Reward_Inputs_v1.zip -d /home1/mgunasek/ee451-katherine-pilot
git status --short
git diff --check
```

The ZIP contains only the five new files listed at the end of this document.
`unzip -n` preserves an existing file if a name is already present. Review that
case rather than assuming a partially extracted patch is installed.

Run tests in a CPU compute allocation. If an allocation is already active, reuse
it; the allocation command is only for a login shell without a current job:

```bash
srun --account=prasanna_933 --partition=main --nodes=1 --ntasks=1 \
  --cpus-per-task=2 --mem=16G --time=00:30:00 --pty bash -l
```

Inside the allocation, reuse the existing environment:

```bash
module load gcc/13.3.0 cuda/12.6.3 conda
eval "$(conda shell.bash hook)"
conda activate ee451-pilot
cd /home1/mgunasek/ee451-katherine-pilot
source scripts/carc_env.sh
python -m unittest discover -s tests -p 'test_reward_inputs.py' -v
python scripts/reward_from_ids.py validate \
  --work-root "$EE451_PILOT_ROOT" --config configs/reward_external_pilot.json
```

The example configuration references the last evidenced run's saved pool,
validation and selected-ID files. If any is missing, preserve the error and
locate the actual run files. Do not regenerate selections or replace IDs with
another pool just to make validation pass. The full prepared dataset and these
run ID files were not included in the handoff ZIP; they are expected on CARC.

## Input protocol

All ID files must be nonempty JSON arrays of unique strings. Integer selector
row indices must first be mapped through the exact ordered pool file by the
integration owner. Relative input paths resolve under `--work-root`; absolute
paths are accepted. The config path resolves from the shell's current directory.

Selected-ID order is retained. A different order can change minibatch composition
after seeded shuffling, so agree on an ordering policy for the final comparison.
Every method must supply exactly `budget` IDs from the same training pool.
Names such as `test-123` describe source rows in the pooled pilot: current split
metadata, not the prefix, determines eligibility.

The prepared manifest must match the reviewed dataset/encoder/source hashes and
reward model/revision. Training settings may differ from those recorded when
models/data were cached. Only train and validation files are read; there is no
held-out-test modeling action. The shared final split/ID/feature protocol is
still a team decision. This patch does not settle it.

No embedding or selection stage runs here. In particular, the unnormalized
`pilot_features.npy` in `feature/cuda` is not silently substituted for the measured
CARC pilot's features. Record the provenance of any new external selections.

## Later GPU smoke check

Only after CPU validation passes, request/reuse an allocated GPU compute session
and activate the existing environment as above. Check the allocation/device:

```bash
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
nvidia-smi
```

Then the optional integration smoke command is:

```bash
python -u scripts/reward_from_ids.py train \
  --work-root "$EE451_PILOT_ROOT" --config configs/reward_external_pilot.json
```

The example uses one epoch, both existing 200-pair selections and checkpoint
saving to exercise the new input/configuration path. It is a pipeline/checkpoint
smoke check, not a replacement for the completed three-epoch feasibility result
or a frozen final quality comparison. Do not launch it as part of CPU validation.

Each training run writes a new `runs/reward_ids_seed.../` directory with stage
JSONL, a summary, resolved config, ordered input IDs, histories, optional best
weights and checkpoint metadata. Training and validation times are separate;
external embedding/selection costs are absent. Models are loaded offline from
the existing pinned cache. No setup/download/prepare command is needed.

## Files in this patch

- `scripts/reward_inputs.py`
- `scripts/reward_from_ids.py`
- `configs/reward_external_pilot.json`
- `tests/test_reward_inputs.py`
- `docs/REWARD_FROM_IDS.md`
