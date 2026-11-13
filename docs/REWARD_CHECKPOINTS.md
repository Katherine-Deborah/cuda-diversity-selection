# Reward checkpoint reload verification

This three-file additive increment requires the installed external-ID reward
runner. It preserves the existing training code and checkpoint-saving behavior.
It does not retrain models, evaluate test, or overwrite a training summary.

## Current measured evidence

The external-ID integration run at
`runs/reward_ids_seed0_20261113T020808Z_a296ed95` passed on CARC P100 job
14068858, node e21-03, source commit `7e71ae983a47e0cc08ff69805225529458a615b1`.
It used N=2,000, k=200 per method, one epoch and 1,000 validation pairs.

| Method | Validation accuracy | Pairwise loss | Optimizer steps |
|---|---|---|---|
| Random | 50.7% | 0.6925278902 | 7 |
| NumPy k-center | 48.3% | 0.6946240664 | 7 |

Both complete epoch-one metric records exactly match the earlier three-epoch
pilot's epoch one. Pool/validation content and initial head hashes match too.
This supports correct integration of the new input path at the default settings.
The new run saved both checkpoint files with recorded SHA-256 hashes and model,
configuration, validation-ID and selection metadata. Reload verification is the
next check; successful serialization alone does not establish it.

Measured new-run wall time was 78.331 seconds and peak PyTorch allocation was
2.288 GiB. The earlier 138.339-second run included three training epochs and
embedding/selection stages, so these are different workloads. First-use setup
also affected method timing. These figures do not establish selector speedup or
model-quality superiority. Test evaluation remains unused.

## Tokenizer warning

The original `reward.encode` first tokenizes the entire context/response with
`add_special_tokens=False`, then manually retains recent context and the start
of the response. That first step can log a 552 > 512 warning. The assembled
sequence is subsequently capped at the configured 256 tokens, including special
tokens. This verifier checks actual encoded lengths for validation and both
training selections before evaluating a reloaded model. A length above the
configured cap causes a failure. No warning suppression or tokenization change
is needed to perform this check.

## Install and run

Upload `EE451_Checkpoint_Verification_v1.zip` to your CARC home directory and run
from the repository, preserving any files already present:

```bash
unzip -n ~/EE451_Checkpoint_Verification_v1.zip -d /home1/mgunasek/ee451-katherine-pilot
python -m pytest -q tests
```

The 16 new regression tests exercise metadata/hash mismatches, ordered selected
IDs, comparison tolerances and token-length limits. GPU reloading is not tested
in the preparation workspace; run the check below on CARC.

Reuse an active GPU allocation and the activated `ee451-pilot` environment.
If the allocation has ended, obtain a new valid GPU session before `verify`.

```bash
cd /home1/mgunasek/ee451-katherine-pilot
source scripts/carc_env.sh
hostname
printf 'SLURM_JOB_ID=%s\n' "${SLURM_JOB_ID:-none}"
nvidia-smi
ee451_reward_run="$EE451_PILOT_ROOT/runs/reward_ids_seed0_20261113T020808Z_a296ed95"
python -u scripts/verify_reward_checkpoints.py verify \
  --work-root "$EE451_PILOT_ROOT" --run-dir "$ee451_reward_run"
```

The command verifies the saved source/configuration, prepared data, ordered IDs,
checkpoint sidecars and weight-file hashes. It loads each state dictionary with
strict key matching, evaluates the same validation IDs and compares with the
saved best-epoch metrics (absolute accuracy tolerance 1e-7, loss tolerance 1e-6).
It writes a new `checkpoint_reload_TIMESTAMP.json` and stage JSONL beside the
original run, which is otherwise retained. It does not save or overwrite weights.

Success ends with `CHECKPOINT RELOAD VERIFICATION PASSED`. Share the printed
`RELOAD REPORT:` JSON. If metrics differ, retain the report and investigate the
data/software/hardware/configuration before changing tolerances or training again.

For CPU-only metadata inspection, use `inspect` in place of `verify`. That action
does not deserialize weights or import models/Torch, and it does not establish
that the saved model reproduces validation metrics.

Files added:

- `scripts/verify_reward_checkpoints.py`
- `tests/test_reward_checkpoints.py`
- `docs/REWARD_CHECKPOINTS.md`
