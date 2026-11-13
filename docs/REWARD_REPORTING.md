# Reward-model handoff and reporting

## Current completion status

The reward-model engineering increment is ready for review on `feature/reward`:

- The existing profiled training loop accepts validated external pair IDs and
  explicit learning rate, token limits, microbatch/effective batch, epochs and seed.
- Saved checkpoint metadata binds weights to the configured model, selected IDs,
  pool, validation set and source/configuration hashes.
- CARC tests: 49 passed after adding the checkpoint verifier.
- A one-epoch P100 integration run saved both reward models; reload reproduced
  both validation accuracy and pairwise loss exactly (zero absolute differences).
- Actual validation and training input lengths were checked at <=256 tokens.
- Metrics CSVs, validation curves and measured stage-time plots are included here.

The plotting/reporting addition was exercised locally against the supplied CARC
JSONs and the PNGs were visually checked. It adds no model training dependency.
The engineering milestone supports the revised project scope ending at CUDA.
Larger reward experiments remain optional and depend on the team-approved split,
ordered pool, normalization/feature artifact and selected IDs.

## Measured integration pilot

- Run: `reward_ids_seed0_20261113T020808Z_a296ed95`
- Training commit: `7e71ae983a47e0cc08ff69805225529458a615b1`
- Hardware: Tesla P100-PCIE-16GB, two allocated CPU cores, CARC job 14068858.

| Method | Selected pairs | Epochs | Validation pairs | Validation accuracy | Pairwise loss | Reload |
|---|---:|---:|---:|---:|---:|---|
| Random | 200 | 1 | 1,000 | 50.7% | 0.6925278902 | Exact |
| NumPy k-center | 200 | 1 | 1,000 | 48.3% | 0.6946240664 | Exact |

Pool size was 2,000 and training seed was 0. Both methods used the same initial
head hash, architecture, training schedule and validation IDs. The complete first
epoch metric records equal the earlier three-epoch pilot's first epoch, so this
is an integration check using the same examples rather than an independent repeat.

New-run measured wall time was 78.331 seconds and peak PyTorch allocation was
2.288 GiB. Training stages were 7.489 s / 4.710 s; initial plus post-epoch
validation stages were 17.502 s / 14.224 s for random / k-center respectively.
Order and first-use overhead differed. These descriptive costs are not a fair
standalone-method speed comparison. The earlier 138.339-second run included
three epochs plus embeddings/selection, so its workload differs.

Accuracy remains near chance. One seed at k=200 does not establish improved
quality or equivalence. Test was unused, and quality preservation against a
full-pool model remains unmeasured. The saved selected examples came from the
existing NumPy pilot, not newly integrated CUDA outputs.

## Files visible in the project explorer

Pre-generated reports are under:

`results/downloads/reward_ids_seed0_20261113T020808Z_a296ed95/`

- `reward_metrics.csv`: one row per method, best and last validation metrics,
  schedule, training/validation times and checkpoint verification status.
- `learning_curves.csv`: epoch-zero and trained-epoch observations.
- `stages.csv`: measured stage timings and recorded memory peaks.
- `validation_curves.png` / `.svg`: validation accuracy and pairwise loss.
- `training_validation_time.png` / `.svg`: descriptive stage times.
- `report_provenance.json`: report/source hashes and claim limits.

Accuracy columns in the CSV use fractions from 0 to 1; plots display percentages.

The adjacent `..._reports.zip` contains generated reports for direct download.
Refresh the project explorer and right-click the desired file to download it.

Copy the original small run JSONs and prepared manifest into that same visible
folder before committing the final handoff. This also shares the exact ordered
pool, validation and selected-ID arrays without committing model weights/data:

```bash
cd /home1/mgunasek/ee451-katherine-pilot
source scripts/carc_env.sh
ee451_run="$EE451_PILOT_ROOT/runs/reward_ids_seed0_20261113T020808Z_a296ed95"
ee451_downloads="$PWD/results/downloads/$(basename "$ee451_run")"
mkdir -p "$ee451_downloads"
cp -n -- "$ee451_run"/*.json "$ee451_downloads/"
cp -n -- "$EE451_PILOT_ROOT/prepared/manifest.json" "$ee451_downloads/prepared_manifest.json"
```

## Regenerate reports for a future run

CSV export uses only the Python standard library. Plot generation additionally
requires Matplotlib in the environment; the current pilot plots are already
included, so no CARC plotting installation or retraining is needed now.

```bash
python scripts/report_reward_run.py \
  --summary "$ee451_run/summary.json" \
  --reload-report "$ee451_run/checkpoint_reload_20261113T022026Z_e0bd0798.json"
```

The default output is `results/downloads/<run-name>/` inside the checkout, with
an adjacent downloadable report ZIP. Choose the actual reload report for any
new run. Use `--no-plots` for CSV-only export or `--output-dir` for another folder.
Existing generated reports require an explicit `--overwrite` to regenerate.

The reporter checks matched sizes/schedule, the saved best-epoch rule, common
head initialization and reload-report identities before exporting. CRLF editor
downloads are accepted when normalizing them reproduces the original LF hash.
Input JSONs, checkpoints and source datasets are not modified.

## Push and review

Commit the reporter, this documentation and the small report folder on
`feature/reward`, then push that branch to Katherine's repository. Keep the
prepared dataset, embeddings, caches and large checkpoints at their artifact
locations. The result JSONs retain the original training commit and source
hashes even though reporting and verification have later commits.

Suggested PR title: **Add reproducible reward training from selected IDs**

Suggested PR description:

> Adds a training-only runner that validates ordered pool/validation/selected
> pair IDs and applies explicit reward hyperparameters using the existing CARC
> training loop, profiler and checkpoint saver. Adds checkpoint identity/reload
> verification and exports measured pilot metrics, curves and stage timings.
>
> Validation: 49 CARC tests passed; a one-epoch P100 pilot with 200 selected pairs
> per method and 1,000 validation pairs saved both checkpoints. Reload exactly
> reproduced recorded accuracy/loss, and encoded lengths stayed within 256.
> Reporting CSVs and plots were generated and checked from those saved JSONs.
>
> Results are integration-pilot evidence. Final larger reward experiments require
> the shared split, frozen pool/feature artifact and matched selected IDs. Test
> evaluation remains reserved for a frozen final comparison.

## Dependencies to raise with the team

Katherine: confirm one shared split, publish the frozen ordered pool and feature
artifact/normalization with checksums, validation IDs and random selection IDs.
Surpreet: deliver validated k-center indices for that exact ordering so they can
be mapped to pair IDs. The current CUDA branch's on-disk pilot features require
the agreed normalization at use. Matching existing pilot IDs/examples permits
reusing relevant model work; a different pool/selection needs a separately
labelled run. No expanded reward experiment is required by the reduced scope.
