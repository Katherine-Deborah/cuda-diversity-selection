# EE 451 diversity selection — working notes (Phase 1, Section 2, steps 1–2)

Plan: `../EE451_Experimental_Design_Plan.md`. Everything here is reproducible from `src/`.
Status: steps 1–2 mostly done; **encoder NOT final**; nothing committed to git.

## 1. Environment
- Python 3.11.9, torch 2.5.1+cu121, GTX 1650 Ti Max-Q (4 GB, ~3.7 GB free).
- Already installed: datasets, numpy, pandas, scikit-learn, pytest. Added: transformers, sentence-transformers.
- **Gotcha:** latest `transformers` needs torch >= 2.6 and crashes on import with torch 2.5.1.
  Fix = pin `transformers==4.46.3`, `sentence-transformers==3.3.1` (did NOT upgrade torch: big CUDA wheel, risk).
- I accidentally downgraded numpy/scikit-learn during the fix, then restored them (numpy 2.2.6, sklearn 1.9.1).
  Leftover pip warnings (numpyro/jax, streamlit/packaging) are unrelated to this project.

## 2. Dataset: Anthropic HH-RLHF
- 160,800 train rows + 8,552 test rows. Each row = `chosen` and `rejected`, each a full transcript string:
  `\n\nHuman: ...\n\nAssistant: ...\n\nHuman: ...\n\nAssistant: <final reply>`.
- Both transcripts share the same history; they differ only in the final assistant reply (chosen = preferred).

## 3. Parsing (`src/parse.py`, tests in `tests/test_parse.py`)
- Split each transcript at the LAST `\n\nAssistant:` -> context (through last human turn) + final reply.
- Record fields: `id, group, split, source, context, chosen, rejected`.
- Dropped 2,701 pairs (~1.6%): echoes_user 1,392, identical_responses 775, context_mismatch 329, empty_response 194, extra_turn_in_response 11.
- **echoes_user (added after reviewing the 21 odd cases):** a reply that is just the user's last message repeated (or starts with "Human:"), e.g. chosen = "Human: Most people cannot be trusted". Degenerate, so dropped. The 21 'chosen duplicated at end of context' cases were all this.
- 18 train rows contain U+FFFD (replacement char) in the raw data; left as is. (The garbled characters seen in the terminal were mostly console encoding, not the data.)
- Output: `data/{train,val,test}.jsonl`, `data/parse_report.json`. Sample for manual review: `data/inspect_20.md` (**not yet read by a human**).
- Open check: 21 train examples have the chosen text also at the end of the context (likely short repeated phrases; unconfirmed).

## 4. Split logic (and why)
- Many rows come from the same conversation (same opening message, continued to different lengths). Splitting by row
  would put near-identical prompts in train and test = leakage. So split by **prompt group**.
- Group ID = sha1(first human turn). Split = hash(seed + group) -> [0,1): <0.8 train, <0.9 val, else test.
  Deterministic, independent of row order. Script asserts zero cross-split group overlap (it is 0).
- Counts train/val/test = 133,535 / 16,490 / 16,626 pairs (49,511 / 6,121 / 6,157 groups), after the echoes_user rule. (Earlier numbers before that rule: 134,655 / 16,637 / 16,751.)
- **Decisions I made without asking** (revisit with the team):
  1. Pooled official train+test and re-split 80/10/10, so our test set is NOT the official HH-RLHF test split.
  2. Grouping by first human turn can over-merge generic openers ("hi"); safe for leakage, but group sizes are uneven. Not measured.

## 5. Encoder study — 1,000 random train pairs, float32, batch 64 (`src/encoder_study.py`, `src/nn_check.py`)
| | MiniLM-L6 | BGE-small |
|---|---|---|
| max tokens | 256 | 512 |
| context truncated (default end-cut) | 16% (160/1000) | 1.7% (17/1000) |
| response truncated | 0.9% | 0.1% |
| speed | 398 texts/s | 186 texts/s |
| peak GPU mem | 400 MB | 815 MB |
| context->own response in top-1 of 2000 | 76.8% | 72.8% |
| ... in top-10 | 90.0% | 85.8% |
| avg sim chosen vs rejected (same prompt) | 0.48 | 0.72 |
| avg sim chosen vs random response | 0.03 | 0.47 |
| avg nearest-neighbour context sim | 0.53 | 0.76 |
| cos(left-trunc, end-trunc) on over-limit contexts | 0.85 (n=160) | 0.93 (n=17) |

How to read it:
- BGE-small wins on truncation. That is the only clear advantage so far.
- MiniLM is slightly better at matching a context to its own responses (a rough quality proxy), and is 2x faster.
- **BGE similarities are compressed into a high band** (random responses already sit at 0.47). Distances between
  features will be less spread out, which matters for k-center and for the normalization choice in step 4.
- Left vs end truncation changes MiniLM's long-context embeddings a lot (cos 0.85), so the truncation rule matters
  most if MiniLM is chosen. Plan wants recent turns kept -> use `tokenizer.truncation_side="left"` for contexts.
- Nearest-neighbour printouts: `data/nn_check.md`. Caveat: it shows only the LAST human message, but the embedding
  is of the whole context, so some neighbours look unrelated when they are not. Tiny qualitative sample (6).

## 5b. Left truncation implemented and re-measured (`src/embed.py`, `src/remeasure.py`, `data/remeasure.json`)
- `src/embed.py`: `encode_context` left-truncates (keeps recent turns), `encode_response` keeps the start. HF truncates
  BEFORE adding [CLS]/[SEP], so special tokens survive; `python src/embed.py` self-checks this (256 tokens, ends with the real last words).
- Truncation RATES do not change (left vs right cut the same number of tokens): MiniLM 16%, BGE 1.7% of contexts. Only WHAT is kept changes.
- Retrieval (context -> own response, top-1 / top-10 of 2000), end-cut -> left-cut:
  - MiniLM all: 76.8/90.0 -> **78.0/90.1**; on the 160 over-limit contexts top-1: 68.1 -> **75.6** (top-10 85.6 -> 86.3).
  - BGE-small: unchanged (72.8/85.8 all) — only 17 contexts exceed 512 tokens, ranks did not move.
- Takeaway: left truncation helps MiniLM (recent turns hold the relevant request), so the earlier MiniLM numbers were slightly
  understated. MiniLM now beats BGE on retrieval by ~5 points top-1 while being 2x faster; BGE still truncates far less.
  Small samples (n=160 / n=17): treat the long-context numbers as indicative, not significant.

## 6. Where the decision stands
**PROVISIONAL CHOICE (2026-11-07): MiniLM-L6 (`sentence-transformers/all-MiniLM-L6-v2`, 384-d, 256 tokens, left-truncated contexts).**
Chosen to move on; can swap to BGE-small later (same 384-d, so feature shape and CUDA kernel are unaffected). Record the model revision hash when features are frozen.

**Not final.** Earlier I leaned BGE-small on truncation alone; the retrieval check does not support a clear win, so
it is now a genuine toss-up: BGE (less truncation) vs MiniLM (faster, tighter retrieval, wider similarity spread).
Still needed: optionally a subset-quality comparison on validation
data, then team review (Surpreet, Mohan) and an entry in the plan's decision record.

## 7. Next
- (done) left truncation fixed and re-measured, see 5b.
- Plan step 3: symmetric features (prompt, mean, |a-b|) + swap/finite checks.

## 8. Formatting / truncation spec (step 2 deliverable)
- Context = full transcript up to and including the last human turn, turns separated by `

` with `Human:` / `Assistant:` prefixes kept as in the raw data (stripped of outer whitespace).
- Responses = text after the last `

Assistant:`, stripped; embedded WITHOUT any speaker prefix.
- Encoder MiniLM-L6, max 256 tokens INCLUDING [CLS]/[SEP]. Context: left-truncate (keep recent turns). Response: right-truncate (keep the start). Embeddings L2-normalized, float32.
- Dropped: no assistant turn, context mismatch, empty reply, identical replies, echoes_user, extra turn in a reply.
- Reviewed 20 random parsed examples (`data/inspect_20.md`): context/reply boundaries correct; a few chosen/rejected look nearly alike but differ. Only issue found was the echo case above.
- Truncation rate (1000-pair sample): contexts 16%, responses 0.9%.

## 9. Step 3: symmetric features (`src/features.py`, `tests/test_features.py`)
- x = concat(p, (a+b)/2, |a-b|), p/a/b = unit-normalized MiniLM embeddings; dim 3*384 = 1152; float32; NOT yet normalized (step 4).
- Pilot pool: 2,000 random train pairs (seed 0; IDs in `data/pilot_ids.json`, 1,965 distinct prompt groups). Saved: `data/pilot_emb.npz` (P,A,B), `data/pilot_features.npy` (2000x1152, 8.8 MB), `data/features_report.json`.
- Checks passed: shape (2000,1152), all finite, swapping chosen/rejected gives bit-identical x.
- Edge case found: 1 of 2000 pairs has |a-b| = 0 exactly. Cause: both replies are long and identical in their first 256 tokens (they differ only after the cut), so truncation makes the embeddings equal. Not a bug, but it is a zero-difference component; step 4 must define epsilon/zero behavior. Mean |a-b| norm is ~1.0 (min non-zero 0.166).
- Pilot pool is a random sample of the train split, not yet de-duplicated by prompt group (35 groups appear twice).

## 10. Step 4: normalization and distance (`src/normalize.py`, `src/kcenter.py`, `tests/test_normalize.py`, `data/normalize_report.json`)
Options compared on the 2,000-pair pilot embeddings (MiniLM):
- **F (final-vector):** x=[p,m,q] then unit-normalize.  **C (component+final):** unit-normalize p, m, q separately, concat, unit-normalize.
- Rule for both: `x / max(||x||, 1e-6)`, so a zero vector stays zero (no NaN). Distance = squared Euclidean, float32 (== 2-2cos for unit vectors, max error 3e-7 checked).
- Component norms before normalizing: p = 1.00 exactly; m mean 0.85 (0.67-1.0); q mean 1.00, 5-95% range 0.56-1.39, min 0 (the one truncation-identical pair). So q is the component whose magnitude varies most (it measures how different the two replies are).
- Share of mean pairwise squared distance by component (p/m/q): F = 45/36/19%, C = 41/43/16%. Component normalization equalizes NORMS, not contributions to distance; the preference-contrast part q ends up the smallest contributor either way.
- Effect on selection: kNN(10) overlap F vs C = 75%. k-center (k=200) overlap F vs C = 27.5%. For scale: same features but a different start row = 56.5%; chance = 10%. So the choice DOES change which points are picked, more than changing the start row does.
- Amplification risk of C: only 1 of 2000 rows has q < 0.1 and it is exactly 0 (stays 0). Next smallest is 0.166 (x6 boost under C). Small on this sample; could matter at 100x scale.
- **Provisional choice: F (final-vector normalization only).** Reasons: keeps the magnitude information in q (how different the replies are), no blow-up of tiny differences, simpler, and C does not deliver the equal-contribution geometry it is meant to. Revisit if the team wants equal weighting on purpose (then weight the components explicitly instead of normalizing them).
- Open: whether F or C selects better data is untested; that needs the reward-model pilot (compare subsets on validation).

## 11. k-center timing (`src/time_kcenter.py`, `data/kcenter_timing.json`)
- NumPy reference, single machine CPU, d=1152, float32: N=2000,k=200 = 1.4 s; N=4000,k=400 = 5.3 s; N=8000,k=800 = 19.8 s.
- Cost is ~3 ns per (N*k*d), matching the O(N*k*d) estimate (doubling N at fixed 10% quadruples time). N>2000 runs used tiled+noised pilot rows (timing only).
- Projection for NumPy CPU selection only (k = 10% N): N=20k -> ~2.3 min; N=50k -> ~14 min; N=133,535 (all train) -> ~1.7 h. This is the CPU baseline the CUDA version has to beat. Does not include embedding or training.

## 12. Reward-model pilot (`src/reward.py`, `data/reward_pilot_*.json`)
- Model: `distilbert/distilbert-base-uncased` (rev 12040acc...), 66.4M params, full fine-tune, [CLS] -> Linear(768,1). Loss = softplus(-(r+ - r-)). AdamW lr 2e-5, effective batch 32 pairs (micro-batch 8 pairs = 16 sequences x 4 accumulation), fp32.
- Input: [CLS] context [SEP] response [SEP], 256 tokens total; response capped at 128 tokens (start kept), context gets the remainder (recent turns kept). Token split is a pilot choice, not final.
- Tiny-batch check: 16 pairs, loss 0.693 -> 0.044 in 30 steps. The loop works.
- Memory: peak 2.3 GB of 4 GB (fits; room to try 512 tokens? untested). Same seed => identical head init across methods (both start at val acc 0.485).
- Pilot comparison (200 selected pairs from the 2,000 pool, 3 epochs, eval on 1,000 val pairs):
  - random200: val acc 0.499 / 0.495 / 0.497 (after epochs 1/2/3), val loss ~0.69; train loss 0.69 -> 0.64
  - kcenter200 (final-vector features): val acc 0.491 / 0.498 / 0.508, val loss ~0.69-0.70; train loss 0.69 -> 0.65
  - **Both are at chance (50%).** Train loss falls but val does not: 200 pairs is too little to learn a generalizing signal in 3 epochs. The difference between methods (~1 point) is far inside noise (1,000 val pairs => about +-1.6 points at 1 sd). NO conclusion about k-center vs random can be drawn from this.
- Time: ~200 s per method including 4 validation passes over 1,000 pairs; one 200-pair epoch is roughly 20 s of training, so 2,000 pairs is ~3 min/epoch.

- **Learning-signal run (all 2,000 pilot pairs, 2 epochs, eval on the same 1,000 val pairs):** val acc 0.485 (untrained) -> 0.548 (epoch 1) -> **0.558** (epoch 2); val loss 0.6956 -> 0.6869 -> 0.6884; train loss 0.682 -> 0.631. So the setup DOES learn a real, if small, signal once there is enough data (+7 points over untrained; 1,000 val pairs => ~1.6 pt sd). Val loss barely moves and starts rising at epoch 2 while train loss keeps falling: overfitting begins quickly. Peak GPU 2.34 GB.
- **Runtime caveat:** this run took ~60 min (about 24 s/step) because the laptop GPU was power-throttled (300 MHz of 2100 at one point). The earlier 200-pair runs were >2x faster per step. Plug in / set high-performance power mode before timing anything for the cost projection; the 60 min is NOT a representative training cost.
- An earlier attempt at this run was killed by the memory-pressure reaper (low system RAM); closing other apps fixed it.
- Implication: with k = 200 (10% of the 2k pool) both methods are at chance, so the pilot pool is too small to compare selection methods. The main experiment needs a much larger N (and k) than 2k/200 for any difference to be measurable.

- **Re-timed with the GPU unthrottled (12 steps, `python src/reward.py retime`):** median 3.06 s per step of 32 pairs at about 1,750-1,785 MHz (vs 24 s/step when throttled, ~8x faster). 2,000 pairs = ~3.2 min/epoch of training. Peak GPU 2.33 GB. Estimates: k=2,000 x 3 epochs ~10 min/run; 6 runs (2 methods x 3 seeds) ~1 h; k=5,000 ~24 min/run, ~2.4 h for 6. Excludes validation passes and pool embedding.

## 13. Next
- NumPy reference k-center exists (`src/kcenter.py`); time it at N=2000, k=200 and extrapolate (step 5).
- Then reward-model pilot (DistilBERT) on 4 GB.
