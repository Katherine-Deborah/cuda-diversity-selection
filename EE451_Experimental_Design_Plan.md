# EE 451 — Experimental Design Research and Decision Plan

**Project:** CUDA-Accelerated Diversity Selection for Preference Data  
**Team:** Katherine, Surpreet, and Mohan  
**Scope:** Phase 1, Section 2 — Finalize Experimental Design  
**Status:** Proposed recommendations; final choices remain open pending research and pilot measurements.

## Goal

Choose and document a defensible experimental configuration before the main CPU/CUDA implementation and downstream experiments. The configuration must support a fair comparison of selection performance and reward-model quality.

This plan covers the seven experimental-design decisions. Environment setup and dataset preparation are dependencies; the feasibility pilot is used to validate resource-dependent choices.

## Recommended direction at a glance

| Decision | Starting recommendation | Evidence needed before confirming |
| --- | --- | --- |
| Selection encoder | Compare BGE-small-en-v1.5 and all-MiniLM-L6-v2 | Embedding time, memory, truncation, qualitative checks |
| Text representation | Embed shared context and final responses separately | Correct parsing; useful similarity relationships |
| Symmetric features | Prompt + response mean + absolute response difference | Swap invariance; valid finite vectors |
| Normalization and distance | Compare component normalization with final-vector-only normalization; use squared Euclidean and float32 | Component-norm distributions and small validation checks |
| Subset budget | 10% of a fixed training pool | Feasible selection and training costs |
| Reproducibility | Fixed prompt-group split; one debugging seed, target three matched repeats | Saved IDs, configurations, revisions, and environment |
| Reward model | Fully fine-tuned DistilBERT plus scalar head; frozen joint encoder plus head as fallback | Working learning signal, runtime, and memory |

## 1. Research and choose the selection encoder

### Requirements

- Produce fixed-dimensional sentence or passage embeddings.
- Preserve enough prompt and response content.
- Fit the available GPU and preprocessing budget.
- Use a documented tokenizer, pooling method, and model revision.
- Remain frozen during selection.
- Supply identical saved features to all distance-based selection methods.

The selection encoder creates vectors for choosing data. The reward model learns preference scores. They can be different models.

### Candidates

| Encoder | Base dimension | Documented input length | Reason to investigate |
| --- | ---: | --- | --- |
| sentence-transformers/all-MiniLM-L6-v2 | 384 | 256 word pieces by default | Compact, straightforward baseline |
| BAAI/bge-small-en-v1.5 | 384 | 512 tokens | Longer inputs without increasing vector dimension |
| sentence-transformers/all-mpnet-base-v2 | 768 | 384 tokens in its Sentence Transformers configuration | Optional larger representation; doubles final feature dimension |

Input lengths use each model's tokenizer and include the relevant tokenization constraints. Larger input capacity does not establish better preference-subset selection.

### Tasks

- [ ] Read MiniLM and BGE-small model cards, including pooling and input-format instructions.
- [ ] Sample approximately 1,000 training pairs reproducibly.
- [ ] Measure token-length distributions and truncation rates for each tokenizer.
- [ ] Measure embedding wall time, throughput, batch size, and peak GPU memory.
- [ ] Inspect nearest neighbors for several prompts and responses.
- [ ] If affordable, compare subset outcomes using validation data.
- [ ] Record the chosen encoder and rationale.

**Recommendation:** Prefer BGE-small if its longer window retains useful content at acceptable cost; choose MiniLM if its measured efficiency and retained content better fit the project. Benchmark actual throughput rather than assuming a speed ranking.

**Deliverable:** Encoder comparison table and chosen checkpoint revision.

## 2. Define prompt and response representation

### Requirements

Recover a shared conversation context and two alternative final responses from each pair. HH-RLHF chosen/rejected texts may include conversation history.

Each processed record should contain:

- Stable example ID and prompt-group ID.
- Shared context through the final user request.
- Preferred final assistant response.
- Rejected final assistant response.
- Split assignment and source metadata.

### Options

| Approach | Definition | Advantage | Limitation |
| --- | --- | --- | --- |
| Separate embeddings | E(context), E(chosen), E(rejected) | Explicit prompt and response components; reusable prompt embeddings | Response embeddings lack direct context |
| Contextual response embeddings | E(context + chosen), E(context + rejected), optionally E(context) | Captures responses in context | Repeated context and greater truncation pressure |

### Tasks

- [ ] Inspect at least 20 parsed examples.
- [ ] Confirm shared context matches between the two transcripts.
- [ ] Ensure the extracted prompt excludes both final alternative responses.
- [ ] Define handling for malformed pairs and identical responses.
- [ ] Document separators and text formatting.
- [ ] Inspect long examples before choosing truncation.
- [ ] Record the percentage of truncated inputs.

**Recommendation:** Start with separate embeddings. For prompts, preserve recent conversation turns. For responses, start with the beginning; investigate beginning-and-end truncation if important content is frequently lost. Count special tokens within input limits.

**Deliverable:** Exact parsing, formatting, and truncation specification.

## 3. Define symmetric features

Let p, a, and b be the prompt, preferred-response, and rejected-response embeddings.

Construct:

- Response mean: m = (a + b) / 2.
- Absolute response difference: q = abs(a - b).
- Combined feature: x = concatenate(p, m, q).

### Requirements and checks

- [ ] Swapping responses leaves the selection feature unchanged.
- [ ] Every example has the same feature dimension.
- [ ] No NaN or infinite values occur.
- [ ] Identical and nearly identical embeddings are handled explicitly.
- [ ] Response preference labels remain available for reward training.

For base dimension h, the feature dimension is 3h: 1,152 for MiniLM/BGE-small or 2,304 for MPNet.

**Recommendation:** Keep all three components with equal initial weights. Treat their usefulness for reward-data selection as a hypothesis to evaluate, rather than an established guarantee.

**Deliverable:** Feature-construction formula and implementation with small correctness checks.

## 4. Choose normalization and distance

### Compare these normalization choices

Start with unit-normalized encoder outputs, then compare:

| Choice | Procedure | Consequence |
| --- | --- | --- |
| Final-vector normalization | Form mean and difference, concatenate, then unit-normalize | Preserves relative component magnitudes |
| Component and final normalization | Normalize prompt, mean, and difference separately, concatenate, then unit-normalize | Equalizes nonzero component norms |

Component normalization removes the magnitude of response differences and can magnify small differences. Inspect this trade-off rather than selecting it automatically.

### Tasks

- [ ] Plot or summarize prompt, mean, and difference norms.
- [ ] Count zero and near-zero difference components.
- [ ] Choose and document epsilon and zero-vector behavior.
- [ ] Confirm final features are finite and normalized as specified.
- [ ] Fix one normalization procedure before the main experiments.

**Recommendation:** Use component normalization if equal component contribution is the intended geometry; otherwise use final-vector normalization to retain magnitude information. Use squared Euclidean distance and float32 for both CPU and CUDA.

Squared distance is sum_j (x_j - y_j)^2. For unit vectors, it equals 2 - 2 cosine_similarity(x, y). Ordinary Euclidean and squared Euclidean distances produce the same farthest-first choices in exact arithmetic. Squared Euclidean itself does not satisfy the metric triangle inequality; do not transfer metric approximation guarantees to squared coverage values without qualification.

**Deliverable:** Normalization formula, distance definition, precision, and numerical handling rules.

## 5. Set candidate-pool size and subset budget

### Requirements

- Define candidate-pool size N and subset size k explicitly.
- Select from training data only.
- Keep prompt groups separate across training, validation, and test.
- Use identical pools and budgets across selection methods.
- Define whether “full data” means the full experimental pool or all eligible HH-RLHF training data.

### Recommended stages

| Stage | Suggested scale | Purpose |
| --- | --- | --- |
| Encoder study | Approximately 1,000 training pairs | Representation and preprocessing comparison |
| Integration pilot | Approximately 2,000 training pairs; k = 200 | Feasibility and pipeline debugging |
| Main experiment | Fixed pool sized using pilot evidence; k = floor(0.10N) | Controlled method comparison |

Exact incremental k-center requires O(Nkd) work for final feature dimension d. At a fixed subset fraction, doubling N approximately quadruples selection work.

### Tasks

- [ ] Estimate main selection time using both N and k.
- [ ] Estimate embedding, training, and evaluation cost.
- [ ] Choose final N based on resource availability.
- [ ] Save pool IDs and the exact integer budget.
- [ ] Document any pool reduction and how it was sampled.

**Recommendation:** Keep 10% as the primary budget. Do not assume the earlier 20,000-pair suggestion is mandatory. Keep k-means and additional budgets optional. If adding a second budget, use 5% for stronger compression or 20% to investigate an accuracy gap, with decisions based on validation rather than test results.

**Deliverable:** Pool definition, IDs, N, k, and projected cost.

## 6. Establish reproducibility and comparison rules

### Tasks

- [ ] Fix one prompt-group split and save all split IDs.
- [ ] Save candidate-pool and selected-subset IDs.
- [ ] Separate and record split, selection, and training seeds.
- [ ] Debug with one seed; target three matched experimental repeats.
- [ ] Use matched initial reward-model weights across methods within each repeat.
- [ ] Pin dataset/model revisions and record software versions.
- [ ] Record CPU/GPU models, compiler flags, and command lines.
- [ ] Document deterministic settings and any performance effects.

### CPU/CUDA agreement

Use the same feature matrix, row order, initial center, float32 precision, and selected-point masking. For exact distance ties, select the smallest unselected row index. Document distance tolerances and investigate near-tie differences rather than requiring identical real-data indices in every case.

Use numerical comparisons that account for floating-point accumulation differences. Seeds alone do not guarantee cross-platform bitwise reproducibility.

**Recommendation:** Keep the data split fixed while repeating selection and training. Save actual artifacts as well as seeds. Three repeats provide a practical variability estimate, not definitive statistical evidence.

**Deliverable:** Reproducibility checklist and version-controlled configuration.

## 7. Choose reward-model architecture and training settings

### Requirements

Score each prompt-response independently using the same scoring function:

- Preferred score: r_plus = r(context, chosen).
- Rejected score: r_minus = r(context, rejected).
- Loss: -log(sigmoid(r_plus - r_minus)), implemented stably with softplus.

The model must retain prompt-response interactions. A purely linear head on separately concatenated prompt and response embeddings causes the prompt term to cancel in the score difference.

### Architecture options

| Architecture | Advantage | Limitation |
| --- | --- | --- |
| Frozen joint prompt-response encoder + head | Cheap repeated training; cacheable embeddings | Restricted downstream learner |
| Partially fine-tuned encoder + head | Intermediate adaptation and cost | Additional freezing-policy choice |
| Fully fine-tuned DistilBERT + scalar head | Flexible small-model experiment | More training time and memory |

### Recommended starting settings

| Setting | Starting point | Confirmation method |
| --- | --- | --- |
| Backbone | distilbert/distilbert-base-uncased | Training feasibility pilot |
| Trainable parameters | Backbone and scalar head | Memory and training time |
| Learning rate | 2e-5; investigate 1e-5 or 5e-5 if needed | Small validation search |
| Effective batch size | 32 preference pairs | Microbatch and accumulation based on memory |
| Input length | Investigate 256 and 512 tokens | Truncation versus runtime |
| Duration | One pilot epoch; up to three main epochs | Validation learning curves |
| Optimizer | AdamW | Fix configuration across methods |
| Checkpoint | Lowest validation pairwise loss | Same selection rule for every method |
| Training budget | Equal epochs for primary experiment | Report optimizer steps and training time |

These are candidate settings, not known optimal values. A frozen head requires its own learning-rate choice.

### Tasks

- [ ] Verify loss decreases on a tiny training batch.
- [ ] Measure a small random-subset and reference k-center training run.
- [ ] Record validation accuracy, loss, GPU memory, and runtime.
- [ ] Fix token allocation between context and response.
- [ ] Confirm architecture and training configuration.
- [ ] Reserve the test set for final evaluation.

**Recommendation:** Prefer full DistilBERT fine-tuning if repeated runs fit the allocation. Use the frozen joint encoder alternative if necessary, and describe that limitation. Equal epochs compare dataset use under equal passes; equal optimizer steps answer a different question and can be an optional follow-up.

**Deliverable:** Working reward-training pilot and fixed training configuration.

## Execution order and suggested ownership

Assignments below are proposed roles, not confirmed individual commitments.

| Order | Activity | Suggested owner | Dependency |
| --- | --- | --- | --- |
| 1 | Read sources and inspect dataset structure | Workstream B, with team review | Dataset access |
| 2 | Compare encoders and define text representation | Workstream B | Parsed sample |
| 3 | Validate feature construction and distance contract | Workstreams A and B | Representation |
| 4 | Pilot reward-model training | Workstream C | Clean splits; can proceed alongside encoder comparison |
| 5 | Estimate selection and total experiment cost | All workstreams | Pilot measurements |
| 6 | Review choices and freeze configuration | Entire team | Evidence from prior steps |

No calendar deadline is assumed. Complete these gates before launching the main experiments.

## Pilot measurement sheet

| Measurement | Result | Notes |
| --- | --- | --- |
| GPU model and usable memory | GTX 1650 Ti Max-Q, 4 GB (about 3.7 GB free); driver 592.82; torch 2.5.1+cu121 | Laptop GPU; was power-throttled (300 MHz of 2100) during the reward run |
| Parsed training/validation/test counts | 133,535 / 16,490 / 16,626 pairs (49,511 / 6,121 / 6,157 prompt groups) | Official HF train+test pooled and re-split 80/10/10 by prompt group. 2,701 of 169,352 raw pairs dropped (echoes_user 1,392; identical 775; context mismatch 329; empty 194; extra turn 11) |
| Cross-split prompt overlap | 0 in all three split pairs | Checked by assertion in `src/parse.py`. Group = first human turn |
| Encoder, revision, and batch size | all-MiniLM-L6-v2, rev 1110a243fdf4706b3f48f1d95db1a4f5529b4d41, batch 64, float32 | Revision is the hub head when queried, not pinned during runs; pin it before freezing. BGE-small rev 5c38ec7c... was also measured |
| Prompt/response truncation rates | Context 16% (160 of 1,000); response 0.9% (MiniLM, 256 tokens). BGE-small: 1.7% / 0.1% | 1,000-pair sample, special tokens counted. Reward-model input (256 total, response cap 128) truncation not measured |
| Embedding wall time and peak memory | MiniLM 7.5 s for 3,000 texts (398 texts/s), peak 400 MB. BGE-small 16.1 s (186 texts/s), 815 MB | From the 1,000-pair encoder study; the 2,000-pair pilot embedding was not separately timed |
| Feature shape, precision, and storage | 2,000 x 1,152, float32, 8.8 MB (pilot). Full train would be about 0.6 GB | [prompt, mean of replies, absolute difference]; unnormalized on disk, normalized at use |
| Reference k-center N, k, and runtime | N=2,000, k=200: 1.4 s (NumPy, CPU). N=8,000, k=800: 19.8 s. About 3 ns per N*k*d | Extrapolated: N=50k -> 14 min; N=133,535 (k=13,353) -> about 1.7 h. Larger runs used tiled, noised rows (timing only) |
| Reward-training time per epoch | 3.06 s per optimizer step of 32 pairs (median of 10 steps, GPU at about 1,750-1,785 MHz), so about 0.096 s per pair: 2,000 pairs = about 3.2 min per epoch (training only; validation passes extra) | Re-timed with the GPU unthrottled. The earlier 24 s per step came from a power-throttled GPU (300 MHz). 12 steps only, one run |
| Reward-model peak GPU memory | 2.34 GB (DistilBERT full fine-tune, 256 tokens, micro-batch 8 pairs, fp32) | Fits in 4 GB with about 1.6 GB spare |
| Validation accuracy and pairwise loss | 2,000 pairs, 2 epochs: acc 0.485 -> 0.548 -> 0.558; loss 0.6956 -> 0.6869 -> 0.6884 (1,000 val pairs, about 1.6 pt sd). 200 pairs, 3 epochs: acc 0.50 (random) and 0.51 (k-center), at chance | One seed. k-center vs random not distinguishable at k=200 |
| Projected main-experiment cost | Selection (NumPy CPU): N=20k -> 2.3 min, N=50k -> 14 min, N=133,535 -> about 1.7 h. Training at 0.096 s per pair-epoch: k=2,000 x 3 epochs = about 10 min per run; 2 methods x 3 repeats = 6 runs, about 1 h plus validation passes. k=5,000 (N=50k): about 24 min per run, 2.4 h for 6 runs | Training-only estimate from 12 timed steps; excludes validation passes, embedding the pool, and any third method or second budget. Assumes the GPU stays unthrottled |

GPU-resident selector time, end-to-end selector time, and complete pipeline cost must be reported separately. Random sampling does not require selection embeddings; account for shared or cached work consistently.

## Final decision record

| Decision | Chosen value | Evidence/rationale | Owner | Status |
| --- | --- | --- | --- | --- |
| Selection encoder and revision | all-MiniLM-L6-v2, 384-d, 256 tokens, rev 1110a243... | BGE-small truncates far less (1.7% vs 16% of contexts) but MiniLM is 2x faster, uses half the memory, and retrieves the right reply slightly better (78.0% vs 72.8% top-1, left-truncated). Same 384-d, so a later swap keeps the feature shape | Katherine | Provisional; revisit after a subset-quality comparison and team review |
| Parsing and representation | Separate embeddings of context, chosen, rejected. Context = transcript through last human turn; reply = text after last `Assistant:` | 20 parsed examples inspected; 21 echo cases found and handled. Group = first human turn; split by group hash | Katherine | Proposed; test-set choice needs team agreement |
| Truncation and formatting | Context left-truncated (keep recent turns), reply right-truncated, 256 tokens including special tokens | Left truncation raised MiniLM top-1 on over-limit contexts from 68.1% to 75.6% (n=160) | Katherine | Proposed |
| Feature construction | x = [p, (a+b)/2, abs(a-b)], 1,152-d, float32 | Swap-invariant (bit-identical), finite on 2,000 pairs. One pair has a zero difference part (equal embeddings after truncation) | Katherine | Proposed; usefulness of the three parts for selection untested |
| Normalization and distance | Unit-normalize the final vector only; squared Euclidean, float32; x / max(norm, 1e-6) | Component normalization equalizes norms, not distance contributions (shares 41/43/16% vs 45/36/19%). The two choose different points (27.5% k-center overlap at k=200) | Katherine | Provisional; which selects better data is untested |
| Candidate pool and subset budget | Pilot: N=2,000, k=200 (10%). Main: k = floor(0.10 N), N not yet chosen | k=200 is too small to separate methods (both at chance). Selection cost scales with N^2 at fixed 10% (NumPy CPU: 14 min at 50k, about 1.7 h at 133,535) | Katherine | Open; choose N from the training-cost estimate |
| Seeds and numerical rules | Split seed 0 (hash of group); selection start row 0, ties to smallest index; training seed 0 (same init for all methods); target 3 repeats | Only seed 0 run so far; GPU not in deterministic mode, so reruns can differ slightly | Katherine | Partly done; repeats not run |
| Reward architecture and training | DistilBERT-base-uncased (rev 12040acc...) fully fine-tuned + scalar head on [CLS]; AdamW, lr 2e-5, effective batch 32; 256 tokens, reply capped at 128; fp32 | Loss falls on a tiny batch (0.693 to 0.044). 2,000 pairs gives 55.8% val acc. Peak memory 2.34 GB of 4 GB | Katherine | Provisional; training time not yet measured at full GPU speed |
| Evaluation and checkpoint rules | Checkpoint = lowest validation pairwise loss; report accuracy and loss; test set reserved | Not yet exercised: pilot reported per-epoch validation only, test set untouched. Validation loss started rising at epoch 2 | Katherine | Proposed |

## Completion criteria

- [ ] Every decision has a chosen value and rationale.
- [ ] A small pipeline runs from parsed records to selection and reward training.
- [ ] Split isolation, feature symmetry, and finite values are checked.
- [ ] Runtime and memory estimates support the proposed scope.
- [ ] Main methods use identical candidate pools and comparison settings.
- [ ] Test data has not been used to tune the design.
- [ ] The team has reviewed the configuration.
- [ ] The final specification and pilot measurements are version-controlled.

K-center does not need to beat random sampling in the pilot. This section is complete when the design is justified, reproducible, and feasible. Optimized custom CUDA kernels belong to the subsequent implementation phase.

## Research references

1. [Anthropic HH-RLHF dataset](https://huggingface.co/datasets/Anthropic/hh-rlhf) — source format and dataset scope.
2. [Training a Helpful and Harmless Assistant with RLHF](https://arxiv.org/abs/2204.05862) — preference-data context.
3. [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — dimensions, pooling, and default truncation.
4. [BGE-small model card](https://huggingface.co/BAAI/bge-small-en-v1.5) — dimensions, input length, and usage.
5. [MPNet model card](https://huggingface.co/sentence-transformers/all-mpnet-base-v2) and [Sentence Transformers configuration](https://huggingface.co/sentence-transformers/all-mpnet-base-v2/blob/main/sentence_bert_config.json) — optional encoder.
6. [DistilBERT model card](https://huggingface.co/distilbert/distilbert-base-uncased) — reward-backbone candidate.
7. [TRL reward modeling](https://huggingface.co/docs/trl/reward_trainer) — pairwise objective and evaluation.
8. [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) — seeds, determinism, and platform limitations.
9. [Towards Data-Centric RLHF](https://arxiv.org/abs/2409.09603) — scale, label noise, and information content.
10. [Towards Comprehensive Preference Data Collection for Reward Modeling](https://arxiv.org/abs/2406.16486) — data quality and diversity.

Model specifications above reflect the sources reviewed for this plan. Verify installed versions and actual settings during implementation. The references motivate the study; they do not establish that this specific k-center representation will outperform random selection.

