# Paper Notes: Semantic Cloze Multi-Task BabyLM

These notes collect the details likely needed when expanding the short paper draft into the final BabyLM 2026 submission. They intentionally focus on the multi-task semantic cloze experiment and exclude the earlier morph experiments.

## One-Sentence Framing

We study whether a compact BERT-style masked language model trained on BabyLM Strict-Small can gain conceptual plausibility behavior from a lightweight semantic cloze ranking auxiliary objective, while remaining a normal `AutoModelForMaskedLM` at submission time.

## Core Claim

The semantic cloze multi-task final checkpoint improves local fast-eval EWoK over a plain MLM baseline:

```text
MLM baseline EWoK:             50.55
Semantic cloze final EWoK:     52.00
Delta:                         +1.45
```

It also improves reading metrics over the MLM baseline:

```text
Reading eye: 2.47 -> 8.18
Reading SPR: 2.42 -> 3.71
```

Trade-off:

```text
BLiMP drops: 64.33 -> 55.26
Entity tracking rises over MLM baseline but is lower than the semantic-cloze best-validation checkpoint:
  MLM baseline:                 15.56
  Semantic cloze best val:      36.61
  Semantic cloze final:         27.44
```

Careful wording: this is evidence for a shift in behavior from auxiliary objectives, not proof that this objective is globally better.

## Submission Identity

- Challenge: BabyLM 2026
- Track: `strict-small`
- Model name on leaderboard: `semantic-cloze-multitask-bert-strict-small`
- Hugging Face repo: `alonsopg/babylm-2026-semantic-cloze-strict-small`
- HF revision for final model: `main`
- Local experiment: `experiments/multitask_ewok_semantic_cloze_v1`
- Git branch: `multitask-semantic-cloze-clean`
- Current source repo: `git@github.com:alonsopg/BabyLM-2026-submission.git`

## Model Architecture

Compact BERT-style encoder-only masked language model:

| Item | Value |
|---|---:|
| Parameters | 11,201,338 |
| Hidden size | 256 |
| Layers | 4 |
| Attention heads | 4 |
| Intermediate size | 1024 |
| Max position embeddings | 512 |
| Training max sequence length | 128 |
| Tokenizer | `bert-base-uncased` WordPiece |
| Token set size | 30,522 |
| Submission interface | `AutoModelForMaskedLM` |

Important point for paper: no classifier head is required at submission time. The exported checkpoint is an MLM-compatible model.

## Training Data

- Dataset: `BabyLM-community/BabyLM-2026-Strict-Small`
- Track budget: 10M words
- Text column: `text`
- Validation size: 2,000 examples
- No external datasets.
- No official EWoK data.
- No official BLiMP data.
- Auxiliary examples are generated from the BabyLM training data and small hand-written semantic cloze templates.

## Optimization

| Hyperparameter | Value |
|---|---:|
| Optimizer | AdamW |
| Max learning rate | 5e-4 |
| Weight decay | 0.01 |
| Warmup | 6% of steps |
| Scheduler | linear warmup, then linear decay |
| Batch size | 96 sequences |
| Approx. batch size in tokens | 12,288 |
| Max steps | 10,000 |
| Stopped step | 9,500 |
| Gradient clipping | 1.0 |
| Eval interval | 500 steps |
| Eval batches | 40 |
| Precision | fp32 |
| Seed | 1 |
| Device | NVIDIA RTX A6000 |

Early stopping:

| Field | Value |
|---|---:|
| Enabled | true |
| Monitor | `val_mlm_loss` |
| Patience | 2 evals |
| Min delta | 0.01 |
| Start after step | 9,000 |
| Best validation step | 5,000 |
| Best validation MLM loss | 6.810523 |

Leaderboard form caveat:

- The form requires integer FLOPs, so we used `8000000000000000`.
- This corresponds to the earlier rough `0.008` PFLOP estimate.
- GPU hours are approximate: development `12`, training `4`.

## Task Mixture

| Task | Sampling weight | Observed steps |
|---|---:|---:|
| MLM | 0.600 | 6,307 |
| RTD | 0.100 | 800 |
| Connective | 0.075 | 588 |
| Definiteness | 0.075 | 620 |
| Collocation | 0.050 | 435 |
| Grammar min-pair | 0.050 | 362 |
| Semantic cloze ranking | 0.050 | 388 |

The isolated AoA rerun has slightly different observed counts due to rerun stochasticity:

| Task | Rerun steps |
|---|---:|
| MLM | 6,345 |
| RTD | 771 |
| Connective | 598 |
| Definiteness | 562 |
| Collocation | 383 |
| Grammar min-pair | 396 |
| Semantic cloze ranking | 445 |

Use the first table when describing the selected final experiment. Use the second only when explaining the AoA/checkpoint-revision provenance caveat.

## Auxiliary Objectives

### Masked Language Modeling

Standard MLM objective over BabyLM Strict-Small text.

### Replaced-Token Detection

Generated from corpus examples by replacing tokens with plausible alternatives from a local candidate inventory. Used as a token-level discrimination signal.

### Connective Prediction

Masks or predicts discourse connectives from a fixed connective inventory, encouraging sensitivity to local discourse relations.

### Definiteness Prediction

Targets article/determiner behavior around `a`, `an`, and `the`.

### Collocation Discrimination

Ranks or distinguishes frequent corpus bigrams from same-class replacements, encouraging local lexical association knowledge.

### Grammar Minimal-Pair Ranking

Uses grammar minimal-pair examples and ranks the preferred sentence higher through the MLM scoring machinery.

### Semantic Cloze Ranking

The new EWoK-focused idea.

Data:

| Domain | Examples |
|---|---:|
| Affordance | 356 |
| Animate agency | 140 |
| Part-whole | 168 |
| Physical property | 168 |
| Typical location | 168 |
| Total | 1,000 |

Skipped due to multi-token target issues:

```text
4 examples
```

The semantic cloze data does not use official EWoK or BLiMP items.

Loss:

```text
L = softplus(-(score_good - score_bad))
```

Where `score_good` and `score_bad` are MLM log-probabilities at the masked position for the plausible and implausible single-token completions.

Good wording:

> The semantic cloze task uses the MLM head itself as a plausibility scorer, so the auxiliary task changes pretraining but does not require an extra inference-time architecture.

## Local Fast-Eval Comparison

| System | BLiMP | BLiMP Supplement | EWoK | Entity Tracking | Reading Eye | Reading SPR |
|---|---:|---:|---:|---:|---:|---:|
| MLM baseline | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| Semantic cloze best validation | 54.10 | 57.60 | 50.73 | 36.61 | 7.56 | 3.11 |
| Semantic cloze final | 55.26 | 54.40 | 52.00 | 27.44 | 8.18 | 3.71 |

Interpretation:

- Final checkpoint is strongest for EWoK and reading.
- Best-validation checkpoint is better for entity tracking.
- MLM baseline is much stronger on BLiMP.
- This suggests auxiliary objectives shift the evaluation profile rather than uniformly improving all tasks.

## Full Official Eval Scores

Observed from local official evaluation reports:

| Section | Metric | Score |
|---|---|---:|
| BLiMP filtered | average accuracy | 55.35 |
| BLiMP supplement filtered | average accuracy | 53.29 |
| EWoK filtered | average accuracy | 50.45 |
| Entity tracking | average accuracy | 27.55 |
| COMPS | average accuracy | 50.63 |
| Reading | eye-tracking score | 8.18 |
| Reading | SPR score | 3.71 |

Finetuning:

| Task | Accuracy | F1 | MCC |
|---|---:|---:|---:|
| BoolQ | 0.6679 | 0.7755 | 0.2061 |
| MNLI | 0.4269 | n/a | n/a |
| MRPC | 0.6912 | 0.8131 | 0.1300 |
| MultiRC | 0.5858 | 0.2132 | 0.0859 |
| QQP | 0.7044 | 0.5816 | 0.3554 |
| RTE | 0.5468 | 0.5191 | 0.0910 |
| WSC | 0.5769 | 0.1538 | -0.0381 |

## Leaderboard Status

Manual browser submission succeeded on 2026-06-22 after using integer FLOPs.

Live row observed on the Strict-Small leaderboard:

| Metric | Score |
|---|---:|
| Text Average | 33.0 |
| BLiMP | 55.36 |
| BLiMP Supplement | 53.29 |
| EWoK | 0.0 |
| Entity Tracking | 27.55 |
| COMPS | 50.63 |
| Reading | 5.94 |
| AoA | 11.25 |
| SuperGLUE | 60.0 |

Important unresolved issue:

- Leaderboard EWoK displays `0.0`.
- Local artifact contains EWoK predictions.
- Local official eval report gives EWoK `50.45`.
- `fast_eval_results.ewok` has 19 non-null checkpoint entries.

Likely paper handling:

- If unresolved, do not rely on public leaderboard EWoK for the main claim.
- State local official evaluation EWoK clearly.
- Add a footnote or limitation that the leaderboard displayed EWoK as zero despite submitted predictions, pending organizer clarification.

## AoA And Checkpoint Revisions

The official BabyLM AoA and fast-eval trajectory require Hugging Face branches named:

```text
chck_1M, chck_2M, ..., chck_9M,
chck_10M, chck_20M, ..., chck_100M
```

Current state:

- Public HF model repo has all 19 required `chck_*` revisions.
- AoA is populated in the collated artifact.
- Checkpoint fast-eval is populated for BLiMP, BLiMP supplement, EWoK, entity tracking, and reading.
- No `null` entries remain in `fast_eval_results`.

Provenance caveat:

- Original selected `main` final model did not export `chck_*` revisions during training.
- To satisfy AoA/checkpoint requirements, an isolated rerun with the same configuration was used to create real intermediate `chck_*` model states.
- `main` remains the stronger original selected final checkpoint.
- Therefore, `main` and `chck_*` are valid model artifacts but do not come from the exact same local training run.

Suggested wording:

> Because the original selected run did not export the official checkpoint-revision branches, we performed an isolated rerun with the same configuration to populate the required `chck_*` trajectory. The submitted `main` checkpoint remains the stronger selected final model. We therefore treat AoA and checkpoint fast-eval trajectories as complete but note that the trajectory and selected final checkpoint are not from the same local run.

## Reproducibility Pointers

Important files:

| Purpose | Path |
|---|---|
| Main config | `experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml` |
| AoA rerun config | `experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1_aoa_rerun.yaml` |
| Experiment summary | `experiments/multitask_ewok_semantic_cloze_v1/metrics/comparison_summary.md` |
| Fast-eval table | `experiments/multitask_ewok_semantic_cloze_v1/metrics/official_fast_eval_table.md` |
| Submission status | `shared_task_submission/semantic_cloze/LEADERBOARD_SUBMISSION_STATUS.md` |
| Collated predictions | `shared_task_submission/semantic_cloze/artifacts/all_full_preds_and_fast_scores_mlm.json` |
| Paper draft | `paper/semantic-cloze-multitask-babylm.tex` |

Training command:

```bash
cd /home/paperspace/BabyLM-2026-submission
source experiments/multitask_distributional_bert/scripts/env.sh
CUDA_VISIBLE_DEVICES=0 python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml \
  --seed 1
```

Official full zero-shot:

```bash
cd /home/paperspace/babylm-hhm/resources/babylm-eval/strict
source /home/paperspace/BabyLM-2026-submission/experiments/multitask_distributional_bert/scripts/env.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_zero_shot.sh \
  alonsopg/babylm-2026-semantic-cloze-strict-small \
  mlm \
  evaluation_data/full_eval
```

Official finetuning:

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 bash scripts/eval_finetuning.sh \
  --model_path alonsopg/babylm-2026-semantic-cloze-strict-small
```

Official AoA:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_aoa.sh \
  alonsopg/babylm-2026-semantic-cloze-strict-small \
  mlm \
  strict-small \
  evaluation_data/full_eval/aoa/cdi_childes.json \
  results
```

Official checkpoint fast eval:

```bash
for rev in chck_1M chck_2M chck_3M chck_4M chck_5M chck_6M chck_7M chck_8M chck_9M \
  chck_10M chck_20M chck_30M chck_40M chck_50M chck_60M chck_70M chck_80M chck_90M chck_100M; do
  CUDA_VISIBLE_DEVICES=0 bash scripts/eval_zero_shot_fast.sh \
    alonsopg/babylm-2026-semantic-cloze-strict-small \
    "$rev" \
    mlm \
    evaluation_data/fast_eval
done
```

Collation:

```bash
bash scripts/collate_preds.sh \
  alonsopg/babylm-2026-semantic-cloze-strict-small \
  mlm \
  strict-small
```

## Paper Structure To Expand

Recommended final paper structure:

1. Introduction
   - BabyLM Strict-Small problem.
   - Motivation for auxiliary plausibility ranking.
   - Main result and trade-off.

2. Method
   - Model architecture.
   - Multi-task training.
   - Semantic cloze construction.
   - Ranking loss.

3. Experimental Setup
   - Dataset and constraints.
   - Optimization.
   - Evaluation pipeline.
   - Submission artifact and HF model.

4. Results
   - Local fast-eval comparison table.
   - Full official eval table.
   - Finetuning table.
   - Leaderboard row, with EWoK caveat if still unresolved.

5. Analysis
   - Why EWoK/reading may improve.
   - Why BLiMP may drop.
   - Entity tracking checkpoint sensitivity.

6. Limitations
   - Single seed.
   - Small template set.
   - Checkpoint provenance caveat.
   - Leaderboard EWoK display issue, if unresolved.

7. Ethics
   - Official dataset.
   - No external personal data.
   - Research-only small model.

## Claims To Avoid

Avoid saying:

- The method is generally better than MLM.
- The leaderboard EWoK score improved.
- The checkpoint trajectory is from the exact same run as `main`.
- The semantic cloze examples cover all commonsense phenomena.
- The experiment proves cognitive plausibility.

Safer claims:

- The auxiliary objective changes the model's evaluation profile.
- The selected final checkpoint improved local official fast-eval EWoK and reading relative to a local MLM baseline.
- The result suggests semantic cloze ranking is a promising lightweight objective for small-model pretraining.
- The method preserves a standard MLM interface.

## Organizer Message If EWoK Remains Zero

Draft:

```text
Hello BabyLM organizers,

I submitted the Strict-Small model `semantic-cloze-multitask-bert-strict-small`
with HF repo `alonsopg/babylm-2026-semantic-cloze-strict-small`.

The leaderboard row is visible, but EWoK is displayed as 0.0. The submitted
collated JSON contains an `ewok` section with 11 subtasks, and the local official
evaluation report gives EWoK average accuracy 50.45. The checkpoint fast-eval
trajectory also contains 19 non-null EWoK entries.

Could you check whether the leaderboard scorer failed to read the EWoK section
or whether there is a schema/task-name mismatch?

Best,
Alonso
```

