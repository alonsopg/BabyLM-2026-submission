# AoA Rerun Status

Date: 2026-06-20

## Summary

The missing AoA/checkpoint-revision issue was first fixed in a temporary Hugging Face repository, then the validated `chck_*` revisions were uploaded to the official submission model repository.

Temporary model repo:

```text
https://huggingface.co/alonsopg/babylm-2026-semantic-cloze-aoa-rerun
```

The temporary repo contains:

- `main`: final checkpoint from the isolated AoA rerun.
- `chck_1M` through `chck_9M`
- `chck_10M` through `chck_100M` by tens

Official submission model repo:

```text
https://huggingface.co/alonsopg/babylm-2026-semantic-cloze-strict-small
```

The official repo now contains the same 19 `chck_*` revisions. The selected `main` final model was not overwritten.

The official AoA script successfully processed all 19 checkpoint revisions for the official repo and produced:

```text
/home/paperspace/babylm-hhm/resources/babylm-eval/strict/results/babylm-2026-semantic-cloze-strict-small/main/zero_shot/mlm/AoA_word/surprisal.json
```

AoA output:

```text
processed checkpoints: 19
predictions per checkpoint: 8005
total predictions: 152095
```

## What Changed In The Trainer

The trainer now supports optional checkpoint-revision saving through `training.checkpoint_revisions`.

This is disabled unless explicitly enabled in a config. The original selected experiment config is unchanged.

The isolated AoA config is:

```text
experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1_aoa_rerun.yaml
```

It writes local checkpoints under:

```text
experiments/multitask_ewok_semantic_cloze_v1/checkpoints_aoa_rerun/
```

That directory is intentionally ignored by git because it contains large model artifacts.

## Schedule Used

The first attempt used actual non-padding token counts, but this custom multi-task setup would not reach `chck_100M` because many auxiliary batches are short. The final isolated rerun therefore used the official checkpoint names as training-timeline labels with `schedule_by: step_fraction`.

Mapping:

```text
chck_1M   -> step 90
chck_2M   -> step 180
...
chck_10M  -> step 900
chck_20M  -> step 1800
...
chck_100M -> step 9000
```

The checkpoints are real intermediate model states from the rerun, not copies of the final model.

Observed endpoints:

```text
chck_1M:   saved_at_step=90,   seen_tokens=165451
chck_100M: saved_at_step=9000, seen_tokens=15918903
```

## Rerun Training Result

The isolated rerun completed with early stopping at step 9500.

```text
best_val_mlm_loss: 6.79221757253011
best step: 5000
final eval step: 9500
early_stop: true
```

Task counts:

| Task | Count |
|---|---:|
| MLM | 6345 |
| RTD | 771 |
| Connective | 598 |
| Definiteness | 562 |
| Collocation | 383 |
| Grammar min-pair | 396 |
| Semantic cloze ranking | 445 |

## Fast Eval Comparison

The rerun final checkpoint was evaluated locally with the official fast-eval helper and compared to the current submitted final model.

| Metric | Current final | AoA rerun final | Delta |
|---|---:|---:|---:|
| BLiMP | 55.26 | 56.82 | +1.56 |
| BLiMP Supplement | 54.40 | 52.40 | -2.00 |
| EWoK | 52.00 | 50.91 | -1.09 |
| Entity Tracking | 27.44 | 21.52 | -5.92 |
| Reading Eye | 8.18 | 7.91 | -0.27 |
| Reading SPR | 3.71 | 3.60 | -0.11 |

## Decision

Update the official model repo with the isolated rerun trajectory for AoA, but keep the existing selected final model on `main`.

Reason: the temporary AoA rerun proves that the checkpoint-revision/AoA pipeline works, and the `chck_*` branches are real intermediate states rather than final-model copies. The rerun final checkpoint is materially worse than the current final on entity tracking, so the stronger existing final model remains on `main`.

The minimal submission artifact has been regenerated with AoA populated:

```text
shared_task_submission/semantic_cloze_minimal_submission.zip
```

The existing final Hugging Face model on `main` remains untouched:

```text
alonsopg/babylm-2026-semantic-cloze-strict-small
```

## Commands Used

Training:

```bash
cd /home/paperspace/BabyLM-2026-submission
source experiments/multitask_distributional_bert/scripts/env.sh
CUDA_VISIBLE_DEVICES=0 python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1_aoa_rerun.yaml \
  --seed 1
```

Fast eval on rerun final:

```bash
CUDA_VISIBLE_DEVICES=0 experiments/multitask_distributional_bert/scripts/run_official_fast_eval.sh \
  aoa_rerun_final \
  experiments/multitask_ewok_semantic_cloze_v1/checkpoints_aoa_rerun/multitask_ewok_semantic_cloze_v1_aoa_rerun/seed_1/mlm_compatible_checkpoint \
  main
```

AoA on temporary repo:

```bash
cd /home/paperspace/babylm-hhm/resources/babylm-eval/strict
source /home/paperspace/BabyLM-2026-submission/experiments/multitask_distributional_bert/scripts/env.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_aoa.sh \
  alonsopg/babylm-2026-semantic-cloze-aoa-rerun \
  mlm \
  strict-small \
  evaluation_data/full_eval/aoa/cdi_childes.json \
  results
```

AoA on official repo:

```bash
cd /home/paperspace/babylm-hhm/resources/babylm-eval/strict
source /home/paperspace/BabyLM-2026-submission/experiments/multitask_distributional_bert/scripts/env.sh
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_aoa.sh \
  alonsopg/babylm-2026-semantic-cloze-strict-small \
  mlm \
  strict-small \
  evaluation_data/full_eval/aoa/cdi_childes.json \
  results
```
