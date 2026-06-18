# AoA Evaluation Status

Date: 2026-06-18

## Attempted Command

Run from the official BabyLM eval checkout:

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

## Result

The official AoA script started correctly, loaded the AoA word/context data, selected CUDA, and generated the expected strict-small checkpoint list:

```text
chck_1M, chck_2M, chck_3M, chck_4M, chck_5M, chck_6M, chck_7M,
chck_8M, chck_9M, chck_10M, chck_20M, chck_30M, chck_40M,
chck_50M, chck_60M, chck_70M, chck_80M, chck_90M, chck_100M
```

It then queried Hugging Face revisions such as:

```text
https://huggingface.co/alonsopg/babylm-2026-semantic-cloze-strict-small/resolve/chck_1M/config.json
```

Every required `chck_*` revision returned `404 Not Found`, so no AoA surprisal results were generated.

## Current Status

AoA is still missing from the collated submission artifact because the final Hugging Face model repo only contains the final selected model. The official BabyLM AoA pipeline requires checkpoint-revision model states, not only a final checkpoint.

## Required Next Step

To complete AoA honestly, we need real checkpoint revisions for the training trajectory:

1. Recover saved model checkpoints corresponding to the required `chck_*` names, or rerun training with checkpoint export enabled at those milestones.
2. Upload each checkpoint to the Hugging Face model repo as a revision named exactly `chck_1M`, `chck_2M`, ..., `chck_100M`.
3. Re-run `scripts/eval_aoa.sh`.
4. Re-run `scripts/collate_preds.sh` so `aoa` is no longer `null`.

Do not satisfy AoA by copying the final model into every `chck_*` revision; that would produce an artifact but would not represent a real acquisition trajectory.
