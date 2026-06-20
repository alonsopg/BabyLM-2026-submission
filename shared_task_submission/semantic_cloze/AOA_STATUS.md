# AoA Evaluation Status

Date: 2026-06-18

Updated: 2026-06-20

## Original Attempted Command

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

## Original Result

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

Every required `chck_*` revision returned `404 Not Found`, so no AoA surprisal results were generated at that time.

## Current Status

Fixed on 2026-06-20. The official Hugging Face model repo now contains all 19 required `chck_*` revisions, uploaded from an isolated AoA rerun trajectory while leaving the selected `main` final model unchanged.

The official AoA command now resolves the checkpoint revisions successfully and produced:

```text
/home/paperspace/babylm-hhm/resources/babylm-eval/strict/results/babylm-2026-semantic-cloze-strict-small/main/zero_shot/mlm/AoA_word/surprisal.json
```

AoA output:

```text
processed checkpoints: 19
predictions per checkpoint: 8005
total predictions: 152095
```

The collated submission artifact has been regenerated, and `aoa` is no longer `null`.

## Rerun Validation

On 2026-06-20, an isolated AoA rerun was first completed and validated in a temporary Hugging Face repo:

```text
alonsopg/babylm-2026-semantic-cloze-aoa-rerun
```

The official AoA script successfully processed all 19 required checkpoint revisions in that temporary repo and produced `surprisal.json`. After the 404 cause was confirmed, the same 19 checkpoint revisions were uploaded to the official submission repo:

```text
alonsopg/babylm-2026-semantic-cloze-strict-small
```

See `AOA_RERUN.md` for the full record.

## Remaining Caveat

The `chck_*` revisions are real intermediate model states from the isolated rerun, not copies of the final model. However, the selected `main` final model remains the stronger original final checkpoint. This means the AoA trajectory and final `main` checkpoint do not come from the exact same local training run.

Checkpoint-revision fast-eval outputs for BLiMP, BLiMP supplement, EWoK, entity tracking, and reading are still absent for `chck_1M` through `chck_100M`; the collator fills those fast-eval entries with `null`.
