# BabyLM 2026 Minimal Submission: Semantic Cloze Multi-Task

This folder contains the minimal official-eval artifact for the selected multi-task BabyLM system:

- Hugging Face model: https://huggingface.co/alonsopg/babylm-2026-semantic-cloze-strict-small
- HF repo id: `alonsopg/babylm-2026-semantic-cloze-strict-small`
- HF commit: `4530b3e6be2c264f339a59a2223c35bf3e12798a`
- Track target: `strict-small`
- Backend: `mlm`
- Selected local experiment: `experiments/multitask_ewok_semantic_cloze_v1`

## Included Artifact

- `artifacts/all_full_preds_and_fast_scores_mlm.json`
  - Produced by the official BabyLM eval collator.
  - Contains full zero-shot predictions, AoA surprisal results, finetuning predictions, and checkpoint fast-eval results for all required `chck_*` revisions.
  - SHA256: `ce458a181ef04dc47284da51dc7bb6e5b1902666a2d7e725cdee226bb23608d7`

## Official Eval Commands

Official eval checkout:

```bash
cd /home/paperspace/babylm-hhm/resources/babylm-eval/strict
```

Environment:

```bash
source /home/paperspace/BabyLM-2026-submission/experiments/multitask_distributional_bert/scripts/env.sh
```

Full zero-shot:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_zero_shot.sh \
  alonsopg/babylm-2026-semantic-cloze-strict-small \
  mlm \
  evaluation_data/full_eval
```

Full finetuning:

```bash
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES=0 bash scripts/eval_finetuning.sh \
  --model_path alonsopg/babylm-2026-semantic-cloze-strict-small
```

Collation:

```bash
bash scripts/collate_preds.sh \
  alonsopg/babylm-2026-semantic-cloze-strict-small \
  mlm \
  strict-small
```

AoA:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/eval_aoa.sh \
  alonsopg/babylm-2026-semantic-cloze-strict-small \
  mlm \
  strict-small \
  evaluation_data/full_eval/aoa/cdi_childes.json \
  results
```

Checkpoint fast eval:

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

## Full Eval Scores Observed

| Section | Metric | Score |
|---|---:|---:|
| BLiMP filtered | average accuracy | 55.35 |
| BLiMP supplement filtered | average accuracy | 53.29 |
| EWoK filtered | average accuracy | 50.45 |
| Entity tracking | average accuracy | 27.55 |
| COMPS | average accuracy | 50.63 |
| Reading | eye-tracking score | 8.18 |
| Reading | SPR score | 3.71 |

## Finetuning Scores Observed

| Task | Accuracy | F1 | MCC |
|---|---:|---:|---:|
| BoolQ | 0.6679 | 0.7755 | 0.2061 |
| MNLI | 0.4269 | n/a | n/a |
| MRPC | 0.6912 | 0.8131 | 0.1300 |
| MultiRC | 0.5858 | 0.2132 | 0.0859 |
| QQP | 0.7044 | 0.5816 | 0.3554 |
| RTE | 0.5468 | 0.5191 | 0.0910 |
| WSC | 0.5769 | 0.1538 | -0.0381 |

## Submission Status

This is ready as a complete artifact package for the selected final model. The model loads with Hugging Face Transformers as `AutoModelForMaskedLM`, all seven finetuning prediction files are present in the collated JSON, AoA is populated, and checkpoint fast-eval results are populated for all 19 required revisions.

Leaderboard form metadata is prepared in `LEADERBOARD_FORM_PAYLOAD.md` and `leaderboard_form_payload.json`. See `LEADERBOARD_SUBMISSION_STATUS.md` for the 2026-06-22 leaderboard API submission attempt and the remaining manual browser submission step.

Known provenance caveat:

- The required `chck_*` revisions were uploaded from an isolated AoA rerun trajectory while the selected `main` final model remains the stronger original final checkpoint. Thus, the checkpoint trajectory and `main` are valid model artifacts, but they do not come from the exact same local training run.

AoA originally failed on 2026-06-18 because the official script requires Hugging Face revisions named `chck_1M` through `chck_100M`. On 2026-06-20, those revisions were uploaded from an isolated rerun trajectory, the official AoA script processed all 19 checkpoints successfully, and the official fast-eval helper was run over all 19 checkpoint revisions. See `AOA_STATUS.md` and `AOA_RERUN.md`.
