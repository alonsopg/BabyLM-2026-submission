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
  - Contains full zero-shot predictions, finetuning predictions, and available fast-eval results.
  - SHA256: `980297bfd29ceeb0900b89fe35256c59b12023b9bc0a3555abefc4c430af3e9e`

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

This is ready as a minimal artifact for the selected final model. The model loads with Hugging Face Transformers as `AutoModelForMaskedLM`, and all seven finetuning prediction files are present in the collated JSON.

Known incompleteness for a full Challenge-valid package:

- AoA surprisal was not run, so `aoa` is `null` in the collated artifact.
- The required checkpoint-revision fast-eval series (`chck_1M` through `chck_100M`) was not run or uploaded; the collator filled those missing revision entries with `null`.
- The final-model fast-eval results are present from the existing experiment, but not the full checkpoint trajectory.
