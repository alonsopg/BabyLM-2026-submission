# BabyLM 2026 Semantic-Cloze Multi-Task Submission

This branch contains the clean multi-task setup for the selected BabyLM 2026 Strict-Small semantic-cloze experiment. It keeps only the code, data, results, and minimal submission bundle for the best semantic-cloze multi-task model.

## Final Model

- Hugging Face model: https://huggingface.co/alonsopg/babylm-2026-semantic-cloze-strict-small
- HF repo id: `alonsopg/babylm-2026-semantic-cloze-strict-small`
- Track target: `strict-small`
- Backend: `mlm`
- Architecture: compact BERT-style masked language model, 11.2M parameters
- Selected experiment: `experiments/multitask_ewok_semantic_cloze_v1`

## What This Branch Keeps

```text
experiments/multitask_distributional_bert/
  Shared multi-task trainer, data generation helpers, eval table collector,
  and environment scripts.

experiments/multitask_ewok_semantic_cloze_v1/
  The selected semantic-cloze experiment, config, generated auxiliary data,
  compact run records, fast-eval summaries, and notes.

src/
  Minimal shared utilities required by the multi-task trainer: dataset loading,
  random MLM masking, model construction, and MLM validation loss evaluation.

shared_task_submission/semantic_cloze/
  Official BabyLM minimal submission artifact, manifest, checksums, and compact
  eval reports.

shared_task_submission/semantic_cloze_minimal_submission.zip
  Uploadable minimal submission bundle.
```

Large model weights and local official-eval resources are not tracked here. The final public model is hosted on Hugging Face.

## Multi-Task Objective

The selected run trains a compact MLM-compatible BERT model with this task mixture:

| Task | Weight |
|---|---:|
| MLM | 0.600 |
| RTD | 0.100 |
| Connective prediction | 0.075 |
| Definiteness prediction | 0.075 |
| Collocation classification | 0.050 |
| Grammar min-pair classification | 0.050 |
| Semantic cloze ranking | 0.050 |

The semantic cloze task uses an MLM-head pairwise ranking loss. Each example contains one `[MASK]`, one plausible single-token completion, and one implausible single-token completion. The loss is:

```text
softplus(-(log p(good | context) - log p(bad | context)))
```

This keeps the trained model compatible with the official BabyLM MLM evaluation path and avoids adding a submission-time classifier head.

## Key Results

Fast local comparison for the selected final checkpoint:

| System | BLiMP | BLiMP Sup. | EWoK | Entity | Eye | SPR |
|---|---:|---:|---:|---:|---:|---:|
| MLM baseline | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| Semantic cloze final | 55.26 | 54.40 | 52.00 | 27.44 | 8.18 | 3.71 |

Official full-eval scores observed for the uploaded model:

| Section | Score |
|---|---:|
| BLiMP filtered | 55.35 |
| BLiMP supplement filtered | 53.29 |
| EWoK filtered | 50.45 |
| Entity tracking | 27.55 |
| COMPS | 50.63 |
| Reading eye-tracking | 8.18 |
| Reading SPR | 3.71 |

Finetuning scores are recorded in `shared_task_submission/semantic_cloze/README.md`.

## Recreate The Selected Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the selected configuration on GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml \
  --seed 1
```

Generate semantic-cloze data if needed:

```bash
python experiments/multitask_ewok_semantic_cloze_v1/scripts/generate_semantic_cloze.py
```

## Minimal Submission

The minimal submission bundle is:

```text
shared_task_submission/semantic_cloze_minimal_submission.zip
```

Bundle checksum:

```text
f02d7c90381a60d91cbbbff8ebbe22c5cc7faad9a29200dd8f21adbb74fc93e7
```

Known incompleteness for a fully challenge-valid package:

- AoA surprisal was not run, so `aoa` is `null` in the collated artifact.
- Checkpoint-revision fast evals from `chck_1M` through `chck_100M` are absent.
- Only the final selected model is currently uploaded publicly on Hugging Face.
