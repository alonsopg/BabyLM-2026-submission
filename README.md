# BabyLM 2026 Submission: Accuracy-Morph Mask Selection

This repository contains the code, paper files, lightweight result exports, and shared-task prediction bundle for a BabyLM 2026 Strict-Small masking-only experiment.

The focused short paper asks:

> Can small language models learn more by masking words they repeatedly get wrong?

The central experimental constraint is that compared approaches use the same data, tokenizer, model architecture, optimizer, schedule, training steps, replacement policy, and total 15% MLM mask rate. The intended difference is only the masking policy.

## Repository Layout

```text
configs/        Experiment configs for random MLM, HHM, CMS-Morph, and Accuracy-Morph variants
src/            Training, masking, model, data, evaluation, and analysis code
scripts/        Experiment runner wrappers
tests/          Unit tests for masking behavior and sketch logic
paper/          Long diagnostic paper, short paper, tables, references, and compiled PDFs
paper_exports/  Lightweight CSV/PNG exports used to populate paper tables
docs/           Method notes and original implementation README
shared_task_submission/
                Final prediction/submission bundle and manifest
```

Large artifacts are intentionally not included: checkpoints, tokenized caches, local BabyLM evaluation resources, raw datasets, and full training output directories.

## Install

```bash
pip install -r requirements.txt
```

The original experiments were run in a CUDA-capable environment. GPU training is recommended.

## Run Tests

```bash
python tests/run_tests.py
```

## Run Experiments

Smoke test:

```bash
python scripts/run_experiments.py --suite smoke --seeds 1
```

Main random-vs-HHM comparison:

```bash
python scripts/run_experiments.py --suite main --seeds 1 2 3
```

Final Accuracy-Morph seed-2 run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.training.train \
  --config configs/accuracy_morph_seed2.yaml \
  --seed 2
```

## Papers

The focused shared-task paper is in `paper/short/`.

```bash
cd paper/short
latexmk -pdf -interaction=nonstopmode babylm_accuracy_morph_short.tex
```

The longer diagnostic report is kept in `paper/`.

```bash
cd paper
latexmk -pdf -interaction=nonstopmode babylm-sketchselect-main.tex
```

The short paper centers on Accuracy-Morph: a correctness-smoothed adaptive mask-selection policy that tracks token and character-trigram exposure/error statistics and redirects a small part of the fixed MLM mask budget toward words the model repeatedly predicts incorrectly.

## Shared-Task Bundle

The prepared submission artifact is:

```text
shared_task_submission/accuracy_morph_seed2_submission.zip
```

The manifest in `shared_task_submission/accuracy_morph_seed2_submission_manifest.json` records included prediction files, hashes, evaluation notes, and known limitations. Full filtered EWoK is included for the final Accuracy-Morph checkpoint; AoA remains partial because only the final checkpoint was retained, while the official strict-small AoA pipeline expects a checkpoint series.
