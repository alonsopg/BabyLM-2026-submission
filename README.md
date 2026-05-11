# BabyLM 2026 Submission: Heavy-Hitter and Accuracy-Morph Masking

This repository contains the code, paper draft, and lightweight result exports for a BabyLM 2026 Strict-Small masking-only study.

The central experimental constraint is that compared systems use the same data, tokenizer, model architecture, optimizer, schedule, training steps, and total MLM mask rate. The intended difference is only the masking policy.

## Repository Layout

```text
configs/        Experiment configs for random MLM, HHM, CMS-Morph, and Accuracy-Morph variants
src/            Training, masking, model, data, evaluation, and analysis code
scripts/        Experiment runner wrappers
tests/          Unit tests for masking behavior and sketch logic
paper/          LaTeX paper draft, tables, references, and compiled PDF
paper_exports/  Lightweight CSV/PNG exports used to populate paper tables
docs/           Method notes and original implementation README
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

## Paper

The current draft is in `paper/`.

```bash
cd paper
latexmk -pdf -interaction=nonstopmode babylm-sketchselect-main.tex
```

The current paper framing is mixed: HHM is a controlled negative/mixed result, while Accuracy-Morph is the most promising follow-up, improving matched validation loss in seeds 1 and 2 and producing the clearest Entity Tracking gains.

