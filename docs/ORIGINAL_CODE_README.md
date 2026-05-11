# BabyLM Heavy-Hitter Masking

This folder contains a reproducible BabyLM Strict-Small experiment for comparing standard random MLM masking with Heavy-Hitter Masking (HHM).

Official resources used:

- Dataset: `BabyLM-community/BabyLM-2026-Strict-Small`
- Track: BabyLM 2026 Strict-Small, 10M words or less
- Evaluation repo: `resources/babylm-eval`

## Environment

Use the CUDA-capable conda environment already present on this machine:

```bash
conda run -n ir-model-compression python tests/run_tests.py
```

## Smoke Test

```bash
conda run -n ir-model-compression python scripts/run_experiments.py --suite smoke --seeds 1
```

## Main Experiments

Minimum paper comparison:

```bash
conda run -n ir-model-compression python scripts/run_experiments.py --suite main --seeds 1 2 3
```

Ablations and controls:

```bash
conda run -n ir-model-compression python scripts/run_experiments.py --suite ablations --seeds 1
```

All configured experiments:

```bash
conda run -n ir-model-compression python scripts/run_experiments.py --suite all --seeds 1 2 3
```

## Outputs

Each run writes to:

```text
outputs/<experiment>/seed_<n>/
```

The directory contains the saved config, metadata, `train_log.csv`, `mask_diagnostics.csv`, final checkpoint, tokenizer files, and error-sketch statistics when applicable.

## Paper Tables

After runs finish:

```bash
conda run -n ir-model-compression python -m src.analysis.aggregate_results
conda run -n ir-model-compression python -m src.analysis.export_latex_tables
conda run -n ir-model-compression python -m src.analysis.plot_training_curves
```

The LaTeX export updates `../babylm-sketchselect/paper_draft/tables/main_results.tex`.

## Fairness Constraint

The intended experimental difference is only the masking policy. Do not change architecture, tokenizer, data split, optimizer, batch size, training steps, or mask rate between baseline and HHM runs.
