# BabyLM Hard-to-Predict Masking Paper

Paper title: **Can Masking Hard-to-Predict Words Improve Small Language Model Pretraining?**

This folder contains the local LaTeX draft for the BabyLM masking paper. The implementation lives in `/home/paperspace/babylm-hhm`.

## Current Status

The current paper is framed as a controlled diagnostic study of hard-to-predict word masking, not as a strong positive method paper.

- Main HHM vs. random comparison: complete for 3 seeds with local validation loss and available official BabyLM strict metrics.
- HHM validation loss is worse than random, but official metrics are slightly higher on average.
- CMS-Morph and Entity-Accuracy-Morph are single-seed follow-ups that do not improve validation loss.
- Accuracy-Morph is the most promising current follow-up: in matched seeds 1 and 2 it improves validation loss and gives the clearest Entity Tracking gain, but still needs broader multi-seed confirmation.

The most important table for the current recommendation is `tables/current_status.tex`.

## Compile

```bash
latexmk -pdf -interaction=nonstopmode babylm-sketchselect-main.tex
```

## Tables

Current table files:

- `tables/main_results.tex`: three-seed random vs. HHM results.
- `tables/improvement_summary.tex`: mean HHM deltas vs. random.
- `tables/cms_morph_pilot.tex`: seed-1 pilot results for CMS-Morph, Accuracy-Morph, and Entity-Accuracy-Morph.
- `tables/accuracy_morph_seed2.tex`: final seed-2 Accuracy-Morph comparison against matched random MLM.
- `tables/accuracy_morph_matched.tex`: matched seed-1/seed-2 Accuracy-Morph summary and paired deltas.
- `tables/accuracy_morph_aggregate.tex`: aggregate Accuracy-Morph context against all available random seeds.
- `tables/current_status.tex`: current interpretation and recommendation for each branch.
- `tables/mask_diagnostics.tex`: HHM mask-source diagnostics.
- `tables/pilot_mask_diagnostics.tex`: final-step mask-budget diagnostics for the seed-1 pilots.
- `tables/significance_tests.tex`: paired tests for the three-seed random-vs-HHM comparison.
- `tables/prelim_ablation_results.tex`: 20-step component ablation checks.

## Result Imports

From `../babylm-hhm`, run:

```bash
conda run -n ir-model-compression python -m src.analysis.aggregate_results
conda run -n ir-model-compression python -m src.analysis.export_latex_tables
conda run -n ir-model-compression python -m src.analysis.plot_training_curves
```

The LaTeX tables in this draft have already been manually updated with the current completed runs and official evaluation results available on this machine.
