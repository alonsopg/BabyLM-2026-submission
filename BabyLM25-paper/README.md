# BabyLM25 Paper Draft

This folder contains a self-contained ACL-style LaTeX draft for the current BabyLM multi-task experiments.

## Current Framing

The paper is framed as a diagnostic negative/partial-result paper:

- Multi-task auxiliary supervision improves Entity Tracking and reading-prediction scores.
- The same multi-task regime substantially hurts BLiMP compared with MLM-only pretraining.
- Two targeted repairs were tested:
  - MLM-head syntax repair with function-word recovery and agreement prediction.
  - BLiMP-style grammatical minimal-pair preference training.
- Neither repair meaningfully recovers BLiMP.

## Compile

```bash
pdflatex babylm25-multitask-repair.tex
bibtex babylm25-multitask-repair
pdflatex babylm25-multitask-repair.tex
pdflatex babylm25-multitask-repair.tex
```

or, if available:

```bash
latexmk -pdf -interaction=nonstopmode babylm25-multitask-repair.tex
```

## Key Files

- `babylm25-multitask-repair.tex`: main paper draft.
- `babylm25-references.bib`: bibliography.
- `tables/main_results.tex`: official fast-evaluation table.
- `notes/current_situation.md`: concise experiment state and interpretation.

## Source Results

The results were copied from:

- `experiments/multitask_distributional_bert/metrics/official_fast_eval_table.md`
- `experiments/multitask_repair_syntax_v1/metrics/comparison_summary.md`
- `experiments/multitask_repair_blmpair_v1/metrics/comparison_summary.md`
