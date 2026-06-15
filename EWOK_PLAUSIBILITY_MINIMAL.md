# EWoK Plausibility Minimal Branch

This branch isolates the `multitask_ewok_plausibility_minimal` experiment from the
larger multi-task BabyLM branch. It starts from `main` and adds only the pieces needed
to reproduce and interpret this specific experiment.

## Included

- `experiments/multitask_ewok_plausibility_minimal/`
  - configuration
  - synthetic data generator
  - generated task JSONL files
  - sample inspection notes
  - official fast-eval results
  - comparison summaries
- `experiments/multitask_mlm_pair_ranking_minimal/`
  - MLM-head pairwise ranking follow-up
  - converter from the previous conceptual-plausibility examples
  - selected-token PLL ranking data
  - official fast-eval results
  - comparison summaries
- `experiments/multitask_ewok_semantic_cloze_v1/`
  - semantic cloze ranking follow-up
  - synthetic single-mask semantic completion examples
  - recovered grammar-minpair auxiliary data needed for the mixture
  - official fast-eval results
  - comparison summaries
- `experiments/run_records/multitask_ewok_plausibility_minimal*/`
  - small run logs, configs, and task counts for the smoke and full runs
- `experiments/run_records/multitask_mlm_pair_ranking_minimal*/`
  - small run logs, configs, and task counts for the smoke and full runs
- `experiments/run_records/multitask_ewok_semantic_cloze_v1*/`
  - small run logs, configs, and task counts for the smoke and full runs
- Minimal shared multi-task utilities under
  `experiments/multitask_distributional_bert/scripts/`
  - `generate_multitask_examples.py`
  - `train_multitask_bert.py`
  - `run_official_fast_eval.sh`
  - `collect_official_eval_table.py`
  - `env.sh`
  - `preflight_gpu.sh`

## Excluded

This branch intentionally excludes the earlier unrelated multi-task experiment folders,
repair variants, baseline run records, and paper-draft material from the broader
`multi-task-babylm` branch.

Large checkpoint artifacts remain local and are ignored by git.

## Result

The experiment trained and evaluated correctly on GPU, but it is a negative result for
the target hypothesis: the synthetic conceptual-plausibility auxiliary task did not
improve EWoK. The best EWoK score was 49.64, below the MLM-only baseline of 50.55 and
below the previous BLiMP-pair repair best of 51.00.

The MLM-head pairwise ranking follow-up also trained and evaluated correctly. It did not
improve EWoK either: best EWoK was 49.55 and final EWoK was 49.18. It did, however,
improve reading scores, reaching Eye 8.73 and SPR 3.67 at the best checkpoint.

The semantic cloze ranking follow-up is the first positive EWoK result in this isolated
series. The best validation checkpoint reached EWoK 50.73, and the final early-stopped
checkpoint reached EWoK 52.00. That is +1.45 over the MLM-only baseline and +1.00 over
the previous branch-best EWoK score. The trade-off is entity tracking: the semantic
cloze best checkpoint reached 36.61, but the final checkpoint dropped to 27.44 while
improving EWoK and SPR.

The semantic cloze weight sweep tested 2.5%, 7.5%, and 10.0% semantic cloze while
leaving all other settings aligned with the 5% run. None of the sweep variants beat the
5% final EWoK score. The best sweep-only EWoK was 51.64 from `semantic_cloze_w100_final`;
the best Entity result was 40.48 from `semantic_cloze_w075_best`; and the best Eye score
was 8.82 from `semantic_cloze_w025_final`. The current recommendation remains the 5%
semantic cloze final checkpoint for EWoK-focused reporting.
