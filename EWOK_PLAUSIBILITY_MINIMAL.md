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
- `experiments/run_records/multitask_ewok_plausibility_minimal*/`
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
