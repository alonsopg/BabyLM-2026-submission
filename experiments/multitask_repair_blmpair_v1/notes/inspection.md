# Inspection Notes

## Previous MLM-only Baseline

- Config: `experiments/multitask_distributional_bert/configs/normal_bert_mlm.yaml`
- Model/tokenizer: same compact BERT setup used throughout this BabyLM run, with `bert-base-uncased` tokenizer, hidden size 256, 4 layers, 4 heads, intermediate size 1024.
- Official strict fast eval: BLiMP 64.33, Supplement 54.00, EWoK 50.55, Entity 15.56, Eye 2.47, SPR 2.42.

## Previous Calibrated Multi-Task Run

- Config: `experiments/multitask_distributional_bert/configs/multitask_bert_calibrated.yaml`
- Mixture: MLM 0.60, RTD 0.20, connective 0.10, definiteness 0.10.
- Training: 10000 steps with a final 2000-step MLM-only calibration phase.
- Early stopping: enabled from step 9000 with patience 2.
- Official strict fast eval: BLiMP 54.80, Supplement 54.80, EWoK 50.27, Entity 34.52, Eye 7.46, SPR 3.19.

## Previous Repair Syntax v1 Run

- Folder: `experiments/multitask_repair_syntax_v1`.
- Mixture: MLM 0.60, RTD 0.10, connective 0.075, definiteness 0.075, collocation 0.05, function-word recovery 0.05, agreement prediction 0.05.
- Removed substitution by setting probability to 0.
- Training completed 10000 steps. Best validation MLM checkpoint was step 9000.
- Official strict fast eval:
  - validation-best: BLiMP 55.28, Entity 34.92, Eye 8.13, SPR 3.50.
  - final: BLiMP 55.51, Entity 32.76, Eye 8.28, SPR 3.63.
- Interpretation: operationally successful, but not a meaningful BLiMP repair.

## Existing Task Mixture

The useful auxiliary tasks to keep are:

- RTD
- connective prediction
- definiteness prediction
- collocation naturalness
- final MLM calibration

The tasks to disable for this run are:

- substitution compatibility: noisy for acceptability contrasts
- function-word recovery: weak BLiMP gain
- agreement prediction: weak BLiMP gain

## Existing Calibration Procedure

The trainer uses `calibration_mlm_steps` to force the last portion of training to MLM-only. This experiment keeps:

- `max_steps: 10000`
- `calibration_mlm_steps: 2000`
- same learning rate, scheduler, batch size, model, tokenizer, and validation settings as the previous calibrated/repair runs.

## Existing Task Generators

Base generator:

```text
experiments/multitask_distributional_bert/scripts/generate_multitask_examples.py
```

It already creates MLM, RTD, connective, definiteness, collocation, and substitution examples. The new generator reuses those functions and adds `grammar_minpair`.

## Existing Evaluation Command

Official strict fast eval wrapper:

```text
experiments/multitask_distributional_bert/scripts/run_official_fast_eval.sh
```

Collector:

```text
experiments/multitask_distributional_bert/scripts/collect_official_eval_table.py
```

## Minimal Changes Needed For BLiMP-Style Pair Objective

1. Add `grammar_minpair` to the trainer's task list.
2. Reuse the existing `collocation_head` and 2-way cross entropy for `grammar_minpair`.
3. Generate high-precision grammatical minimal pairs from corpus patterns and templates.
4. Configure the task mixture as MLM 0.60, RTD 0.10, connective 0.075, definiteness 0.075, collocation 0.05, grammar_minpair 0.10.
5. Keep substitution, function-word recovery, and agreement prediction at 0 probability.
6. Inspect at least 50 `grammar_minpair` examples before training.
