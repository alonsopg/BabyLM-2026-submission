# Inspection notes

## Previous normal MLM run

Config:

```text
experiments/multitask_distributional_bert/configs/normal_bert_mlm.yaml
```

Key settings:

```text
dataset: BabyLM-community/BabyLM-2026-Strict-Small
tokenizer: bert-base-uncased
model: BERT hidden 256, 4 layers, 4 heads, intermediate 1024
batch_size: 96
max_steps: 10000
learning_rate: 5.0e-4
warmup_fraction: 0.06
precision: fp32
eval_every: 500
eval_batches: 40
```

Best local validation:

```text
step 8000
val_loss = 3.0248489379882812
```

Official strict fast eval:

```text
BLiMP 64.33
Supplement 54.00
EWoK 50.55
Entity 15.56
Eye 2.47
SPR 2.42
```

## Previous best multi-task calibrated run

Config:

```text
experiments/multitask_distributional_bert/configs/multitask_bert_calibrated.yaml
```

Task mixture:

```text
MLM 0.60
RTD 0.20
connective 0.10
definiteness 0.10
collocation 0.00
substitution 0.00
```

Training:

```text
max_steps: 10000
calibration_mlm_steps: 2000
early_stopping: enabled
early_stopping_start_after_step: 9000
patience_evals: 2
```

Best local validation:

```text
step 5000
val_mlm_loss = 6.82595952351888
```

Official strict fast eval:

```text
BLiMP 54.80
Supplement 54.80
EWoK 50.27
Entity 34.52
Eye 7.46
SPR 3.19
```

## Current task mixture

The repaired run should use:

```text
MLM 0.60
RTD 0.10
connective 0.075
definiteness 0.075
collocation 0.05
function_word_recovery 0.05
agreement_prediction 0.05
substitution 0.00
```

The two new syntax tasks use the existing MLM head. No new architecture is needed.

## Current calibration setup

The previous best calibrated run used 2000 final MLM-only steps. The repair run keeps:

```text
calibration_mlm_steps: 2000
max_steps: 10000
```

## Existing task generators

Existing generator:

```text
experiments/multitask_distributional_bert/scripts/generate_multitask_examples.py
```

It generates:

```text
mlm
rtd
connective
definiteness
collocation
substitution
```

Substitution remains generated for traceability but is disabled in the repair mixture.

## Existing training entry point

Existing trainer:

```text
experiments/multitask_distributional_bert/scripts/train_multitask_bert.py
```

It supports:

```text
task sampling
MLM-compatible checkpoint export
checkpoint-best
validation MLM loss
early stopping
final MLM-only calibration
```

Minimal change needed: add `function_word_recovery` and `agreement_prediction` as MLM-head tasks.

## Existing evaluation command

Official fast eval wrapper:

```text
experiments/multitask_distributional_bert/scripts/run_official_fast_eval.sh
```

Collector:

```text
experiments/multitask_distributional_bert/scripts/collect_official_eval_table.py
```

The local official bundle is:

```text
/home/paperspace/babylm-hhm/resources/babylm-eval/strict
```

## Minimal changes needed

1. Add repair data generator for function-word recovery and agreement prediction.
2. Keep existing task files but write them under `experiments/multitask_repair_syntax_v1/data/generated`.
3. Disable substitution with probability 0.
4. Include collocation at 0.05 as requested, while documenting prior noise concerns.
5. Extend the trainer so the two new tasks use the MLM head.
6. Smoke test before full training.
