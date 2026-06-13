# Calibrated Follow-Up Result

## Change

After the primary multi-task run, the trainer was corrected so RTD ignores padding and special tokens instead of treating them as easy negative labels.

A follow-up configuration was added:

```text
configs/multitask_bert_calibrated.yaml
```

This run keeps the same 10k-step total budget, but uses:

- steps 1-8000: mixed MLM + RTD + connective + definiteness
- steps 8001-10000: MLM-only calibration
- early stopping active from step 9000

## Result

Seed 1 stopped at step 9500:

```text
final_val_mlm_loss = 6.88095744450887
best_step = 5000
best_val_mlm_loss = 6.82595952351888
early_stop = true
```

This is a small improvement over the first validation-enabled multi-task run:

```text
previous_best_val_mlm_loss = 6.838296095530192
calibrated_best_val_mlm_loss = 6.82595952351888
```

However, the calibrated multi-task model remains far behind the normal MLM baseline:

```text
normal_bert_best_val_loss = 3.0248489379882812
```

## Interpretation

The RTD-label fix made the auxiliary objective more correct and slightly improved the best multi-task checkpoint, but the auxiliary-task mixture still substantially harms MLM validation loss under this setup. The final 2k MLM-only calibration did not recover the model; the best checkpoint remained at step 5000, before calibration.

For downstream BabyLM evaluation, use:

```text
experiments/multitask_distributional_bert/checkpoints/multitask_bert/multitask_bert_calibrated/seed_1/checkpoint-best/mlm_compatible_checkpoint/
```

For the current research direction, the next best experiment is likely a much lighter auxiliary schedule, for example MLM 0.85 / RTD 0.10 / connective 0.025 / definiteness 0.025, or an auxiliary-loss warmup that starts from a partially trained MLM model.
