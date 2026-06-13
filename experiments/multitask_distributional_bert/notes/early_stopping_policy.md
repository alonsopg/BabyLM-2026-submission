# Early Stopping Policy

Early stopping is enabled for future primary runs.

## Normal MLM Baseline

```yaml
monitor: val_loss
start_after_step: 4000
patience_evals: 3
min_delta: 0.01
```

Rationale: the completed baseline kept improving until around step 8000, with noisy validation afterward. Patience 3 avoids stopping too early while still preventing a long tail after the best checkpoint.

## Multi-Task BERT

```yaml
monitor: val_mlm_loss
start_after_step: 4000
patience_evals: 2
min_delta: 0.01
```

Rationale: the validation-enabled multi-task seed-1 run reached its best MLM validation loss at step 5000. Under the earlier patience-4 setting it stopped at step 7000. Patience 2 would stop at step 6000 while retaining the step-5000 `checkpoint-best`.

## Best Checkpoints

Both trainers save:

```text
checkpoint-best/
checkpoint-latest/
checkpoint-final/
```

For BabyLM evaluation, prefer `checkpoint-best` when early stopping is enabled. For the multi-task model, use:

```text
checkpoint-best/mlm_compatible_checkpoint/
```

