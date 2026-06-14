# Minimal inspection notes

## Previous GPU command used

Previous multi-task runs used:

```bash
source experiments/multitask_distributional_bert/scripts/env.sh
python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config <config.yaml> \
  --seed 1
```

Smoke tests used the same command with `--max-steps` and `--name-suffix`.

## Previous best multi-task config

The prior calibrated multi-task setup is:

```text
experiments/multitask_distributional_bert/configs/multitask_bert_calibrated.yaml
```

The strongest repaired compromise was:

```text
experiments/multitask_repair_syntax_v1/configs/multitask_repair_syntax_v1.yaml
```

The previous best EWoK checkpoint was:

```text
multitask_repair_blmpair_v1_best: EWoK 51.00
```

## Previous calibration command

Calibration is implemented in `train_multitask_bert.py` through:

```text
training.calibration_mlm_steps: 2000
```

The final 2000 of 10000 steps are forced to MLM-only.

## Existing pairwise head / pairwise task code

`collocation` and `grammar_minpair` use the existing two-way `collocation_head` over the pooled `[CLS]` representation. The new `conceptual_plausibility_choice` task reuses this head and cross-entropy loss.

## Minimal files changed for this run

- `experiments/multitask_distributional_bert/scripts/train_multitask_bert.py`
- `experiments/multitask_ewok_plausibility_minimal/configs/multitask_ewok_plausibility_minimal.yaml`
- `experiments/multitask_ewok_plausibility_minimal/scripts/generate_plausibility_examples.py`

No EWoK evaluation data is read or copied by the generator.
