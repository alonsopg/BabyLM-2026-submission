# Inspection Notes

## Branch

`multitask-semantic-cloze-clean`

## Selected Experiment

`experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml`

The selected run keeps a compact MLM-compatible BERT backbone and adds one semantic cloze ranking auxiliary task. The model submitted to BabyLM remains a plain `BertForMaskedLM` checkpoint.

## Multi-Task Sampler

`experiments/multitask_distributional_bert/scripts/train_multitask_bert.py` samples active tasks according to config probabilities until the final MLM-only calibration phase.

## MLM Head And Loss Path

The MLM task uses `model.mlm(...)` directly. The `semantic_cloze_ranking` task also uses `model.mlm(...)` directly, gathering log probabilities at the single `[MASK]` position and optimizing `softplus(-(score_good - score_bad))`.

## GPU Command

```bash
source experiments/multitask_distributional_bert/scripts/env.sh
python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml \
  --seed 1
```

## Calibration

Final MLM-only calibration is implemented by:

`training.calibration_mlm_steps: 2000`

## Fast Evaluation Command

```bash
experiments/multitask_distributional_bert/scripts/run_official_fast_eval.sh \
  <run_name> <checkpoint_dir> main
```
