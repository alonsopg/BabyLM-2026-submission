# Inspection Notes

## Previous GPU Command Used

```bash
source experiments/multitask_distributional_bert/scripts/env.sh
python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_plausibility_minimal/configs/multitask_ewok_plausibility_minimal.yaml \
  --seed 1
```

Smoke runs use the same command with `--max-steps` and `--name-suffix`.

## Previous EWoK Plausibility Config

`experiments/multitask_ewok_plausibility_minimal/configs/multitask_ewok_plausibility_minimal.yaml`

That run used `conceptual_plausibility_choice` at 15% through the classifier-style
collocation head.

## Existing Conceptual Plausibility Data Path

`experiments/multitask_ewok_plausibility_minimal/data/generated/conceptual_plausibility_choice.jsonl`

The new run reuses this file and converts each row into a good/bad MLM-scored pair. No
EWoK or BLiMP evaluation data is used.

## Existing Pairwise Head Implementation

`collocation`, `grammar_minpair`, and the previous `conceptual_plausibility_choice` use
`collocation_head` over the pooled `[CLS]` representation with cross-entropy.

## Existing MLM Head / MLM Loss Implementation

`mlm`, `function_word_recovery`, and `agreement_prediction` call `model.mlm(...)` with
masked-token labels. The new `mlm_pair_ranking` task also calls `model.mlm`, but builds
selected-token masked variants for the good and bad contexts and applies:

`softplus(-(score_good - score_bad))`

where each score is mean selected-token log probability.

## Existing Calibration Command

The final MLM-only calibration phase is controlled by:

`training.calibration_mlm_steps: 2000`

## Existing Evaluation Command

```bash
experiments/multitask_distributional_bert/scripts/run_official_fast_eval.sh \
  <run_name> <checkpoint_dir> main
```

## Minimal Files Changed

- `experiments/multitask_distributional_bert/scripts/train_multitask_bert.py`
- `experiments/multitask_mlm_pair_ranking_minimal/configs/multitask_mlm_pair_ranking_minimal.yaml`
- `experiments/multitask_mlm_pair_ranking_minimal/scripts/build_mlm_pair_ranking_data.py`
