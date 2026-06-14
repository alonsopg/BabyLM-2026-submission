# Inspection Notes

## Branch

`ewok-plausibility-minimal`

## Previous BLiMP-Pair Repair Config

The local previous BLiMP-pair repair run used the same BERT-style model, tokenizer,
optimizer, batch size, 10k total steps, and final 2k-step MLM-only calibration. Its task
mix had `grammar_minpair` enabled and produced the best EWoK score so far, 51.00.

For this isolated branch, only the lightweight `grammar_minpair.jsonl` data file was
recovered into the new experiment folder. The older repair experiment folder is not
added to git.

## Previous EWoK Plausibility Config

`experiments/multitask_ewok_plausibility_minimal/configs/multitask_ewok_plausibility_minimal.yaml`

That run used the failed 15% classifier-head conceptual-plausibility task. This new run
sets conceptual plausibility to 0%.

## Existing Multi-Task Sampler

`experiments/multitask_distributional_bert/scripts/train_multitask_bert.py` samples
active tasks according to config probabilities until the final MLM-only calibration
phase.

## Existing MLM Head And Loss Path

The MLM task uses `model.mlm(...)` directly. The new `semantic_cloze_ranking` task also
uses `model.mlm(...)` directly, gathering log probabilities at the single `[MASK]`
position and optimizing `softplus(-(score_good - score_bad))`.

## Existing GPU Command

```bash
source experiments/multitask_distributional_bert/scripts/env.sh
python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml \
  --seed 1
```

## Existing Calibration Command

Final MLM-only calibration is implemented by:

`training.calibration_mlm_steps: 2000`

## Existing Evaluation Command

```bash
experiments/multitask_distributional_bert/scripts/run_official_fast_eval.sh \
  <run_name> <checkpoint_dir> main
```

## Minimal Files Changed

- `experiments/multitask_distributional_bert/scripts/train_multitask_bert.py`
- `experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml`
- `experiments/multitask_ewok_semantic_cloze_v1/scripts/generate_semantic_cloze.py`
