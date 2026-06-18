# Inspection notes

## Branch

- Branch: `ewok-plausibility-minimal`
- Starting commit: `71f3318 Add semantic cloze weight sweep for EWoK`

## Base semantic cloze implementation

The existing trainer implements `semantic_cloze_ranking` as an MLM-head pairwise
ranking task. Each row has one `[MASK]`, a good single-token completion, and a bad
single-token completion.

## Base semantic cloze config

- Base config: `experiments/multitask_ewok_semantic_cloze_v1/configs/multitask_ewok_semantic_cloze_v1.yaml`
- Current best EWoK checkpoint: `multitask_ewok_semantic_cloze_v1_final`, EWoK 52.00

## Base semantic cloze data

- Original semantic data: `experiments/multitask_ewok_semantic_cloze_v1/data/semantic_cloze_ranking.jsonl`
- Grammar-minpair data: `experiments/multitask_ewok_semantic_cloze_v1/data/grammar_minpair.jsonl`

## Existing MLM-head ranking loss

The relational task reuses the same path:

```text
score_good = log P_MLM(good | prompt)
score_bad = log P_MLM(bad | prompt)
loss = softplus(-(score_good - score_bad))
```

No classifier head, CLS pooling, architecture change, tokenizer change, corpus change,
or evaluator change was added.

## Existing GPU command

```bash
source experiments/multitask_distributional_bert/scripts/env.sh
python experiments/multitask_distributional_bert/scripts/train_multitask_bert.py \
  --config experiments/multitask_ewok_relational_cloze_v1/configs/multitask_ewok_relational_cloze_v1.yaml \
  --seed 1
```

CUDA check through the BabyLM environment:

```text
torch 2.5.1+cu121
True NVIDIA RTX A6000
```

## Existing calibration command

Calibration is handled inside the trainer with `training.calibration_mlm_steps: 2000`.

## Existing evaluation command

```bash
experiments/multitask_distributional_bert/scripts/run_official_fast_eval.sh \
  <run_name> <checkpoint_dir> main
```

## Minimal files changed

- Added `relational_semantic_cloze` to the existing semantic cloze task path in the trainer.
- Added a high-precision relational cloze generator and generated data files.
- Added one aligned config with 2.5% original semantic cloze and 2.5% relational cloze.
