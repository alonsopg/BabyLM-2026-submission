# Multi-Task Distributional BERT Experiment

Fresh experiment workspace for comparing:

1. `normal_bert_mlm`: standard BabyLM Strict-Small BERT masked language modeling.
2. `multitask_bert`: the same BERT encoder trained with MLM plus offline-generated distributional auxiliary tasks.

This folder intentionally reuses only the stable training/evaluation infrastructure from the previous BabyLM codebase. The old HHM, CMS-Morph, and Accuracy-Morph methods are references only; they are not the main method for this experiment.

## Current Status

Prepared and run:

- controlled normal MLM baseline config
- first-pass multi-task config stub
- GPU preflight and batch-size probe
- normal baseline launch wrapper using the existing trainer
- evaluation wrapper using the existing evaluation entry point
- codebase inspection notes
- offline auxiliary task generator
- multi-task JSONL dataset loader
- shared BERT encoder with MLM and auxiliary heads
- multi-task trainer
- MLM-compatible checkpoint export
- generated auxiliary task datasets and sample inspections
- validation-enabled multi-task run with `checkpoint-best`
- calibrated follow-up run with fixed RTD padding/special-token labels

## Reused Code

The baseline run uses the existing training stack:

```bash
python -m src.training.train \
  --config experiments/multitask_distributional_bert/configs/normal_bert_mlm.yaml \
  --seed 1
```

That stack already provides:

- BabyLM Strict-Small loading from `BabyLM-community/BabyLM-2026-Strict-Small`
- `bert-base-uncased` tokenizer
- compact BERT MLM model
- random 15 percent MLM masking with 80/10/10 replacement
- AdamW, linear warmup/decay, logging, validation loss, and checkpoint saving

## Environment

This experiment uses a dedicated conda environment cloned from the CUDA-working `ir-model-compression` environment:

```text
babylm-26
```

It has PyTorch `2.5.1+cu121`, which is compatible with the installed NVIDIA driver. The currently active `iwslt-2026` environment has PyTorch `2.11.0+cu130` and fails CUDA initialization on this machine, so the experiment scripts source `scripts/env.sh` and activate `babylm-26` automatically.

## Preflight

Run this before training:

```bash
experiments/multitask_distributional_bert/scripts/preflight_gpu.sh
```

If CUDA is unavailable, stop. Do not launch long training on CPU.

## Normal Baseline Smoke Run

```bash
experiments/multitask_distributional_bert/scripts/train_normal_bert.sh \
  --seed 1 \
  --max-steps 100 \
  --max-train-rows 2000 \
  --name-suffix smoke
```

## Full Normal Baseline

```bash
experiments/multitask_distributional_bert/scripts/train_normal_bert.sh --seed 1
```

Outputs go to:

```text
experiments/multitask_distributional_bert/checkpoints/normal_bert/normal_bert_mlm/seed_1/
```

## Evaluation Wrapper

The existing evaluator currently records an evaluation handoff note for the official BabyLM evaluation repo:

```bash
experiments/multitask_distributional_bert/scripts/run_eval.sh \
  experiments/multitask_distributional_bert/checkpoints/normal_bert/normal_bert_mlm/seed_1/checkpoint-final
```

For the current best multi-task follow-up checkpoint:

```bash
experiments/multitask_distributional_bert/scripts/run_eval.sh \
  experiments/multitask_distributional_bert/checkpoints/multitask_bert/multitask_bert_calibrated/seed_1/checkpoint-best/mlm_compatible_checkpoint
```

## Current Validation Results

Local MLM validation loss currently favors the normal MLM baseline by a large margin:

```text
normal_bert_mlm best val_loss: 3.0248489379882812 at step 8000
multitask_bert best val_mlm_loss: 6.838296095530192 at step 5000
multitask_bert_calibrated best val_mlm_loss: 6.82595952351888 at step 5000
```

The calibrated follow-up fixes RTD padding/special-token labels and slightly improves the multi-task best checkpoint, but the final MLM-only calibration does not close the gap to the normal baseline.

## Fairness Contract

The primary comparison controls total optimization budget:

- same corpus
- same tokenizer
- same model size
- same sequence length
- same optimizer and schedule
- same total training steps
- same seeds
- same evaluation pipeline
- same checkpoint-saving format

The multi-task model should not receive extra steps in the primary comparison.
