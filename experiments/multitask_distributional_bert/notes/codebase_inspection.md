# Codebase Inspection

## Environment

No project-local virtualenv was found in the cleaned BabyLM repo. This experiment now uses a dedicated conda environment cloned from the CUDA-working `ir-model-compression` environment:

```text
babylm-26
```

Environment check:

```text
/home/paperspace/miniconda3/envs/babylm-26/bin/python
torch 2.5.1+cu121
cuda True
GPU NVIDIA RTX A6000
```

The currently active `iwslt-2026` environment is not suitable for this experiment because its PyTorch build is `2.11.0+cu130`, which fails CUDA initialization with the installed driver.

Before training, run:

```bash
experiments/multitask_distributional_bert/scripts/preflight_gpu.sh
```

This checks both `nvidia-smi` and `torch.cuda.is_available()`.

## Training Entry Points

Reusable baseline entry point:

```bash
python -m src.training.train --config <config.yaml> --seed <seed>
```

Source:

```text
src/training/train.py
```

The old suite launcher is:

```text
scripts/run_experiments.py
```

It launches configs such as `baseline_random_mlm.yaml`, `hhm_entity_error.yaml`, and ablations. For this new experiment, use only the normal random MLM path as the baseline.

## Evaluation Entry Points

Current local evaluator:

```bash
python -m src.evaluation.evaluate --checkpoint <checkpoint> --babylm-eval resources/babylm-eval
```

Source:

```text
src/evaluation/evaluate.py
```

This currently writes an evaluation handoff note. The official BabyLM evaluation pipeline lives under:

```text
resources/babylm-eval/
```

## Dataset Loading

Source:

```text
src/data/load_dataset.py
```

The previous experiments use:

```text
BabyLM-community/BabyLM-2026-Strict-Small
```

with text column:

```text
text
```

The loader filters empty examples, creates a train/validation split, tokenizes with the configured tokenizer, and groups token streams into fixed-length chunks.

## Existing Collators

Random MLM baseline:

```text
src/masking/random_collator.py
```

Behavior:

- 15 percent mask rate from config
- excludes special tokens
- 80 percent `[MASK]`
- 10 percent random token
- 10 percent unchanged
- returns `mask_metadata` for diagnostics

Other old masking methods exist but should not be used as the main method:

```text
src/masking/heavy_hitter_collator.py
src/masking/cms_morph_collator.py
src/masking/accuracy_morph_collator.py
src/masking/entity_accuracy_morph_collator.py
```

## Existing Model Builder

Source:

```text
src/models/build_model.py
```

It builds a compact `BertForMaskedLM` with:

- `bert-base-uncased` tokenizer
- hidden size 256
- 4 layers
- 4 attention heads
- intermediate size 1024
- max positions 512

## Existing Checkpoint Format

During training:

```text
checkpoint-latest/
  model/
  training_state.pt
  collator_state.json
```

Final checkpoint:

```text
checkpoint-final/
  config.json
  model.safetensors or pytorch_model.bin
  tokenizer files
```

The multi-task model must additionally export:

```text
full_multitask_checkpoint/
mlm_compatible_checkpoint/
```

where `mlm_compatible_checkpoint/` loads as a standard `BertForMaskedLM`.

## Previous Experiment Commands

Normal MLM baseline equivalent:

```bash
python -m src.training.train --config configs/baseline_random_mlm.yaml --seed 1
```

New rescued baseline command:

```bash
experiments/multitask_distributional_bert/scripts/train_normal_bert.sh --seed 1
```

Smoke version:

```bash
experiments/multitask_distributional_bert/scripts/train_normal_bert.sh \
  --seed 1 \
  --max-steps 100 \
  --max-train-rows 2000 \
  --name-suffix smoke
```

## Reusable Components

- dataset loading and grouping
- tokenizer/model construction for normal MLM
- random MLM collator
- optimizer/scheduler/logging/checkpoint loop
- validation MLM-loss evaluation
- checkpoint save/resume logic

## Required New Components

- offline auxiliary task generator
- task quality inspection logs and 50-example samples
- JSONL-backed multi-task dataset
- task-aware collator/batching
- shared BERT encoder with auxiliary heads
- mixed-task training loop with controlled task probabilities
- MLM-compatible export from the multi-task checkpoint
- final comparison summary over normal MLM and multi-task runs
