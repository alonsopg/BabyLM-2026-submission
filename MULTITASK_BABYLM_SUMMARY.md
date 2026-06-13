# Multi-Task BabyLM Experiment Summary

This branch packages the current BabyLM 2026 multi-task experiments, repair attempts, run records, generated task data, evaluation metrics, and ACL-style paper draft.

## Current Scientific Situation

The experiments show a stable trade-off:

- Standard MLM is strongest on BLiMP.
- Multi-task training improves Entity Tracking and reading-prediction metrics.
- Two targeted syntax/BLiMP repairs did not recover the BLiMP loss.

The most compact conclusion is:

> Auxiliary objectives help compact BabyLM-scale BERT models on Entity Tracking and reading-prediction behavior, but they introduce a persistent BLiMP degradation that is not fixed by simple syntax-focused MLM tasks or by a template-based BLiMP-style minimal-pair preference task.

## Main Results

| System | BLiMP | Supplement | EWoK | Entity | Eye | SPR | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `normal_bert_mlm_final` | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 | MLM-only baseline |
| `multitask_bert_calibrated_best` | 54.80 | 54.80 | 50.27 | 34.52 | 7.46 | 3.19 | previous calibrated multi-task |
| `multitask_bert_light_aux_best` | 54.59 | 52.00 | 50.09 | 29.60 | 8.07 | 3.34 | reduced auxiliary variant |
| `multitask_repair_syntax_v1_best` | 55.28 | 52.80 | 50.09 | 34.92 | 8.13 | 3.50 | best Entity among repaired systems |
| `multitask_repair_syntax_v1_final` | 55.51 | 52.40 | 49.55 | 32.76 | 8.28 | 3.63 | best BLiMP/reading repaired checkpoint |
| `multitask_repair_blmpair_v1_best` | 54.80 | 51.60 | 51.00 | 27.54 | 8.02 | 3.05 | BLiMP-pair validation-best |
| `multitask_repair_blmpair_v1_final` | 54.81 | 52.40 | 50.73 | 25.99 | 7.96 | 3.11 | BLiMP-pair final |

## Model and Training Details

All main experiments use the same compact BERT-style masked language model:

- Tokenizer: `bert-base-uncased`
- Hidden size: 256
- Transformer layers: 4
- Attention heads: 4
- Intermediate size: 1024
- Maximum position embeddings: 512
- Maximum sequence length: 128
- Batch size: 96
- Optimizer: AdamW
- Learning rate: `5.0e-4`
- Weight decay: `0.01`
- Warmup fraction: `0.06`
- Max gradient norm: `1.0`
- Precision: fp32
- Full training budget: 10000 steps
- Validation split: 2000 examples
- Validation schedule: every 500 steps for the multi-task runs

The training corpus is `BabyLM-community/BabyLM-2026-Strict-Small`.

GPU note:

- PyTorch CUDA detected `NVIDIA RTX A6000`.
- `nvidia-smi` failed locally with an NVML driver/library mismatch.
- The multi-task trainer refuses to run on CPU, so the recorded full runs were not silent CPU fallbacks.

## Experiment Families

### 1. MLM-only baseline

Path:

- `experiments/multitask_distributional_bert/configs/normal_bert_mlm.yaml`

This is the strongest BLiMP model:

- BLiMP: 64.33
- Entity: 15.56
- Eye: 2.47
- SPR: 2.42

Interpretation:

- MLM-only preserves grammatical minimal-pair behavior better than the multi-task variants.
- It performs poorly on Entity Tracking and reading-prediction metrics.

### 2. Calibrated multi-task baseline

Path:

- `experiments/multitask_distributional_bert/configs/multitask_bert_calibrated.yaml`

Task mixture:

- MLM: 60%
- Replaced-token detection: 20%
- Connective prediction: 10%
- Definiteness prediction: 10%
- Final MLM-only calibration: 2000 steps

Result:

- BLiMP drops from 64.33 to 54.80.
- Entity improves from 15.56 to 34.52.
- Eye improves from 2.47 to 7.46.
- SPR improves from 2.42 to 3.19.

Interpretation:

- Multi-tasking creates useful auxiliary behavior, but at a large syntactic cost.

### 3. Syntax repair v1

Path:

- `experiments/multitask_repair_syntax_v1/`

Task mixture:

- MLM: 60%
- Replaced-token detection: 10%
- Connective prediction: 7.5%
- Definiteness prediction: 7.5%
- Collocation naturalness: 5%
- Function-word recovery: 5%
- Agreement prediction: 5%
- Substitution: 0%
- Final MLM-only calibration: 2000 steps

New tasks:

- `function_word_recovery`: masks function words and predicts them through the MLM head.
- `agreement_prediction`: masks local agreement-bearing tokens such as `is/are`, `was/were`, `has/have`, `do/does`, and demonstratives.

Result:

- BLiMP improves only slightly, peaking at 55.51.
- Entity and reading remain strong.

Interpretation:

- Syntax-flavored MLM repair is operationally useful but does not solve the BLiMP regression.

### 4. BLiMP-pair repair v1

Path:

- `experiments/multitask_repair_blmpair_v1/`

Task mixture:

- MLM: 60%
- Replaced-token detection: 10%
- Connective prediction: 7.5%
- Definiteness prediction: 7.5%
- Collocation naturalness: 5%
- Grammar minimal pairs: 10%
- Substitution: 0%
- Function-word recovery: 0%
- Agreement prediction: 0%
- Final MLM-only calibration: 2000 steps

New task:

- `grammar_minpair`
- Input format: `[CLS] good sentence [SEP] bad sentence [SEP]`
- Target: `0` if the first sentence is better, `1` if the second sentence is better.
- Head: reuses the existing two-way collocation pair-classification head over `[CLS]`.

Generator history:

- The first corpus-corruption generator was rejected before training because determiner-number examples were too noisy, especially `that` as complementizer/pronoun.
- The final generator uses high-precision templates for:
  - subject-verb agreement
  - determiner-noun agreement
  - auxiliary agreement
  - NPI licensing
  - simple reflexive binding

Generated `grammar_minpair` examples:

- Total: 30820
- Subject-verb agreement: 17500
- Determiner-noun agreement: 7040
- Auxiliary agreement: 5976
- NPI licensing: 220
- Reflexive binding: 84

Result:

- BLiMP-pair best: 54.80
- BLiMP-pair final: 54.81

Interpretation:

- Direct grammatical pair supervision did not transfer to BLiMP.
- The pairwise classifier quickly solved the template task, but the MLM head still did not recover BLiMP-style sentence preferences.

## Included In This Branch

The branch includes:

- Experiment code and scripts.
- Config files.
- Generated task JSONL files.
- Sample files used for manual inspection.
- Metrics and official fast-evaluation tables.
- Run records copied from checkpoint directories:
  - `config.yaml`
  - `train_log.csv`
  - `task_counts.json`
- Notes documenting inspection and interpretation.
- ACL-style paper draft in `BabyLM25-paper/`.
- Existing repository files are preserved.

Important run-record location:

- `experiments/run_records/`

Important paper location:

- `BabyLM25-paper/babylm25-multitask-repair.tex`
- `BabyLM25-paper/babylm25-multitask-repair.pdf`

## Not Included As Git Objects

Large local artifacts are intentionally not committed:

- model checkpoints
- `.safetensors`
- `.bin`
- `.pt`
- optimizer/training-state blobs

Reason:

- The local checkpoint tree is about 2.9 GB.
- Some training-state files are close to GitHub's hard file-size limit.
- The existing `.gitignore` already excludes these artifact types.

The branch preserves small run logs and configs in `experiments/run_records/` so the training history remains inspectable without uploading model weights.

Local evaluation symlinks under `experiments/multitask_distributional_bert/eval_models/` are also not committed; the evaluation wrapper recreates them from checkpoint paths when run locally.

## Current Recommendation

For a BabyLM paper:

- Frame this as a diagnostic negative/partial-result paper.
- The strongest claim is not that multi-tasking wins overall.
- The real result is the trade-off:
  - better Entity/Reading behavior,
  - worse BLiMP,
  - failed simple repair attempts.

Best current model by goal:

- BLiMP: `normal_bert_mlm_final`
- Entity Tracking: `multitask_repair_syntax_v1_best`
- Reading: `multitask_repair_syntax_v1_final`
- Best compromise among repaired multi-task systems: `multitask_repair_syntax_v1_final`

Recommended next experiment:

- Increase MLM exposure while keeping a smaller grammar-pair dose.
- Example mixture:
  - MLM 70%
  - RTD 7.5%
  - connective 5%
  - definiteness 5%
  - collocation 2.5%
  - grammar-minpair 10%
  - same final 2000-step MLM calibration

Hypothesis for that follow-up:

- If BLiMP loss is mainly reduced effective MLM exposure, increasing MLM share should recover more BLiMP than adding more template grammar supervision.
