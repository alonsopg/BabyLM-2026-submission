# EWoK Plausibility Minimal Experiment Summary

Run date: 2026-06-14

## Goal

Test whether a small, explicit conceptual-plausibility auxiliary task can improve EWoK
without using EWoK evaluation data. The new task was added at 15% sampling weight and
trained with the same BabyLM strict-small data pipeline, tokenizer, model size,
calibration behavior, official fast eval path, CUDA setup, and early-stopping logic used
by the prior multi-task experiments.

## Training Setup

- Experiment: `multitask_ewok_plausibility_minimal`
- Model: BERT-style masked LM, hidden size 256, 4 layers, 4 attention heads
- Tokenizer: `bert-base-uncased`
- Dataset source: `BabyLM-community/BabyLM-2026-Strict-Small`
- Max steps: 10,000
- GPU: NVIDIA RTX A6000
- Early stopping: enabled, patience 2, `start_after_step: 9000`
- Best validation checkpoint: step 9500, validation MLM loss 6.8477
- Final checkpoint: step 10000, validation MLM loss 6.8702

The 300-step smoke run completed before full training. It sampled
`conceptual_plausibility_choice` 37 times out of 300 steps and exported a loadable
`BertForMaskedLM` checkpoint.

## New Auxiliary Task

`conceptual_plausibility_choice` is a two-choice task:

`[CLS] Target: ... [SEP] Context A: ... [SEP] Context B: ... [SEP]`

The target label is `0` when Context A is more plausible and `1` when Context B is more
plausible. Choice order is randomized.

The task has 50,000 generated examples:

| Domain | Examples |
| --- | ---: |
| selectional_preference | 12,000 |
| affordance | 10,000 |
| spatial_containment | 6,500 |
| part_whole | 6,000 |
| animate_agency | 15,500 |

No EWoK data was used to generate the task. The manifest records `ewok_data_used: false`.

## Official Fast Eval Results

| System | BLiMP | BLiMP Supplement | EWoK | Entity Tracking | Reading Eye | Reading SPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal_bert_mlm_final | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| multitask_bert_calibrated_best | 54.80 | 54.80 | 50.27 | 34.52 | 7.46 | 3.19 |
| multitask_repair_syntax_v1_final | 55.51 | 52.40 | 49.55 | 32.76 | 8.28 | 3.63 |
| multitask_repair_blmpair_v1_best | 54.80 | 51.60 | 51.00 | 27.54 | 8.02 | 3.05 |
| multitask_repair_blmpair_v1_final | 54.81 | 52.40 | 50.73 | 25.99 | 7.96 | 3.11 |
| multitask_ewok_plausibility_minimal_best | 55.46 | 56.40 | 49.64 | 37.10 | 7.87 | 3.22 |
| multitask_ewok_plausibility_minimal_final | 55.70 | 54.00 | 49.55 | 35.81 | 7.79 | 3.25 |

## Interpretation

This variant trained correctly, used the GPU, respected the task weights, and produced
valid checkpoints. However, it did not improve EWoK. The best EWoK score was 49.64,
which is -0.91 below the MLM-only baseline and -1.36 below the previous BLiMP-pair repair
best. The final checkpoint was essentially identical on EWoK at 49.55.

The new task did help or preserve some non-target metrics: entity tracking was the best
among the compared runs, and BLiMP was higher than the previous multi-task variants. But
the target hypothesis was not supported: explicit high-precision plausibility templates
did not transfer to EWoK in this formulation.

Likely failure modes:

- The pairwise auxiliary head can learn the template task without forcing the MLM scoring
  geometry used by EWoK to change in the right way.
- The generated contrasts are too easy and too lexically regular, so they improve the
  auxiliary classifier rather than broad conceptual representations.
- EWoK requires more diverse context/target contrast coverage than these five template
  families provide.

## Recommendation

Do not use this run as the main BabyLM submission result. It is useful as a documented
negative result. If continuing this direction, the next attempt should either make the
plausibility supervision MLM-compatible directly or generate harder, more EWoK-shaped
minimal pairs while still avoiding evaluation leakage.
