# MLM-Head Pairwise Ranking Experiment Summary

Run date: 2026-06-14

## Goal

Test whether the previous EWoK-oriented conceptual-plausibility signal transfers better
when trained through the MLM head instead of a separate classifier-style pairwise head.

The experiment keeps the previous setup fixed: same model size, tokenizer, BabyLM
strict-small corpus, task mixture, final MLM-only calibration phase, and official fast
evaluation command. The only substantive change is replacing the 15% classifier-head
`conceptual_plausibility_choice` task with a 15% `mlm_pair_ranking` task.

## Method

The new task reuses:

`experiments/multitask_ewok_plausibility_minimal/data/generated/conceptual_plausibility_choice.jsonl`

Each row is converted into:

- `good_sequence`: `Target: ... Context: <plausible context>`
- `bad_sequence`: `Target: ... Context: <implausible context>`

The trainer scores selected non-special context tokens only, capped at six positions per
context. For each selected token, the model masks that token and gathers the MLM
log-probability of the original token. Scores are mean selected-token log probabilities.

The loss is:

`softplus(-(score_good - score_bad))`

No official EWoK or BLiMP evaluation data is used.

## Training

- GPU: NVIDIA RTX A6000
- Smoke run: 100 steps, completed successfully
- Full run: stopped early at step 9500
- Best validation checkpoint: step 8000
- Best validation MLM loss: 6.9339
- Final stopped checkpoint validation MLM loss: 6.9400
- `mlm_pair_ranking` batches before calibration: 1220
- Final MLM-only calibration started after step 8000
- Early stopping triggered after two non-improving eligible evals at steps 9000 and 9500

## Official Fast Eval Results

| System | BLiMP | BLiMP Supplement | EWoK | Entity Tracking | Reading Eye | Reading SPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal_bert_mlm_final | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| multitask_repair_blmpair_v1_best | 54.80 | 51.60 | 51.00 | 27.54 | 8.02 | 3.05 |
| multitask_ewok_plausibility_minimal_best | 55.46 | 56.40 | 49.64 | 37.10 | 7.87 | 3.22 |
| multitask_ewok_plausibility_minimal_final | 55.70 | 54.00 | 49.55 | 35.81 | 7.79 | 3.25 |
| multitask_mlm_pair_ranking_minimal_best | 55.95 | 47.20 | 49.55 | 33.50 | 8.73 | 3.67 |
| multitask_mlm_pair_ranking_minimal_final | 56.45 | 50.40 | 49.18 | 32.93 | 8.45 | 3.56 |

## Deltas

Best MLM-head ranking checkpoint:

- EWoK vs MLM baseline: 49.55 - 50.55 = -1.00
- EWoK vs previous BLiMP-pair best: 49.55 - 51.00 = -1.45
- EWoK vs classifier-head plausibility best: 49.55 - 49.64 = -0.09
- EWoK vs classifier-head plausibility final: 49.55 - 49.55 = +0.00

Final stopped MLM-head ranking checkpoint:

- EWoK vs MLM baseline: 49.18 - 50.55 = -1.37
- EWoK vs previous BLiMP-pair best: 49.18 - 51.00 = -1.82
- EWoK vs classifier-head plausibility best: 49.18 - 49.64 = -0.46
- EWoK vs classifier-head plausibility final: 49.18 - 49.55 = -0.37

## Interpretation

This is a negative result for the head-mismatch hypothesis. The new objective trained
correctly through the MLM head, used the GPU, respected the task mixture, and produced
valid checkpoints. However, EWoK did not improve. The best EWoK score was 49.55, below
the MLM baseline and below the previous BLiMP-pair repair best.

The result does show a different trade-off: reading scores improved substantially. The
best checkpoint reached Eye 8.73 and SPR 3.67, higher than the classifier-head
plausibility run and the previous BLiMP-pair best. Entity tracking remained well above
the MLM baseline but below the classifier-head plausibility run.

Conclusion: the previous EWoK failure is not explained only by classifier-head mismatch.
The likely remaining causes are synthetic-template mismatch, insufficient conceptual
coverage, small-model capacity, or limited transfer from this style of plausibility
supervision under the BabyLM constraints.
