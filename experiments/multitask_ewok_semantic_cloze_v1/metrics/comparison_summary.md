# Semantic Cloze Ranking EWoK Experiment

## Question

Can a small MLM-head semantic cloze ranking auxiliary task improve EWoK without using
official EWoK or BLiMP data, external datasets, or a classifier head?

## Setup

- Branch: `ewok-plausibility-minimal`
- Experiment: `multitask_ewok_semantic_cloze_v1`
- Model/tokenizer/corpus/eval/calibration: unchanged from the earlier isolated runs
- Training device: NVIDIA RTX A6000
- Early stopping: enabled, patience 2, started after step 9000
- Stopped at step 9500
- Best validation checkpoint: step 5000, validation MLM loss 6.810523

Task mixture:

| Task | Weight |
| --- | ---: |
| MLM | 0.600 |
| RTD | 0.100 |
| connective | 0.075 |
| definiteness | 0.075 |
| collocation | 0.050 |
| grammar_minpair | 0.050 |
| semantic_cloze_ranking | 0.050 |

Observed full-run task counts:

| Task | Steps |
| --- | ---: |
| MLM | 6307 |
| RTD | 800 |
| connective | 588 |
| definiteness | 620 |
| collocation | 435 |
| grammar_minpair | 362 |
| semantic_cloze_ranking | 388 |

## Semantic Cloze Task

The added task is an MLM-head pairwise ranking objective. Each example contains one
`[MASK]`, one good single-token completion, and one bad single-token completion. The
loss is `softplus(-(score_good - score_bad))`, where scores are MLM log-probabilities
at the masked position. This keeps the experiment compatible with the official MLM
evaluation path and avoids adding a classifier head.

The generated semantic cloze data has 1000 examples across affordance, animate agency,
part-whole, physical-property, and typical-location templates. It does not use official
EWoK or BLiMP items.

## Results

| System | BLiMP | BLiMP Supplement | EWoK | Entity Tracking | Reading Eye | Reading SPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal_bert_mlm_final | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| multitask_repair_blmpair_v1_best | 54.80 | 51.60 | 51.00 | 27.54 | 8.02 | 3.05 |
| multitask_repair_blmpair_v1_final | 54.81 | 52.40 | 50.73 | 25.99 | 7.96 | 3.11 |
| multitask_ewok_plausibility_minimal_best | 55.46 | 56.40 | 49.64 | 37.10 | 7.87 | 3.22 |
| multitask_ewok_plausibility_minimal_final | 55.70 | 54.00 | 49.55 | 35.81 | 7.79 | 3.25 |
| multitask_mlm_pair_ranking_minimal_best | 55.95 | 47.20 | 49.55 | 33.50 | 8.73 | 3.67 |
| multitask_mlm_pair_ranking_minimal_final | 56.45 | 50.40 | 49.18 | 32.93 | 8.45 | 3.56 |
| multitask_ewok_semantic_cloze_v1_best | 54.10 | 57.60 | 50.73 | 36.61 | 7.56 | 3.11 |
| multitask_ewok_semantic_cloze_v1_final | 55.26 | 54.40 | 52.00 | 27.44 | 8.18 | 3.71 |

## Interpretation

This is a positive result for the target metric. The final semantic-cloze checkpoint
reaches EWoK 52.00, which is +1.45 over the MLM baseline and +1.00 over the previous
best EWoK result in this isolated branch. It also gives the best self-paced reading
score in the comparison, SPR 3.71.

The trade-off is that entity tracking drops from the semantic-cloze best checkpoint
36.61 to 27.44 at the final checkpoint. The best checkpoint is stronger for supplement
and entity tracking, while the final checkpoint is clearly better for EWoK and SPR.

## Next Step

Keep this line of work. The most useful next experiment is a small sweep around the
semantic-cloze weight and schedule, for example 2.5%, 5%, and 7.5%, while preserving
the no-official-EWoK/no-classifier constraint. The result suggests the semantic signal
needs enough late training to transfer to EWoK, so evaluating both best-validation and
final checkpoints remains important.
