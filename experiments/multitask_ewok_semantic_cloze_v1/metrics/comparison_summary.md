# Semantic Cloze Ranking EWoK Experiment

## Question

Can a small MLM-head semantic cloze ranking auxiliary task improve EWoK without using official EWoK or BLiMP data, external datasets, or a submission-time classifier head?

## Setup

- Branch: `multitask-semantic-cloze-clean`
- Experiment: `multitask_ewok_semantic_cloze_v1`
- Model: compact BERT-style MLM, 11.2M parameters
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

The added task is an MLM-head pairwise ranking objective. Each example contains one `[MASK]`, one good single-token completion, and one bad single-token completion. The loss is `softplus(-(score_good - score_bad))`, where scores are MLM log-probabilities at the masked position.

The generated semantic cloze data has 1000 examples across affordance, animate agency, part-whole, physical-property, and typical-location templates. It does not use official EWoK or BLiMP items.

## Fast Eval Results

| System | BLiMP | BLiMP Supplement | EWoK | Entity Tracking | Reading Eye | Reading SPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| normal_bert_mlm_final | 64.33 | 54.00 | 50.55 | 15.56 | 2.47 | 2.42 |
| multitask_ewok_semantic_cloze_v1_best | 54.10 | 57.60 | 50.73 | 36.61 | 7.56 | 3.11 |
| multitask_ewok_semantic_cloze_v1_final | 55.26 | 54.40 | 52.00 | 27.44 | 8.18 | 3.71 |

## Interpretation

The final semantic-cloze checkpoint is the selected system because it reaches EWoK 52.00, a +1.45 gain over the MLM baseline, and gives the strongest reading scores among the retained systems. The trade-off is lower BLiMP and lower entity tracking than the semantic-cloze best-validation checkpoint.
